/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The address of a bigarray's first byte, which the test's path answers as
   memory it mapped. */

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/mlvalues.h>

value rig_mlx5_test_address(value v_b) {
  return Val_long((intnat)Caml_ba_data_val(v_b));
}
