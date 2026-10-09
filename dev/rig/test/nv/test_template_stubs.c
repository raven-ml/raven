/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device state's template filled as the writer fills it, with no GPU. */

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/mlvalues.h>

#include "rig_nv_stubs.h"

/* Template [v_t] of the state [v_self], filled with the arguments: its
   words as little-endian bytes on a little-endian host. */
value rig_nv_test_fill(value v_self, value v_t, value v_a0, value v_a1,
                       value v_a2) {
  const struct device *d = (const struct device *)Long_val(v_self);
  uint64_t args[3] = {(uint64_t)Int64_val(v_a0), (uint64_t)Int64_val(v_a1),
                      (uint64_t)Int64_val(v_a2)};
  uint32_t w[TEMPLATE_WORDS];
  int n = rig_nv_fill(&d->t[Int_val(v_t)], args, w);
  value s = caml_alloc_string(4 * (mlsize_t)n);
  memcpy(Bytes_val(s), w, 4 * (size_t)n);
  return s;
}

/* The structure qmd[0][0] of the launch [v_launch], filled with the values
   [v_values], an int64 array. */
value rig_nv_test_fill_structure(value v_launch, value v_values) {
  const struct launch *l = (const struct launch *)Nativeint_val(v_launch);
  uint64_t values[VALUES];
  for (int i = 0; i < VALUES; i++)
    values[i] = (uint64_t)Int64_val(Field(v_values, i));
  uint8_t w[STRUCTURE_BYTES];
  rig_nv_fill_structure(&l->qmd[0][0], values, w);
  value s = caml_alloc_string(l->qmd[0][0].nbytes);
  memcpy(Bytes_val(s), w, l->qmd[0][0].nbytes);
  return s;
}
