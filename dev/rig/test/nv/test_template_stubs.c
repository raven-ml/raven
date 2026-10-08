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
