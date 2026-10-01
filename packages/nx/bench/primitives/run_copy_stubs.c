/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The run-copy twin of Nx.Ragged.take: a gather of variable-length runs of
   bytes, one loop for the lengths and one memcpy per run. */

#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

/* The number of bytes the runs at [perm] hold. */
value bench_run_copy_total(value v_offsets, value v_perm) {
  const int64_t *off = Caml_ba_data_val(v_offsets);
  const int64_t *perm = Caml_ba_data_val(v_perm);
  intnat k = Caml_ba_array_val(v_perm)->dim[0];
  int64_t total = 0;
  for (intnat j = 0; j < k; j++) total += off[perm[j] + 1] - off[perm[j]];
  return Val_long(total);
}

/* Copies the runs at [perm] one after the other into [out], and their offsets
   into [out_offsets]. */
value bench_run_copy(value v_offsets, value v_values, value v_perm,
                     value v_out_offsets, value v_out) {
  const int64_t *off = Caml_ba_data_val(v_offsets);
  const uint8_t *values = Caml_ba_data_val(v_values);
  const int64_t *perm = Caml_ba_data_val(v_perm);
  int64_t *out_off = Caml_ba_data_val(v_out_offsets);
  uint8_t *out = Caml_ba_data_val(v_out);
  intnat k = Caml_ba_array_val(v_perm)->dim[0];
  int64_t pos = 0;
  for (intnat j = 0; j < k; j++) {
    int64_t lo = off[perm[j]], len = off[perm[j] + 1] - lo;
    out_off[j] = pos;
    memcpy(out + pos, values + lo, (size_t)len);
    pos += len;
  }
  out_off[k] = pos;
  return Val_unit;
}
