/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Patterns of bytes in host memory, written and checked in C, and the
   driver's C entries. Every stub holds the runtime: none blocks. */

#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "rig_nv.h"

#define Ptr_val(v) ((void *)Long_val(v))

/* Byte [i] of the pattern of [seed]. */
static uint8_t pattern_byte(uint64_t seed, uint64_t i) {
  uint64_t h = (i + (seed << 40)) * 0x9E3779B97F4A7C1ULL;
  return (uint8_t)((h ^ (h >> 29)) >> 17);
}

value rig_nv_test_pattern(value v_p, value v_n, value v_seed) {
  uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++) p[i] = pattern_byte(seed, i);
  return Val_unit;
}

value rig_nv_test_mismatch(value v_p, value v_n, value v_seed) {
  const uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++)
    if (p[i] != pattern_byte(seed, i)) return Val_long(i);
  return Val_long(-1);
}

/* Hands the device whose C state is [v_self] the parts [v_parts], each
   [(queue, words)], as the value [v_v] with no waits, through its room
   check and submit. Raises [Failure] unless the parts fit and the submit
   succeeds. */
value rig_nv_test_submit(value v_self, value v_v, value v_parts) {
  void *self = (void *)Nativeint_val(v_self);
  int n = (int)Wosize_val(v_parts);
  size_t nwords = 0;
  for (int i = 0; i < n; i++)
    nwords += Wosize_val(Field(Field(v_parts, i), 1));
  struct rig_part *p = calloc((size_t)n + 1, sizeof *p);
  void *scratch = calloc((size_t)n + 1, RIG_SCRATCH);
  uint32_t *words = calloc(nwords + 1, sizeof *words);
  if (p == NULL || scratch == NULL || words == NULL) {
    free(p);
    free(scratch);
    free(words);
    caml_raise_out_of_memory();
  }
  uint32_t *w = words;
  for (int i = 0; i < n; i++) {
    value k = Field(v_parts, i), v_words = Field(k, 1);
    size_t m = Wosize_val(v_words);
    for (size_t j = 0; j < m; j++) w[j] = (uint32_t)Long_val(Field(v_words, j));
    p[i] = (struct rig_part){
        .queue = Int_val(Field(k, 0)), .kind = RIG_WORDS, .words = {w, m}};
    w += m;
  }
  const char *failure = NULL;
  int room = rig_nv_room(self, p, n, scratch, NULL);
  int r = room != RIG_FITS ? RIG_FAILED
                           : rig_nv_submit(self, (uint64_t)Long_val(v_v), NULL,
                                           0, p, n, scratch, NULL, NULL, 0,
                                           NULL, &failure);
  free(p);
  free(scratch);
  free(words);
  if (room != RIG_FITS)
    caml_failwith("rig_nv_test_submit: the parts do not fit");
  if (r == RIG_FAILED) caml_failwith(failure != NULL ? failure : "");
  return Val_unit;
}
