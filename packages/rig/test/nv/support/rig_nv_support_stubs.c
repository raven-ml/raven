/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Patterns of bytes in host memory, written and checked in C, and the
   driver's C entries. Every stub holds the runtime: none blocks. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "rig_edge.h"

#define Ptr_val(v) ((void *)Long_val(v))

/* Bytes [8k] to [8k + 7] of the pattern of [seed], in memory order: a hash
   of [k], each byte xored with [seed]'s low byte, so that the patterns of
   seeds that differ modulo 256 differ in every byte. A word at a time, a
   pattern of the 2 GiB a copy engine moves at once is written or checked in
   about 0.2 s, where a byte at a time took 0.75 s. */
static uint64_t pattern_word(uint64_t seed, uint64_t k) {
  uint64_t h = k * 0x9E3779B97F4A7C1ULL;
  return h ^ (h >> 29) ^ (UINT64_C(0x0101010101010101) * (seed & 0xFF));
}

value rig_nv_test_pattern(value v_p, value v_n, value v_seed) {
  uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  uint64_t k = 0;
  for (; 8 * k + 8 <= n; k++) {
    uint64_t w = pattern_word(seed, k);
    memcpy(p + (8 * k), &w, 8);
  }
  uint64_t w = pattern_word(seed, k);
  memcpy(p + (8 * k), &w, n - (8 * k));
  return Val_unit;
}

/* The index of the first of the [m] bytes at [p] that differs from the
   first [m] bytes of the word [w], or [-1]. */
static int64_t first_differing(const uint8_t *p, uint64_t w, uint64_t m) {
  uint8_t b[8];
  memcpy(b, &w, 8);
  for (uint64_t j = 0; j < m; j++)
    if (p[j] != b[j]) return (int64_t)j;
  return -1;
}

value rig_nv_test_mismatch(value v_p, value v_n, value v_seed) {
  const uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  uint64_t k = 0;
  for (; 8 * k + 8 <= n; k++) {
    uint64_t got, w = pattern_word(seed, k);
    memcpy(&got, p + (8 * k), 8);
    if (got != w)
      return Val_long((int64_t)(8 * k) + first_differing(p + (8 * k), w, 8));
  }
  int64_t j = first_differing(p + (8 * k), pattern_word(seed, k), n - (8 * k));
  return Val_long(j < 0 ? -1 : (int64_t)(8 * k) + j);
}

/* The parts [v_parts], each [(queue, words)], in C, with their scratch
   and words, which [parts_free] frees. Raises [Out_of_memory]. */
struct parts {
  int n;
  struct rig_part *p;
  void *scratch;
  uint32_t *words;
};

static void parts_free(struct parts *x) {
  free(x->p);
  free(x->scratch);
  free(x->words);
}

static struct parts parts_of(value v_parts) {
  struct parts x = {.n = (int)Wosize_val(v_parts)};
  size_t nwords = 0;
  for (int i = 0; i < x.n; i++)
    nwords += Wosize_val(Field(Field(v_parts, i), 1));
  x.p = calloc((size_t)x.n + 1, sizeof *x.p);
  x.scratch = calloc((size_t)x.n + 1, RIG_SCRATCH);
  x.words = calloc(nwords + 1, sizeof *x.words);
  if (x.p == NULL || x.scratch == NULL || x.words == NULL) {
    parts_free(&x);
    caml_raise_out_of_memory();
  }
  uint32_t *w = x.words;
  for (int i = 0; i < x.n; i++) {
    value k = Field(v_parts, i), v_words = Field(k, 1);
    size_t m = Wosize_val(v_words);
    for (size_t j = 0; j < m; j++) w[j] = (uint32_t)Long_val(Field(v_words, j));
    x.p[i] = (struct rig_part){
        .queue = Int_val(Field(k, 0)), .kind = RIG_WORDS, .words = {w, m}};
    w += m;
  }
  return x;
}

/* What the room check of the device whose C state is [v_self] answers for
   the parts [v_parts], each [(queue, words)]: rig_edge.h's RIG_FITS,
   RIG_LATER, RIG_NEVER or RIG_GROW. */
value rig_nv_test_room(value v_self, value v_parts) {
  void *self = (void *)Nativeint_val(v_self);
  struct parts x = parts_of(v_parts);
  int room = rig_driver_of(self)->room(self, 0, x.p, x.n, x.scratch, NULL);
  parts_free(&x);
  return Val_int(room);
}

/* Calls the trim of the device whose C state is [v_self], as rig does at a
   free_cache: nothing where the trim is NULL. */
value rig_nv_test_trim(value v_self) {
  void *self = (void *)Nativeint_val(v_self);
  const struct rig_driver *d = rig_driver_of(self);
  if (d->trim != NULL) d->trim(self);
  return Val_unit;
}

/* Hands the device whose C state is [v_self] the parts [v_parts], each
   [(queue, words)], as the value [v_v] with no waits, through its room
   check and submit. Raises [Failure] unless the parts fit and the submit
   succeeds. */
value rig_nv_test_submit(value v_self, value v_v, value v_parts) {
  void *self = (void *)Nativeint_val(v_self);
  const struct rig_driver *d = rig_driver_of(self);
  struct parts x = parts_of(v_parts);
  const char *failure = NULL;
  int room = d->room(self, 0, x.p, x.n, x.scratch, NULL);
  int r = room != RIG_FITS
              ? RIG_FAILED
              : d->submit(self, (uint64_t)Long_val(v_v), NULL, 0, x.p, x.n,
                          x.scratch, NULL, NULL, 0, NULL, &failure);
  parts_free(&x);
  if (room != RIG_FITS)
    caml_failwith("rig_nv_test_submit: the parts do not fit");
  if (r == RIG_FAILED) caml_failwith(failure != NULL ? failure : "");
  return Val_unit;
}
