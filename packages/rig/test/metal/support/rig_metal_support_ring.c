/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's ring driven by hand: the commits of a submitter and the
   completions of Metal's handler, in whatever order the suite draws. The
   commit numbered k (from 0) takes the value after the last one taken when
   it ends a submission, writes its times at k's cells, and completes with
   the times 10k+1 and 10k+2. Every stub holds the runtime: a commit, the
   one call that can wait, is made only when a slot is free. */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_metal_ring.h"

#define most 4096 /* commits of a ring */

struct test_ring {
  struct rig_metal_ring ring;
  struct rig_metal_slot slots[8];
  uint64_t word, values;
  int commits;
  int commit_of[8]; /* the commit each slot holds */
  uint64_t times[most][2];
};

#define Ring_val(v) (*(struct test_ring **)Data_custom_val(v))

static void finalize_ring(value v) { free(Ring_val(v)); }

static struct custom_operations ring_ops = {
    "rig_metal_test.ring",   finalize_ring,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

value rig_metal_test_ring(value v_n) {
  CAMLparam1(v_n);
  CAMLlocal1(v);
  int n = Int_val(v_n);
  if (n < 1 || n > 8) caml_invalid_argument("Rig_metal_support.ring");
  struct test_ring *t = calloc(1, sizeof *t);
  if (t == NULL) caml_raise_out_of_memory();
  v = caml_alloc_custom(&ring_ops, sizeof t, 0, 1);
  Ring_val(v) = t;
  rig_metal_ring_init(&t->ring, t->slots, n, &t->word);
  CAMLreturn(v);
}

value rig_metal_test_commit(value v_ring, value v_last) {
  struct test_ring *t = Ring_val(v_ring);
  if (t->commits == most) caml_invalid_argument("Rig_metal_support.commit");
  int k = t->commits++, i = rig_metal_ring_take(&t->ring);
  struct rig_metal_slot *s = &t->ring.slots[i];
  s->v = Bool_val(v_last) ? ++t->values : 0;
  s->start = &t->times[k][0];
  s->end = &t->times[k][1];
  t->commit_of[i] = k;
  return Val_int(i);
}

value rig_metal_test_complete(value v_ring, value v_slot, value v_failed) {
  struct test_ring *t = Ring_val(v_ring);
  uint64_t k = (uint64_t)t->commit_of[Int_val(v_slot)];
  char why[32];
  snprintf(why, sizeof why, "command buffer %d failed", (int)k);
  rig_metal_ring_complete(&t->ring, Int_val(v_slot),
                             Bool_val(v_failed) ? why : NULL, 10 * k + 1,
                             10 * k + 2);
  return Val_unit;
}

value rig_metal_test_word(value v_ring) {
  return Val_long(
      (intnat)__atomic_load_n(&Ring_val(v_ring)->word, __ATOMIC_ACQUIRE));
}

/* The times of commit [k]. */
value rig_metal_test_times(value v_ring, value v_k) {
  CAMLparam2(v_ring, v_k);
  CAMLlocal1(v);
  uint64_t *times = Ring_val(v_ring)->times[Int_val(v_k)];
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_long((intnat)times[0]));
  Store_field(v, 1, Val_long((intnat)times[1]));
  CAMLreturn(v);
}

static value failure_option(const char *failure) {
  CAMLparam0();
  CAMLlocal1(v);
  if (failure == NULL) CAMLreturn(Val_none);
  v = caml_copy_string(failure);
  CAMLreturn(caml_alloc_some(v));
}

value rig_metal_test_failure(value v_ring) {
  return failure_option(rig_metal_ring_failure(&Ring_val(v_ring)->ring));
}

/* What sleep answers once the word changed, a failure is recorded, or at
   once. */
value rig_metal_test_sleep(value v_ring) {
  struct test_ring *t = Ring_val(v_ring);
  return failure_option(rig_metal_ring_sleep(&t->ring, t->word, 0));
}

value rig_metal_test_stop(value v_ring) {
  struct test_ring *t = Ring_val(v_ring);
  return Val_bool(rig_metal_ring_stop(&t->ring, t->values));
}
