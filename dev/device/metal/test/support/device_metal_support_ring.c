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

#include "device_metal_ring.h"

#define most 4096 /* commits and releases of a ring */

struct release {
  struct device_metal_release node;
  int id;
  struct test_ring *ring;
};

struct test_ring {
  struct device_metal_ring ring;
  struct device_metal_slot slots[8];
  uint64_t word, values;
  int commits, releases, nran;
  int commit_of[8]; /* the commit each slot holds */
  uint64_t times[most][2];
  struct release release[most];
  int ran[most];
};

#define Ring_val(v) (*(struct test_ring **)Data_custom_val(v))

static void finalize_ring(value v) { free(Ring_val(v)); }

static struct custom_operations ring_ops = {
    "device_metal_test.ring",   finalize_ring,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default,
};

value device_metal_test_ring(value v_n) {
  CAMLparam1(v_n);
  CAMLlocal1(v);
  int n = Int_val(v_n);
  if (n < 1 || n > 8) caml_invalid_argument("Device_metal_support.ring");
  struct test_ring *t = calloc(1, sizeof *t);
  if (t == NULL) caml_raise_out_of_memory();
  v = caml_alloc_custom(&ring_ops, sizeof t, 0, 1);
  Ring_val(v) = t;
  device_metal_ring_init(&t->ring, t->slots, n, &t->word);
  CAMLreturn(v);
}

value device_metal_test_commit(value v_ring, value v_last) {
  struct test_ring *t = Ring_val(v_ring);
  if (t->commits == most) caml_invalid_argument("Device_metal_support.commit");
  int k = t->commits++, i = device_metal_ring_take(&t->ring);
  struct device_metal_slot *s = &t->ring.slots[i];
  s->v = Bool_val(v_last) ? ++t->values : 0;
  s->start = &t->times[k][0];
  s->end = &t->times[k][1];
  t->commit_of[i] = k;
  return Val_int(i);
}

value device_metal_test_complete(value v_ring, value v_slot, value v_failed) {
  struct test_ring *t = Ring_val(v_ring);
  uint64_t k = (uint64_t)t->commit_of[Int_val(v_slot)];
  char why[32];
  snprintf(why, sizeof why, "command buffer %d failed", (int)k);
  device_metal_ring_complete(&t->ring, Int_val(v_slot),
                             Bool_val(v_failed) ? why : NULL, 10 * k + 1,
                             10 * k + 2);
  return Val_unit;
}

static void run_release(struct device_metal_release *node) {
  struct release *r = (struct release *)node;
  r->ring->ran[r->ring->nran++] = r->id;
}

value device_metal_test_defer(value v_ring) {
  struct test_ring *t = Ring_val(v_ring);
  if (t->releases == most) caml_invalid_argument("Device_metal_support.defer");
  int id = t->releases++;
  t->release[id] = (struct release){{run_release, NULL}, id, t};
  device_metal_ring_defer(&t->ring, &t->release[id].node);
  return Val_int(id);
}

value device_metal_test_word(value v_ring) {
  return Val_long(
      (intnat)__atomic_load_n(&Ring_val(v_ring)->word, __ATOMIC_ACQUIRE));
}

value device_metal_test_ran(value v_ring) {
  CAMLparam1(v_ring);
  CAMLlocal1(v);
  struct test_ring *t = Ring_val(v_ring);
  v = caml_alloc_tuple(t->nran);
  for (int i = 0; i < t->nran; i++) Store_field(v, i, Val_int(t->ran[i]));
  CAMLreturn(t->nran ? v : Atom(0));
}

/* The times of commit [k]. */
value device_metal_test_times(value v_ring, value v_k) {
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

value device_metal_test_failure(value v_ring) {
  return failure_option(device_metal_ring_failure(&Ring_val(v_ring)->ring));
}

/* What sleep answers once the word changed, a failure is recorded, or at
   once. */
value device_metal_test_sleep(value v_ring) {
  struct test_ring *t = Ring_val(v_ring);
  return failure_option(device_metal_ring_sleep(&t->ring, t->word, 0));
}

value device_metal_test_stop(value v_ring) {
  struct test_ring *t = Ring_val(v_ring);
  return Val_bool(device_metal_ring_stop(&t->ring, t->values));
}
