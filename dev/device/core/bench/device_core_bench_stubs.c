/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The driver alone: a device's room and submit entries called directly, with
   a fill that adds 1 to a word of its own, as the core calls them for a
   submission of no part or of that fill. Nothing here releases the runtime:
   the floor is a driver whose calls return at once. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>

#include "nx_edge.h"

struct floor {
  void *self;
  nx_room_fn *room;
  nx_submit_fn *submit;
  uint64_t v;
  uint64_t word;
  uint64_t handle;
  struct nx_part fill;
};

#define Floor_val(v) ((struct floor *)Nativeint_val(v))

value device_core_bench_floor_new(value v_self, value v_room, value v_submit,
                                  value v_fill) {
  struct floor *f = calloc(1, sizeof *f);
  if (f == NULL) caml_raise_out_of_memory();
  f->self = (void *)Nativeint_val(v_self);
  f->room = (nx_room_fn *)Nativeint_val(v_room);
  f->submit = (nx_submit_fn *)Nativeint_val(v_submit);
  f->fill.fill = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  f->fill.arg = &f->word;
  f->handle = (uint64_t)(uintptr_t)&f->word;
  return caml_copy_nativeint((intnat)f);
}

/* Hands the device the next value with [v_parts] parts, 0 or the fill, as
   one room check and one submit. */
value device_core_bench_floor_submit(value v_f, value v_parts) {
  struct floor *f = Floor_val(v_f);
  int n = Int_val(v_parts);
  const char *failure = NULL;
  if (f->room(f->self, &f->fill, n) != NX_FITS) abort();
  if (f->submit(f->self, ++f->v, NULL, 0, &f->fill, n, &f->handle, n,
                &failure) != NX_OK)
    abort();
  return Val_unit;
}
