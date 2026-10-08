/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The driver alone: a device's room and submit entries called directly, with
   a fill that adds 1 to a word of its own, as the core calls them for a
   submission of no part or of that fill, naming the fill's word or the
   handles it was given. A submit holds the runtime, even for a driver whose
   submit may block, except a turn's: it takes the floor's mutex as the core
   takes a device's turn, by try-lock and otherwise with the runtime
   released. */

#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/threads.h>

#ifdef _WIN32
#include <windows.h>
typedef SRWLOCK turn;
static void turn_init(turn *t) { InitializeSRWLock(t); }
static int turn_try(turn *t) { return TryAcquireSRWLockExclusive(t); }
static void turn_lock(turn *t) { AcquireSRWLockExclusive(t); }
static void turn_unlock(turn *t) { ReleaseSRWLockExclusive(t); }
#else
#include <pthread.h>
typedef pthread_mutex_t turn;
static void turn_init(turn *t) { pthread_mutex_init(t, NULL); }
static int turn_try(turn *t) { return pthread_mutex_trylock(t) == 0; }
static void turn_lock(turn *t) { pthread_mutex_lock(t); }
static void turn_unlock(turn *t) { pthread_mutex_unlock(t); }
#endif

#include "rig_edge.h"

struct floor {
  void *self;
  rig_room_fn *room;
  rig_submit_fn *submit;
  uint64_t v;
  uint64_t word;
  uint64_t handle;
  uint64_t *handles; /* NULL: the fill's word, for each part */
  int nhandles;
  turn *turn;
  struct rig_part fill;
};

#define Floor_val(v) ((struct floor *)Nativeint_val(v))

value rig_bench_floor_new(value v_self, value v_room, value v_submit,
                                  value v_fill) {
  struct floor *f = calloc(1, sizeof *f);
  if (f == NULL) caml_raise_out_of_memory();
  f->self = (void *)Nativeint_val(v_self);
  f->room = (rig_room_fn *)Nativeint_val(v_room);
  f->submit = (rig_submit_fn *)Nativeint_val(v_submit);
  f->fill.fill = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  f->fill.arg = &f->word;
  f->handle = (uint64_t)(uintptr_t)&f->word;
  f->turn = malloc(sizeof *f->turn);
  if (f->turn == NULL) caml_raise_out_of_memory();
  turn_init(f->turn);
  return caml_copy_nativeint((intnat)f);
}

/* Makes [v_g] take [v_f]'s turn. */
value rig_bench_floor_share(value v_f, value v_g) {
  Floor_val(v_g)->turn = Floor_val(v_f)->turn;
  return Val_unit;
}

/* Makes each submit name the handles [v_handles]. */
value rig_bench_floor_handles(value v_f, value v_handles) {
  struct floor *f = Floor_val(v_f);
  int n = (int)Wosize_val(v_handles);
  uint64_t *h = malloc((size_t)(n == 0 ? 1 : n) * sizeof *h);
  if (h == NULL) caml_raise_out_of_memory();
  for (int i = 0; i < n; i++) h[i] = (uint64_t)Nativeint_val(Field(v_handles, i));
  free(f->handles);
  f->handles = h;
  f->nhandles = n;
  return Val_unit;
}

static void submit(struct floor *f, int n) {
  const char *failure = NULL;
  const uint64_t *h = f->handles != NULL ? f->handles : &f->handle;
  int nh = f->handles != NULL ? f->nhandles : n;
  if (f->room(f->self, &f->fill, n) != RIG_FITS) abort();
  if (f->submit(f->self, ++f->v, NULL, 0, &f->fill, n, h, nh, &failure) !=
      RIG_OK)
    abort();
}

/* Hands the device the next value with [v_parts] parts, 0 or the fill, as
   one room check and one submit. */
value rig_bench_floor_submit(value v_f, value v_parts) {
  submit(Floor_val(v_f), Int_val(v_parts));
  return Val_unit;
}

/* As [rig_bench_floor_submit], under the floor's turn. It releases the
   runtime while it waits for the turn. */
value rig_bench_floor_turn_submit(value v_f, value v_parts) {
  struct floor *f = Floor_val(v_f);
  int released = 0;
  if (!turn_try(f->turn)) {
    caml_enter_blocking_section_no_pending();
    released = 1;
    turn_lock(f->turn);
  }
  submit(f, Int_val(v_parts));
  turn_unlock(f->turn);
  if (released) caml_leave_blocking_section();
  return Val_unit;
}
