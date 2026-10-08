/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the NV bench: each row's submissions through
   rig_nv_room and rig_nv_submit, called from C in a loop on the
   device the row opened, then a spin until the device's timeline word
   holds the last value. The submission is the least sequence the hardware
   needs (the driver's words in a segment, a ring entry, GP_PUT, the
   doorbell), so a row's distance to its floor is what rig and the
   driver's OCaml side add.

   A failing call raises Failure. Every stub holds the runtime. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <rig_nv.h>

#define Ptr_val(v) ((void *)Nativeint_val(v))

/* The device, its timeline word, and the last value given to it. */
static void *self;
static _Atomic uint64_t *word;
static uint64_t last;

value rig_nv_bench_start(value v_self, value v_word, value v_last) {
  self = Ptr_val(v_self);
  word = (_Atomic uint64_t *)Long_val(v_word);
  last = (uint64_t)Long_val(v_last);
  return Val_unit;
}

static void submit(const struct rig_wait *waits, int nwaits,
                   const struct rig_part *parts, int nparts) {
  const char *failure = NULL;
  if (rig_nv_room(self, parts, nparts) != RIG_FITS)
    caml_failwith("rig_nv_room: the parts do not fit");
  if (rig_nv_submit(self, ++last, waits, nwaits, parts, nparts, NULL, 0,
                       &failure) != RIG_OK)
    caml_failwith(failure);
}

static void spin(void) {
  while (atomic_load_explicit(word, memory_order_acquire) < last) {
  }
}

/* [v_k] submissions of no parts, then the wait for the last. */
value rig_nv_bench_release(value v_k) {
  for (long i = 0; i < Long_val(v_k); i++) submit(NULL, 0, NULL, 0);
  spin();
  return Val_unit;
}

/* A part of no entries on COPY:0 at even values, no part at odd ones, so
   that the release changes channel at each value. */
value rig_nv_bench_switch(value unit) {
  struct rig_part copy = {.queue = 1};
  (void)unit;
  submit(NULL, 0, &copy, (last & 1) == 0);
  spin();
  return Val_unit;
}

/* [v_n] waits for the 64-bit word at [v_at] to reach 1, then a release. */
value rig_nv_bench_waits(value v_at, value v_n) {
  struct rig_wait waits[16];
  int n = Int_val(v_n);
  if (n > 16) caml_invalid_argument("rig_nv_bench_waits: more than 16");
  for (int i = 0; i < n; i++)
    waits[i] = (struct rig_wait){(uint64_t)Long_val(v_at), 1, RIG_WORD};
  submit(waits, n, NULL, 0);
  spin();
  return Val_unit;
}

/* The ring entry of two words [v_lo], [v_hi] on COMPUTE:0. */
value rig_nv_bench_entry(value v_lo, value v_hi) {
  uint32_t words[2] = {(uint32_t)Long_val(v_lo), (uint32_t)Long_val(v_hi)};
  struct rig_part p = {.queue = 0, .words = words, .n = 2};
  submit(NULL, 0, &p, 1);
  spin();
  return Val_unit;
}

/* A copy of [v_n] bytes from the GPU address [v_src] to [v_dst] on
   COPY:0. */
value rig_nv_bench_copy(value v_dst, value v_src, value v_n) {
  struct rig_part p = {.queue = 1,
                      .copy_dst = (uint64_t)Long_val(v_dst),
                      .copy_src = (uint64_t)Long_val(v_src),
                      .copy_bytes = (uint64_t)Long_val(v_n)};
  submit(NULL, 0, &p, 1);
  spin();
  return Val_unit;
}
