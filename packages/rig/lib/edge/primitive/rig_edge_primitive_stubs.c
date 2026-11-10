/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The OCaml primitives of rig_edge.h, each a call of the C function it
   names. */

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "rig_edge.h"

/* Clock and words */

intnat caml_rig_edge_now(value unit) {
  (void)unit;
  return (intnat)rig_now_ns();
}

value caml_rig_edge_now_byte(value unit) {
  return Val_long(caml_rig_edge_now(unit));
}

/* rig's array of words: its address and its bytes. */
value caml_rig_edge_words(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_long((intnat)rig_words()));
  Store_field(v, 1, Val_long((intnat)RIG_WORDS_BYTES));
  CAMLreturn(v);
}

/* Locks */

intnat caml_rig_edge_holds(value unit) {
  (void)unit;
  return rig_holds();
}

value caml_rig_edge_holds_byte(value unit) {
  return Val_long(caml_rig_edge_holds(unit));
}

value caml_rig_edge_hold(value unit) {
  (void)unit;
  rig_hold();
  return Val_unit;
}

value caml_rig_edge_unhold(value unit) {
  (void)unit;
  rig_unhold();
  return Val_unit;
}

/* Signals */

value caml_rig_edge_mask(value unit) {
  (void)unit;
  rig_mask();
  return Val_unit;
}

/* The outermost end also leaves a blocking section, as Unix.sigprocmask
   does: a signal the runtime recorded while the thread blocked it was
   skipped by the poll points since, each clearing what makes the next one
   look, and the leave sets that again while any is recorded. The external
   is not noalloc, so its return arms the next poll point to run the
   handler. A wait's end needs none of this: its blocking step leaves a
   blocking section, and caml_rig_edge_block runs the handlers. */
static void rearm(void) {
  caml_enter_blocking_section_no_pending();
  caml_leave_blocking_section();
}

value caml_rig_edge_unmask(value unit) {
  (void)unit;
  if (rig_unmask()) rearm();
  return Val_unit;
}

value caml_rig_edge_unblock(value unit) {
  (void)unit;
  rig_unblock();
  return Val_unit;
}

/* A wait's end runs the handlers of the signals that came during it, before
   it blocks them again, so that their exceptions end the wait. */
value caml_rig_edge_block(value unit) {
  (void)unit;
  caml_result r = caml_process_pending_actions_res();
  rig_block();
  return caml_get_value_or_raise(r);
}

value caml_rig_edge_shield(value unit) {
  (void)unit;
  rig_shield();
  return Val_unit;
}

value caml_rig_edge_unshield(value unit) {
  (void)unit;
  if (rig_unshield()) rearm();
  return Val_unit;
}
