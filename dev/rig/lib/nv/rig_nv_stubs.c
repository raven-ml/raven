/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's C state, for the OCaml side: made at open from memory the
   path gave, and read by the wait loop.

   A device is named by the address of its state, an OCaml int. Every stub
   holds the runtime, except those whose comment says they release it,
   which do so without running pending signal handlers, so no OCaml code
   runs between the call's answer and its caller. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "rig_nv_stubs.h"

/* The platform */

/* The wait loop reads the word and the notifiers each POLL_MS
   milliseconds: a fault is found within POLL_MS of the RM's report, and the
   wake latency falls on waits rig has already spun on, which are
   long. */
#define POLL_MS 1

#if defined(_WIN32)
#include <windows.h>
static int64_t now_ms(void) { return (int64_t)GetTickCount64(); }
static void poll_pause(void) { Sleep(POLL_MS); }
#else
#include <time.h>
static int64_t now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
static void poll_pause(void) {
  struct timespec t = {0, POLL_MS * 1000000L};
  nanosleep(&t, NULL);
}
#endif

#define Device_val(v) ((struct device *)Long_val(v))
#define Pointer_val(v) ((void *)Long_val(v))

/* Runs [stmt] with the runtime released, running no pending action. */
#define RELEASED(stmt)                                                         \
  do {                                                                         \
    caml_enter_blocking_section_no_pending();                                  \
    stmt;                                                                      \
    caml_leave_blocking_section();                                             \
  } while (0)

/* Making a device */

/* A device whose timeline word is the host's 64-bit word at [v_word], the
   GPU's at [v_word_gpu], followed by its two join words, and whose
   notifications hold their error at [v_info32] and its status at
   [v_status]; or [0] without memory. The state lives for the process. */
value caml_rig_nv_create(value v_word, value v_word_gpu, value v_info32,
                            value v_status) {
  struct device *d = calloc(1, sizeof *d);
  if (d == NULL) return Val_long(0);
  d->word = Pointer_val(v_word);
  d->word_gpu = (uint64_t)Long_val(v_word_gpu);
  d->info32_at = (uint32_t)Long_val(v_info32);
  d->status_at = (uint32_t)Long_val(v_status);
  for (int q = 0; q < CHANNELS; q++) d->ch[q].owes_setup = 1;
  return Val_long((intnat)d);
}

/* The fields of [caml_rig_nv_channel]'s ints. */
enum {
  ch_ring,
  ch_entries,
  ch_gp_put,
  ch_token,
  ch_segments,
  ch_segments_gpu,
  ch_size,
  ch_notifier,
  ch_fields
};

/* Sets channel [v_q] of [v_self] from [v_ints], host addresses and sizes
   as above; [entries] and [size] are powers of two. [true], or [false]
   without memory for its marks. */
value caml_rig_nv_channel(value v_self, value v_q, value v_ints) {
  struct channel *c = &Device_val(v_self)->ch[Int_val(v_q)];
#define F(i) Long_val(Field(v_ints, i))
  c->ring = Pointer_val(Field(v_ints, ch_ring));
  c->entries = (uint64_t)F(ch_entries);
  c->gp_put = Pointer_val(Field(v_ints, ch_gp_put));
  c->token = (uint32_t)F(ch_token);
  c->segments = Pointer_val(Field(v_ints, ch_segments));
  c->segments_gpu = (uint64_t)F(ch_segments_gpu);
  c->size = (uint64_t)F(ch_size);
  c->notifier = Pointer_val(Field(v_ints, ch_notifier));
#undef F
  c->marks = malloc(c->entries * sizeof *c->marks);
  return Val_bool(c->marks != NULL);
}

value caml_rig_nv_doorbell(value v_self, value v_at) {
  Device_val(v_self)->doorbell = Pointer_val(v_at);
  return Val_unit;
}

/* The ints of a hole: its index, slot and width, its number of operations,
   then for each of HOLE_OPS operations its shift and its addend. */
enum { hole_at, hole_slot, hole_wide, hole_nops, hole_ops, hole_fields =
       hole_ops + 2 * HOLE_OPS };

/* Sets template [v_k] to the words [v_words] with the holes [v_holes]. The
   OCaml side checks the counts against TEMPLATE_WORDS and TEMPLATE_HOLES. */
value caml_rig_nv_template(value v_self, value v_k, value v_words,
                              value v_holes) {
  struct template *t = &Device_val(v_self)->t[Int_val(v_k)];
  t->nwords = (int)Wosize_val(v_words);
  for (int i = 0; i < t->nwords; i++)
    t->words[i] = (uint32_t)Long_val(Field(v_words, i));
  t->nholes = (int)(Wosize_val(v_holes) / hole_fields);
  for (int i = 0; i < t->nholes; i++) {
    struct hole *h = &t->holes[i];
#define F(f) Long_val(Field(v_holes, i * hole_fields + (f)))
    h->at = (uint16_t)F(hole_at);
    h->slot = (uint8_t)F(hole_slot);
    h->wide = (uint8_t)F(hole_wide);
    h->nops = (uint8_t)F(hole_nops);
    for (int j = 0; j < HOLE_OPS; j++) {
      h->shift[j] = (uint8_t)F(hole_ops + 2 * j);
      h->n[j] = (uint64_t)F(hole_ops + 2 * j + 1);
    }
#undef F
  }
  return Val_unit;
}

/* An entry is its segment's address plus [v_base] plus its words times
   [v_word]. */
value caml_rig_nv_entry(value v_self, value v_base, value v_word) {
  struct device *d = Device_val(v_self);
  d->entry_base = (uint64_t)Long_val(v_base);
  d->entry_word = (uint64_t)Long_val(v_word);
  return Val_unit;
}

/* The word of the BAR a submission reads while BAR regions are live. */
value caml_rig_nv_bar(value v_self, value v_at) {
  Device_val(v_self)->bar = Pointer_val(v_at);
  return Val_unit;
}

value caml_rig_nv_bar_live(value v_self, value v_delta) {
  atomic_fetch_add(&Device_val(v_self)->bar_live, Long_val(v_delta));
  return Val_unit;
}

/* Zeroes the [v_n] bytes of host memory at [v_at]. */
value caml_rig_nv_zero(value v_at, value v_n) {
  memset(Pointer_val(v_at), 0, (size_t)Long_val(v_n));
  return Val_unit;
}

/* Owed words */

/* Replaces the pending local memory [v_expected] with [v_desired]: [true],
   or [false] if a submission took [v_expected] meanwhile. */
value caml_rig_nv_offer_local(value v_self, value v_expected,
                                 value v_desired) {
  uint64_t expected = (uint64_t)Long_val(v_expected);
  return Val_bool(atomic_compare_exchange_strong(
      &Device_val(v_self)->local, &expected, (uint64_t)Long_val(v_desired)));
}

/* The pending local memory, or [0] if none is pending. */
value caml_rig_nv_pending_local(value v_self) {
  return Val_long(
      atomic_load_explicit(&Device_val(v_self)->local, memory_order_acquire));
}

/* Makes [v_packed] the pending local memory; none is pending. */
value caml_rig_nv_set_local(value v_self, value v_packed) {
  atomic_store_explicit(&Device_val(v_self)->local, (uint64_t)Long_val(v_packed),
                        memory_order_release);
  return Val_unit;
}

value caml_rig_nv_owe_invalidate(value v_self) {
  atomic_store(&Device_val(v_self)->invalidate, 1);
  return Val_unit;
}

/* Timeline */

value caml_rig_nv_signaled(value v_self) {
  return Val_long(
      atomic_load_explicit(Device_val(v_self)->word, memory_order_acquire));
}

/* The error the RM wrote into channel [q]'s notifier, its code and its
   status: [code lsl 16 lor status], or [0] if none. */
static uint64_t notification(const struct device *d, int q) {
  volatile const uint8_t *n = d->ch[q].notifier;
  uint32_t code = *(volatile const uint32_t *)(n + d->info32_at);
  uint16_t status = *(volatile const uint16_t *)(n + d->status_at);
  return (uint64_t)code << 16 | status;
}

value caml_rig_nv_notification(value v_self, value v_q) {
  return Val_long(notification(Device_val(v_self), Int_val(v_q)));
}

/* Waits until the word differs from [v_seen], a channel's notifier holds
   an error, or [v_ms] milliseconds passed, reading both each POLL_MS:
   [true] if a notifier holds an error. Releases the runtime. */
value caml_rig_nv_watch(value v_self, value v_seen, value v_ms) {
  struct device *d = Device_val(v_self);
  uint64_t seen = (uint64_t)Long_val(v_seen);
  int64_t ms = Long_val(v_ms);
  int faulted = 0;
  RELEASED({
    int64_t start = now_ms();
    while (atomic_load_explicit(d->word, memory_order_acquire) == seen) {
      faulted = notification(d, COMPUTE) != 0 || notification(d, COPY) != 0;
      if (faulted || now_ms() - start >= ms) break;
      poll_pause();
    }
  });
  return Val_bool(faulted);
}

value caml_rig_nv_last(value v_self) {
  return Val_long(atomic_load_explicit(&Device_val(v_self)->last,
                                       memory_order_acquire));
}

/* The monotonic clock, in milliseconds. */
value caml_rig_nv_now_ms(value unit) {
  (void)unit;
  return Val_long(now_ms());
}

/* Raises the word to the last value submitted, by compare-and-set: it
   never moves backwards. */
value caml_rig_nv_raise(value v_self) {
  struct device *d = Device_val(v_self);
  uint64_t last = atomic_load_explicit(&d->last, memory_order_relaxed);
  uint64_t w = atomic_load_explicit(d->word, memory_order_acquire);
  while ((int64_t)(w - last) < 0 &&
         !atomic_compare_exchange_weak(d->word, &w, last)) {
  }
  return Val_unit;
}

/* Ends the channels' state once the RM freed them: their marks, and every
   pointer into the memory the path gives back. */
value caml_rig_nv_end(value v_self) {
  struct device *d = Device_val(v_self);
  for (int q = 0; q < CHANNELS; q++) {
    free(d->ch[q].marks);
    memset(&d->ch[q], 0, sizeof d->ch[q]);
  }
  d->doorbell = NULL;
  d->bar = NULL;
  return Val_unit;
}

/* The edge */

/* Assigned to the edge's types, so a signature that drifts from rig_edge.h
   is a compile error. */
static rig_room_fn *const room_entry = rig_nv_room;
static rig_submit_fn *const submit_entry = rig_nv_submit;

value caml_rig_nv_room_entry(value unit) {
  (void)unit;
  return Val_long((intnat)room_entry);
}

value caml_rig_nv_submit_entry(value unit) {
  (void)unit;
  return Val_long((intnat)submit_entry);
}
