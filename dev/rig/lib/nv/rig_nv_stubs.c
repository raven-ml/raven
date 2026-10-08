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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
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

/* Frees the state of a device whose open failed, which nothing holds. */
value caml_rig_nv_destroy(value v_self) {
  struct device *d = Device_val(v_self);
  for (int q = 0; q < CHANNELS; q++) free(d->ch[q].marks);
  free(d);
  return Val_unit;
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

/* Rig_nv_abi.Packet's terms and the words that hold one, by tag: a term
   is the value, a writer's value number (0), or an addition (1) or a right
   shift (2) of a term and a constant; a word is W32 (1) or W64 (2). */
enum { TERM_VALUE, TERM_ADD, TERM_SHIFT };
enum { WORD_W64 = 2 };

static void refuse(const char *why) {
  char msg[96];
  snprintf(msg, sizeof msg, "Rig_nv.make: a template %s", why);
  caml_invalid_argument(msg);
}

/* Sets [h] to the word [v_w] at index [at]: its value number, then its
   operations in the order they apply, the innermost first. */
static void read_hole(struct hole *h, intnat at, value v_w) {
  value t = Field(v_w, 0);
  int n = 0;
  for (value u = t; Tag_val(u) != TERM_VALUE; u = Field(u, 0)) n++;
  if (n > HOLE_OPS) refuse("hole takes too many operations");
  h->at = (uint16_t)at;
  h->wide = Tag_val(v_w) == WORD_W64;
  h->nops = (uint8_t)n;
  for (int i = n - 1; i >= 0; i--, t = Field(t, 0)) {
    if (Tag_val(t) == TERM_ADD) {
      h->shift[i] = 0;
      h->n[i] = (uint64_t)Int64_val(Field(t, 1));
      continue;
    }
    intnat shift = Long_val(Field(t, 1));
    if (shift < 0 || shift > 63) refuse("hole's shift is outside 0 to 63");
    h->shift[i] = (uint8_t)shift;
    h->n[i] = 0;
  }
  intnat slot = Long_val(Field(t, 0));
  if (slot < 0 || slot > 2) refuse("hole reads a value outside 0 to 2");
  h->slot = (uint8_t)slot;
}

/* Sets template [v_k] to the words [v_words], little-endian bytes, with
   the holes [v_holes], the list Rig_nv_abi.Packet.template answers over
   the writer's values 0, 1 and 2. */
value caml_rig_nv_template(value v_self, value v_k, value v_words,
                           value v_holes) {
  struct template *t = &Device_val(v_self)->t[Int_val(v_k)];
  size_t bytes = caml_string_length(v_words);
  if (bytes > sizeof t->words) refuse("exceeds 16 words");
  t->nwords = (int)(bytes / 4);
  memcpy(t->words, String_val(v_words), bytes);
  t->nholes = 0;
  for (value l = v_holes; l != Val_emptylist; l = Field(l, 1)) {
    if (t->nholes == TEMPLATE_HOLES) refuse("has more than 6 holes");
    value h = Field(l, 0);
    read_hole(&t->holes[t->nholes++], Long_val(Field(h, 0)), Field(h, 1));
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
  atomic_store_explicit(&Device_val(v_self)->local,
                        (uint64_t)Long_val(v_packed),
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

/* Each value's hand-over writes its release: a commit does nothing. */
static int rig_nv_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

/* Assigned to the edge's types, so a signature that drifts from rig_edge.h
   is a compile error. */
static rig_room_fn *const room_entry = rig_nv_room;
static rig_submit_fn *const submit_entry = rig_nv_submit;
static rig_commit_fn *const commit_entry = rig_nv_commit;

value caml_rig_nv_room_entry(value unit) {
  (void)unit;
  return Val_long((intnat)room_entry);
}

value caml_rig_nv_submit_entry(value unit) {
  (void)unit;
  return Val_long((intnat)submit_entry);
}

value caml_rig_nv_commit_entry(value unit) {
  (void)unit;
  return Val_long((intnat)commit_entry);
}
