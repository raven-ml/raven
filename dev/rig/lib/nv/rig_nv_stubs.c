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
#include <caml/alloc.h>
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

/* Each value's hand-over writes its release: a commit does nothing. */
static int rig_nv_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

static const struct rig_driver driver = {rig_nv_room, rig_nv_submit,
                                         rig_nv_commit};

/* A device whose timeline word is the host's 64-bit word at [v_word], the
   GPU's at [v_word_gpu], followed by its two join words, and whose
   notifications hold their error at [v_info32] and its status at
   [v_status]; or [0] without memory. The state lives for the process. */
value caml_rig_nv_create(value v_word, value v_word_gpu, value v_info32,
                         value v_status) {
  struct device *d = calloc(1, sizeof *d);
  if (d == NULL) return Val_long(0);
  d->driver = &driver;
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

/* Sets COMPUTE's launch ring to the [v_size] bytes, a power of two, at the
   host address [v_host] and the GPU's [v_gpu], in the GPU's memory through
   its BAR if [v_bar]. */
value caml_rig_nv_launches(value v_self, value v_host, value v_gpu,
                           value v_size, value v_bar) {
  struct channel *c = &Device_val(v_self)->ch[COMPUTE];
  c->launches = Pointer_val(v_host);
  c->launches_gpu = (uint64_t)Long_val(v_gpu);
  c->launches_size = (uint64_t)Long_val(v_size);
  c->bar = Bool_val(v_bar);
  return Val_unit;
}

value caml_rig_nv_doorbell(value v_self, value v_at) {
  Device_val(v_self)->doorbell = Pointer_val(v_at);
  return Val_unit;
}

/* A hole's record, as Template lays it out: 64-bit words, the index of its
   first word, its value's slot, its words (1 or 2), its operations' count,
   then each operation and its constant. */
enum { OP_ADD, OP_SHIFT };

static void refuse(const char *why) {
  char msg[96];
  snprintf(msg, sizeof msg, "Rig_nv.make: a template %s", why);
  caml_invalid_argument(msg);
}

/* Sets template [v_k] to the words [v_words], little-endian bytes, and the
   holes [v_holes], records as above, once they fit the template's bounds:
   a refused template leaves the one before. */
value caml_rig_nv_template(value v_self, value v_k, value v_words,
                           value v_holes) {
  struct template c;
  size_t bytes = caml_string_length(v_words);
  if (bytes > sizeof c.words) refuse("exceeds 16 words");
  c.nwords = (int)(bytes / 4);
  memcpy(c.words, String_val(v_words), 4 * (size_t)c.nwords);
  size_t n = caml_string_length(v_holes) / 8;
  uint64_t f[4 + 2 * HOLE_OPS];
  c.nholes = 0;
  for (size_t i = 0; i < n; i += 4 + 2 * (size_t)f[3]) {
    if (c.nholes == TEMPLATE_HOLES) refuse("has more than 6 holes");
    if (n - i < 4) refuse("hole is cut short");
    memcpy(f, String_val(v_holes) + 8 * i, 4 * 8);
    if (f[3] > HOLE_OPS) refuse("hole takes too many operations");
    if (n - i - 4 < 2 * f[3]) refuse("hole is cut short");
    memcpy(f + 4, String_val(v_holes) + 8 * (i + 4), 2 * 8 * f[3]);
    if (f[1] > 2) refuse("hole reads a value outside 0 to 2");
    if (f[2] < 1 || f[2] > 2 || f[0] + f[2] > (uint64_t)c.nwords)
      refuse("hole lies outside its words");
    struct hole *h = &c.holes[c.nholes++];
    h->at = (uint16_t)f[0];
    h->slot = (uint8_t)f[1];
    h->wide = f[2] == 2;
    h->nops = (uint8_t)f[3];
    for (int j = 0; j < h->nops; j++) {
      uint64_t op = f[4 + 2 * j], k = f[5 + 2 * j];
      if (op > OP_SHIFT) refuse("hole takes an unknown operation");
      if (op == OP_SHIFT && k > 63) refuse("hole's shift is outside 0 to 63");
      h->shift[j] = op == OP_SHIFT ? (uint8_t)k : 0;
      h->n[j] = op == OP_ADD ? k : 0;
    }
  }
  Device_val(v_self)->t[Int_val(v_k)] = c;
  return Val_unit;
}

/* Launches */

static void refuse_launch(struct launch *l, const char *why) {
  char msg[96];
  free(l);
  snprintf(msg, sizeof msg, "Rig_nv.entry: a structure %s", why);
  caml_invalid_argument(msg);
}

/* Sets [s] to the bytes [v_bytes] and the fields [v_fields]: 64-bit words,
   each field's byte, bits, slot and operations' count, then each operation
   and its constant. A word's width is the narrowest of 1, 2, 4 and 8 bytes
   that holds its bits. */
static void structure(struct launch *l, struct structure *s, value v_bytes,
                      value v_fields) {
  size_t bytes = caml_string_length(v_bytes);
  if (bytes > STRUCTURE_BYTES) refuse_launch(l, "exceeds 1 KiB");
  s->nbytes = (uint32_t)bytes;
  memcpy(s->bytes, String_val(v_bytes), bytes);
  size_t n = caml_string_length(v_fields) / 8;
  uint64_t f[4 + 2 * HOLE_OPS];
  s->nfields = 0;
  for (size_t i = 0; i < n; i += 4 + 2 * (size_t)f[3]) {
    if (s->nfields == STRUCTURE_FIELDS) refuse_launch(l, "has too many holes");
    if (n - i < 4) refuse_launch(l, "hole is cut short");
    memcpy(f, String_val(v_fields) + 8 * i, 4 * 8);
    if (f[3] > HOLE_OPS) refuse_launch(l, "hole takes too many operations");
    if (n - i - 4 < 2 * f[3]) refuse_launch(l, "hole is cut short");
    memcpy(f + 4, String_val(v_fields) + 8 * (i + 4), 2 * 8 * f[3]);
    if (f[2] >= VALUES) refuse_launch(l, "hole reads no launch value");
    if (f[1] < 1 || f[1] > 64) refuse_launch(l, "hole's bits are not 1 to 64");
    unsigned width = f[1] <= 8 ? 1 : f[1] <= 16 ? 2 : f[1] <= 32 ? 4 : 8;
    if (f[0] + width > bytes) refuse_launch(l, "hole lies outside its bytes");
    struct field *h = &s->fields[s->nfields++];
    h->at = (uint16_t)f[0];
    h->width = (uint8_t)width;
    h->bits = (uint8_t)f[1];
    h->slot = (uint8_t)f[2];
    h->nops = (uint8_t)f[3];
    for (int j = 0; j < h->nops; j++) {
      uint64_t op = f[4 + 2 * j], k = f[5 + 2 * j];
      if (op > OP_SHIFT) refuse_launch(l, "hole takes an unknown operation");
      if (op == OP_SHIFT && k > 63) refuse_launch(l, "hole's shift is past 63");
      h->shift[j] = op == OP_SHIFT ? (uint8_t)k : 0;
      h->n[j] = op == OP_ADD ? k : 0;
    }
  }
}

/* The fields of [caml_rig_nv_launch]'s ints. */
enum { l_params_at, l_bank0_bytes, l_max, l_max_threads = l_max + 6,
       l_max_shared, l_fields };

/* A function set up for launch: [v_structures] holds the bytes and fields
   of qmd[0][0], qmd[0][1], qmd[1][0], qmd[1][1] and bank0, in turn, and
   [v_ints] its numbers, as above. The caller frees it with
   [caml_rig_nv_launch_free]. */
value caml_rig_nv_launch(value v_structures, value v_ints) {
  struct launch *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  struct structure *s[5] = {&l->qmd[0][0], &l->qmd[0][1], &l->qmd[1][0],
                            &l->qmd[1][1], &l->bank0};
  for (int i = 0; i < 5; i++)
    structure(l, s[i], Field(v_structures, 2 * i),
              Field(v_structures, 2 * i + 1));
#define F(i) ((uint32_t)Long_val(Field(v_ints, i)))
  l->params_at = F(l_params_at);
  l->bank0_bytes = F(l_bank0_bytes);
  for (int i = 0; i < 6; i++) l->max[i] = F(l_max + i);
  l->max_threads = F(l_max_threads);
  l->max_shared = F(l_max_shared);
#undef F
  if (l->params_at > STRUCTURE_BYTES)
    refuse_launch(l, "places parameters past 1 KiB");
  return caml_copy_nativeint((intnat)l);
}

value caml_rig_nv_launch_free(value v_launch) {
  free((void *)Nativeint_val(v_launch));
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

/* Raises the word to the last value submitted, by compare-and-set: it
   never moves backwards. */
value caml_rig_nv_raise(value v_self) {
  struct device *d = Device_val(v_self);
  rig_raise(d->word, atomic_load_explicit(&d->last, memory_order_relaxed));
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

