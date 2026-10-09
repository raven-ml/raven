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
#include <rig_nv_stubs.h>

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
                       &failure) == RIG_FAILED)
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
  struct rig_part copy = {.queue = 1, .kind = RIG_WORDS};
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
  struct rig_part p = {.queue = 0, .kind = RIG_WORDS, .words = {words, 2}};
  submit(NULL, 0, &p, 1);
  spin();
  return Val_unit;
}

/* The most ring entries [entries] takes. */
#define ENTRIES 64

/* The ring entries of two words each, [v_words] one after the other, on
   COMPUTE:0, a submission per entry, each with its release. Then the wait
   for the last. */
value rig_nv_bench_entries(value v_words) {
  uint32_t words[2 * ENTRIES];
  struct rig_part parts[ENTRIES];
  int n = (int)Wosize_val(v_words) / 2;
  if (n > ENTRIES) caml_invalid_argument("rig_nv_bench_entries: too many");
  for (int i = 0; i < n; i++) {
    words[2 * i] = (uint32_t)Long_val(Field(v_words, 2 * i));
    words[2 * i + 1] = (uint32_t)Long_val(Field(v_words, 2 * i + 1));
    parts[i] = (struct rig_part){
        .queue = 0, .kind = RIG_WORDS, .words = {&words[2 * i], 2}};
  }
  for (int i = 0; i < n; i++) submit(NULL, 0, &parts[i], 1);
  spin();
  return Val_unit;
}

/* The writer's sequence, for a floor that rings the doorbell where
   rig_nv_submit does not: rig_nv_ring.c's helpers, copied. */

static void store_fence(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("sfence" ::: "memory");
#elif defined(__aarch64__)
  __asm__ __volatile__("dsb st" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
}

static void close_segment(const struct device *d, struct channel *c) {
  if (c->open_words == 0) return;
  uint64_t at = c->segments_gpu + (c->open & (c->size - 1));
  c->ring[c->put & (c->entries - 1)] =
      at + d->entry_base + c->open_words * d->entry_word;
  c->put++;
  c->open = c->written;
  c->open_words = 0;
}

static void emit(const struct device *d, struct channel *c, int k, uint64_t a,
                 uint64_t b) {
  const struct template *t = &d->t[k];
  uint64_t bytes = 4 * (uint64_t)t->nwords;
  uint64_t start = c->open & (c->size - 1);
  if (start + 4 * c->open_words + bytes > c->size) {
    close_segment(d, c);
    uint64_t end = c->written & (c->size - 1);
    if (end + bytes > c->size) c->written += c->size - end;
    c->open = c->written;
  }
  uint64_t at = c->written & (c->size - 1);
  const uint64_t values[3] = {a, b, 0};
  rig_nv_fill(t, values, (uint32_t *)(c->segments + at));
  c->written += bytes;
  c->open_words += (uint64_t)t->nwords;
}

static void full_fence(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#elif defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#else
  atomic_thread_fence(memory_order_seq_cst);
#endif
}

/* Makes the channel's entries up to [c->put] the GPU's and rings, as
   rig_nv_submit does. */
static void ring(struct device *d, struct channel *c) {
  store_fence();
  *c->gp_put = (uint32_t)(c->put & (c->entries - 1));
  if (atomic_load_explicit(&d->bar_live, memory_order_acquire) > 0) {
    full_fence();
    (void)*d->bar;
  } else
    store_fence();
  *d->doorbell = c->token;
}

/* Ends the value [v] on [c]: its release into the timeline word and its
   mark. */
static void release(struct device *d, struct channel *c, uint64_t v) {
  emit(d, c, T_RELEASE, d->word_gpu, v);
  close_segment(d, c);
  c->released = v;
  c->marks[(c->first + c->count++) & (c->entries - 1)] =
      (struct mark){v, c->put, c->written};
  atomic_store_explicit(&d->last, v, memory_order_release);
}

/* The ring entries of two words each, [v_words] one after the other, on
   COMPUTE:0, as values written as the driver writes them, each entry put
   and rung: with [v_each], each value ends with its release, which waits
   for idle; otherwise each ends with the wait for idle that orders the next
   value's launches after it, and one release ends them all. Then the wait
   for the last. The rows' gap is the releases' cost on the GPU; their gap
   to [rig_nv_bench_entries] is the driver's on the host. */
value rig_nv_bench_rung(value v_words, value v_each) {
  struct device *d = self;
  struct channel *c = &d->ch[COMPUTE];
  uint32_t words[2 * ENTRIES];
  struct rig_part parts[ENTRIES];
  int n = (int)Wosize_val(v_words) / 2, each = Bool_val(v_each);
  if (n > ENTRIES) caml_invalid_argument("rig_nv_bench_rung: too many");
  for (int i = 0; i < n; i++) {
    words[2 * i] = (uint32_t)Long_val(Field(v_words, 2 * i));
    words[2 * i + 1] = (uint32_t)Long_val(Field(v_words, 2 * i + 1));
    parts[i] = (struct rig_part){
        .queue = 0, .kind = RIG_WORDS, .words = {&words[2 * i], 2}};
  }
  /* The words a channel owes come with a submission of rig_nv_submit's. */
  if (c->owes_setup || c->released != last ||
      atomic_load_explicit(&d->local, memory_order_acquire) != 0 ||
      atomic_load_explicit(&d->invalidate, memory_order_acquire) != 0)
    submit(NULL, 0, NULL, 0);
  if (rig_nv_room(self, parts, n) != RIG_FITS)
    caml_failwith("rig_nv_room: the parts do not fit");
  c->open = c->written;
  c->open_words = 0;
  for (int i = 0; i < n; i++) {
    close_segment(d, c);
    c->ring[c->put++ & (c->entries - 1)] =
        (uint64_t)words[2 * i] | (uint64_t)words[2 * i + 1] << 32;
    if (each || i + 1 == n)
      release(d, c, ++last);
    else {
      emit(d, c, T_IDLE, 0, 0);
      close_segment(d, c);
    }
    ring(d, c);
  }
  spin();
  return Val_unit;
}

/* A copy of [v_n] bytes from the GPU address [v_src] to [v_dst] on
   COPY:0. */
value rig_nv_bench_copy(value v_dst, value v_src, value v_n) {
  struct rig_part p = {.queue = 1,
                      .kind = RIG_COPY,
                      .copy = {.dst = (uint64_t)Long_val(v_dst),
                               .src = (uint64_t)Long_val(v_src),
                               .bytes = (uint64_t)Long_val(v_n)}};
  submit(NULL, 0, &p, 1);
  spin();
  return Val_unit;
}
