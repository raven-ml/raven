/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Rig_amd's C state, filled from OCaml. No stub here releases the runtime:
   none blocks. A device's state is named in OCaml by its address, an int. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_amd_stubs.h"

#define Device_val(v) ((struct rig_amd *)Long_val(v))
#define Host_val(v) ((void *)Long_val(v))

static intnat at(value a, int i) { return Long_val(Field(a, i)); }

/* State */

/* Each value's hand-over writes its release: a commit does nothing. */
static int rig_amd_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

static const struct rig_driver driver = {rig_amd_room, rig_amd_submit,
                                         rig_amd_commit};

value caml_rig_amd_create(value unit) {
  (void)unit;
  struct rig_amd *d = calloc(1, sizeof *d);
  if (d == NULL) caml_raise_out_of_memory();
  d->driver = &driver;
  return Val_long((intnat)d);
}

/* The word, the slots (set to low32(-1), which no early wait compares
   with) and the segment, at their host and GPU addresses. */
value caml_rig_amd_memory(value v_self, value v_word, value v_word_gpu,
                             value v_slots, value v_slots_gpu) {
  struct rig_amd *d = Device_val(v_self);
  d->word = Host_val(v_word);
  atomic_store_explicit(d->word, 0, memory_order_release);
  d->word_gpu = (uint64_t)Long_val(v_word_gpu);
  d->slots = Host_val(v_slots);
  d->slots_gpu = (uint64_t)Long_val(v_slots_gpu);
  for (int i = 0; i < RIG_AMD_SLOTS; i++) d->slots[2 * i] = UINT32_MAX;
  return Val_unit;
}

value caml_rig_amd_segment(value v_self, value v_host, value v_gpu,
                              value v_size) {
  struct rig_amd *d = Device_val(v_self);
  d->segment.host = Host_val(v_host);
  d->segment.gpu = (uint64_t)Long_val(v_gpu);
  d->segment.size = (uint64_t)Long_val(v_size);
  return Val_unit;
}

/* Ring [q]: its words, size in bytes, write-position word and doorbell, all
   host addresses, and its kind (RING_PM4, RING_AQL, RING_SDMA). */
value caml_rig_amd_ring(value v_self, value v_q, value v_words,
                           value v_bytes, value v_write, value v_doorbell,
                           value v_kind) {
  struct rig_amd_ring *r = &Device_val(v_self)->rings[Int_val(v_q)];
  r->words = Host_val(v_words);
  r->size = (uint64_t)Long_val(v_bytes) / 4;
  r->write = Host_val(v_write);
  r->doorbell = Host_val(v_doorbell);
  r->kind = Int_val(v_kind);
  return Val_unit;
}

value caml_rig_amd_ring_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_amd_ring(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6]);
}

/* A hole's record, as Template lays it out: 64-bit words, the index of its
   first word, its argument, its words (1 or 2), its operations' count, then
   each operation and its constant. */

#define TEMPLATE "Rig_amd.make: a packet template"
#define LAUNCH "Rig_amd.entry: a launch"

static void refuse(const char *what, const char *why) {
  char msg[96];
  snprintf(msg, sizeof msg, "%s %s", what, why);
  caml_invalid_argument(msg);
}

/* Reads the hole records [v_holes] of [what], whose words are [n], into
   [h], at most [max] of them, each reading an argument below [nargs]:
   their count. */
static int holes(const char *what, value v_holes, struct rig_amd_hole *h,
                 int max, int nargs, int n) {
  size_t len = caml_string_length(v_holes) / 8;
  uint64_t f[4 + 2 * RIG_AMD_HOLE_OPS];
  int count = 0;
  for (size_t i = 0; i < len; i += 4 + 2 * (size_t)f[3]) {
    if (count == max) refuse(what, "has too many holes");
    if (len - i < 4) refuse(what, "hole is cut short");
    memcpy(f, String_val(v_holes) + 8 * i, 4 * 8);
    if (f[3] > RIG_AMD_HOLE_OPS) refuse(what, "hole takes too many operations");
    if (len - i - 4 < 2 * f[3]) refuse(what, "hole is cut short");
    memcpy(f + 4, String_val(v_holes) + 8 * (i + 4), 2 * 8 * f[3]);
    if (f[1] >= (uint64_t)nargs) refuse(what, "hole reads an unknown argument");
    if (f[2] < 1 || f[2] > 2 || f[0] + f[2] > (uint64_t)n)
      refuse(what, "hole lies outside its words");
    struct rig_amd_hole *o = &h[count++];
    o->at = (uint8_t)f[0];
    o->arg = (uint8_t)f[1];
    o->wide = (uint8_t)f[2];
    o->nops = (uint8_t)f[3];
    for (int j = 0; j < o->nops; j++) {
      uint64_t op = f[4 + 2 * j], k = f[5 + 2 * j];
      if (op > OP_OR) refuse(what, "hole takes an unknown operation");
      if (op == OP_SHIFT && k > 63) refuse(what, "hole's shift is outside 0 to 63");
      o->op[j] = (uint8_t)op;
      o->k[j] = k;
    }
  }
  return count;
}

/* Template [t]: its words as little-endian bytes, and its holes, records as
   above, once they fit the template's bounds: a refused template leaves the
   one before. */
value caml_rig_amd_template(value v_self, value v_t, value v_words,
                            value v_holes) {
  struct rig_amd_template c;
  size_t bytes = caml_string_length(v_words);
  if (bytes > sizeof c.words) refuse(TEMPLATE, "exceeds 16 words");
  c.n = (int)(bytes / 4);
  memcpy(c.words, String_val(v_words), 4 * (size_t)c.n);
  c.nholes = holes(TEMPLATE, v_holes, c.holes, RIG_AMD_TEMPLATE_HOLES,
                   RIG_AMD_TEMPLATE_ARGS, c.n);
  Device_val(v_self)->templates[Int_val(v_t)] = c;
  return Val_unit;
}

/* The bytes of each implicit argument a launch writes. */
static const int hidden_bytes[RIG_AMD_HIDDEN] = {4, 4, 4, 2, 2, 2, 2,
                                                 2, 2, 8, 8, 8, 2, 4};

/* A launch (struct rig_amd_launch): its dispatch's words and holes, as a
   template's; [v_lds], the index of its LDS word, that word without LDS, the
   word's increment per granule, the granule's bytes and the function's own
   LDS bytes; [v_max], its most threads per group and shared bytes and its
   arguments' bytes; and [v_hidden], the offsets of its implicit arguments,
   each within its arguments, or -1. Its address, an int, which
   caml_rig_amd_launch_free frees. */
value caml_rig_amd_launch(value v_words, value v_holes, value v_lds,
                          value v_max, value v_hidden) {
  struct rig_amd_launch c;
  size_t bytes = caml_string_length(v_words);
  if (bytes > sizeof c.words) refuse(LAUNCH, "exceeds its words");
  c.n = (int)(bytes / 4);
  memcpy(c.words, String_val(v_words), 4 * (size_t)c.n);
  c.nholes = holes(LAUNCH, v_holes, c.holes, RIG_AMD_LAUNCH_HOLES,
                   RIG_AMD_LAUNCH_ARGS, c.n);
  c.lds_at = (uint32_t)at(v_lds, 0);
  c.lds_word = (uint32_t)at(v_lds, 1);
  c.lds_unit = (uint32_t)at(v_lds, 2);
  c.lds_granule = (uint32_t)at(v_lds, 3);
  c.group = (uint32_t)at(v_lds, 4);
  c.max_threads = (uint32_t)at(v_max, 0);
  c.max_shared = (uint32_t)at(v_max, 1);
  c.kernarg = (uint32_t)at(v_max, 2);
  if (c.lds_at >= (uint32_t)c.n || c.lds_granule == 0)
    refuse(LAUNCH, "LDS word lies outside its words");
  if (Wosize_val(v_hidden) != RIG_AMD_HIDDEN)
    refuse(LAUNCH, "names another number of implicit arguments");
  for (int i = 0; i < RIG_AMD_HIDDEN; i++) {
    intnat h = at(v_hidden, i);
    if (h < -1 || (h >= 0 && h + hidden_bytes[i] > (intnat)c.kernarg))
      refuse(LAUNCH, "implicit argument lies outside its arguments");
    c.hidden[i] = (int32_t)h;
  }
  struct rig_amd_launch *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  *l = c;
  return Val_long((intnat)l);
}

value caml_rig_amd_launch_free(value v_l) {
  free((void *)Long_val(v_l));
  return Val_unit;
}

/* The words of launch [v_l]'s dispatch, as the hand-over places them with
   the arguments [v_args] and [v_shared] bytes of shared memory, as
   little-endian bytes on a little-endian host. */
value caml_rig_amd_dispatch(value v_l, value v_args, value v_shared) {
  const struct rig_amd_launch *l = (const void *)Long_val(v_l);
  uint64_t args[RIG_AMD_LAUNCH_ARGS];
  uint32_t w[RIG_AMD_LAUNCH_WORDS];
  if (Wosize_val(v_args) != RIG_AMD_LAUNCH_ARGS)
    caml_invalid_argument("Rig_amd.dispatch: expected 8 arguments");
  for (int i = 0; i < RIG_AMD_LAUNCH_ARGS; i++) args[i] = (uint64_t)at(v_args, i);
  int n = rig_amd_dispatch(l, args, (uint32_t)Long_val(v_shared), w);
  value s = caml_alloc_string(4 * (mlsize_t)n);
  memcpy(Bytes_val(s), w, 4 * (size_t)n);
  return s;
}

value caml_rig_amd_max_copy(value v_self, value v_n) {
  Device_val(v_self)->max_copy = (uint64_t)Long_val(v_n);
  return Val_unit;
}

/* Adds [delta] to the regions that need the HDP register at host address
   [reg] flushed. The caller serialises calls for one device. */
value caml_rig_amd_hdp(value v_self, value v_reg, value v_delta) {
  struct rig_amd *d = Device_val(v_self);
  volatile uint32_t *reg = Host_val(v_reg);
  struct rig_amd_hdp *free = NULL;
  for (int i = 0; i < RIG_AMD_HDPS; i++) {
    struct rig_amd_hdp *h = &d->hdps[i];
    if (h->reg == reg) {
      atomic_fetch_add_explicit(&h->count, Int_val(v_delta),
                                memory_order_acq_rel);
      return Val_true;
    }
    if (free == NULL && h->reg == NULL) free = h;
  }
  if (free == NULL) return Val_false;
  free->reg = reg;
  atomic_store_explicit(&free->count, Int_val(v_delta), memory_order_release);
  return Val_true;
}

/* Work */

value caml_rig_amd_last(value v_self) {
  return Val_long(atomic_load_explicit(&Device_val(v_self)->last,
                                       memory_order_acquire));
}

value caml_rig_amd_place_entry(value unit) {
  (void)unit;
  return Val_long((intnat)rig_amd_place);
}

value caml_rig_amd_segment_entry(value unit) {
  (void)unit;
  return Val_long((intnat)rig_amd_segment);
}

/* Timeline */

value caml_rig_amd_signaled(value v_self) {
  struct rig_amd *d = Device_val(v_self);
  return Val_long(atomic_load_explicit(d->word, memory_order_acquire));
}

/* Raises the word to the last value submit was given, with a release
   store, unless it is there already: the queues are destroyed, so nothing
   else writes it. */
value caml_rig_amd_settle(value v_self) {
  struct rig_amd *d = Device_val(v_self);
  rig_raise(d->word, atomic_load_explicit(&d->last, memory_order_relaxed));
  return Val_unit;
}

/* Publishes a new scratch at the GPU address [base], with the writes of
   it to an AQL queue's descriptor: the 32-bit [values] at the GPU addresses
   [at], which the next submission places. Answers the value whose
   submission placed the publication this one replaces, or 0 if none did,
   read under the lock. */
value caml_rig_amd_scratch(value v_self, value v_base, value v_at,
                           value v_values) {
  struct rig_amd *d = Device_val(v_self);
  int n = (int)Wosize_val(v_at), idle = 0;
  if (n > RIG_AMD_SCRATCH_WRITES)
    caml_invalid_argument("Rig_amd: too many scratch writes");
  while (!atomic_compare_exchange_weak(&d->scratch_lock, &idle, 1)) idle = 0;
  uint64_t took = atomic_load_explicit(&d->scratch_taken, memory_order_relaxed);
  for (int i = 0; i < n; i++) {
    d->scratch_at[i] = (uint64_t)at(v_at, i);
    d->scratch_value[i] = (uint32_t)at(v_values, i);
  }
  d->scratch_n = n;
  d->scratch_next = (uint64_t)Long_val(v_base);
  atomic_store_explicit(&d->scratch_taken, 0, memory_order_relaxed);
  atomic_store_explicit(&d->scratch_ready, 1, memory_order_relaxed);
  atomic_store_explicit(&d->scratch_lock, 0, memory_order_release);
  return Val_long((intnat)took);
}

/* The value whose submission placed the last scratch writes, or 0. */
value caml_rig_amd_scratch_taken(value v_self) {
  return Val_long(atomic_load_explicit(&Device_val(v_self)->scratch_taken,
                                       memory_order_acquire));
}

value caml_rig_amd_poke32(value v_host, value v_off, value v_v) {
  uint8_t *p = Host_val(v_host);
  *(volatile uint32_t *)(p + Long_val(v_off)) = (uint32_t)Long_val(v_v);
  return Val_unit;
}

value caml_rig_amd_zero(value v_host, value v_n) {
  memset(Host_val(v_host), 0, (size_t)Long_val(v_n));
  return Val_unit;
}

/* Makes [v] the value after the device's last one; the device is idle. A
   ring that released the last value is taken to have released v - 1, which
   the word now holds, and each slot gets low32(v - 1 - age), last written
   at v - 1 - age: the writer refreshes a slot once it is more than 2^31
   values old. */
value caml_rig_amd_renumber(value v_self, value v_v, value v_age) {
  struct rig_amd *d = Device_val(v_self);
  uint64_t prev = (uint64_t)Long_val(v_v) - 1;
  uint64_t written = prev - (uint64_t)Long_val(v_age);
  for (int i = 0; i < RIG_AMD_SLOTS; i++) {
    d->slots[2 * i] = (uint32_t)written;
    d->slot_last[i] = written;
  }
  uint64_t last = atomic_load_explicit(&d->last, memory_order_relaxed);
  for (int q = 0; q < RIG_AMD_QUEUES; q++)
    if (d->rings[q].released == last) d->rings[q].released = prev;
  atomic_store_explicit(&d->last, prev, memory_order_relaxed);
  atomic_store_explicit(d->word, prev, memory_order_release);
  return Val_unit;
}
