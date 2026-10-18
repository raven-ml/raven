/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Rig_amd's C state, filled from OCaml. Only the first measure of the
   clock releases the runtime, while it may wait for the GPU; no other stub
   blocks. A device's
   state is named in OCaml by its address, an int. */

#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
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

#include "rig_amd_stubs.h"

#define Edge_val(v) ((struct rig_amd_edge *)Long_val(v))
#define Device_val(v) (Edge_val(v)->d)
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
                                         rig_amd_commit, rig_amd_settle};

/* RIG_AMD_WAITS, the most waits a submission takes. */
value caml_rig_amd_max_waits(value unit) {
  (void)unit;
  return Val_int(RIG_AMD_WAITS);
}

/* The live edges, linked from their create to their end, under [edges_mu],
   which no one takes while holding an edge's [hdp_mu]. */
static pthread_mutex_t edges_mu = PTHREAD_MUTEX_INITIALIZER;
static struct rig_amd_edge *edges = NULL;

static void link_edge(struct rig_amd_edge *e) {
  pthread_mutex_lock(&edges_mu);
  e->next = edges;
  edges = e;
  pthread_mutex_unlock(&edges_mu);
}

static void unlink_edge(struct rig_amd_edge *e) {
  pthread_mutex_lock(&edges_mu);
  struct rig_amd_edge **at = &edges;
  while (*at != NULL && *at != e) at = &(*at)->next;
  if (*at == e) *at = e->next;
  pthread_mutex_unlock(&edges_mu);
}

value caml_rig_amd_create(value unit) {
  (void)unit;
  struct rig_amd_edge *e = calloc(1, sizeof *e);
  struct rig_amd *d = calloc(1, sizeof *d);
  if (e == NULL || d == NULL || pthread_mutex_init(&e->hdp_mu, NULL) != 0) {
    free(e);
    free(d);
    caml_raise_out_of_memory();
  }
  e->driver = &driver;
  e->d = d;
  link_edge(e);
  return Val_long((intnat)e);
}

/* Frees the run state of a stopped device: its edge stays, and rig frees a
   word of its own. */
value caml_rig_amd_end(value v_self) {
  struct rig_amd_edge *e = Edge_val(v_self);
  unlink_edge(e);
  pthread_mutex_lock(&e->hdp_mu);
  struct rig_amd *d = e->d;
  e->d = NULL;
  pthread_mutex_unlock(&e->hdp_mu);
  free(d);
  return Val_unit;
}

/* Frees the state of a device whose open failed: nothing holds it. */
value caml_rig_amd_destroy(value v_self) {
  struct rig_amd_edge *e = Edge_val(v_self);
  unlink_edge(e);
  free(e->d);
  pthread_mutex_destroy(&e->hdp_mu);
  free(e);
  return Val_unit;
}

value caml_rig_amd_copy_launch(value v_self, value v_launch) {
  Device_val(v_self)->copy_launch = (const void *)Long_val(v_launch);
  return Val_unit;
}

value caml_rig_amd_write_code(value v_host, value v_bytes) {
  memcpy(Host_val(v_host), String_val(v_bytes), caml_string_length(v_bytes));
  return Val_unit;
}

/* The word, at its host and GPU addresses. */
value caml_rig_amd_memory(value v_self, value v_word, value v_word_gpu) {
  struct rig_amd *d = Device_val(v_self);
  d->word = Host_val(v_word);
  Edge_val(v_self)->word = d->word;
  d->word_gpu = (uint64_t)Long_val(v_word_gpu);
  return Val_unit;
}

/* The stamp ring, of RIG_AMD_STAMPS slots, at the host address [v_host]
   and the GPU's [v_gpu], zeroed, followed by the record of a stamp of the
   measure at open; and the clock's rate [v_hz], positive. Raises
   [Invalid_argument] if its [v_bytes] bytes hold less. */
value caml_rig_amd_stamps(value v_self, value v_host, value v_gpu,
                          value v_bytes, value v_hz) {
  struct rig_amd *d = Device_val(v_self);
  if ((uint64_t)Long_val(v_bytes) <
      RIG_AMD_STAMPS * sizeof(struct rig_amd_stamps) + sizeof(uint64_t))
    caml_invalid_argument("Rig_amd: the stamp ring is too small");
  d->stamps = Host_val(v_host);
  d->stamps_gpu = (uint64_t)Long_val(v_gpu);
  d->clock = (volatile uint64_t *)(d->stamps + RIG_AMD_STAMPS);
  d->clock_gpu = d->stamps_gpu + RIG_AMD_STAMPS * sizeof(struct rig_amd_stamps);
  d->clock_hz = (uint64_t)Long_val(v_hz);
  return Val_unit;
}

/* Sets the clock's source to the path's [struct rig_amd_clock] at
   [v_source], or none if it is [0], and takes the first measure
   (rig_amd_measure): whether it could. Releases the runtime, as a measure
   without a source waits for the GPU. */
value caml_rig_amd_clock(value v_self, value v_source) {
  struct rig_amd *d = Device_val(v_self);
  d->source = Host_val(v_source);
  caml_enter_blocking_section_no_pending();
  int ok = rig_amd_measure(d);
  caml_leave_blocking_section();
  return Val_bool(ok);
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

/* A hole's record, as Template lays it out: six 64-bit words, the index of
   its first word, its words (1 or 2), its argument, then its add, shift and
   set. Template refuses a record outside the bounds; C checks only what
   keeps its reads and the ring's writes in their arrays, with a fixed
   message. */

#define OUT_OF_BOUNDS "Rig_amd: a template or launch is out of bounds"
#define LAUNCH "Rig_amd.entry: a launch's "
#define HOLE_WORDS 6

/* Reads the hole records [v_holes], for words [n] long, into [h], at most
   [max] of them, each reading an argument below [nargs]: their count. */
static int holes(value v_holes, struct rig_amd_hole *h, int max, int nargs,
                 int n) {
  size_t len = caml_string_length(v_holes) / (8 * HOLE_WORDS);
  if (len > (size_t)max || caml_string_length(v_holes) % (8 * HOLE_WORDS))
    caml_invalid_argument(OUT_OF_BOUNDS);
  for (size_t i = 0; i < len; i++) {
    uint64_t f[HOLE_WORDS];
    memcpy(f, String_val(v_holes) + 8 * HOLE_WORDS * i, sizeof f);
    if (f[1] < 1 || f[1] > 2 || f[0] > (uint64_t)n ||
        f[0] + f[1] > (uint64_t)n || f[2] >= (uint64_t)nargs || f[4] > 63)
      caml_invalid_argument(OUT_OF_BOUNDS);
    h[i] = (struct rig_amd_hole){.at = (uint8_t)f[0],
                                 .wide = (uint8_t)f[1],
                                 .arg = (uint8_t)f[2],
                                 .add = f[3],
                                 .shift = (uint8_t)f[4],
                                 .set = f[5]};
  }
  return (int)len;
}

/* Template [t]: its words as little-endian bytes, and its holes, records as
   above: a refused template leaves the one before. */
value caml_rig_amd_template(value v_self, value v_t, value v_words,
                            value v_holes) {
  struct rig_amd_template c;
  size_t bytes = caml_string_length(v_words);
  if (bytes > sizeof c.words) caml_invalid_argument(OUT_OF_BOUNDS);
  c.n = (int)(bytes / 4);
  memcpy(c.words, String_val(v_words), 4 * (size_t)c.n);
  c.nholes = holes(v_holes, c.holes, RIG_AMD_TEMPLATE_HOLES,
                   RIG_AMD_TEMPLATE_ARGS, c.n);
  Device_val(v_self)->templates[Int_val(v_t)] = c;
  return Val_unit;
}

/* The bytes of each implicit argument a launch writes. */
static const int hidden_bytes[RIG_AMD_HIDDEN] = {4, 4, 4, 2, 2, 2, 2,
                                                 2, 2, 8, 8, 8, 2, 4};

/* Reads into [c] what a launch of either form holds: its dispatch's words
   and holes, as a template's; [v_max], the function's own LDS bytes, its most
   threads per group and shared bytes, its arguments' bytes and its explicit
   ones'; and
   [v_hidden], the offsets of its implicit arguments, each within its
   arguments, or -1. */
static void launch_of(struct rig_amd_launch *c, value v_words, value v_holes,
                      value v_max, value v_hidden) {
  size_t bytes = caml_string_length(v_words);
  if (bytes > sizeof c->words) caml_invalid_argument(OUT_OF_BOUNDS);
  c->n = (int)(bytes / 4);
  memcpy(c->words, String_val(v_words), 4 * (size_t)c->n);
  c->nholes = holes(v_holes, c->holes, RIG_AMD_LAUNCH_HOLES,
                    RIG_AMD_LAUNCH_ARGS, c->n);
  c->group = (uint32_t)at(v_max, 0);
  c->max_threads = (uint32_t)at(v_max, 1);
  c->max_shared = (uint32_t)at(v_max, 2);
  c->kernarg = (uint32_t)at(v_max, 3);
  c->explicit_size = (uint32_t)at(v_max, 4);
  if (Wosize_val(v_hidden) != RIG_AMD_HIDDEN)
    caml_invalid_argument(OUT_OF_BOUNDS);
  for (int i = 0; i < RIG_AMD_HIDDEN; i++) {
    intnat h = at(v_hidden, i);
    if (h < -1 || (h >= 0 && h + hidden_bytes[i] > (intnat)c->kernarg))
      caml_invalid_argument(LAUNCH "implicit argument lies outside its "
                                   "arguments");
    c->hidden[i] = (int32_t)h;
  }
}

/* The launch [c] in memory of its own: its address, an int, which
   caml_rig_amd_launch_free frees. */
static value keep(const struct rig_amd_launch *c) {
  struct rig_amd_launch *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  *l = *c;
  return Val_long((intnat)l);
}

/* A launch on a PM4 queue: [v_lds], the index of its LDS word, that word
   without LDS, the word's increment per granule and the granule's bytes; the
   rest as launch_of reads them. */
value caml_rig_amd_launch_pm4(value v_words, value v_holes, value v_lds,
                              value v_max, value v_hidden) {
  struct rig_amd_launch c = {0};
  launch_of(&c, v_words, v_holes, v_max, v_hidden);
  c.pm4.lds_at = (uint32_t)at(v_lds, 0);
  c.pm4.lds_word = (uint32_t)at(v_lds, 1);
  c.pm4.lds_unit = (uint32_t)at(v_lds, 2);
  c.pm4.lds_granule = (uint32_t)at(v_lds, 3);
  if (c.pm4.lds_at >= (uint32_t)c.n || c.pm4.lds_granule == 0)
    caml_invalid_argument(LAUNCH "LDS word lies outside its words");
  return keep(&c);
}

/* A launch on an AQL queue, whose words are one packet: [v_fields], the
   bytes of its workgroup sizes along x, y and z and of its group segment
   size; the rest as launch_of reads them. */
value caml_rig_amd_launch_aql(value v_words, value v_holes, value v_fields,
                              value v_max, value v_hidden) {
  struct rig_amd_launch c = {0};
  launch_of(&c, v_words, v_holes, v_max, v_hidden);
  c.packet = 1;
  for (int k = 0; k < 3; k++) c.aql.threads[k] = (uint32_t)at(v_fields, k);
  c.aql.group = (uint32_t)at(v_fields, 3);
  for (int k = 0; k < 3; k++)
    if (c.aql.threads[k] + 2 > 4 * (uint32_t)c.n)
      caml_invalid_argument(LAUNCH "workgroup size lies outside its words");
  if (c.aql.group + 4 > 4 * (uint32_t)c.n)
    caml_invalid_argument(LAUNCH "group segment size lies outside its "
                                 "words");
  return keep(&c);
}

value caml_rig_amd_launch_free(value v_l) {
  free((void *)Long_val(v_l));
  return Val_unit;
}

/* Frees each launch of the list [v_launches]. */
value caml_rig_amd_launches_free(value v_launches) {
  for (value l = v_launches; Is_block(l); l = Field(l, 1))
    free((void *)Long_val(Field(l, 0)));
  return Val_unit;
}

value caml_rig_amd_most_parts(value v_self) {
  return Val_long(rig_amd_most_parts(Device_val(v_self)));
}

value caml_rig_amd_max_copy(value v_self, value v_n) {
  Device_val(v_self)->max_copy = (uint64_t)Long_val(v_n);
  return Val_unit;
}

/* The device's own HDP register, the path's, which takes the first entry
   for good with no region counted, so that the device's own [Mapped] memory
   always finds one and views of other GPUs' take at most the rest. */
value caml_rig_amd_set_hdp(value v_self, value v_reg) {
  struct rig_amd_edge *e = Edge_val(v_self);
  volatile uint32_t *reg = Host_val(v_reg);
  e->d->hdps[0].src = e;
  atomic_store_explicit(&e->d->hdps[0].reg, reg, memory_order_release);
  atomic_store_explicit(&e->hdp, reg, memory_order_release);
  return Val_unit;
}

/* Adds [delta] to the regions of [d] that need [src]'s HDP register flushed,
   taking a free entry for a new register: [0] if none is free, or if [src]
   has stopped. A count down of a register that [src]'s stop took out counts
   nothing. The caller holds [d]'s edge's [hdp_mu]. */
static int count_hdp(struct rig_amd *d, const struct rig_amd_edge *src,
                     int delta) {
  struct rig_amd_hdp *free = NULL;
  for (int i = 0; i < RIG_AMD_HDPS; i++) {
    struct rig_amd_hdp *h = &d->hdps[i];
    if (h->src == src) {
      atomic_fetch_add_explicit(&h->count, delta, memory_order_acq_rel);
      return 1;
    }
    if (free == NULL && i > 0 &&
        (atomic_load(&h->reg) == NULL || atomic_load(&h->count) == 0))
      free = h;
  }
  if (delta < 0) return 1;
  volatile uint32_t *reg =
      atomic_load_explicit(&src->hdp, memory_order_seq_cst);
  if (free == NULL || reg == NULL) return 0;
  free->src = src;
  atomic_store_explicit(&free->reg, reg, memory_order_release);
  atomic_store_explicit(&free->count, delta, memory_order_release);
  return 1;
}

/* [count_hdp] on [v_self] of [v_src]'s register, each of whose stops may run
   on another domain: [true] once [v_self] has stopped. */
value caml_rig_amd_hdp(value v_self, value v_src, value v_delta) {
  struct rig_amd_edge *e = Edge_val(v_self);
  pthread_mutex_lock(&e->hdp_mu);
  int counted =
      e->d == NULL || count_hdp(e->d, Edge_val(v_src), Int_val(v_delta));
  pthread_mutex_unlock(&e->hdp_mu);
  return Val_bool(counted);
}

/* Takes [v_self]'s HDP register out of every other live device's entries,
   before its path unmaps it, and returns once no submission may still store
   to it. [hdp] goes first, so no entry for the register starts after its
   walk (count_hdp reads it under the same lock), and each submission either
   reads the entry's register as NULL or set [flushing] before this reads it
   (hand_over). A device's [hdp_mu] keeps its state from its end. */
value caml_rig_amd_drop(value v_self) {
  struct rig_amd_edge *self = Edge_val(v_self);
  if (atomic_exchange_explicit(&self->hdp, NULL, memory_order_seq_cst) ==
      NULL)
    return Val_unit;
  pthread_mutex_lock(&edges_mu);
  for (struct rig_amd_edge *e = edges; e != NULL; e = e->next) {
    if (e == self) continue;
    pthread_mutex_lock(&e->hdp_mu);
    struct rig_amd *d = e->d;
    int took = 0;
    for (int i = 1; d != NULL && i < RIG_AMD_HDPS; i++) {
      struct rig_amd_hdp *h = &d->hdps[i];
      if (h->src != self) continue;
      h->src = NULL;
      atomic_store_explicit(&h->count, 0, memory_order_relaxed);
      atomic_store_explicit(&h->reg, NULL, memory_order_relaxed);
      took = 1;
    }
    if (took) {
      rig_amd_barrier();
      while (atomic_load_explicit(&d->flushing, memory_order_acquire))
        sched_yield();
    }
    pthread_mutex_unlock(&e->hdp_mu);
  }
  pthread_mutex_unlock(&edges_mu);
  return Val_unit;
}

/* Work */

value caml_rig_amd_last(value v_self) {
  return Val_long(atomic_load_explicit(&Device_val(v_self)->last,
                                       memory_order_acquire));
}

/* Timeline */

value caml_rig_amd_signaled(value v_self) {
  return Val_long(
      atomic_load_explicit(Edge_val(v_self)->word, memory_order_acquire));
}

/* Raises the word to the last value submit was given, with a release
   store, unless it is there already: the queues are destroyed, so nothing
   else writes it. */
value caml_rig_amd_raise(value v_self) {
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

/* Copies the 64-bit word at [v_from] to [v_to], with acquire and release
   order. */
value caml_rig_amd_copy_word(value v_to, value v_from) {
  uint64_t w = atomic_load_explicit((_Atomic uint64_t *)Host_val(v_from),
                                    memory_order_acquire);
  atomic_store_explicit((_Atomic uint64_t *)Host_val(v_to), w,
                        memory_order_release);
  return Val_unit;
}

/* Makes [v] the value after the device's last one; the device is idle. A
   ring that released the last value is taken to have released v - 1, which
   the word now holds. */
value caml_rig_amd_renumber(value v_self, value v_v) {
  struct rig_amd *d = Device_val(v_self);
  uint64_t prev = (uint64_t)Long_val(v_v) - 1;
  uint64_t last = atomic_load_explicit(&d->last, memory_order_relaxed);
  for (int q = 0; q < RIG_AMD_QUEUES; q++)
    if (d->rings[q].released == last) d->rings[q].released = prev;
  atomic_store_explicit(&d->last, prev, memory_order_relaxed);
  atomic_store_explicit(d->word, prev, memory_order_release);
  return Val_unit;
}

/* The words placed on ring [q] since it was made, and its size in words. */
value caml_rig_amd_placed(value v_self, value v_q) {
  CAMLparam2(v_self, v_q);
  CAMLlocal1(r);
  const struct rig_amd_ring *q = &Device_val(v_self)->rings[Int_val(v_q)];
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long(q->put));
  Store_field(r, 1, Val_long(q->size));
  CAMLreturn(r);
}
