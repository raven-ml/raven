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

/* A device's functions, and a measuring device's, whose facts list
   counters (Rig_edge: [measures] is NULL iff they list none). */
static const struct rig_driver driver = {rig_amd_room, rig_amd_submit,
                                         rig_amd_commit, rig_amd_settle, NULL};
static const struct rig_driver measuring = {rig_amd_room, rig_amd_submit,
                                            rig_amd_commit, rig_amd_settle,
                                            rig_amd_measures};

/* The header's values that Rig_amd and Template read rather than state,
   each by its name: caml_rig_amd_<name> is [v]. */
#define CONSTANT(name, v)                                                     \
  value caml_rig_amd_##name(value unit) {                                     \
    (void)unit;                                                               \
    return Val_int(v);                                                        \
  }

CONSTANT(ring_pm4, RING_PM4)
CONSTANT(ring_aql, RING_AQL)
CONSTANT(ring_sdma, RING_SDMA)
CONSTANT(template_words, RIG_AMD_TEMPLATE_WORDS)
CONSTANT(template_holes, RIG_AMD_TEMPLATE_HOLES)
CONSTANT(template_args, RIG_AMD_TEMPLATE_ARGS)
CONSTANT(launch_words, RIG_AMD_LAUNCH_WORDS)
CONSTANT(launch_holes, RIG_AMD_LAUNCH_HOLES)
CONSTANT(launch_args, RIG_AMD_LAUNCH_ARGS)
CONSTANT(templates, RIG_AMD_TEMPLATES)
CONSTANT(t_wait, T_WAIT)
CONSTANT(t_flush, T_FLUSH)
CONSTANT(t_acquire, T_ACQUIRE)
CONSTANT(t_wait64, T_WAIT64)
CONSTANT(t_signal, T_SIGNAL)
CONSTANT(t_release, T_RELEASE)
CONSTANT(t_write, T_WRITE)
CONSTANT(s_poll, S_POLL)
CONSTANT(s_fence, S_FENCE)
CONSTANT(s_trap, S_TRAP)
CONSTANT(s_copy, S_COPY)
CONSTANT(a_ib, A_IB)
CONSTANT(t_invalidate, T_INVALIDATE)
CONSTANT(t_stamp, T_STAMP)
CONSTANT(s_stamp, S_STAMP)
CONSTANT(s_fill, S_FILL)
CONSTANT(l_args, L_ARGS)
CONSTANT(l_scratch, L_SCRATCH)
CONSTANT(l_threads, L_THREADS)
CONSTANT(l_groups, L_GROUPS)
CONSTANT(l_grid, L_GRID)
CONSTANT(h_blocks, H_BLOCKS)
CONSTANT(h_group, H_GROUP)
CONSTANT(h_remainder, H_REMAINDER)
CONSTANT(h_offset, H_OFFSET)
CONSTANT(h_dims, H_DIMS)
CONSTANT(h_lds, H_LDS)
CONSTANT(hidden, RIG_AMD_HIDDEN)
CONSTANT(program_words, RIG_AMD_PROGRAM_WORDS)
CONSTANT(program_holes, RIG_AMD_PROGRAM_HOLES)
CONSTANT(record_samples, RIG_AMD_RECORD_SAMPLES)

/* Queue [v_q] of a device, which Rig_amd numbers as the header does. */
static int queue_of(value v_q) {
  int q = Int_val(v_q);
  if (q < 0 || q >= RIG_AMD_QUEUES)
    caml_invalid_argument("Rig_amd: no such queue");
  return q;
}

value caml_rig_amd_create(value unit) {
  (void)unit;
  struct rig_amd_edge *e = calloc(1, sizeof *e);
  struct rig_amd *d = calloc(1, sizeof *d);
  if (e == NULL || d == NULL || pthread_mutex_init(&d->grown_mu, NULL) != 0) {
    free(e);
    free(d);
    caml_raise_out_of_memory();
  }
  e->driver = &driver;
  e->d = d;
  return Val_long((intnat)e);
}

/* Frees the run state of a stopped device: its edge stays, and rig frees a
   word of its own. */
value caml_rig_amd_end(value v_self) {
  struct rig_amd_edge *e = Edge_val(v_self);
  struct rig_amd *d = e->d;
  e->d = NULL;
  rig_amd_counting_free(d->counting);
  free(d->grown);
  pthread_mutex_destroy(&d->grown_mu);
  free(d);
  return Val_unit;
}

/* Frees the state of a device whose open failed: nothing holds it. */
value caml_rig_amd_destroy(value v_self) {
  struct rig_amd_edge *e = Edge_val(v_self);
  rig_amd_counting_free(e->d->counting);
  free(e->d->grown);
  pthread_mutex_destroy(&e->d->grown_mu);
  free(e->d);
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
   (rig_amd_measure): whether it could. Releases the runtime, as every
   measure waits for stamps of the GPU's compute queue. */
value caml_rig_amd_clock(value v_self, value v_source) {
  struct rig_amd *d = Device_val(v_self);
  d->source = Host_val(v_source);
  caml_enter_blocking_section_no_pending();
  int ok = rig_amd_measure(d);
  caml_leave_blocking_section();
  return Val_bool(ok);
}

/* The argument segment, at open: [v_size] bytes, a power of two, at the
   host address [v_host] and the GPU's [v_gpu]. */
value caml_rig_amd_segment(value v_self, value v_host, value v_gpu,
                           value v_size) {
  struct rig_amd *d = Device_val(v_self);
  d->segment.host = Host_val(v_host);
  d->segment.gpu = (uint64_t)Long_val(v_gpu);
  d->segment.size = (uint64_t)Long_val(v_size);
  d->installed[0] = d->segment.gpu;
  return Val_unit;
}

/* The compute queue's AQL descriptor, at open: at the host address
   [v_host] and the GPU's [v_gpu]. */
value caml_rig_amd_descriptor(value v_self, value v_host, value v_gpu) {
  struct rig_amd *d = Device_val(v_self);
  d->desc = Host_val(v_host);
  d->desc_gpu = (uint64_t)Long_val(v_gpu);
  return Val_unit;
}

/* Growth (rig_amd_ring.c) */

/* What the submission whose rig_amd_room last answered RIG_GROW on this
   thread needs, as rig_amd_asked numbers it: the bytes the segment must
   have, then each ring, 0 for those the device's or the growth's handed
   hold; the record is consumed. Under the device's growth lock. */
value caml_rig_amd_needed(value v_self) {
  CAMLparam1(v_self);
  CAMLlocal1(v);
  struct rig_amd *d = Device_val(v_self);
  int mine = rig_amd_asked.d == d;
  rig_amd_asked.d = NULL;
  uint64_t w[1 + RIG_AMD_QUEUES] = {0};
  pthread_mutex_lock(&d->grown_mu);
  const struct rig_amd_grown *g = d->grown;
  if (mine && rig_amd_asked.bytes > d->segment.size &&
      (g == NULL || rig_amd_asked.bytes > g->size))
    w[0] = rig_amd_asked.bytes;
  for (int q = 0; q < RIG_AMD_QUEUES; q++) {
    uint64_t want = mine ? rig_amd_asked.rings[q] : 0;
    if (want > 4 * d->rings[q].size &&
        (g == NULL || want > 4 * g->ring[q].size))
      w[1 + q] = want;
  }
  pthread_mutex_unlock(&d->grown_mu);
  v = caml_alloc_tuple(1 + RIG_AMD_QUEUES);
  for (int i = 0; i <= RIG_AMD_QUEUES; i++)
    Store_field(v, i, Val_long((intnat)w[i]));
  CAMLreturn(v);
}

/* The fields of a grown queue in [caml_rig_amd_hand]'s [v_rings]. */
enum { q_words, q_bytes, q_write, q_doorbell, q_id, q_fields };

/* Hands the next room that needs it what a growth made, in place of what
   one handed before and no room installed, whose parts it keeps where this
   one makes none: [v_segment], the segment's host and GPU addresses and
   bytes, 0 bytes for none; [v_rings], for each queue, a new queue's ring's
   host address and bytes, its write-position word's and doorbell's host
   addresses and its name, 0 bytes for none; and [v_desc], the host and GPU
   addresses of a new AQL compute queue's descriptor, or 0. [false] without
   host memory for its record. Under the device's growth lock. */
value caml_rig_amd_hand(value v_self, value v_segment, value v_rings,
                        value v_desc) {
  struct rig_amd *d = Device_val(v_self);
  struct rig_amd_grown *g = malloc(sizeof *g);
  if (g == NULL) return Val_false;
  pthread_mutex_lock(&d->grown_mu);
  struct rig_amd_grown *handed = d->grown;
  if (handed != NULL) *g = *handed;
  else memset(g, 0, sizeof *g);
  if (at(v_segment, 2) > 0) {
    g->host = Host_val(Field(v_segment, 0));
    g->gpu = (uint64_t)at(v_segment, 1);
    g->size = (uint64_t)at(v_segment, 2);
  }
  for (int q = 0; q < RIG_AMD_QUEUES; q++) {
#define F(i) at(v_rings, q * q_fields + (i))
    if (F(q_bytes) == 0) continue;
    g->ring[q].words = (volatile uint32_t *)F(q_words);
    g->ring[q].size = (uint64_t)F(q_bytes) / 4;
    g->ring[q].write = (volatile uint64_t *)F(q_write);
    g->ring[q].doorbell = (volatile uint64_t *)F(q_doorbell);
    g->ring[q].id = (uint64_t)F(q_id);
#undef F
  }
  if (at(v_desc, 0) != 0) {
    g->desc = Host_val(Field(v_desc, 0));
    g->desc_gpu = (uint64_t)at(v_desc, 1);
  }
  d->grown = g;
  pthread_mutex_unlock(&d->grown_mu);
  free(handed);
  return Val_true;
}

/* Takes back from what a growth handed and no room installed its queue of
   ring [v_q], which then holds no work, so that the growth that replaces it
   destroys it before it makes its own. Under the device's growth lock. */
value caml_rig_amd_unhand(value v_self, value v_q) {
  struct rig_amd *d = Device_val(v_self);
  int q = queue_of(v_q);
  pthread_mutex_lock(&d->grown_mu);
  struct rig_amd_grown *g = d->grown;
  if (g != NULL && g->ring[q].size > 0) {
    memset(&g->ring[q], 0, sizeof g->ring[q]);
    if (q == RIG_AMD_COMPUTE) g->desc = NULL;
    int left = g->size > 0;
    for (int r = 0; r < RIG_AMD_QUEUES; r++) left |= g->ring[r].size > 0;
    if (!left) {
      d->grown = NULL;
      free(g);
    }
  }
  pthread_mutex_unlock(&d->grown_mu);
  return Val_unit;
}

/* The names of what is installed and of what is handed, 0 for none, in
   rig_amd_asked's order: the segment's, then each queue's ring's, read at
   one instant. */
value caml_rig_amd_growth(value v_self) {
  CAMLparam1(v_self);
  CAMLlocal1(v);
  struct rig_amd *d = Device_val(v_self);
  uint64_t names[2 * (1 + RIG_AMD_QUEUES)] = {0};
  pthread_mutex_lock(&d->grown_mu);
  for (int i = 0; i <= RIG_AMD_QUEUES; i++) names[i] = d->installed[i];
  const struct rig_amd_grown *g = d->grown;
  if (g != NULL) {
    names[1 + RIG_AMD_QUEUES] = g->size > 0 ? g->gpu : 0;
    for (int q = 0; q < RIG_AMD_QUEUES; q++)
      names[2 + RIG_AMD_QUEUES + q] = g->ring[q].size > 0 ? g->ring[q].id : 0;
  }
  pthread_mutex_unlock(&d->grown_mu);
  v = caml_alloc_tuple(2 * (1 + RIG_AMD_QUEUES));
  for (int i = 0; i < 2 * (1 + RIG_AMD_QUEUES); i++)
    Store_field(v, i, Val_long((intnat)names[i]));
  CAMLreturn(v);
}

/* Ring [q], at open: its words, its size and the most bytes of a ring of
   its path, its write-position word and doorbell, all host addresses, its
   kind (RING_PM4, RING_AQL, RING_SDMA), and its name, its GPU address. */
value caml_rig_amd_ring(value v_self, value v_q, value v_words,
                        value v_bytes, value v_most, value v_write,
                        value v_doorbell, value v_kind, value v_id) {
  struct rig_amd *d = Device_val(v_self);
  int q = queue_of(v_q), kind = Int_val(v_kind);
  if (kind != RING_PM4 && kind != RING_AQL && kind != RING_SDMA)
    caml_invalid_argument("Rig_amd: no such ring kind");
  struct rig_amd_ring *r = &d->rings[q];
  r->words = Host_val(v_words);
  r->size = (uint64_t)Long_val(v_bytes) / 4;
  r->most = (uint64_t)Long_val(v_most) / 4;
  r->write = Host_val(v_write);
  r->doorbell = Host_val(v_doorbell);
  r->kind = kind;
  d->installed[1 + q] = (uint64_t)Long_val(v_id);
  return Val_unit;
}

value caml_rig_amd_ring_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_amd_ring(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5], argv[6], argv[7], argv[8]);
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
    h[i] = (struct rig_amd_hole){.at = (uint16_t)f[0],
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
  if (Int_val(v_t) < 0 || Int_val(v_t) >= RIG_AMD_TEMPLATES ||
      bytes > sizeof c.words)
    caml_invalid_argument(OUT_OF_BOUNDS);
  c.n = (int)(bytes / 4);
  memcpy(c.words, String_val(v_words), 4 * (size_t)c.n);
  c.nholes = holes(v_holes, c.holes, RIG_AMD_TEMPLATE_HOLES,
                   RIG_AMD_TEMPLATE_ARGS, c.n);
  Device_val(v_self)->templates[Int_val(v_t)] = c;
  return Val_unit;
}

/* A macro that names RIG_AMD_HIDDEN after the enumerator would size the
   offsets past the widths. */
_Static_assert(sizeof rig_amd_hidden_bytes == RIG_AMD_HIDDEN,
               "an implicit argument's bytes for each of RIG_AMD_HIDDEN");

/* Reads into [c] what a launch of either form holds: its dispatch's words
   and holes, as a template's; [v_max], the function's own LDS bytes, its most
   threads per group and shared bytes, its arguments' bytes and its explicit
   ones', and
   the lanes of its waves; and [v_hidden], the offsets of its implicit
   arguments, each within its arguments, or -1. */
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
  c->lanes = (uint32_t)at(v_max, 5);
  if (c->lanes != 32 && c->lanes != 64) caml_invalid_argument(OUT_OF_BOUNDS);
  if (Wosize_val(v_hidden) != RIG_AMD_HIDDEN)
    caml_invalid_argument(OUT_OF_BOUNDS);
  for (int i = 0; i < RIG_AMD_HIDDEN; i++) {
    intnat h = at(v_hidden, i);
    int w = rig_amd_hidden_bytes[i];
    /* An enumerator the table gives no width would leave its bytes
       unwritten. */
    if (w != 2 && w != 4 && w != 8) caml_invalid_argument(OUT_OF_BOUNDS);
    if (h < -1 || (h >= 0 && h + w > (intnat)c->kernarg))
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

/* Reads program [p] from its words [v_words] and holes [v_holes], records
   as above whose argument is the record's address. Answers 0 if out of
   memory. */
static int program_of(struct rig_amd_program *p, value v_words,
                      value v_holes) {
  size_t bytes = caml_string_length(v_words);
  if (bytes > 4 * RIG_AMD_PROGRAM_WORDS || bytes % 4 != 0)
    caml_invalid_argument(OUT_OF_BOUNDS);
  p->n = (int)(bytes / 4);
  p->words = malloc(bytes + 4);
  p->holes = malloc(RIG_AMD_PROGRAM_HOLES * sizeof *p->holes);
  if (p->words == NULL || p->holes == NULL) return 0;
  memcpy(p->words, String_val(v_words), bytes);
  p->nholes = holes(v_holes, p->holes, RIG_AMD_PROGRAM_HOLES, 1, p->n);
  return 1;
}

/* Has the device measure counters, before its first submission: the
   filter kernel's launch [v_filter], which it copies; in [v_programs] the
   words and holes of the start and stop programs around a launch, then
   of the drain's; and in [v_values] the 8-byte values of a launch's
   samples, of a drain's, of SQ_WAVES, the samples' first, and of
   SQ_WAVES_SAVED, the next. */
value caml_rig_amd_counting(value v_self, value v_filter, value v_programs,
                            value v_values) {
  CAMLparam4(v_self, v_filter, v_programs, v_values);
  struct rig_amd *d = Device_val(v_self);
  struct rig_amd_counting *c = calloc(1, sizeof *c);
  if (c == NULL) caml_raise_out_of_memory();
  c->filter = *(const struct rig_amd_launch *)Long_val(v_filter);
  struct rig_amd_program *p[4] = {&c->start, &c->stop, &c->drain_start,
                                  &c->drain_stop};
  /* The state holds [c] while a refused program raises, so that the
     state's free frees it. */
  rig_amd_counting_free(d->counting);
  d->counting = c;
  int most = 0, made = 1;
  for (int i = 0; i < 4 && made; i++) {
    made = program_of(p[i], Field(v_programs, 2 * i),
                      Field(v_programs, 2 * i + 1));
    if (p[i]->n > most) most = p[i]->n;
  }
  c->scratch = malloc(4 * (size_t)most + 4);
  c->values = (uint64_t)at(v_values, 0);
  c->drain_values = (uint64_t)at(v_values, 1);
  c->waves = (uint64_t)at(v_values, 2);
  c->saved = (uint64_t)at(v_values, 3);
  atomic_init(&c->forced, -1);
  if (!made || c->scratch == NULL) {
    d->counting = NULL;
    rig_amd_counting_free(c);
    caml_raise_out_of_memory();
  }
  Edge_val(v_self)->driver = &measuring;
  CAMLreturn(Val_unit);
}

/* Has the next bracket store [v_word] as its filter word in place of its
   filter wave's. */
value caml_rig_amd_stale_filter(value v_self, value v_word) {
  struct rig_amd_counting *c = Device_val(v_self)->counting;
  if (c == NULL) caml_invalid_argument("Rig_amd: the device measures nothing");
  atomic_store(&c->forced, (int64_t)(uint32_t)Long_val(v_word));
  return Val_unit;
}

/* Sets the bytes one S_COPY moves and one S_FILL sets. */
value caml_rig_amd_copies(value v_self, value v_copy, value v_fill) {
  struct rig_amd *d = Device_val(v_self);
  d->max_copy = (uint64_t)Long_val(v_copy);
  d->max_fill = (uint64_t)Long_val(v_fill);
  return Val_unit;
}

/* Sets the sub-window's layout from [v_w], the fields of
   [struct rig_amd_window] in order. */
value caml_rig_amd_window(value v_self, value v_w) {
  struct rig_amd_window *w = &Device_val(v_self)->window;
  int *bits[] = {&w->element, &w->src_x,     &w->src_pitch, &w->dst_x,
                 &w->dst_pitch, &w->width, &w->rows};
  w->header = (uint32_t)Long_val(Field(v_w, 0));
  for (int i = 0; i < 7; i++) *bits[i] = (int)Long_val(Field(v_w, 1 + i));
  w->most_width = (uint64_t)Long_val(Field(v_w, 8));
  w->most_rows = (uint64_t)Long_val(Field(v_w, 9));
  w->most_pitch = (uint64_t)Long_val(Field(v_w, 10));
  return Val_unit;
}

/* The device's HDP register, the path's, set at open before any
   submission. */
value caml_rig_amd_set_hdp(value v_self, value v_reg) {
  Edge_val(v_self)->d->hdp = Host_val(v_reg);
  return Val_unit;
}

/* Rig_amd.flush: the calling thread's earlier stores through the memory BAR
   reach the GPU's memory before it returns. The full barrier drains the
   thread's write-combining buffers before the store to the HDP register,
   whose read back returns once the HDP wrote out what it held, as
   hand_over's flush does. rig counts the call, so the device's stop, after
   which the path may unmap the register, never runs during it. */
value caml_rig_amd_flush(value v_self) {
  struct rig_amd *d = Edge_val(v_self)->d;
  if (d->hdp != NULL) {
    rig_amd_barrier();
    d->hdp[0] = 0;
    (void)d->hdp[0];
  }
  return Val_unit;
}

/* Adds [v_delta] to the regions of [v_self] another function was given
   ([Rig_amd.dma]), whose writes reach the GPU's memory through its host data
   path: a submission flushes that path while any is counted (hand_over). */
value caml_rig_amd_given(value v_self, value v_delta) {
  struct rig_amd *d = Edge_val(v_self)->d;
  atomic_fetch_add_explicit(&d->given, Int_val(v_delta), memory_order_acq_rel);
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
   it to an AQL queue's descriptor: the 32-bit [values] at the offsets [at]
   of the descriptor, which the next submission places. Answers the value
   whose submission placed the publication this one replaces, or 0 if none
   did, read under the lock. */
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

/* The words placed on ring [q] since it was made, and its size in words,
   read under the growth lock, under which an install, in another domain's
   submit, writes both. A submit places words without it: while one runs,
   the count may be an earlier one. */
value caml_rig_amd_placed(value v_self, value v_q) {
  CAMLparam2(v_self, v_q);
  CAMLlocal1(r);
  struct rig_amd *d = Device_val(v_self);
  const struct rig_amd_ring *q = &d->rings[queue_of(v_q)];
  pthread_mutex_lock(&d->grown_mu);
  uint64_t put = q->put, size = q->size;
  pthread_mutex_unlock(&d->grown_mu);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long((intnat)put));
  Store_field(r, 1, Val_long((intnat)size));
  CAMLreturn(r);
}

/* The bytes of the argument segment installed, read under the growth lock,
   under which an install writes them. */
value caml_rig_amd_segment_bytes(value v_self) {
  struct rig_amd *d = Device_val(v_self);
  pthread_mutex_lock(&d->grown_mu);
  uint64_t size = d->segment.size;
  pthread_mutex_unlock(&d->grown_mu);
  return Val_long((intnat)size);
}
