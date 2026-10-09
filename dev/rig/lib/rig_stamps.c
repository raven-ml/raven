/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stamps, the prepared form of submissions and runs. None of these
   stubs blocks or releases the runtime. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_stubs.h"

#define Stamps_val(v) ((struct rig_stamps *)Long_val(v))
/* A prepared submission and a run are each held by a custom block whose
   finaliser frees it, with no OCaml, once it is collected. */
#define Sub_val(v) (*(struct rig_sub **)Data_custom_val(v))
#define Run_val(v) (*(struct rig_run **)Data_custom_val(v))

/* A run buffer's address where its memory has none: the -1 OCaml passes. */
#define NO_ADDRESS UINT64_MAX

/* Stamps */

static struct rig_stamps *chunk(void) {
  struct rig_stamps *s = calloc(1, sizeof *s);
  if (s == NULL) caml_raise_out_of_memory();
  return s;
}

value caml_rig_stamps_new(value unit) {
  (void)unit;
  struct rig_stamps *s = chunk();
  atomic_store(&s->refs, 1);
  return Val_long((intnat)s);
}

value caml_rig_stamps_ref(value v_s) {
  atomic_fetch_add(&Stamps_val(v_s)->refs, 1);
  return Val_unit;
}

/* Drops a reference to the stamps [s], freeing them with the last. */
static void unref(struct rig_stamps *s) {
  if (atomic_fetch_sub(&s->refs, 1) != 1) return;
  while (s != NULL) {
    struct rig_stamps *next = atomic_load(&s->next);
    free(s);
    s = next;
  }
}

value caml_rig_stamps_unref(value v_s) {
  unref(Stamps_val(v_s));
  return Val_unit;
}

/* The use word of the device [index] in the stamps [s]: its existing one,
   or an empty one, claimed with the point (index, 0), in a chunk added if
   every word is taken. A raise through it allocates nothing. */
static _Atomic uint64_t *reserve_any(struct rig_stamps *s, int index) {
  for (;;) {
    for (int i = 0; i < RIG_USES; i++) {
      uint64_t p = atomic_load(&s->use[i]);
      if (p != 0 && RIG_INDEX(p) == index) return &s->use[i];
      if (p == 0) {
        uint64_t mine = RIG_POINT(index, 0);
        if (atomic_compare_exchange_strong(&s->use[i], &p, mine))
          return &s->use[i];
        if (RIG_INDEX(p) == index) return &s->use[i];
      }
    }
    struct rig_stamps *next = atomic_load(&s->next);
    if (next == NULL) {
      struct rig_stamps *grown = chunk();
      if (!atomic_compare_exchange_strong(&s->next, &next, grown)) free(grown);
      else next = grown;
    }
    s = next;
  }
}

/* [reserve_any], first trying the first word, which a memory one device
   uses holds. */
static inline _Atomic uint64_t *reserve(struct rig_stamps *s, int index) {
  uint64_t p = atomic_load_explicit(&s->use[0], memory_order_relaxed);
  if (p != 0 && RIG_INDEX(p) == index) return &s->use[0];
  return reserve_any(s, index);
}

/* The point the use word [w] holds: none while it holds the (index, 0) its
   reservation stored, before the first raise. */
static uint64_t use_point(_Atomic uint64_t *w) {
  uint64_t p = atomic_load(w);
  return RIG_VALUE(p) == 0 ? 0 : p;
}

/* Raises a submission's device's use word to its value [p]. Only that
   device's submissions raise the word, one at a time under its turn, each
   with a greater value than the last, so the raise is a store. */
static void raise_own(_Atomic uint64_t *slot, uint64_t p) {
  atomic_store_explicit(slot, p, memory_order_release);
}

/* Makes [p] the last write. A write of another device replaces the last
   write: its caller ordered the two writes. */
static void raise_last_write(struct rig_stamps *s, uint64_t p) {
  uint64_t cur = atomic_load(&s->write);
  for (;;) {
    uint64_t next = RIG_INDEX(cur) == RIG_INDEX(p) && cur > p ? cur : p;
    if (next == cur || atomic_compare_exchange_weak(&s->write, &cur, next))
      break;
  }
}

/* The [v_k]th point of the stamps: 0 the last write, then the uses in
   order; 0 for an empty slot, -1 past the last. */
value caml_rig_stamps_get(value v_s, value v_k) {
  struct rig_stamps *s = Stamps_val(v_s);
  intnat k = Long_val(v_k);
  if (k == 0) return Val_long((intnat)atomic_load(&s->write));
  k -= 1;
  for (; s != NULL; s = atomic_load(&s->next)) {
    if (k < RIG_USES) return Val_long((intnat)use_point(&s->use[k]));
    k -= RIG_USES;
  }
  return Val_long(-1);
}

/* Forgets every point of the stamps [v_s] but those of the device
   [v_index]: memory taken out of that device's cache, whose
   other points its cache reached before it took the memory in. Nothing else
   names the stamps then, so the stores race with nothing. */
value caml_rig_stamps_keep(value v_s, value v_index) {
  struct rig_stamps *s = Stamps_val(v_s);
  int index = Int_val(v_index);
  uint64_t w = atomic_load(&s->write);
  if (w != 0 && RIG_INDEX(w) != index) atomic_store(&s->write, 0);
  for (; s != NULL; s = atomic_load(&s->next))
    for (int i = 0; i < RIG_USES; i++) {
      uint64_t p = atomic_load(&s->use[i]);
      if (p != 0 && RIG_INDEX(p) != index) atomic_store(&s->use[i], 0);
    }
  return Val_unit;
}

/* Prepared submissions */

/* [n] zeroed elements of [size] bytes, or none for no element; clears [ok]
   if memory ran out. */
static void *zalloc(size_t n, size_t size, int *ok) {
  if (n == 0) return NULL;
  void *p = calloc(n, size);
  if (p == NULL) *ok = 0;
  return p;
}

static void sub_free(struct rig_sub *s) {
  free(s->parts);
  free(s->after);
  free(s->fixed);
  free(s->refs);
  free(s);
}

static void sub_finalize(value v) { sub_free(Sub_val(v)); }

static struct custom_operations sub_ops = {
    "rig.submission",   sub_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The ids of submissions: a run keeps its handles while it serves the
   submission it collected them for. */
static _Atomic uint64_t next_id = 1;

/* A prepared submission on the device [v_d] of [v_nparts] parts whose
   [after] lists hold [v_nafter] indices in all, naming [v_nfixed] buffers,
   whose runs read [v_nreads] buffers and write [v_nwrites], and whose
   launches hold [v_nrefs] refs in all. */
value caml_rig_sub_new(value v_d, value v_nparts, value v_nafter,
                       value v_nfixed, value v_nreads, value v_nwrites,
                       value v_nrefs) {
  struct rig_sub *s = calloc(1, sizeof *s);
  if (s == NULL) caml_raise_out_of_memory();
  s->id = atomic_fetch_add(&next_id, 1);
  s->dev = (struct rig_device *)Long_val(v_d);
  s->nparts = Int_val(v_nparts);
  s->nfixed = Int_val(v_nfixed);
  s->nreads = Int_val(v_nreads);
  s->nwrites = Int_val(v_nwrites);
  /* Every array, then one check: a refusal frees what was made. */
  int ok = 1;
  s->parts = zalloc((size_t)s->nparts, sizeof *s->parts, &ok);
  s->after = zalloc((size_t)Int_val(v_nafter), sizeof *s->after, &ok);
  s->fixed = zalloc((size_t)s->nfixed, sizeof *s->fixed, &ok);
  s->nrefs = Int_val(v_nrefs);
  s->refs = zalloc((size_t)s->nrefs, sizeof *s->refs, &ok);
  if (!ok) {
    sub_free(s);
    caml_raise_out_of_memory();
  }
  value v = caml_alloc_custom(&sub_ops, sizeof(struct rig_sub *), 0, 1);
  Sub_val(v) = s;
  return v;
}

value caml_rig_sub_new_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_new(argv[0], argv[1], argv[2], argv[3], argv[4],
                          argv[5], argv[6]);
}

/* Part [v_i] on queue [v_queue], after the parts in [v_after], whose
   indices are stored from [v_at] on in the submission's [after]. Its work
   is set by one of the three below. */
value caml_rig_sub_part(value v_s, value v_i, value v_queue, value v_after,
                        value v_at) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_part *p = &s->parts[Int_val(v_i)];
  int at = Int_val(v_at), n = (int)Wosize_val(v_after);
  for (int j = 0; j < n; j++) s->after[at + j] = Int_val(Field(v_after, j));
  p->queue = Int_val(v_queue);
  p->after = n == 0 ? NULL : &s->after[at];
  p->nafter = n;
  return Val_unit;
}

value caml_rig_sub_words(value v_s, value v_i, value v_host, value v_n) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->kind = RIG_WORDS;
  p->words.at = (const uint32_t *)Long_val(v_host);
  p->words.n = (size_t)Long_val(v_n);
  return Val_unit;
}

value caml_rig_sub_fill(value v_s, value v_i, value v_fill, value v_arg,
                        value v_units, value v_bytes) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->kind = RIG_FILL;
  p->fill.fn = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  p->fill.arg = (void *)Long_val(v_arg);
  p->fill.ring_units = (size_t)Long_val(v_units);
  p->fill.segment_bytes = (size_t)Long_val(v_bytes);
  return Val_unit;
}

value caml_rig_sub_fill_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_fill(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5]);
}

value caml_rig_sub_copy(value v_s, value v_i, value v_args) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->kind = RIG_COPY;
  p->copy.dst = (uint64_t)Nativeint_val(Field(v_args, 0));
  p->copy.dst_offset = (uint64_t)Long_val(Field(v_args, 1));
  p->copy.src = (uint64_t)Nativeint_val(Field(v_args, 2));
  p->copy.src_offset = (uint64_t)Long_val(Field(v_args, 3));
  p->copy.bytes = (uint64_t)Long_val(Field(v_args, 4));
  return Val_unit;
}

/* The side of part [v_i]'s copy that is memory of this process
   ([copy_local]). */
value caml_rig_sub_copy_local(value v_s, value v_i, value v_side) {
  Sub_val(v_s)->parts[Int_val(v_i)].copy.local = Int_val(v_side);
  return Val_unit;
}

/* The bytes of a block of [params] parameter bytes: its header, then its
   parameters, up to the 16 bytes the next block starts on. */
static size_t block_bytes(size_t params) {
  return sizeof(struct rig_block) + ((params + 15) & ~(size_t)15);
}

/* Part [v_i] launches the function [v_code], [v_launch] with [v_params]
   parameter bytes and the refs [v_refs], stored from [v_at] on in the
   submission's [refs]. Its block follows the block of the launch before
   it. Answers the block, as RIG_BLOCK_START and RIG_BLOCK_PARAMS read it. */
value caml_rig_sub_launch(value v_s, value v_i, value v_code, value v_launch,
                          value v_params, value v_refs, value v_at) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_part *p = &s->parts[Int_val(v_i)];
  int at = Int_val(v_at), n = (int)Wosize_val(v_refs);
  for (int j = 0; j < n; j++) {
    value r = Field(v_refs, j);
    s->refs[at + j] = (struct rig_ref){(uint32_t)Long_val(Field(r, 0)),
                                       (uint32_t)Long_val(Field(r, 1))};
  }
  size_t start = s->args, params = (size_t)Long_val(v_params);
  p->kind = RIG_LAUNCH;
  p->launch.code = (uint64_t)Long_val(v_code);
  p->launch.launch = (const void *)Nativeint_val(v_launch);
  p->launch.block = (uint32_t)start;
  p->launch.params = (uint32_t)params;
  p->launch.refs = n == 0 ? NULL : &s->refs[at];
  p->launch.nrefs = n;
  s->args = start + block_bytes(params);
  return Val_long((intnat)((start << RIG_BLOCK_BITS) | params));
}

value caml_rig_sub_launch_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_launch(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6]);
}

/* Fixed buffer [v_k]: the stamps and handle of a memory a part names, and
   whether the part writes it. */
value caml_rig_sub_fixed(value v_s, value v_k, value v_stamps, value v_handle,
                         value v_write) {
  struct rig_fixed *f = &Sub_val(v_s)->fixed[Int_val(v_k)];
  f->stamps = Stamps_val(v_stamps);
  f->handle = (uint64_t)Nativeint_val(v_handle);
  f->write = Bool_val(v_write);
  return Val_unit;
}

/* Runs */

static void run_free(struct rig_run *r) {
  free(r->fixed);
  free(r->slots);
  free(r->addresses);
  free(r->args);
  free(r->points);
  free(r->waits);
  free(r->producers);
  free(r->handles);
  free(r->seen);
  free(r);
}

static void run_finalize(value v) { run_free(Run_val(v)); }

static struct custom_operations run_ops = {
    "rig.run",                  run_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

value caml_rig_run_new(value unit) {
  (void)unit;
  struct rig_run *r = calloc(1, sizeof *r);
  if (r == NULL) caml_raise_out_of_memory();
  value v = caml_alloc_custom(&run_ops, sizeof(struct rig_run *), 0, 1);
  Run_val(v) = r;
  return v;
}

/* Grows the array [*a] of [*c] elements of [size] bytes to hold [want],
   zeroing what it adds: whether it could. */
static int fit(void **a, int *c, int want, size_t size) {
  if (want <= *c) return 1;
  void *p = realloc(*a, (size_t)want * size);
  if (p == NULL) return 0;
  memset((char *)p + (size_t)*c * size, 0, (size_t)(want - *c) * size);
  *a = p;
  *c = want;
  return 1;
}

/* Fits the run [r] to the submission [s]: its uses, slots and handles, and
   a table of handles of twice their bound. Whether memory sufficed. A run
   that served [s] last fits it: its arrays only grow. */
static int run_fit(struct rig_run *r, const struct rig_sub *s) {
  if (s->id == r->sub) return 1;
  int nslots = s->nreads + s->nwrites, nhandles = s->nfixed + nslots;
  int bits = 1;
  while ((1 << bits) < 2 * nhandles) bits++;
  if (bits > r->seen_bits) {
    struct rig_seen *seen = calloc((size_t)1 << bits, sizeof *seen);
    if (seen == NULL) return 0;
    free(r->seen);
    r->seen = seen;
    r->seen_bits = bits;
    r->epoch = 0;
  }
  /* The addresses grow with the slots, which [fit] counts. */
  int caddresses = r->cslots;
  if (!fit((void **)&r->fixed, &r->cfixed, s->nfixed, sizeof *r->fixed) ||
      !fit((void **)&r->addresses, &caddresses, nslots,
           sizeof *r->addresses) ||
      !fit((void **)&r->slots, &r->cslots, nslots, sizeof *r->slots) ||
      !fit((void **)&r->handles, &r->chandles, nhandles, sizeof *r->handles))
    return 0;
  r->sub = s->id;
  r->nslots = nslots;
  r->handles_stale = 1;
  return 1;
}

/* Takes the run [v_r] for a submit of [v_s]: 0 if another submit holds it,
   1 if it fits [v_s], 2 if [caml_rig_run_fit] must fit it first, and 3,
   leaving it, if it does not hold [v_s]'s last block. */
value caml_rig_run_take(value v_r, value v_s) {
  struct rig_run *r = Run_val(v_r);
  const struct rig_sub *s = Sub_val(v_s);
  int idle = 0;
  if (!atomic_compare_exchange_strong(&r->busy, &idle, 1)) return Val_int(0);
  if (r->nargs < s->args) {
    atomic_store(&r->busy, 0);
    return Val_int(3);
  }
  return Val_int(s->id == r->sub ? 1 : 2);
}

/* Fits the taken run [v_r] to [v_s], giving it back if memory ran out. */
value caml_rig_run_fit(value v_r, value v_s) {
  struct rig_run *r = Run_val(v_r);
  if (!run_fit(r, Sub_val(v_s))) {
    atomic_store(&r->busy, 0);
    caml_raise_out_of_memory();
  }
  return Val_unit;
}

/* Forgets the run's buffers but their handles, its hold and its waits, and
   gives it back. */
value caml_rig_run_give(value v_r) {
  struct rig_run *r = Run_val(v_r);
  for (int k = 0; k < r->nslots; k++) r->slots[k].stamps = NULL;
  r->hold = NULL;
  r->npoints = r->nwaits = 0;
  atomic_store(&r->busy, 0);
  return Val_unit;
}

/* Names the run's [v_k]th buffer, at [v_address] as the device's work
   addresses it, -1 for memory with no address: a read below [nreads], else
   a write. */
value caml_rig_run_slot(value v_r, value v_k, value v_stamps, value v_handle,
                        value v_address) {
  struct rig_run *r = Run_val(v_r);
  int k = Int_val(v_k);
  struct rig_slot *slot = &r->slots[k];
  uint64_t h = (uint64_t)Nativeint_val(v_handle);
  r->addresses[k] = (uint64_t)Long_val(v_address);
  slot->stamps = Stamps_val(v_stamps);
  if (slot->handle != h) {
    slot->handle = h;
    r->handles_stale = 1;
  }
  return Val_unit;
}

static void grow(void **a, int *c, int want, size_t size) {
  if (want <= *c) return;
  int n = want < 8 ? 8 : 2 * want;
  void *p = realloc(*a, (size_t)n * size);
  if (p == NULL) caml_raise_out_of_memory();
  *a = p;
  *c = n;
}

/* Keeps the greatest point per device, leaving out [own]'s: its own order
   covers its work. */
static void add_point(struct rig_run *r, int own, uint64_t p) {
  if (p == 0 || RIG_INDEX(p) == own) return;
  for (int i = 0; i < r->npoints; i++)
    if (RIG_INDEX(r->points[i]) == RIG_INDEX(p)) {
      if (r->points[i] < p) r->points[i] = p;
      return;
    }
  grow((void **)&r->points, &r->cpoints, r->npoints + 1, sizeof *r->points);
  r->points[r->npoints++] = p;
}

/* Adds [h] to the handles once. A lookup in [seen], a table of twice their
   bound, takes a probe or two where a scan of the handles takes one per
   handle: a submission of 25 buffers would make 300 compares. */
static void add_handle(struct rig_run *r, uint64_t h) {
  if (h == 0) return;
  uint64_t mask = ((uint64_t)1 << r->seen_bits) - 1;
  uint64_t i = (h * UINT64_C(0x9E3779B97F4A7C15)) >> (64 - r->seen_bits);
  for (; r->seen[i].epoch == r->epoch; i = (i + 1) & mask)
    if (r->seen[i].handle == h) return;
  r->seen[i] = (struct rig_seen){h, r->epoch};
  r->handles[r->nhandles++] = h;
}

static void add_uses(struct rig_run *r, int own, struct rig_stamps *st) {
  for (; st != NULL; st = atomic_load(&st->next))
    for (int i = 0; i < RIG_USES; i++)
      add_point(r, own, use_point(&st->use[i]));
}

/* Adds the points the use of the memory of the stamps [st] follows: its
   last write, and every use if the work writes it. Is [own]'s use word in
   the stamps, reserved. */
static _Atomic uint64_t *add_memory(struct rig_run *r, int own,
                                    struct rig_stamps *st, int write) {
  _Atomic uint64_t *use = reserve(st, own);
  uint64_t w = atomic_load_explicit(&st->write, memory_order_acquire);
  if (w != 0 && RIG_INDEX(w) != own) add_point(r, own, w);
  if (write) add_uses(r, own, st);
  return use;
}

/* The handles of the memory [s]'s work names in [r], each once. */
static void collect_handles(const struct rig_sub *s, struct rig_run *r,
                            int nslots) {
  r->nhandles = 0;
  r->epoch++;
  for (int k = 0; k < s->nfixed; k++) add_handle(r, s->fixed[k].handle);
  for (int k = 0; k < nslots; k++) add_handle(r, r->slots[k].handle);
  r->handles_stale = 0;
}

/* Collects in the run [v_r] the points [v_s]'s work follows, the greatest
   per other device, with the points of [v_waits], and the handles of the
   memory it names, each once, unless every handle is the last collect's,
   and reserves the use words its raise stores to, and [own]'s in the
   hold's stamps [v_hold], 0 for none: a hold orders nothing, so its points
   are not followed. Answers the number of points. */
value caml_rig_run_collect(value v_s, value v_r, value v_waits,
                           value v_hold) {
  const struct rig_sub *s = Sub_val(v_s);
  struct rig_run *r = Run_val(v_r);
  int own = s->dev->index, nslots = s->nreads + s->nwrites;
  for (int k = 0; k < s->nrefs; k++)
    if (r->addresses[s->refs[k].slot] == NO_ADDRESS)
      caml_invalid_argument(
          "Rig.submit: a launch addresses a run buffer whose memory has no "
          "address");
  r->npoints = r->nwaits = 0;
  r->hold = Stamps_val(v_hold);
  for (int k = 0; k < s->nfixed; k++)
    r->fixed[k] = add_memory(r, own, s->fixed[k].stamps, s->fixed[k].write);
  for (int k = 0; k < nslots; k++)
    r->slots[k].use = add_memory(r, own, r->slots[k].stamps, k >= s->nreads);
  for (mlsize_t k = 0; k < Wosize_val(v_waits); k++)
    add_point(r, own, (uint64_t)Long_val(Field(v_waits, k)));
  if (r->hold != NULL) r->hold_use = reserve(r->hold, own);
  if (r->handles_stale) collect_handles(s, r, nslots);
  return Val_int(r->npoints);
}

value caml_rig_run_point(value v_r, value v_i) {
  return Val_long((intnat)Run_val(v_r)->points[Int_val(v_i)]);
}

/* Adds an in-queue wait on the producer [v_producer]'s value [v_value], at
   [v_at] by the kind [v_kind] (RIG_WORD, RIG_OBJECT). */
value caml_rig_run_wait(value v_r, value v_producer, value v_at,
                        value v_value, value v_kind) {
  struct rig_run *r = Run_val(v_r);
  if (r->nwaits == r->cwaits) {
    int c = r->cwaits == 0 ? 8 : 2 * r->cwaits;
    struct rig_wait *waits = realloc(r->waits, (size_t)c * sizeof *waits);
    if (waits == NULL) caml_raise_out_of_memory();
    r->waits = waits;
    int *producers = realloc(r->producers, (size_t)c * sizeof *producers);
    if (producers == NULL) caml_raise_out_of_memory();
    r->producers = producers;
    r->cwaits = c;
  }
  r->waits[r->nwaits] = (struct rig_wait){(uint64_t)Long_val(v_at),
                                          (uint64_t)Long_val(v_value),
                                          Int_val(v_kind)};
  r->producers[r->nwaits] = Int_val(v_producer);
  r->nwaits++;
  return Val_unit;
}

/* Raises the stamps [s]'s work names in [r] to [p]. */
void rig_sub_raise(const struct rig_sub *s, struct rig_run *r, uint64_t p) {
  for (int k = 0; k < s->nfixed; k++) {
    if (s->fixed[k].write) raise_last_write(s->fixed[k].stamps, p);
    raise_own(r->fixed[k], p);
  }
  for (int k = 0; k < s->nreads; k++) raise_own(r->slots[k].use, p);
  for (int k = s->nreads; k < s->nreads + s->nwrites; k++) {
    raise_last_write(r->slots[k].stamps, p);
    raise_own(r->slots[k].use, p);
  }
  if (r->hold != NULL) raise_own(r->hold_use, p);
}

value caml_rig_run_value(value v_r) {
  return Val_long((intnat)Run_val(v_r)->v);
}

value caml_rig_run_no_room_at(value v_r) {
  return Val_long((intnat)Run_val(v_r)->no_room_at);
}

value caml_rig_run_producer(value v_r) {
  return Val_int(Run_val(v_r)->producer);
}

/* Blocks */

/* The [size] bytes at byte [at] of the run [v_r]'s block [b], growing the
   run to hold the whole block. Raises if a submit is using the run, or if
   [at] is negative or the bytes end past the block, its header's bytes and
   then its parameters. A store raises before it writes. */
static uint8_t *block_at(value v_r, intnat b, intnat at, intnat size) {
  struct rig_run *r = Run_val(v_r);
  if (atomic_load_explicit(&r->busy, memory_order_relaxed))
    caml_invalid_argument("Rig.Submission.Run: a submit is using the run");
  size_t params = RIG_BLOCK_PARAMS(b);
  if (at < 0 || (size_t)(at + size) > sizeof(struct rig_block) + params)
    caml_invalid_argument(
        "Rig.Submission.Run: the bytes lie outside the block's parameters");
  size_t start = RIG_BLOCK_START(b), end = start + block_bytes(params);
  if (end > r->cargs) {
    size_t c = 2 * r->cargs > end ? 2 * r->cargs : end;
    uint8_t *args = realloc(r->args, c);
    if (args == NULL) caml_raise_out_of_memory();
    memset(args + r->cargs, 0, c - r->cargs);
    r->args = args;
    r->cargs = c;
  }
  if (end > r->nargs) r->nargs = end;
  return r->args + start + at;
}

/* The parameters' byte [i], as [block_at] counts. */
#define PARAM(i) ((intnat)sizeof(struct rig_block) + (i))

/* A header field holds a uint32_t. */
static uint32_t header_word(intnat v) {
  if (v < 0 || v > (intnat)UINT32_MAX)
    caml_invalid_argument(
        "Rig.Submission.Run: a size is negative or above 2^32 - 1");
  return (uint32_t)v;
}

static void store_axes(value v_r, intnat b, size_t field, intnat x, intnat y,
                       intnat z) {
  uint32_t a[3] = {header_word(x), header_word(y), header_word(z)};
  memcpy(block_at(v_r, b, (intnat)field, sizeof a), a, sizeof a);
}

value caml_rig_run_groups(value v_r, intnat b, intnat x, intnat y, intnat z) {
  store_axes(v_r, b, offsetof(struct rig_block, groups), x, y, z);
  return Val_unit;
}

value caml_rig_run_groups_byte(value v_r, value b, value x, value y,
                               value z) {
  return caml_rig_run_groups(v_r, Long_val(b), Long_val(x), Long_val(y),
                             Long_val(z));
}

value caml_rig_run_threads(value v_r, intnat b, intnat x, intnat y,
                           intnat z) {
  store_axes(v_r, b, offsetof(struct rig_block, threads), x, y, z);
  return Val_unit;
}

value caml_rig_run_threads_byte(value v_r, value b, value x, value y,
                                value z) {
  return caml_rig_run_threads(v_r, Long_val(b), Long_val(x), Long_val(y),
                              Long_val(z));
}

value caml_rig_run_shared(value v_r, intnat b, intnat n) {
  uint32_t w = header_word(n);
  memcpy(block_at(v_r, b, offsetof(struct rig_block, shared), sizeof w), &w,
         sizeof w);
  return Val_unit;
}

value caml_rig_run_shared_byte(value v_r, value b, value n) {
  return caml_rig_run_shared(v_r, Long_val(b), Long_val(n));
}

/* A negative [i] stays negative through PARAM's sum, which [block_at]
   refuses only below the header: refuse it here. */
static uint8_t *param_at(value v_r, intnat b, intnat i, intnat size) {
  if (i < 0)
    caml_invalid_argument(
        "Rig.Submission.Run: the bytes lie outside the block's parameters");
  return block_at(v_r, b, PARAM(i), size);
}

value caml_rig_run_int32(value v_r, intnat b, intnat i, intnat v) {
  uint32_t x = (uint32_t)v;
  memcpy(param_at(v_r, b, i, sizeof x), &x, sizeof x);
  return Val_unit;
}

value caml_rig_run_int32_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_int32(v_r, Long_val(b), Long_val(i), Long_val(v));
}

value caml_rig_run_int64(value v_r, intnat b, intnat i, intnat v) {
  int64_t x = v;
  memcpy(param_at(v_r, b, i, sizeof x), &x, sizeof x);
  return Val_unit;
}

value caml_rig_run_int64_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_int64(v_r, Long_val(b), Long_val(i), Long_val(v));
}

value caml_rig_run_float32(value v_r, intnat b, intnat i, double v) {
  float x = (float)v;
  memcpy(param_at(v_r, b, i, sizeof x), &x, sizeof x);
  return Val_unit;
}

value caml_rig_run_float32_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_float32(v_r, Long_val(b), Long_val(i), Double_val(v));
}

value caml_rig_run_float64(value v_r, intnat b, intnat i, double v) {
  memcpy(param_at(v_r, b, i, sizeof v), &v, sizeof v);
  return Val_unit;
}

value caml_rig_run_float64_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_float64(v_r, Long_val(b), Long_val(i), Double_val(v));
}
