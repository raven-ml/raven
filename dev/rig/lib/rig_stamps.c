/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stamps and the prepared form of submissions. None of these stubs
   blocks or releases the runtime. */

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
/* A prepared submission is held by a custom block whose finaliser frees
   it, with no OCaml, once the submission is collected. */
#define Sub_val(v) (*(struct rig_sub **)Data_custom_val(v))

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

/* Drops a reference to the stamps, freeing them with the last. */
value caml_rig_stamps_unref(value v_s) {
  struct rig_stamps *s = Stamps_val(v_s);
  if (atomic_fetch_sub(&s->refs, 1) != 1) return Val_unit;
  while (s != NULL) {
    struct rig_stamps *next = atomic_load(&s->next);
    free(s);
    s = next;
  }
  return Val_unit;
}

/* The use word of the device [index] in the stamps [s]: its existing one,
   or an empty one, claimed with the point (index, 0), in a chunk added if
   every word is taken. A raise through it allocates nothing. */
static _Atomic uint64_t *reserve(struct rig_stamps *s, int index) {
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

static void raise_max(_Atomic uint64_t *slot, uint64_t p) {
  uint64_t cur = atomic_load(slot);
  while (cur < p && !atomic_compare_exchange_weak(slot, &cur, p)) {
  }
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

/* Raises [v_dst] with every point of [v_src]: the stamps of memory put in a
   hold. */
value caml_rig_stamps_absorb(value v_dst, value v_src) {
  struct rig_stamps *dst = Stamps_val(v_dst), *src = Stamps_val(v_src);
  uint64_t w = atomic_load(&src->write);
  for (; src != NULL; src = atomic_load(&src->next))
    for (int i = 0; i < RIG_USES; i++) {
      uint64_t p = use_point(&src->use[i]);
      if (p == 0) continue;
      raise_max(reserve(dst, RIG_INDEX(p)), p);
    }
  if (w != 0) {
    uint64_t cur = atomic_load(&dst->write);
    if (cur == 0) atomic_compare_exchange_strong(&dst->write, &cur, w);
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

/* A prepared submission on the device [v_d] of [v_nparts] parts whose
   [after] lists hold [v_nafter] indices in all, naming [v_nfixed] buffers,
   with [v_nreads] read, [v_nwrites] write and [v_nwaits] wait slots. */
static void sub_free(struct rig_sub *s);

static void sub_finalize(value v) {
  rig_guard_destroy(Sub_val(v));
  sub_free(Sub_val(v));
}

static struct custom_operations sub_ops = {
    "rig.submission",   sub_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

value caml_rig_sub_new(value v_d, value v_nparts, value v_nafter,
                       value v_nfixed, value v_nreads, value v_nwrites,
                       value v_nwaits) {
  struct rig_sub *s = calloc(1, sizeof *s);
  if (s == NULL) caml_raise_out_of_memory();
  s->dev = (struct rig_device *)Long_val(v_d);
  s->nparts = Int_val(v_nparts);
  s->nfixed = Int_val(v_nfixed);
  s->nreads = Int_val(v_nreads);
  s->nwrites = Int_val(v_nwrites);
  s->nwait_slots = Int_val(v_nwaits);
  int nhandles = s->nfixed + s->nreads + s->nwrites;
  s->seen_bits = 1;
  while ((1 << s->seen_bits) < 2 * nhandles) s->seen_bits++;
  /* Every array, then one check: a refusal frees what was made. */
  int ok = 1;
  s->parts = zalloc((size_t)s->nparts, sizeof *s->parts, &ok);
  s->after = zalloc((size_t)Int_val(v_nafter), sizeof *s->after, &ok);
  s->fixed = zalloc((size_t)s->nfixed, sizeof *s->fixed, &ok);
  s->fixed_write = zalloc((size_t)s->nfixed, 1, &ok);
  s->slots = zalloc((size_t)(s->nreads + s->nwrites), sizeof *s->slots, &ok);
  s->wait_slots = zalloc((size_t)s->nwait_slots, sizeof *s->wait_slots, &ok);
  s->handles = zalloc((size_t)nhandles, sizeof *s->handles, &ok);
  s->seen = zalloc(nhandles == 0 ? 0 : (size_t)1 << s->seen_bits,
                   sizeof *s->seen, &ok);
  if (!ok) {
    sub_free(s);
    caml_raise_out_of_memory();
  }
  rig_guard_init(s);
  value v = caml_alloc_custom(&sub_ops, sizeof(struct rig_sub *), 0, 1);
  Sub_val(v) = s;
  return v;
}

value caml_rig_sub_new_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_new(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6]);
}

static void sub_free(struct rig_sub *s) {
  free(s->parts);
  free(s->after);
  free(s->fixed);
  free(s->fixed_write);
  free(s->slots);
  free(s->wait_slots);
  free(s->points);
  free(s->waits);
  free(s->producers);
  free(s->handles);
  free(s->seen);
  free(s->claims);
  free(s);
}


/* Part [v_i] on queue [v_queue], after the parts in [v_after], whose
   indices are stored from [v_at] on in the submission's [after]. Its work
   is set by one of the three below. */
value caml_rig_sub_part(value v_s, value v_i, value v_queue,
                                value v_after, value v_at) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_part *p = &s->parts[Int_val(v_i)];
  int at = Int_val(v_at), n = (int)Wosize_val(v_after);
  for (int j = 0; j < n; j++) s->after[at + j] = Int_val(Field(v_after, j));
  p->queue = Int_val(v_queue);
  p->after = n == 0 ? NULL : &s->after[at];
  p->nafter = n;
  return Val_unit;
}

value caml_rig_sub_words(value v_s, value v_i, value v_host,
                                 value v_n) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->words = (const uint32_t *)Long_val(v_host);
  p->n = (size_t)Long_val(v_n);
  return Val_unit;
}

value caml_rig_sub_fill(value v_s, value v_i, value v_fill,
                                value v_arg, value v_units, value v_bytes) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->fill = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  p->arg = (void *)Long_val(v_arg);
  p->ring_units = (size_t)Long_val(v_units);
  p->segment_bytes = (size_t)Long_val(v_bytes);
  return Val_unit;
}

value caml_rig_sub_fill_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_fill(argv[0], argv[1], argv[2], argv[3],
                                   argv[4], argv[5]);
}

value caml_rig_sub_copy(value v_s, value v_i, value v_args) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->copy_dst = (uint64_t)Nativeint_val(Field(v_args, 0));
  p->copy_dst_offset = (uint64_t)Long_val(Field(v_args, 1));
  p->copy_src = (uint64_t)Nativeint_val(Field(v_args, 2));
  p->copy_src_offset = (uint64_t)Long_val(Field(v_args, 3));
  p->copy_bytes = (uint64_t)Long_val(Field(v_args, 4));
  return Val_unit;
}

/* Fixed buffer [v_k]: the stamps and handle of a memory a part names, and
   whether the part writes it. */
value caml_rig_sub_fixed(value v_s, value v_k, value v_stamps,
                                 value v_handle, value v_write) {
  struct rig_sub *s = Sub_val(v_s);
  int k = Int_val(v_k);
  s->fixed[k].stamps = Stamps_val(v_stamps);
  s->fixed[k].handle = (uint64_t)Nativeint_val(v_handle);
  s->fixed_write[k] = (unsigned char)Bool_val(v_write);
  return Val_unit;
}

/* Read slot [v_k], or write slot [v_k - nreads]; stamps 0 unsets it. */
value caml_rig_sub_slot(value v_s, value v_k, value v_stamps,
                                value v_handle) {
  struct rig_slot *slot = &Sub_val(v_s)->slots[Int_val(v_k)];
  slot->stamps = Stamps_val(v_stamps);
  slot->handle = (uint64_t)Nativeint_val(v_handle);
  return Val_unit;
}

value caml_rig_sub_wait_slot(value v_s, value v_k, value v_p) {
  Sub_val(v_s)->wait_slots[Int_val(v_k)] = (uint64_t)Long_val(v_p);
  return Val_unit;
}

value caml_rig_sub_hold(value v_s, value v_stamps) {
  Sub_val(v_s)->hold = Stamps_val(v_stamps);
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
static void add_point(struct rig_sub *s, int own, uint64_t p) {
  if (p == 0 || RIG_INDEX(p) == own) return;
  for (int i = 0; i < s->npoints; i++)
    if (RIG_INDEX(s->points[i]) == RIG_INDEX(p)) {
      if (s->points[i] < p) s->points[i] = p;
      return;
    }
  grow((void **)&s->points, &s->cpoints, s->npoints + 1, sizeof *s->points);
  s->points[s->npoints++] = p;
}

/* Adds [h] to the handles once. A lookup in [seen], a table of twice their
   bound, takes a probe or two where a scan of the handles takes one per
   handle: a submission of 25 buffers would make 300 compares. */
static void add_handle(struct rig_sub *s, uint64_t h) {
  if (h == 0) return;
  uint64_t mask = ((uint64_t)1 << s->seen_bits) - 1;
  uint64_t i = (h * UINT64_C(0x9E3779B97F4A7C15)) >> (64 - s->seen_bits);
  for (; s->seen[i].epoch == s->epoch; i = (i + 1) & mask)
    if (s->seen[i].handle == h) return;
  s->seen[i] = (struct rig_seen){h, s->epoch};
  s->handles[s->nhandles++] = h;
}

/* Adds the points the use of [sl] follows: its last write, and every use if
   the work writes it. Reserves [own]'s use word in its stamps and adds its
   handle. */
static void add_slot(struct rig_sub *s, int own, struct rig_slot *sl,
                     int write) {
  struct rig_stamps *st = sl->stamps;
  sl->use = reserve(st, own);
  add_point(s, own, atomic_load(&st->write));
  if (write)
    for (; st != NULL; st = atomic_load(&st->next))
      for (int i = 0; i < RIG_USES; i++)
        add_point(s, own, use_point(&st->use[i]));
  add_handle(s, sl->handle);
}

/* Collects the points [s]'s work follows, the greatest per other device,
   and the handles of the memory it names, each once, and reserves the use
   words its raise stores to. Answers the number of points, or -1 if a read
   or write slot is unset. */
value caml_rig_sub_collect(value v_s) {
  struct rig_sub *s = Sub_val(v_s);
  int own = s->dev->index, nslots = s->nreads + s->nwrites;
  s->npoints = s->nwaits = s->nhandles = 0;
  s->epoch++;
  for (int k = 0; k < s->nfixed; k++)
    add_slot(s, own, &s->fixed[k], s->fixed_write[k]);
  for (int k = 0; k < nslots; k++) {
    if (s->slots[k].stamps == NULL) return Val_int(-1);
    add_slot(s, own, &s->slots[k], k >= s->nreads);
  }
  for (int k = 0; k < s->nwait_slots; k++)
    add_point(s, own, s->wait_slots[k]);
  if (s->hold != NULL) s->hold_use = reserve(s->hold, own);
  return Val_int(s->npoints);
}

value caml_rig_sub_point(value v_s, value v_i) {
  return Val_long((intnat)Sub_val(v_s)->points[Int_val(v_i)]);
}

/* Adds an in-queue wait on the producer [v_producer]'s value [v_value], at
   [v_at] by the kind [v_kind] (RIG_WORD, RIG_OBJECT). */
value caml_rig_sub_wait(value v_s, value v_producer, value v_at,
                                value v_value, value v_kind) {
  struct rig_sub *s = Sub_val(v_s);
  if (s->nwaits == s->cwaits) {
    int c = s->cwaits == 0 ? 8 : 2 * s->cwaits;
    struct rig_wait *waits = realloc(s->waits, (size_t)c * sizeof *waits);
    if (waits == NULL) caml_raise_out_of_memory();
    s->waits = waits;
    int *producers = realloc(s->producers, (size_t)c * sizeof *producers);
    if (producers == NULL) caml_raise_out_of_memory();
    s->producers = producers;
    s->cwaits = c;
  }
  s->waits[s->nwaits] = (struct rig_wait){(uint64_t)Long_val(v_at),
                                         (uint64_t)Long_val(v_value),
                                         Int_val(v_kind)};
  s->producers[s->nwaits] = Int_val(v_producer);
  s->nwaits++;
  return Val_unit;
}

/* Raises the stamps [s]'s work names to [p]. */
void rig_sub_raise(struct rig_sub *s, uint64_t p) {
  for (int k = 0; k < s->nfixed; k++) {
    if (s->fixed_write[k]) raise_last_write(s->fixed[k].stamps, p);
    raise_own(s->fixed[k].use, p);
  }
  for (int k = 0; k < s->nreads; k++) raise_own(s->slots[k].use, p);
  for (int k = s->nreads; k < s->nreads + s->nwrites; k++) {
    raise_last_write(s->slots[k].stamps, p);
    raise_own(s->slots[k].use, p);
  }
  if (s->hold != NULL) raise_own(s->hold_use, p);
}

/* Unsets [s]'s slots and forgets its waits. */
value caml_rig_sub_clear(value v_s) {
  struct rig_sub *s = Sub_val(v_s);
  if (s->nreads + s->nwrites > 0)
    memset(s->slots, 0, (size_t)(s->nreads + s->nwrites) * sizeof *s->slots);
  if (s->nwait_slots > 0)
    memset(s->wait_slots, 0, (size_t)s->nwait_slots * sizeof *s->wait_slots);
  s->npoints = s->nwaits = s->nhandles = 0;
  return Val_unit;
}

value caml_rig_sub_value(value v_s) {
  return Val_long((intnat)Sub_val(v_s)->v);
}

value caml_rig_sub_no_room_at(value v_s) {
  return Val_long((intnat)Sub_val(v_s)->no_room_at);
}

value caml_rig_sub_producer(value v_s) {
  return Val_int(Sub_val(v_s)->producer);
}

value caml_rig_sub_claims(value v_s) {
  struct rig_sub *s = Sub_val(v_s);
  value a = caml_alloc_tuple((mlsize_t)s->nclaims);
  for (int i = 0; i < s->nclaims; i++)
    Store_field(a, (mlsize_t)i, Val_int(s->claims[i]));
  return a;
}
