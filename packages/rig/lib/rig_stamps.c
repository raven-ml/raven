/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stamps, the prepared form of submissions and runs. None of these
   stubs blocks or releases the runtime. */

#define _GNU_SOURCE

#include <stdio.h>
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

/* A chunk's use words, or its keep words if [keep]. */
static _Atomic uint64_t *words(struct rig_stamps *s, int keep) {
  return keep ? s->keep : s->use;
}

/* The use word, or the keep word if [keep], of the device [index] in the
   stamps [s]: its existing one, or an empty one, claimed with the point
   (index, 0), in a chunk added if every word is taken. A raise through it
   allocates nothing. */
static _Atomic uint64_t *reserve_any(struct rig_stamps *s, int index,
                                     int keep) {
  for (;;) {
    _Atomic uint64_t *w = words(s, keep);
    for (int i = 0; i < RIG_USES; i++) {
      uint64_t p = atomic_load(&w[i]);
      if (p != 0 && RIG_INDEX(p) == index) return &w[i];
      if (p == 0) {
        uint64_t mine = RIG_POINT(index, 0);
        if (atomic_compare_exchange_strong(&w[i], &p, mine)) return &w[i];
        if (RIG_INDEX(p) == index) return &w[i];
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
  return reserve_any(s, index, 0);
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

/* [caml_rig_stamps_get]'s answer past the last point: a point of index 0,
   which no work has. Every other word is a point, -1 among them. */
#define PAST_LAST RIG_POINT(0, 1)

/* The [v_k]th point of the stamps: 0 the last write, then the uses in
   order; 0 for an empty slot, PAST_LAST past the last. */
value caml_rig_stamps_get(value v_s, value v_k) {
  struct rig_stamps *s = Stamps_val(v_s);
  intnat k = Long_val(v_k);
  if (k == 0) return Val_long((intnat)atomic_load(&s->write));
  k -= 1;
  for (; s != NULL; s = atomic_load(&s->next)) {
    if (k < RIG_USES) return Val_long((intnat)use_point(&s->use[k]));
    k -= RIG_USES;
  }
  return Val_long(PAST_LAST);
}

/* The [v_k]th keep point of the stamps, in order; 0 for an empty slot,
   PAST_LAST past the last. */
value caml_rig_stamps_kept(value v_s, value v_k) {
  struct rig_stamps *s = Stamps_val(v_s);
  intnat k = Long_val(v_k);
  for (; s != NULL; s = atomic_load(&s->next)) {
    if (k < RIG_USES) return Val_long((intnat)use_point(&s->keep[k]));
    k -= RIG_USES;
  }
  return Val_long(PAST_LAST);
}

/* Forgets every point of memory taken out of its device's cache: all its
   points settled before it entered the cache. No live operation reads or
   writes these stamps, so the stores race with nothing. */
value caml_rig_stamps_clear(value v_s) {
  struct rig_stamps *s = Stamps_val(v_s);
  atomic_store(&s->write, 0);
  for (; s != NULL; s = atomic_load(&s->next))
    for (int i = 0; i < RIG_USES; i++) {
      atomic_store(&s->use[i], 0);
      atomic_store(&s->keep[i], 0);
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
  rig_device_unref(s->dev);
  free(s->parts);
  free(s->scratch);
  free(s->after);
  free(s->fixed);
  free(s->uses);
  free(s->refs);
  free(s->codes);
  free(s);
}

static void sub_finalize(value v) { sub_free(Sub_val(v)); }

static struct custom_operations sub_ops = {
    "rig.submission",   sub_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The ids of submissions, which identify their blocks and fitted runs. */
static _Atomic uint64_t next_id = 1;

/* A prepared submission on the device [v_d] of [v_nparts] parts whose
   [after] lists hold [v_nafter] indices in all, naming [v_nfixed] buffers,
   whose runs use one buffer per access of [v_access] (an rig_use),
   and whose launches and fills hold [v_nrefs] buffer refs and [v_ncodes]
   entry refs in all. It holds the device's record until it is freed. */
value caml_rig_sub_new(value v_d, value v_nparts, value v_nafter,
                       value v_nfixed, value v_access, value v_nrefs,
                       value v_ncodes) {
  struct rig_sub *s = calloc(1, sizeof *s);
  if (s == NULL) caml_raise_out_of_memory();
  s->id = atomic_fetch_add(&next_id, 1);
  s->dev = Device_val(v_d);
  rig_device_ref(s->dev);
  s->nparts = Int_val(v_nparts);
  s->nfixed = Int_val(v_nfixed);
  s->nslots = (int)Wosize_val(v_access);
  /* Every array, then one check: a refusal frees what was made. */
  int ok = 1;
  s->parts = zalloc((size_t)s->nparts, sizeof *s->parts, &ok);
  s->scratch = zalloc((size_t)s->nparts, RIG_SCRATCH, &ok);
  s->after = zalloc((size_t)Int_val(v_nafter), sizeof *s->after, &ok);
  s->fixed = zalloc((size_t)s->nfixed, sizeof *s->fixed, &ok);
  s->uses = zalloc((size_t)s->nslots, sizeof *s->uses, &ok);
  s->nrefs = Int_val(v_nrefs);
  s->refs = zalloc((size_t)s->nrefs, sizeof *s->refs, &ok);
  s->ncodes = Int_val(v_ncodes);
  s->codes = zalloc((size_t)s->ncodes, sizeof *s->codes, &ok);
  if (!ok) {
    sub_free(s);
    caml_raise_out_of_memory();
  }
  for (int k = 0; k < s->nslots; k++)
    s->uses[k] = (uint8_t)Int_val(Field(v_access, k));
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

/* Both endpoint facts are replaced whenever a copy part is set. */
value caml_rig_sub_copy_facts(value v_s, value v_i, value v_side,
                             value v_device_memory) {
  struct rig_part *p = &Sub_val(v_s)->parts[Int_val(v_i)];
  p->copy.local = Int_val(v_side);
  p->copy.device_memory = Bool_val(v_device_memory);
  return Val_unit;
}

/* The bytes of a block of [params] parameter bytes: its header, then its
   parameters, up to the 16 bytes the next block starts on. */
static size_t block_bytes(size_t params) {
  return sizeof(struct rig_block) + ((params + 15) & ~(size_t)15);
}

/* Places the parameters [a] of a launch or a fill, of [v_params] bytes
   with the refs [v_refs], stored from [v_at] on in [s]'s refs, in a block
   after the last one's. Answers the block, as RIG_BLOCK_START and
   RIG_BLOCK_PARAMS read it. */
static intnat sub_args(struct rig_sub *s, struct rig_args *a, value v_params,
                       value v_refs, value v_at) {
  size_t start = s->args, params = (size_t)Long_val(v_params);
  size_t bytes = block_bytes(params);
  if (start > UINT32_MAX || bytes > UINT32_MAX - start)
    caml_invalid_argument(
        "Rig.Submission.make: run blocks exceed 2^32 - 1 bytes");
  int at = Int_val(v_at), n = (int)Wosize_val(v_refs);
  for (int j = 0; j < n; j++) {
    value r = Field(v_refs, j);
    s->refs[at + j] = (struct rig_ref){(uint32_t)Long_val(Field(r, 0)),
                                       (uint32_t)Long_val(Field(r, 1))};
  }
  a->block = (uint32_t)start;
  a->params = (uint32_t)params;
  a->refs = n == 0 ? NULL : &s->refs[at];
  a->nrefs = n;
  s->args = start + bytes;
  return (intnat)((start << RIG_BLOCK_BITS) | params);
}

/* Part [v_i] launches the function [v_code], [v_launch] with [v_params]
   parameter bytes and the refs [v_refs], stored from [v_at] on. */
value caml_rig_sub_launch(value v_s, value v_i, value v_code, value v_launch,
                          value v_params, value v_refs, value v_at) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_part *p = &s->parts[Int_val(v_i)];
  p->kind = RIG_LAUNCH;
  p->launch.code = (uint64_t)Long_val(v_code);
  p->launch.launch = (const void *)Nativeint_val(v_launch);
  return Val_long(sub_args(s, &p->launch.args, v_params, v_refs, v_at));
}

value caml_rig_sub_launch_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_launch(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6]);
}

/* The submission's [v_k]th entry ref writes [v_code] at byte [v_at] of the
   parameters of the block [v_block], as [sub_args] answers it. */
value caml_rig_sub_code(value v_s, value v_k, value v_block, value v_at,
                        value v_code) {
  struct rig_sub *s = Sub_val(v_s);
  s->codes[Int_val(v_k)] = (struct rig_code){
      RIG_BLOCK_START(Long_val(v_block)) + sizeof(struct rig_block) +
          (size_t)Long_val(v_at),
      (uint64_t)Long_val(v_code)};
  return Val_unit;
}

/* Part [v_i] calls the fill [v_fill] on [v_params] parameter bytes and the
   refs [v_refs], stored from [v_at] on. */
value caml_rig_sub_fill(value v_s, value v_i, value v_fill, value v_params,
                        value v_refs, value v_at) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_part *p = &s->parts[Int_val(v_i)];
  p->kind = RIG_FILL;
  p->fill.fn = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  return Val_long(sub_args(s, &p->fill.args, v_params, v_refs, v_at));
}

value caml_rig_sub_fill_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_fill(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5]);
}

/* The id of the submission [v_s], which its blocks carry. */
value caml_rig_sub_id(value v_s) { return Val_long((intnat)Sub_val(v_s)->id); }

/* Fixed buffer [v_k]: the stamps of a memory a part names and how the
   part uses it, an rig_use. */
value caml_rig_sub_fixed(value v_s, value v_k, value v_stamps, value v_use) {
  struct rig_fixed *f = &Sub_val(v_s)->fixed[Int_val(v_k)];
  f->stamps = Stamps_val(v_stamps);
  f->use = (enum rig_use)Int_val(v_use);
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

/* Fits the run [r] to the submission [s]: its uses, slots and addresses,
   and its blocks, which [s] then owns, its new bytes 0. Whether memory
   sufficed. A run that served [s] last fits its slots: its arrays only
   grow. */
static int run_fit(struct rig_run *r, const struct rig_sub *s) {
  if (s->args > 0 && r->nargs < s->args) {
    if (s->args > r->cargs) {
      uint8_t *args = realloc(r->args, s->args);
      if (args == NULL) return 0;
      memset(args + r->cargs, 0, s->args - r->cargs);
      r->args = args;
      r->cargs = s->args;
    }
    r->owner = s->id;
    r->nargs = s->args;
  }
  if (s->id == r->sub) return 1;
  int nslots = s->nslots;
  /* The addresses grow with the slots, which [fit] counts. */
  int caddresses = r->cslots;
  if (!fit((void **)&r->fixed, &r->cfixed, s->nfixed, sizeof *r->fixed) ||
      !fit((void **)&r->addresses, &caddresses, nslots,
           sizeof *r->addresses) ||
      !fit((void **)&r->slots, &r->cslots, nslots, sizeof *r->slots))
    return 0;
  r->sub = s->id;
  r->nslots = nslots;
  return 1;
}

/* Takes the run [v_r] for a submit of [v_s]: 0 if another submit holds it,
   4 if a store does, 1 if it fits [v_s], 2 if [caml_rig_run_fit] must fit
   it first, its slots or its blocks, and 3, leaving it, if it holds another
   submission's blocks. */
value caml_rig_run_take(value v_r, value v_s) {
  struct rig_run *r = Run_val(v_r);
  const struct rig_sub *s = Sub_val(v_s);
  int idle = RUN_IDLE;
  if (!atomic_compare_exchange_strong(&r->busy, &idle, RUN_SUBMIT))
    return Val_int(idle == RUN_SUBMIT ? 0 : 4);
  if (s->args > 0 && r->owner != s->id && r->nargs > 0) {
    atomic_store(&r->busy, RUN_IDLE);
    return Val_int(3);
  }
  int grow = s->args > 0 && r->nargs < s->args;
  return Val_int(s->id == r->sub && !grow ? 1 : 2);
}

/* Fits the taken run [v_r] to [v_s], giving it back if memory ran out. */
value caml_rig_run_fit(value v_r, value v_s) {
  struct rig_run *r = Run_val(v_r);
  if (!run_fit(r, Sub_val(v_s))) {
    atomic_store(&r->busy, RUN_IDLE);
    caml_raise_out_of_memory();
  }
  return Val_unit;
}

/* Forgets the run's buffers, its hold and its waits, and gives it back. */
value caml_rig_run_give(value v_r) {
  struct rig_run *r = Run_val(v_r);
  for (int k = 0; k < r->nslots; k++) r->slots[k].stamps = NULL;
  r->hold = NULL;
  r->npoints = r->nwaits = 0;
  atomic_store(&r->busy, RUN_IDLE);
  return Val_unit;
}

/* Names the run's [v_k]th buffer, at [v_address] as the device's work
   addresses it, -1 for memory with no address. */
value caml_rig_run_slot(value v_r, value v_k, value v_stamps, value v_address) {
  struct rig_run *r = Run_val(v_r);
  int k = Int_val(v_k);
  struct rig_slot *slot = &r->slots[k];
  r->addresses[k] = (uint64_t)Long_val(v_address);
  slot->stamps = Stamps_val(v_stamps);
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

static void add_uses(struct rig_run *r, int own, struct rig_stamps *st) {
  for (; st != NULL; st = atomic_load(&st->next))
    for (int i = 0; i < RIG_USES; i++)
      add_point(r, own, use_point(&st->use[i]));
}

/* Adds the points the use [u] of the memory of the stamps [st] follows: its
   last write, and every use if the work writes it; none if it is unordered.
   Is [own]'s use word in the stamps, reserved, its keep word for an
   unordered use. */
static _Atomic uint64_t *add_memory(struct rig_run *r, int own,
                                    struct rig_stamps *st, enum rig_use u) {
  if (u == RIG_USE_UNORDERED) return reserve_any(st, own, 1);
  _Atomic uint64_t *use = reserve(st, own);
  uint64_t w = atomic_load_explicit(&st->write, memory_order_acquire);
  if (w != 0 && RIG_INDEX(w) != own) add_point(r, own, w);
  if (u == RIG_USE_WRITE) add_uses(r, own, st);
  return use;
}

/* Writes each entry ref's value into the run [v_r]'s blocks, which the
   taken run holds whole. Collects in [v_r] the points [v_s]'s work follows,
   the greatest per other device, with the points of [v_waits], and reserves
   the use words its raise stores to, and [own]'s in the hold's stamps
   [v_hold], 0 for none: a hold orders nothing, so its points are not
   followed. Answers the number of points. */
value caml_rig_run_collect(value v_s, value v_r, value v_waits,
                           value v_hold) {
  const struct rig_sub *s = Sub_val(v_s);
  struct rig_run *r = Run_val(v_r);
  int own = s->dev->index, nslots = s->nslots;
  for (int k = 0; k < s->nrefs; k++)
    if (r->addresses[s->refs[k].slot] == NO_ADDRESS)
      caml_invalid_argument(
          "Rig.submit: a ref names a run buffer whose memory has no "
          "address");
  for (int k = 0; k < s->ncodes; k++)
    memcpy(r->args + s->codes[k].at, &s->codes[k].code, sizeof(uint64_t));
  r->npoints = r->nwaits = 0;
  r->hold = Stamps_val(v_hold);
  for (int k = 0; k < s->nfixed; k++)
    r->fixed[k] = add_memory(r, own, s->fixed[k].stamps, s->fixed[k].use);
  for (int k = 0; k < nslots; k++)
    r->slots[k].use =
        add_memory(r, own, r->slots[k].stamps, (enum rig_use)s->uses[k]);
  for (mlsize_t k = 0; k < Wosize_val(v_waits); k++)
    add_point(r, own, Unsigned_long_val(Field(v_waits, k)));
  if (r->hold != NULL) r->hold_use = reserve(r->hold, own);
  return Val_int(r->npoints);
}

/* Makes the run's next submit time its value, iff [v_timed]. */
value caml_rig_run_timed(value v_r, value v_timed) {
  struct rig_run *r = Run_val(v_r);
  r->timed = Bool_val(v_timed);
  r->pair = -1;
  return Val_unit;
}

/* The time pair the run's last submit took, -1 for none. */
value caml_rig_run_pair(value v_r) { return Val_int(Run_val(v_r)->pair); }

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
    if (s->fixed[k].use == RIG_USE_WRITE)
      raise_last_write(s->fixed[k].stamps, p);
    raise_own(r->fixed[k], p);
  }
  for (int k = 0; k < s->nslots; k++) {
    if (s->uses[k] == RIG_USE_WRITE) raise_last_write(r->slots[k].stamps, p);
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

/* A block, as Submission.block makes it: its submission's id, where it lies
   in a run, the most threads a group of its function has or 0 for none, and
   the function's name. */
#define BLOCK_SUB(v) ((uint64_t)Long_val(Field(v, 0)))
#define BLOCK_AT(v) Long_val(Field(v, 1))
#define BLOCK_MOST(v) Long_val(Field(v, 2))
#define BLOCK_FN(v) String_val(Field(v, 3))

/* Stores the [size] bytes at [src] at byte [at] of the run [v_r]'s block
   [v_b], growing the run to hold the whole block. It takes the run as a
   submit does, so that no submit reads the blocks and no other store grows
   them meanwhile. A run holds one submission's blocks: a store into
   another's first zeroes every byte the run holds. Raises, storing nothing,
   if a submit or another store is using the run, or if [at] is negative or
   the bytes end past the block, its header's bytes and then its
   parameters. */
static void block_store(value v_r, value v_b, intnat at, const void *src,
                        intnat size) {
  struct rig_run *r = Run_val(v_r);
  intnat b = BLOCK_AT(v_b);
  uint64_t sub = BLOCK_SUB(v_b);
  size_t params = RIG_BLOCK_PARAMS(b);
  if (at < 0 || (size_t)(at + size) > sizeof(struct rig_block) + params)
    caml_invalid_argument(
        "Rig.Submission.Run: the bytes lie outside the block's parameters");
  int idle = RUN_IDLE;
  if (!atomic_compare_exchange_strong(&r->busy, &idle, RUN_STORE))
    caml_invalid_argument(idle == RUN_SUBMIT
                              ? "Rig.Submission.Run: a submit is using the run"
                              : "Rig.Submission.Run: a store is using the run");
  if (r->owner != sub) {
    if (r->nargs > 0) memset(r->args, 0, r->nargs);
    r->nargs = 0;
    r->owner = sub;
  }
  size_t start = RIG_BLOCK_START(b), end = start + block_bytes(params);
  if (end > r->cargs) {
    size_t c = 2 * r->cargs > end ? 2 * r->cargs : end;
    uint8_t *args = realloc(r->args, c);
    if (args == NULL) {
      atomic_store(&r->busy, RUN_IDLE);
      caml_raise_out_of_memory();
    }
    memset(args + r->cargs, 0, c - r->cargs);
    r->args = args;
    r->cargs = c;
  }
  if (end > r->nargs) r->nargs = end;
  memcpy(r->args + start + at, src, (size_t)size);
  atomic_store(&r->busy, RUN_IDLE);
}

/* The parameters' byte [i], as [block_store] counts. */
#define PARAM(i) ((intnat)sizeof(struct rig_block) + (i))

/* A header field holds a uint32_t. */
static uint32_t header_word(intnat v) {
  if (v < 0 || v > (intnat)UINT32_MAX)
    caml_invalid_argument(
        "Rig.Submission.Run: a size is negative or above 2^32 - 1");
  return (uint32_t)v;
}

static void store_axes(value v_r, value b, size_t field, intnat x, intnat y,
                       intnat z) {
  uint32_t a[3] = {header_word(x), header_word(y), header_word(z)};
  block_store(v_r, b, (intnat)field, a, sizeof a);
}

value caml_rig_run_groups(value v_r, value b, intnat x, intnat y, intnat z) {
  store_axes(v_r, b, offsetof(struct rig_block, groups), x, y, z);
  return Val_unit;
}

value caml_rig_run_groups_byte(value v_r, value b, value x, value y,
                               value z) {
  return caml_rig_run_groups(v_r, b, Long_val(x), Long_val(y),
                             Long_val(z));
}

/* Refuses [x * y * z] threads per group past the most of the block's
   function, where it has one, leaving a size outside a header word to
   [store_axes]'s refusal. Each size is below 2^32, so the product of two fits
   64 bits, and the third multiplies a product no larger than the most, which
   is below 2^32 too. */
static void check_threads(value b, intnat x, intnat y, intnat z) {
  intnat most = BLOCK_MOST(b);
  if (most == 0 || x < 0 || y < 0 || z < 0 || x > (intnat)UINT32_MAX ||
      y > (intnat)UINT32_MAX || z > (intnat)UINT32_MAX)
    return;
  uint64_t xy = (uint64_t)x * (uint64_t)y;
  if (xy <= (uint64_t)most && xy * (uint64_t)z <= (uint64_t)most) return;
  char why[256];
  snprintf(why, sizeof why,
           "Rig.Submission.Run.threads: %s runs at most %ld threads per group; "
           "the block sets %ldx%ldx%ld",
           BLOCK_FN(b), (long)most, (long)x, (long)y, (long)z);
  caml_invalid_argument(why);
}

value caml_rig_run_threads(value v_r, value b, intnat x, intnat y,
                           intnat z) {
  check_threads(b, x, y, z);
  store_axes(v_r, b, offsetof(struct rig_block, threads), x, y, z);
  return Val_unit;
}

value caml_rig_run_threads_byte(value v_r, value b, value x, value y,
                                value z) {
  return caml_rig_run_threads(v_r, b, Long_val(x), Long_val(y),
                              Long_val(z));
}

value caml_rig_run_shared(value v_r, value b, intnat n) {
  uint32_t w = header_word(n);
  block_store(v_r, b, offsetof(struct rig_block, shared), &w, sizeof w);
  return Val_unit;
}

value caml_rig_run_shared_byte(value v_r, value b, value n) {
  return caml_rig_run_shared(v_r, b, Long_val(n));
}

/* A negative [i] stays negative through PARAM's sum, which [block_store]
   refuses only below the header: refuse it here. */
static void param_store(value v_r, value b, intnat i, const void *src,
                        intnat size) {
  if (i < 0)
    caml_invalid_argument(
        "Rig.Submission.Run: the bytes lie outside the block's parameters");
  block_store(v_r, b, PARAM(i), src, size);
}

value caml_rig_run_int32(value v_r, value b, intnat i, intnat v) {
  uint32_t x = (uint32_t)v;
  param_store(v_r, b, i, &x, sizeof x);
  return Val_unit;
}

value caml_rig_run_int32_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_int32(v_r, b, Long_val(i), Long_val(v));
}

value caml_rig_run_int64(value v_r, value b, intnat i, intnat v) {
  int64_t x = v;
  param_store(v_r, b, i, &x, sizeof x);
  return Val_unit;
}

value caml_rig_run_int64_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_int64(v_r, b, Long_val(i), Long_val(v));
}

value caml_rig_run_float32(value v_r, value b, intnat i, double v) {
  float x = (float)v;
  param_store(v_r, b, i, &x, sizeof x);
  return Val_unit;
}

value caml_rig_run_float32_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_float32(v_r, b, Long_val(i), Double_val(v));
}

value caml_rig_run_float64(value v_r, value b, intnat i, double v) {
  param_store(v_r, b, i, &v, sizeof v);
  return Val_unit;
}

value caml_rig_run_float64_byte(value v_r, value b, value i, value v) {
  return caml_rig_run_float64(v_r, b, Long_val(i), Double_val(v));
}
