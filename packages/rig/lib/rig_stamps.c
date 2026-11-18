/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Stamps, the prepared form of submissions and the calls that serve their
   submits. None of these stubs blocks or releases the runtime. */

#define _GNU_SOURCE

#include <stdint.h>
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
   it, with no OCaml, once it is collected. A call is a pointer that OCaml
   holds as an int. */
#define Sub_val(v) (*(struct rig_sub **)Data_custom_val(v))
#define Call_val(v) ((struct rig_call *)Long_val(v))

/* A submit buffer's address where its memory has none: the -1 OCaml
   passes. */
#define UNADDRESSED UINT64_MAX

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

/* The write word beside the use word [use]: its chunk's write[i] for its
   use[i]. */
static _Atomic uint64_t *write_word(_Atomic uint64_t *use) {
  ptrdiff_t at = (ptrdiff_t)offsetof(struct rig_stamps, write) -
                 (ptrdiff_t)offsetof(struct rig_stamps, use);
  return (_Atomic uint64_t *)((char *)use + at);
}

/* Whether [p] is a point of an earlier device of its slot: at most the
   value [base] the slot's device started after, read from its slot's line,
   never its device. */
static int below(uint64_t p, uint64_t base) {
  return RIG_VALUE(p) != 0 && RIG_VALUE(p) <= base;
}

/* Whether [p] is the word of the device of slot [index] that started after
   [base]: its reservation, which holds no point, or a point above [base].
   An earlier device's point of the slot is another device's. */
static int is_own(uint64_t p, int index, uint64_t base) {
  return p != 0 && RIG_INDEX(p) == index &&
         (RIG_VALUE(p) == 0 || RIG_VALUE(p) > base);
}

/* The word of the device of slot [index] that started after [base] among
   the use words of [s], or its keep words if [keep], or NULL. */
static _Atomic uint64_t *own_word(struct rig_stamps *s, int index,
                                  uint64_t base, int keep) {
  for (; s != NULL; s = atomic_load(&s->next)) {
    _Atomic uint64_t *w = words(s, keep);
    for (int i = 0; i < RIG_USES; i++)
      if (is_own(atomic_load(&w[i]), index, base)) return &w[i];
  }
  return NULL;
}

/* Whether another device's reservation may take the word [w] of the stamps
   [s], which holds [p]:
   - an empty word;
   - a point of a device whose stop was claimed, done: a submit checks that
     its device is not lost under its turn before it raises, so no raise
     stores to the word again;
   - an earlier device's point of its slot that is done;
   - one a loss dropped: a keep word, whose point is settled, or a use word
     once its write word's point, if a loss dropped it too, is folded into
     [s]'s head. A dropped read of an earlier device, which stopped before
     its slot passed on, counts as done ([rig_use_followed]). The fold comes
     before the take, so a walk that misses the point in the word finds it
     in the head. The first use word is not taken while the second is
     empty, so that a folded point implies a second use word: memory whose
     first word is its device's and second is empty has nothing folded.
   A reservation that holds no point is never taken: only the next device
   of its slot adopts it, under that device's turn. */
static int takeable(struct rig_stamps *s, _Atomic uint64_t *w, uint64_t p,
                    int keep) {
  if (p == 0) return 1;
  if (RIG_VALUE(p) == 0) return 0;
  if (!below(p, rig_line_base(RIG_INDEX(p)))) {
    int st = atomic_load_explicit(&rig_lines[RIG_INDEX(p)].state,
                                  memory_order_acquire);
    return (st == RIG_STOPPING || st == RIG_ENDED) && rig_point_done(p);
  }
  if (!rig_point_lost(p) || keep) return 1;
  if (w == &s->use[0] &&
      atomic_load_explicit(&s->use[1], memory_order_acquire) == 0)
    return 0;
  uint64_t wp = atomic_load(write_word(w)), none = 0;
  if (below(wp, rig_line_base(RIG_INDEX(wp))) && rig_point_lost(wp))
    atomic_compare_exchange_strong(&s->lost_write, &none, wp);
  return 1;
}

/* The use word, or the keep word if [keep], of the device of slot [index] that
   started after [base] in the stamps [s]: its existing one, or the first
   another may take ([takeable]), claimed with the point (index, 0), in a chunk
   added if none is. A memory's words thus number the devices whose work on it
   may still matter, and a walk's cost follows them. A raise through it
   allocates nothing. A word is taken only by compare-and-set, so a reservation
   that races a reclaim of the same word takes it once. */
static _Atomic uint64_t *reserve_any(struct rig_stamps *s, int index,
                                     uint64_t base, int keep) {
  uint64_t mine = RIG_POINT(index, 0);
  for (;;) {
    _Atomic uint64_t *own = own_word(s, index, base, keep);
    if (own != NULL) return own;
    _Atomic uint64_t *free_ = NULL;
    uint64_t seen = 0;
    struct rig_stamps *last = s;
    for (struct rig_stamps *c = s; c != NULL && free_ == NULL;
         c = atomic_load(&c->next)) {
      _Atomic uint64_t *w = words(c, keep);
      for (int i = 0; i < RIG_USES && free_ == NULL; i++) {
        uint64_t p = atomic_load(&w[i]);
        if (takeable(s, &w[i], p, keep)) {
          free_ = &w[i];
          seen = p;
        }
      }
      last = c;
    }
    if (free_ != NULL) {
      if (atomic_compare_exchange_strong(free_, &seen, mine)) return free_;
      continue;
    }
    struct rig_stamps *grown = chunk(), *none = NULL;
    if (!atomic_compare_exchange_strong(&last->next, &none, grown)) free(grown);
  }
}

/* The point the use word [w] holds: none while it holds the (index, 0) its
   reservation stored, before the first raise. */
static uint64_t use_point(_Atomic uint64_t *w) {
  uint64_t p = atomic_load(w);
  return RIG_VALUE(p) == 0 ? 0 : p;
}

/* Raises a submission's device's use or write word to its value [p]. Only
   that device's submissions raise the word, one at a time under its turn,
   each with a greater value than the last, so the raise is a store. */
static void raise_own(_Atomic uint64_t *slot, uint64_t p) {
  atomic_store_explicit(slot, p, memory_order_release);
}

/* A walk of stamps reads each chunk's words in turn, stepping to the next
   chunk itself: each read is one call, whatever the number of chunks. */

value caml_rig_stamps_uses(value unit) {
  (void)unit;
  return Val_int(RIG_USES);
}

/* rig_stamps_followed of the chunk [v_c]'s words [v_i], for [v_every], with
   OCaml's min_int for its -1: a point from index 32768 on is a negative
   OCaml int, and min_int, whose value bits are 0, is no point. */
value caml_rig_stamps_followed(value v_c, value v_i, value v_every) {
  int64_t p = rig_stamps_followed(Stamps_val(v_c), Int_val(v_i),
                                  Bool_val(v_every));
  return Val_long(p < 0 ? Min_long : (intnat)p);
}

/* The point of the chunk [v_c]'s keep word [v_i], 0 for none. */
value caml_rig_stamps_keep(value v_c, value v_i) {
  return Val_long((intnat)use_point(&Stamps_val(v_c)->keep[Int_val(v_i)]));
}

/* The chunk after [v_c], 0 for none. */
value caml_rig_stamps_next(value v_c) {
  return Val_long((intnat)atomic_load(&Stamps_val(v_c)->next));
}

/* Forgets every point of memory taken out of its device's cache: all its
   points settled before it entered the cache. No live buffer reads or
   writes these stamps. A wait that began on a buffer a free then killed may
   still read them (Memory.release_early); the words are atomic, so it reads
   points of this use or the next, and only waits longer. */
value caml_rig_stamps_clear(value v_s) {
  struct rig_stamps *s = Stamps_val(v_s);
  atomic_store(&s->lost_write, 0);
  for (; s != NULL; s = atomic_load(&s->next))
    for (int i = 0; i < RIG_USES; i++) {
      atomic_store(&s->use[i], 0);
      atomic_store(&s->write[i], 0);
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
  free(s->parts);
  free(s->scratch);
  free(s->after);
  free(s->fixed);
  free(s->uses);
  free(s->refs);
  free(s->codes);
  free(s->copies);
  free(s->blocks);
  free(s);
}

static void sub_finalize(value v) { sub_free(Sub_val(v)); }

static struct custom_operations sub_ops = {
    "rig.submission",   sub_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The ids of submissions, which identify the calls fitted to them. */
static _Atomic uint64_t next_id = 1;

/* A prepared submission on the device [v_d] of [v_nparts] parts whose
   [after] lists hold [v_nafter] indices in all, naming [v_nfixed] buffers,
   whose submits use one buffer per access of [v_access] (an rig_use),
   whose [v_nblocks] launches and fills hold [v_nrefs] buffer refs and
   [v_ncodes] entry refs in all, and [v_ncopies] of whose parts are
   copies. */
value caml_rig_sub_new(value v_d, value v_nparts, value v_nafter,
                       value v_nfixed, value v_access, value v_nblocks,
                       value v_nrefs, value v_ncodes, value v_ncopies) {
  struct rig_sub *s = calloc(1, sizeof *s);
  if (s == NULL) caml_raise_out_of_memory();
  s->id = atomic_fetch_add(&next_id, 1);
  s->dev = Device_val(v_d);
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
  s->ncopies = Int_val(v_ncopies);
  s->copies = zalloc((size_t)s->ncopies, sizeof *s->copies, &ok);
  s->nblocks = Int_val(v_nblocks);
  s->blocks = zalloc((size_t)s->nblocks, sizeof *s->blocks, &ok);
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
                          argv[5], argv[6], argv[7], argv[8]);
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

/* The submission's [v_k]th copy is part [v_i], from the submit's slot
   [v_src] into its slot [v_dst], each -1 for a buffer that
   [caml_rig_sub_side] sets. */
value caml_rig_sub_copy(value v_s, value v_k, value v_i, value v_src,
                        value v_dst) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_copy *c = &s->copies[Int_val(v_k)];
  s->parts[Int_val(v_i)].kind = RIG_COPY;
  c->part = Int_val(v_i);
  c->src = Int_val(v_src);
  c->dst = Int_val(v_dst);
  c->boxed = 0;
  return Val_unit;
}

/* The submission's [v_k]th copy is the box [v_box], a Def.box whose fields
   C reads in order, between its two buffers. */
value caml_rig_sub_box(value v_s, value v_k, value v_box) {
  struct rig_copy *c = &Sub_val(v_s)->copies[Int_val(v_k)];
  c->boxed = 1;
  c->box.src_offset = (uint64_t)Long_val(Field(v_box, 0));
  c->box.dst_offset = (uint64_t)Long_val(Field(v_box, 1));
  c->box.width = (uint64_t)Long_val(Field(v_box, 2));
  c->box.rows = (uint64_t)Long_val(Field(v_box, 3));
  c->box.src_pitch = (uint64_t)Long_val(Field(v_box, 4));
  c->box.dst_pitch = (uint64_t)Long_val(Field(v_box, 5));
  return Val_unit;
}

/* The submission's [v_k]th copy is part [v_i], a fill of the submit's slot
   [v_dst], or of a buffer that [caml_rig_sub_side] sets for -1, with the
   byte [v_byte]. */
value caml_rig_sub_set(value v_s, value v_k, value v_i, value v_dst,
                       value v_byte) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_copy *c = &s->copies[Int_val(v_k)];
  s->parts[Int_val(v_i)].kind = RIG_COPY;
  memset(c, 0, sizeof *c);
  c->part = Int_val(v_i);
  c->src = -1;
  c->dst = Int_val(v_dst);
  c->fill = 1;
  c->byte = (uint8_t)Int_val(v_byte);
  return Val_unit;
}

/* Sets the buffer [v_b] as the [v_k]th copy's destination if [v_dst], its
   source otherwise, passed as this process's memory iff [v_local], with the
   copy.device_memory fact [v_device_memory]. The side's alias is read once,
   here: a copy runs on no io device, so an io memory end is a borrow, and
   borrowing io memory of any byte asked for its pages, which an entry
   publishes once and keeps. */
value caml_rig_sub_side(value v_s, value v_k, value v_dst, value v_b,
                        value v_local, value v_device_memory) {
  struct rig_side *side =
      &Sub_val(v_s)->copies[Int_val(v_k)].ends[Bool_val(v_dst)];
  rig_buffer_side(v_b, Bool_val(v_local), side);
  side->device_memory = Bool_val(v_device_memory);
  return Val_unit;
}

value caml_rig_sub_side_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_side(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5]);
}

/* Sets the fields of [p], the copy [c] from [src] into [dst], or its fill
   of [dst], which reads no memory. */
static void set_copy(struct rig_part *p, const struct rig_copy *c,
                     const struct rig_side *src, const struct rig_side *dst) {
  if (c->fill) {
    memset(&p->copy, 0, sizeof p->copy);
    p->copy.dst = dst->handle;
    p->copy.dst_offset = dst->offset;
    p->copy.width = p->copy.dst_pitch = dst->bytes;
    p->copy.rows = 1;
    p->copy.device_memory = dst->device_memory;
    p->copy.fill = 1;
    p->copy.byte = c->byte;
    return;
  }
  p->copy.dst = dst->handle;
  p->copy.src = src->handle;
  if (c->boxed) {
    p->copy.dst_offset = dst->offset + c->box.dst_offset;
    p->copy.src_offset = src->offset + c->box.src_offset;
    p->copy.width = c->box.width;
    p->copy.rows = c->box.rows;
    p->copy.src_pitch = c->box.src_pitch;
    p->copy.dst_pitch = c->box.dst_pitch;
  } else {
    p->copy.dst_offset = dst->offset;
    p->copy.src_offset = src->offset;
    p->copy.width = src->bytes;
    p->copy.rows = 1;
    p->copy.src_pitch = p->copy.dst_pitch = src->bytes;
  }
  p->copy.local = src->local   ? RIG_LOCAL_SRC
                  : dst->local ? RIG_LOCAL_DST
                               : RIG_LOCAL_NONE;
  p->copy.device_memory = src->device_memory && dst->device_memory;
}

/* Sets each copy between two buffers in the submission's parts, which make
   checked, and counts the copies with a slot end, which each submit sets. */
value caml_rig_sub_seal(value v_s) {
  struct rig_sub *s = Sub_val(v_s);
  for (int k = 0; k < s->ncopies; k++) {
    struct rig_copy *c = &s->copies[k];
    if (c->src < 0 && c->dst < 0)
      set_copy(&s->parts[c->part], c, &c->ends[0], &c->ends[1]);
    else
      s->nslotted++;
  }
  return Val_unit;
}

/* Places the parameters [a] of a launch or a fill, of [v_params] bytes
   with the refs [v_refs], stored from [v_at] on in [s]'s refs, in a block
   after the last one's. [v_params] is below 2^62, so [rig_block_bytes] does
   not wrap. */
static void sub_args(struct rig_sub *s, struct rig_args *a, value v_params,
                     value v_refs, value v_at) {
  uint64_t start = s->args, params = (uint64_t)Long_val(v_params);
  uint64_t bytes = rig_block_bytes(params);
  if (start > UINT32_MAX || bytes > UINT32_MAX - start)
    caml_invalid_argument(
        "Rig.Submission.make: the blocks of the launches and fills take more "
        "than 2^32 - 1 bytes");
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
  sub_args(s, &p->launch.args, v_params, v_refs, v_at);
  return Val_unit;
}

value caml_rig_sub_launch_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_launch(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6]);
}

/* Part [v_i] calls the fill [v_fill] on [v_params] parameter bytes and the
   refs [v_refs], stored from [v_at] on. */
value caml_rig_sub_fill(value v_s, value v_i, value v_fill, value v_params,
                        value v_refs, value v_at) {
  struct rig_sub *s = Sub_val(v_s);
  struct rig_part *p = &s->parts[Int_val(v_i)];
  p->kind = RIG_FILL;
  p->fill.fn = (int (*)(void *, void *, uint64_t))Nativeint_val(v_fill);
  sub_args(s, &p->fill.args, v_params, v_refs, v_at);
  return Val_unit;
}

value caml_rig_sub_fill_byte(value *argv, int argn) {
  (void)argn;
  return caml_rig_sub_fill(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5]);
}

/* The parameters of [s]'s launch or fill [j]. */
static const struct rig_args *block_args(const struct rig_sub *s, int j) {
  const struct rig_part *p = &s->parts[s->blocks[j].part];
  return p->kind == RIG_LAUNCH ? &p->launch.args : &p->fill.args;
}

/* The submission's launch or fill [v_j] is its part [v_i], which
   [caml_rig_sub_launch] or [caml_rig_sub_fill] set, whose function runs at
   most [v_most] threads a group, 0 for none. */
value caml_rig_sub_block(value v_s, value v_j, value v_i, value v_most) {
  struct rig_sub *s = Sub_val(v_s);
  s->blocks[Int_val(v_j)] =
      (struct rig_sub_block){Int_val(v_i), 0, (int64_t)Long_val(v_most)};
  return Val_unit;
}

/* The submission's [v_k]th entry ref writes [v_code] at byte [v_at] of the
   parameters of its launch or fill [v_j], which [caml_rig_sub_block] set. */
value caml_rig_sub_code(value v_s, value v_k, value v_j, value v_at,
                        value v_code) {
  struct rig_sub *s = Sub_val(v_s);
  int j = Int_val(v_j);
  uint32_t at = (uint32_t)Long_val(v_at);
  s->codes[Int_val(v_k)] = (struct rig_code){
      (size_t)block_args(s, j)->block + sizeof(struct rig_block) + at,
      (uint64_t)Long_val(v_code)};
  if (at + 8 > s->blocks[j].coded) s->blocks[j].coded = at + 8;
  return Val_unit;
}

/* Fixed buffer [v_k]: the stamps of a memory a part names and how the
   part uses it, an rig_use. [v_k] lies among the buffers counted at
   [caml_rig_sub_new]. */
value caml_rig_sub_fixed(value v_s, value v_k, value v_stamps, value v_use) {
  struct rig_sub *s = Sub_val(v_s);
  intnat k = Long_val(v_k);
  if (k < 0 || k >= s->nfixed)
    caml_invalid_argument(
        "Rig.Submission.make: a fixed buffer past those counted");
  struct rig_fixed *f = &s->fixed[k];
  f->stamps = Stamps_val(v_stamps);
  f->use = (enum rig_use)Int_val(v_use);
  return Val_unit;
}

/* Calls */

/* Every call made, the newest first. A call is never freed, so the list only
   grows at its head, and its links are fixed once published: a walk needs
   no lock. Its length is the most submits that ever ran at once. */
static struct rig_call *_Atomic calls;

static void call_free(struct rig_call *c) {
  free(c->fixed);
  free(c->slots);
  free(c->addresses);
  free(c->sides);
  free(c->parts);
  free(c->points);
  free(c->waits);
  free(c->producers);
  free(c->args);
  free(c->marks);
  free(c->spare);
  free(c);
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

/* [fit] for the [*c] bytes at [*a], whose count needs a size_t. */
static int fit_bytes(uint8_t **a, size_t *c, size_t want) {
  if (want <= *c) return 1;
  uint8_t *p = realloc(*a, want);
  if (p == NULL) return 0;
  memset(p + *c, 0, want - *c);
  *a = p;
  *c = want;
  return 1;
}

/* The [x]th range of the bytes that the blocks [m] may hold other than 0,
   two per block in increasing order: its header and its parameters up to
   its mark, then its place, as rig_block_bytes lays them out. */
static void marked(const struct rig_mark *m, int x, uint64_t *from,
                   uint64_t *to) {
  const struct rig_mark *b = &m[x / 2];
  uint64_t params = b->at + sizeof(struct rig_block);
  if (x % 2 == 0) {
    *from = b->at;
    *to = params + b->high;
  } else {
    *from = params + rig_round16(b->params);
    *to = *from + rig_round16(b->params);
  }
}

/* Moves the call [c]'s marks from the submission it served last to [s]:
   clears the marked bytes past [s]'s blocks, and marks each block of [s]
   as far as marked bytes reach into its parameters, rounded up to 16, so
   that its next copy clears them. Its headers and places may hold any
   bytes. The ranges and [s]'s blocks both lie in increasing order, so one
   walk over the ranges serves every block.

   For example, in place of a block of 48 parameter bytes whose last submit
   wrote 16, a block of 16 is marked up to byte 16, and a block of 64 up to
   byte 64, since the old block's place lies at bytes 48 to 96 of the new
   block's parameters. */
static void call_switch(struct rig_call *c, const struct rig_sub *s) {
  int n = 2 * c->nmarks, x = 0;
  uint64_t from, to;
  for (int j = 0; j < s->nblocks; j++) {
    const struct rig_args *a = block_args(s, j);
    uint64_t start = a->block + sizeof(struct rig_block);
    uint64_t end = start + rig_round16(a->params), reach = start;
    /* A range that ends before this block's parameters ends before every
       later block's. */
    for (; x < n; x++) {
      marked(c->marks, x, &from, &to);
      if (to > start) break;
    }
    for (int y = x; y < n; y++) {
      marked(c->marks, y, &from, &to);
      if (from >= end) break;
      if (to > from) reach = to < end ? to : end;
    }
    c->spare[j] =
        (struct rig_mark){a->block, a->params, (uint32_t)(reach - start)};
  }
  for (int y = n - 1; y >= 0; y--) {
    marked(c->marks, y, &from, &to);
    if (to <= s->args) break;
    if (from < s->args) from = s->args;
    memset(c->args + from, 0, to - from);
  }
  struct rig_mark *m = c->marks;
  c->marks = c->spare;
  c->spare = m;
  int cm = c->cmarks;
  c->cmarks = c->cspare;
  c->cspare = cm;
  c->nmarks = s->nblocks;
}

/* Fits the call [c] to the submission [s]: its fixed buffers, slots,
   addresses and sides, the parts of a submission with copies, and the
   blocks and their marks. Whether memory sufficed. A call that served [s]
   last fits it already: its arrays only grow. */
static int call_fit(struct rig_call *c, const struct rig_sub *s) {
  if (s->id == c->sub) return 1;
  int nslots = s->nslots;
  /* The addresses and sides grow with the slots, which [fit] counts. */
  int caddresses = c->cslots, csides = c->cslots;
  if (!fit((void **)&c->fixed, &c->cfixed, s->nfixed, sizeof *c->fixed) ||
      !fit((void **)&c->addresses, &caddresses, nslots,
           sizeof *c->addresses) ||
      !fit((void **)&c->sides, &csides, nslots, sizeof *c->sides) ||
      !fit((void **)&c->slots, &c->cslots, nslots, sizeof *c->slots))
    return 0;
  if (s->nslotted > 0 &&
      !fit((void **)&c->parts, &c->cparts, s->nparts, sizeof *c->parts))
    return 0;
  if (!fit_bytes(&c->args, &c->cargs, s->args) ||
      !fit((void **)&c->marks, &c->cmarks, s->nblocks, sizeof *c->marks) ||
      !fit((void **)&c->spare, &c->cspare, s->nblocks, sizeof *c->spare))
    return 0;
  call_switch(c, s);
  c->sub = s->id;
  c->nslots = nslots;
  return 1;
}

/* A free call, taken and fitted to [s], made if none is free; NULL if the
   host has no memory for it. */
static struct rig_call *call_take(const struct rig_sub *s) {
  struct rig_call *c = atomic_load_explicit(&calls, memory_order_acquire);
  for (; c != NULL; c = c->next) {
    int seen = 0;
    if (atomic_load_explicit(&c->taken, memory_order_relaxed) == 0 &&
        atomic_compare_exchange_strong(&c->taken, &seen, 1))
      break;
  }
  if (c == NULL) {
    if ((c = calloc(1, sizeof *c)) == NULL) return NULL;
    atomic_init(&c->taken, 1);
    if (!call_fit(c, s)) {
      call_free(c);
      return NULL;
    }
    c->next = atomic_load(&calls);
    while (!atomic_compare_exchange_weak(&calls, &c->next, c)) {
    }
    return c;
  }
  if (call_fit(c, s)) return c;
  atomic_store_explicit(&c->taken, 0, memory_order_release);
  return NULL;
}

/* Begins a submit of [v_s]: takes a free call fitted to it. Raises
   Out_of_memory if the host has no memory for the call. */
value caml_rig_submit_begin(value v_s) {
  struct rig_call *c = call_take(Sub_val(v_s));
  if (c == NULL) caml_raise_out_of_memory();
  return Val_long((intnat)c);
}

/* Ends the submit that [caml_rig_submit_begin] began: forgets the call
   [v_c]'s buffers and waits, and gives it back. */
value caml_rig_submit_end(value v_c) {
  struct rig_call *c = Call_val(v_c);
  for (int k = 0; k < c->nslots; k++) c->slots[k].stamps = NULL;
  c->npoints = c->nwaits = c->checks = 0;
  atomic_store_explicit(&c->taken, 0, memory_order_release);
  return Val_unit;
}

/* What [caml_rig_call_args] refuses in a submit's values. */
enum { ARGS_COUNT, ARGS_LONG, ARGS_SIZE, ARGS_THREADS };

static int header_fits(intnat v) { return v >= 0 && v <= (intnat)UINT32_MAX; }

/* Copies [v_args], the values of [v_s]'s launches and fills in part order,
   into the blocks of the call [v_c], fitted to [v_s]: a launch's grid,
   threads and shared bytes into its header, then each element's [params],
   its parameters past them 0. It clears only up to the block's mark, the
   most an earlier submit may have written there, and moves the mark to
   what this one wrote, the entry refs' bytes included, which the collect
   writes next. Each element is read once, and a record and the length of
   its bytes never change, so what is checked is what is copied, whatever
   another domain or a callback stores into the array meanwhile. Answers -1,
   or a refusal: 4 times the index of the element refused, the length of
   [v_args] for ARGS_COUNT, plus the ARGS_ refusal. The blocks before it
   are written, which no work reads.

   A size is below 2^32, so the product of two fits 64 bits, and the third
   multiplies a product no larger than the most, which is below 2^32 too. */
value caml_rig_call_args(value v_s, value v_c, value v_args) {
  const struct rig_sub *s = Sub_val(v_s);
  struct rig_call *c = Call_val(v_c);
  intnat n = (intnat)Wosize_val(v_args);
  if (n != s->nblocks) return Val_long(4 * n + ARGS_COUNT);
  for (intnat j = 0; j < n; j++) {
    value a = Field(v_args, j), v_params = Field(a, 3);
    const struct rig_sub_block *b = &s->blocks[j];
    const struct rig_args *ra = block_args(s, (int)j);
    size_t len = caml_string_length(v_params);
    if (len > ra->params) return Val_long(4 * j + ARGS_LONG);
    uint8_t *block = c->args + ra->block;
    if (s->parts[b->part].kind == RIG_LAUNCH) {
      value g = Field(a, 0), t = Field(a, 1);
      intnat h[7] = {Long_val(Field(g, 0)), Long_val(Field(g, 1)),
                     Long_val(Field(g, 2)), Long_val(Field(t, 0)),
                     Long_val(Field(t, 1)), Long_val(Field(t, 2)),
                     Long_val(Field(a, 2))};
      for (int k = 0; k < 7; k++)
        if (!header_fits(h[k])) return Val_long(4 * j + ARGS_SIZE);
      uint64_t most = (uint64_t)b->most;
      uint64_t xy = (uint64_t)h[3] * (uint64_t)h[4];
      if (most > 0 && (xy > most || xy * (uint64_t)h[5] > most))
        return Val_long(4 * j + ARGS_THREADS);
      struct rig_block *head = (struct rig_block *)block;
      for (int k = 0; k < 3; k++) {
        head->groups[k] = (uint32_t)h[k];
        head->threads[k] = (uint32_t)h[3 + k];
      }
      head->shared = (uint32_t)h[6];
      head->unused = 0;
    }
    uint8_t *params = block + sizeof(struct rig_block);
    memcpy(params, Bytes_val(v_params), len);
    struct rig_mark *m = &c->marks[j];
    if (m->high > len) memset(params + len, 0, m->high - len);
    m->high = len > b->coded ? (uint32_t)len : b->coded;
  }
  return Val_long(-1);
}

/* Names the call's [v_k]th buffer, at [v_address] as the device's work
   addresses it, -1 for memory with no address. */
value caml_rig_call_slot(value v_c, value v_k, value v_stamps,
                         value v_address) {
  struct rig_call *c = Call_val(v_c);
  int k = Int_val(v_k);
  c->addresses[k] = (uint64_t)Long_val(v_address);
  c->slots[k].stamps = Stamps_val(v_stamps);
  return Val_unit;
}

/* Sets the call's [v_k]th side, for copies, from its buffer [v_b], which a
   copy passes as this process's memory iff [v_local], and whose
   copy.device_memory fact is [v_device_memory]. */
value caml_rig_call_side(value v_c, value v_k, value v_b, value v_local,
                         value v_device_memory) {
  struct rig_side *side = &Call_val(v_c)->sides[Int_val(v_k)];
  rig_buffer_side(v_b, Bool_val(v_local), side);
  side->device_memory = Bool_val(v_device_memory);
  return Val_unit;
}

/* Sets the call's parts from [s]'s, the fields of each copy with a slot end
   from the sides of its ends, or refuses a copy whose buffers overlap or are
   both this process's memory, or, other than a box, differ in size. */
static void set_copies(const struct rig_sub *s, struct rig_call *r) {
  memcpy(r->parts, s->parts, (size_t)s->nparts * sizeof *r->parts);
  for (int k = 0; k < s->ncopies; k++) {
    const struct rig_copy *c = &s->copies[k];
    if (c->src < 0 && c->dst < 0) continue;
    const struct rig_side *src = c->src < 0 ? &c->ends[0] : &r->sides[c->src];
    const struct rig_side *dst = c->dst < 0 ? &c->ends[1] : &r->sides[c->dst];
    if (c->fill) {
      if (dst->local)
        caml_invalid_argument(
            "Rig.submit: a fill's buffer is not its device's memory");
      set_copy(&r->parts[c->part], c, src, dst);
      continue;
    }
    if (!c->boxed && src->bytes != dst->bytes)
      caml_invalid_argument("Rig.submit: a copy's buffers differ in size");
    if (rig_sides_overlap(src, dst))
      caml_invalid_argument("Rig.submit: a copy's buffers overlap");
    if (src->local && dst->local)
      caml_invalid_argument(
          "Rig.submit: neither of a copy's buffers is its device's memory");
    set_copy(&r->parts[c->part], c, src, dst);
  }
}

static void grow(void **a, int *c, int want, size_t size) {
  if (want <= *c) return;
  int n = want < 8 ? 8 : 2 * want;
  void *p = realloc(*a, (size_t)n * size);
  if (p == NULL) caml_raise_out_of_memory();
  *a = p;
  *c = n;
}

/* Keeps the greatest point per device, leaving out the points of [own], the
   device of slot [own] that started after [own_base]: its own order covers
   its work. Two points of one slot are judged by its line only when they
   differ: an earlier device's point that is done is left out, and one a
   loss dropped is kept apart, since a later device's points of the slot
   follow no earlier one's, and the submit raises its loss. The line is read
   after the stamp word that held the point; a base read too early takes it
   for the slot's device's, whose points lie above every earlier one's,
   which only waits for more. Inlined into each walk of a submit's
   buffers. */
static inline __attribute__((always_inline)) void
add_point(struct rig_call *r, int own, uint64_t own_base, uint64_t p) {
  if (p == 0) return;
  if (RIG_INDEX(p) == own && RIG_VALUE(p) > own_base) return;
  uint64_t base = 0;
  int based = 0;
  for (int i = 0; i < r->npoints; i++) {
    uint64_t q = r->points[i];
    if (q == p) return;
    if (RIG_INDEX(q) != RIG_INDEX(p)) continue;
    if (!based) {
      base = rig_line_base(RIG_INDEX(p));
      based = 1;
      if (below(p, base) && rig_point_done(p)) return;
    }
    if (below(p, base) || below(q, base)) continue;
    if (q < p) r->points[i] = p;
    return;
  }
  grow((void **)&r->points, &r->cpoints, r->npoints + 1, sizeof *r->points);
  r->points[r->npoints++] = p;
}

/* Adds the points the use [u] of the memory of the stamps [st] follows:
   each device's last write, and every use if the work writes it; none if it
   is unordered. Is [own]'s use word in the stamps, reserved, its keep word
   for an unordered use. The walk is rig_stamps_followed's, and it finds
   [own]'s use word on the way, before that word's write word is read: its
   device order covers its own work, and the write word beside its use word
   holds its own write or a spent device's done point. The points folded
   into the head come after the words. [own] is the device of slot [own]
   that started after [base].

   The host, of index 0, reserves no word and gets NULL: its work is done
   when its submit returns, before a raise could store its point, so a
   point of it orders nothing. Its reservation would be (0, 0), the empty
   word, which another device's reservation takes. */
static _Atomic uint64_t *add_memory(struct rig_call *r, int own,
                                    uint64_t base, struct rig_stamps *st,
                                    enum rig_use u) {
  int host = own == 0;
  if (u == RIG_USE_UNORDERED)
    return host ? NULL : reserve_any(st, own, base, 1);
  int every = u == RIG_USE_WRITE;
  /* Memory that only [own] uses, the common case, has nothing to follow:
     its first use word is [own]'s and its second empty, so nothing is
     folded ([takeable]). */
  uint64_t first = atomic_load_explicit(&st->use[0], memory_order_acquire);
  if (is_own(first, own, base) &&
      atomic_load_explicit(&st->use[1], memory_order_acquire) == 0)
    return &st->use[0];
  _Atomic uint64_t *mine = NULL;
  for (struct rig_stamps *c = st; c != NULL;
       c = atomic_load_explicit(&c->next, memory_order_acquire)) {
    int end = 0;
    for (int i = 0; i < RIG_USES && !end; i++) {
      uint64_t w = atomic_load_explicit(&c->use[i], memory_order_acquire);
      if (w == 0) end = 1;
      else if (is_own(w, own, base)) {
        if (mine == NULL) mine = &c->use[i];
      } else {
        uint64_t p =
            every ? rig_use_followed(w, &c->write[i])
                  : atomic_load_explicit(&c->write[i], memory_order_acquire);
        if (RIG_VALUE(p) != 0) add_point(r, own, base, p);
      }
    }
    if (end) break;
  }
  uint64_t folded = rig_stamps_folded(st);
  if (folded != 0) add_point(r, own, base, folded);
  if (host) return NULL;
  return mine != NULL ? mine : reserve_any(st, own, base, 0);
}

/* Writes each entry ref's value into the call [v_c]'s blocks, after the
   values ([caml_rig_call_args]), and sets its parts for its copies.
   Collects in [v_c] the points [v_s]'s work follows, the greatest per other
   device, with the points of [v_waits], and reserves the use words its
   raise stores to. Answers the number of points. */
value caml_rig_call_collect(value v_s, value v_c, value v_waits) {
  const struct rig_sub *s = Sub_val(v_s);
  struct rig_call *r = Call_val(v_c);
  uint8_t *args = r->args;
  int own = s->dev->index, nslots = s->nslots;
  uint64_t base = s->dev->base;
  for (int k = 0; k < s->nrefs; k++)
    if (r->addresses[s->refs[k].slot] == UNADDRESSED)
      caml_invalid_argument(
          "Rig.submit: a ref names a buffer whose memory has no "
          "address");
  if (s->nslotted > 0) set_copies(s, r);
  for (int k = 0; k < s->ncodes; k++)
    memcpy(args + s->codes[k].at, &s->codes[k].code, sizeof(uint64_t));
  r->npoints = r->nwaits = r->checks = 0;
  for (int k = 0; k < s->nfixed; k++)
    r->fixed[k] =
        add_memory(r, own, base, s->fixed[k].stamps, s->fixed[k].use);
  for (int k = 0; k < nslots; k++)
    r->slots[k].use = add_memory(r, own, base, r->slots[k].stamps,
                                 (enum rig_use)s->uses[k]);
  for (mlsize_t k = 0; k < Wosize_val(v_waits); k++)
    add_point(r, own, base, Unsigned_long_val(Field(v_waits, k)));
  return Val_int(r->npoints);
}

/* Makes the call's submit time its value, iff [v_timed]. */
value caml_rig_call_timed(value v_c, value v_timed) {
  struct rig_call *c = Call_val(v_c);
  c->timed = Bool_val(v_timed);
  c->pair = -1;
  return Val_unit;
}

/* The time pair the call's submit took, -1 for none. */
value caml_rig_call_pair(value v_c) { return Val_int(Call_val(v_c)->pair); }

value caml_rig_call_point(value v_c, value v_i) {
  return Val_long((intnat)Call_val(v_c)->points[Int_val(v_i)]);
}

/* Adds an in-queue wait on the producer [v_producer]'s value [v_value], whose
   word is at [v_at]. */
value caml_rig_call_wait(value v_c, value v_producer, value v_at,
                         value v_value) {
  struct rig_call *r = Call_val(v_c);
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
                                          (uint64_t)Long_val(v_value)};
  r->producers[r->nwaits] = Int_val(v_producer);
  r->nwaits++;
  return Val_unit;
}

/* Notes that a wait of the call reads a word through a mapping its device
   may give back, which its submit then checks ([record_waits]). */
value caml_rig_call_check(value v_c) {
  Call_val(v_c)->checks = 1;
  return Val_unit;
}

/* Raises the stamps [s]'s work names in [r] to [p]. */
void rig_sub_raise(const struct rig_sub *s, struct rig_call *r, uint64_t p) {
  for (int k = 0; k < s->nfixed; k++) {
    if (s->fixed[k].use == RIG_USE_WRITE) raise_own(write_word(r->fixed[k]), p);
    raise_own(r->fixed[k], p);
  }
  for (int k = 0; k < s->nslots; k++) {
    if (s->uses[k] == RIG_USE_WRITE) raise_own(write_word(r->slots[k].use), p);
    raise_own(r->slots[k].use, p);
  }
}

value caml_rig_call_value(value v_c) {
  return Val_long((intnat)Call_val(v_c)->v);
}

value caml_rig_call_no_room_at(value v_c) {
  return Val_long((intnat)Call_val(v_c)->no_room_at);
}

value caml_rig_call_producer(value v_c) {
  return Val_long((intnat)Call_val(v_c)->producer);
}
