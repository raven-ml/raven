/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills and the C entries, for the AMD suite. */

#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_amd.h"

/* A fill: words to place, in two calls if [split] is inside them, and
   segment bytes to take, through the capability's functions, then its own
   answer. [address] is where its last call took the bytes. It lives in a
   bigarray, rig's buffer for the fill's argument. */
struct fill {
  int (*place)(void *queue, const uint32_t *words, size_t n);
  int (*segment)(void *queue, size_t n, void **host, uint64_t *address);
  size_t n, split, bytes;
  uint64_t address;
  int code;
  uint32_t words[];
};

static int fill(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct fill *f = arg;
  size_t first = f->split > 0 && f->split < f->n ? f->split : f->n;
  int e = f->place(queue, f->words, first);
  if (e == 0 && first < f->n)
    e = f->place(queue, f->words + first, f->n - first);
  if (e) return e;
  if (f->bytes > 0) {
    void *host;
    e = f->segment(queue, f->bytes, &host, &f->address);
    if (e) return e;
  }
  return f->code;
}

value rig_amd_test_fill_entry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)fill);
}

value rig_amd_test_fill_arg(value v_place, value v_segment, value v_ws,
                            value v_split, value v_bytes, value v_code) {
  CAMLparam3(v_place, v_segment, v_ws);
  CAMLlocal1(v_arg);
  size_t n = Wosize_val(v_ws);
  v_arg = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                             (intnat)(sizeof(struct fill) + n * sizeof(uint32_t)));
  struct fill *f = Caml_ba_data_val(v_arg);
  f->place = (int (*)(void *, const uint32_t *, size_t))Nativeint_val(v_place);
  f->segment =
      (int (*)(void *, size_t, void **, uint64_t *))Nativeint_val(v_segment);
  f->n = n;
  f->split = (size_t)Long_val(v_split);
  f->address = 0;
  f->bytes = (size_t)Long_val(v_bytes);
  f->code = Int_val(v_code);
  for (size_t i = 0; i < n; i++)
    f->words[i] = (uint32_t)Long_val(Field(v_ws, i));
  CAMLreturn(v_arg);
}

value rig_amd_test_fill_arg_byte(value *argv, int argn) {
  (void)argn;
  return rig_amd_test_fill_arg(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5]);
}

value rig_amd_test_fill_address(value v_arg) {
  return Val_long((intnat)((struct fill *)Caml_ba_data_val(v_arg))->address);
}

value rig_amd_test_data(value v_ba) {
  return Val_long((intnat)Caml_ba_data_val(v_ba));
}

/* The C entries. A part is the ints Rig_amd_support.Edge makes: the queue,
   its kind, the fill, its argument, ring units, segment bytes, the copy's
   destination, source and bytes, the counts of [after] indices and of
   words, then the indices, then the words. A launch's fill is its launch,
   its argument its code, its ring units its block's offset in the args,
   its segment bytes its parameters' and its words its refs, each two words,
   at then slot. */

enum {
  part_queue,
  part_kind,
  part_fill,
  part_arg,
  part_units,
  part_bytes,
  part_dst,
  part_src,
  part_copy,
  part_nafter,
  part_nwords,
  part_after
};

static intnat at(value a, int i) { return Long_val(Field(a, i)); }

/* The parts and waits in C, in one allocation the caller frees. */
static void *edge(value v_waits, value v_parts, struct rig_wait **wp,
                  struct rig_part **pp) {
  int nwaits = (int)(Wosize_val(v_waits) / 2);
  int nparts = (int)Wosize_val(v_parts);
  size_t nafter = 0, nwords = 0;
  for (int i = 0; i < nparts; i++) {
    value p = Field(v_parts, i);
    nafter += (size_t)at(p, part_nafter);
    nwords += (size_t)at(p, part_nwords);
  }
  size_t size = nwaits * sizeof(struct rig_wait) +
                nparts * sizeof(struct rig_part) + nafter * sizeof(int) +
                nwords * sizeof(uint32_t) + 1;
  char *mem = malloc(size);
  if (mem == NULL) caml_raise_out_of_memory();
  struct rig_wait *w = (struct rig_wait *)mem;
  struct rig_part *p = (struct rig_part *)(w + nwaits);
  int *after = (int *)(p + nparts);
  uint32_t *words = (uint32_t *)(after + nafter);
  for (int i = 0; i < nwaits; i++) {
    w[i].kind = RIG_WORD;
    w[i].at = (uint64_t)at(v_waits, 2 * i);
    w[i].value = (uint64_t)at(v_waits, 2 * i + 1);
  }
  for (int i = 0; i < nparts; i++) {
    value k = Field(v_parts, i);
    int na = (int)at(k, part_nafter), nw = (int)at(k, part_nwords);
    p[i] = (struct rig_part){.queue = (int)at(k, part_queue),
                             .kind = (int)at(k, part_kind),
                             .after = na > 0 ? after : NULL,
                             .nafter = na};
    switch (p[i].kind) {
    case RIG_WORDS:
      p[i].words.at = nw > 0 ? words : NULL;
      p[i].words.n = (size_t)nw;
      break;
    case RIG_FILL:
      p[i].fill.fn = (int (*)(void *, void *, uint64_t))at(k, part_fill);
      p[i].fill.arg = (void *)at(k, part_arg);
      p[i].fill.ring_units = (size_t)at(k, part_units);
      p[i].fill.segment_bytes = (size_t)at(k, part_bytes);
      break;
    case RIG_COPY:
      p[i].copy.dst = (uint64_t)at(k, part_dst);
      p[i].copy.src = (uint64_t)at(k, part_src);
      p[i].copy.bytes = (uint64_t)at(k, part_copy);
      break;
    case RIG_LAUNCH:
      p[i].launch.code = (uint64_t)at(k, part_arg);
      p[i].launch.launch = (const void *)at(k, part_fill);
      p[i].launch.block = (uint32_t)at(k, part_units);
      p[i].launch.params = (uint32_t)at(k, part_bytes);
      p[i].launch.refs = (const struct rig_ref *)words;
      p[i].launch.nrefs = nw / 2;
      break;
    }
    for (int j = 0; j < na; j++) *after++ = (int)at(k, part_after + j);
    for (int j = 0; j < nw; j++)
      *words++ = (uint32_t)at(k, part_after + na + j);
  }
  *wp = w;
  *pp = p;
  return mem;
}

/* The driver whose C state is [v_self]. */
static const struct rig_driver *driver_of(value v_self) {
  return rig_driver_of((void *)Nativeint_val(v_self));
}

/* What the room check of the device [v_self] answers for [v_parts]. */
value rig_amd_test_room(value v_self, value v_parts, value v_args) {
  struct rig_wait *w;
  struct rig_part *p;
  void *mem = edge(Atom(0), v_parts, &w, &p);
  int r = driver_of(v_self)->room((void *)Nativeint_val(v_self), p,
                                  (int)Wosize_val(v_parts),
                                  (const uint8_t *)String_val(v_args));
  free(mem);
  return Val_int(r);
}

/* What the submit of the device [v_self] answers for [v_parts] as the value
   [v_v] after the waits [v_waits] (address, value pairs), its launches'
   blocks in [v_args] and its slots' addresses [v_slots]: [None], or
   [Some why] for RIG_FAILED. */
value rig_amd_test_submit(value v_self, value v_v, value v_waits,
                          value v_parts, value v_args, value v_slots) {
  CAMLparam5(v_self, v_v, v_waits, v_parts, v_args);
  CAMLxparam1(v_slots);
  CAMLlocal1(v_why);
  struct rig_wait *w;
  struct rig_part *p;
  int nslots = (int)Wosize_val(v_slots);
  uint64_t *slots = malloc((size_t)(nslots + 1) * sizeof *slots);
  if (slots == NULL) caml_raise_out_of_memory();
  for (int i = 0; i < nslots; i++) slots[i] = (uint64_t)at(v_slots, i);
  void *mem = edge(v_waits, v_parts, &w, &p);
  const char *failure = NULL;
  int r = driver_of(v_self)->submit(
      (void *)Nativeint_val(v_self), (uint64_t)Long_val(v_v), w,
      (int)(Wosize_val(v_waits) / 2), p, (int)Wosize_val(v_parts),
      (const uint8_t *)String_val(v_args), slots, nslots, NULL, 0, &failure);
  free(mem);
  free(slots);
  if (r != RIG_FAILED) CAMLreturn(Val_none);
  v_why = caml_copy_string(failure ? failure : "");
  CAMLreturn(caml_alloc_some(v_why));
}

value rig_amd_test_submit_byte(value *argv, int argn) {
  (void)argn;
  return rig_amd_test_submit(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5]);
}
