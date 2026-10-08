/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Device_amd's C state, filled from OCaml, and the OCaml forms of room and
   submit. No stub here releases the runtime: none blocks. A device's state
   is named in OCaml by its address, an int. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "device_amd_stubs.h"

#define Device_val(v) ((struct device_amd *)Long_val(v))
#define Host_val(v) ((void *)Nativeint_val(v))

static intnat at(value a, int i) { return Long_val(Field(a, i)); }

/* State */

value caml_device_amd_create(value unit) {
  (void)unit;
  struct device_amd *d = calloc(1, sizeof *d);
  if (d == NULL) caml_raise_out_of_memory();
  return Val_long((intnat)d);
}

/* The word, the slots (set to low32(-1), which no early wait compares
   with) and the segment, at their host and GPU addresses. */
value caml_device_amd_memory(value v_self, value v_word, value v_word_gpu,
                             value v_slots, value v_slots_gpu) {
  struct device_amd *d = Device_val(v_self);
  d->word = Host_val(v_word);
  atomic_store_explicit(d->word, 0, memory_order_release);
  d->word_gpu = (uint64_t)Long_val(v_word_gpu);
  d->slots = Host_val(v_slots);
  d->slots_gpu = (uint64_t)Long_val(v_slots_gpu);
  for (int i = 0; i < DEVICE_AMD_SLOTS; i++) d->slots[2 * i] = UINT32_MAX;
  return Val_unit;
}

value caml_device_amd_segment(value v_self, value v_host, value v_gpu,
                              value v_size) {
  struct device_amd *d = Device_val(v_self);
  d->segment.host = Host_val(v_host);
  d->segment.gpu = (uint64_t)Long_val(v_gpu);
  d->segment.size = (uint64_t)Long_val(v_size);
  return Val_unit;
}

/* Ring [q]: its words, size in bytes, write-position word and doorbell, all
   host addresses, and its kind (RING_PM4, RING_AQL, RING_SDMA). */
value caml_device_amd_ring(value v_self, value v_q, value v_words,
                           value v_bytes, value v_write, value v_doorbell,
                           value v_kind) {
  struct device_amd_ring *r = &Device_val(v_self)->rings[Int_val(v_q)];
  r->words = Host_val(v_words);
  r->size = (uint64_t)Long_val(v_bytes) / 4;
  r->write = Host_val(v_write);
  r->doorbell = Host_val(v_doorbell);
  r->kind = Int_val(v_kind);
  return Val_unit;
}

value caml_device_amd_ring_byte(value *argv, int argn) {
  (void)argn;
  return caml_device_amd_ring(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6]);
}

/* Template [t]: its words as little-endian bytes, and its holes, each as the
   ints at, wide, arg, nops, then (op, k) per operation. */
value caml_device_amd_template(value v_self, value v_t, value v_words,
                               value v_holes) {
  struct device_amd_template *t =
      &Device_val(v_self)->templates[Int_val(v_t)];
  t->n = (int)(caml_string_length(v_words) / 4);
  memcpy(t->words, String_val(v_words), 4 * (size_t)t->n);
  int n = (int)Wosize_val(v_holes), k = 0;
  t->nholes = 0;
  while (k < n) {
    struct device_amd_hole *h = &t->holes[t->nholes++];
    h->at = (uint8_t)at(v_holes, k);
    h->wide = (uint8_t)at(v_holes, k + 1);
    h->arg = (uint8_t)at(v_holes, k + 2);
    h->nops = (uint8_t)at(v_holes, k + 3);
    k += 4;
    for (int i = 0; i < h->nops; i++, k += 2) {
      h->op[i] = (uint8_t)at(v_holes, k);
      h->k[i] = (uint64_t)at(v_holes, k + 1);
    }
  }
  return Val_unit;
}

value caml_device_amd_max_copy(value v_self, value v_n) {
  Device_val(v_self)->max_copy = (uint64_t)Long_val(v_n);
  return Val_unit;
}

/* Adds [delta] to the regions that need the HDP register at host address
   [reg] flushed. The caller serialises calls for one device. */
value caml_device_amd_hdp(value v_self, value v_reg, value v_delta) {
  struct device_amd *d = Device_val(v_self);
  volatile uint32_t *reg = Host_val(v_reg);
  struct device_amd_hdp *free = NULL;
  for (int i = 0; i < DEVICE_AMD_HDPS; i++) {
    struct device_amd_hdp *h = &d->hdps[i];
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

/* Work. A part is the ints Device_amd.part makes: the device's state, the
   queue, fill, arg, ring units, segment bytes, copy_dst, copy_dst_offset,
   copy_src, copy_src_offset, copy_bytes, the number of [after] indices and
   of words, then the indices, then the words. */

enum {
  part_self,
  part_queue,
  part_fill,
  part_arg,
  part_units,
  part_bytes,
  part_dst,
  part_dst_offset,
  part_src,
  part_src_offset,
  part_copy_bytes,
  part_nafter,
  part_nwords,
  part_after
};

/* The parts and waits in C, in one allocation the caller frees. */
static void *edge(value v_waits, value v_parts, struct nx_wait **wp,
                  struct nx_part **pp) {
  int nwaits = (int)(Wosize_val(v_waits) / 3);
  int nparts = (int)Wosize_val(v_parts);
  size_t nafter = 0, nwords = 0;
  for (int i = 0; i < nparts; i++) {
    value p = Field(v_parts, i);
    nafter += (size_t)at(p, part_nafter);
    nwords += (size_t)at(p, part_nwords);
  }
  size_t size = nwaits * sizeof(struct nx_wait) +
                nparts * sizeof(struct nx_part) + nafter * sizeof(int) +
                nwords * sizeof(uint32_t);
  char *mem = size == 0 ? NULL : malloc(size);
  if (size != 0 && mem == NULL) caml_raise_out_of_memory();
  struct nx_wait *w = (struct nx_wait *)mem;
  struct nx_part *p = (struct nx_part *)(w + nwaits);
  int *after = (int *)(p + nparts);
  uint32_t *words = (uint32_t *)(after + nafter);
  for (int i = 0; i < nwaits; i++) {
    w[i].kind = (int)at(v_waits, 3 * i);
    w[i].at = (uint64_t)at(v_waits, 3 * i + 1);
    w[i].value = (uint64_t)at(v_waits, 3 * i + 2);
  }
  for (int i = 0; i < nparts; i++) {
    value k = Field(v_parts, i);
    int na = (int)at(k, part_nafter), nw = (int)at(k, part_nwords);
    p[i] = (struct nx_part){
        .queue = (int)at(k, part_queue),
        .words = nw > 0 ? words : NULL,
        .n = (size_t)nw,
        .fill = (int (*)(void *, void *, uint64_t))at(k, part_fill),
        .arg = (void *)at(k, part_arg),
        .ring_units = (size_t)at(k, part_units),
        .segment_bytes = (size_t)at(k, part_bytes),
        .copy_dst = (uint64_t)at(k, part_dst),
        .copy_dst_offset = (uint64_t)at(k, part_dst_offset),
        .copy_src = (uint64_t)at(k, part_src),
        .copy_src_offset = (uint64_t)at(k, part_src_offset),
        .copy_bytes = (uint64_t)at(k, part_copy_bytes),
        .after = na > 0 ? after : NULL,
        .nafter = na};
    for (int j = 0; j < na; j++) *after++ = (int)at(k, part_after + j);
    for (int j = 0; j < nw; j++)
      *words++ = (uint32_t)at(k, part_after + na + j);
  }
  *wp = w;
  *pp = p;
  return mem;
}

value caml_device_amd_room(value v_self, value v_parts) {
  struct nx_wait *w;
  struct nx_part *p;
  void *mem = edge(Atom(0), v_parts, &w, &p);
  int r = device_amd_room(Device_val(v_self), p, (int)Wosize_val(v_parts));
  free(mem);
  return Val_int(r);
}

value caml_device_amd_submit(value v_self, value v_v, value v_waits,
                             value v_parts) {
  struct nx_wait *w;
  struct nx_part *p;
  void *mem = edge(v_waits, v_parts, &w, &p);
  const char *failure = NULL;
  int r = device_amd_submit(Device_val(v_self), (uint64_t)Long_val(v_v), w,
                            (int)(Wosize_val(v_waits) / 3), p,
                            (int)Wosize_val(v_parts), NULL, 0, &failure);
  free(mem);
  return Val_int(r);
}

value caml_device_amd_failure(value v_self) {
  return caml_copy_string(Device_val(v_self)->failure);
}

value caml_device_amd_last(value v_self) {
  return Val_long(Device_val(v_self)->last);
}

value caml_device_amd_room_entry(value unit) {
  (void)unit;
  return Val_long((intnat)device_amd_room);
}

value caml_device_amd_submit_entry(value unit) {
  (void)unit;
  return Val_long((intnat)device_amd_submit);
}

value caml_device_amd_place_entry(value unit) {
  (void)unit;
  return Val_long((intnat)device_amd_place);
}

value caml_device_amd_segment_entry(value unit) {
  (void)unit;
  return Val_long((intnat)device_amd_segment);
}

/* Timeline */

value caml_device_amd_signaled(value v_self) {
  struct device_amd *d = Device_val(v_self);
  return Val_long(atomic_load_explicit(d->word, memory_order_acquire));
}

/* Raises the word to the last value submit was given, with a release
   store, unless it is there already: the queues are destroyed, so nothing
   else writes it. */
value caml_device_amd_settle(value v_self) {
  struct device_amd *d = Device_val(v_self);
  uint64_t seen = atomic_load_explicit(d->word, memory_order_acquire);
  while (seen < d->last &&
         !atomic_compare_exchange_weak_explicit(d->word, &seen, d->last,
                                                memory_order_release,
                                                memory_order_acquire)) {
  }
  return Val_unit;
}

/* Publishes the writes of a new AQL scratch to the queue's descriptor: the
   32-bit [values] at the GPU addresses [at], which the next submission
   places. */
value caml_device_amd_scratch(value v_self, value v_at, value v_values) {
  struct device_amd *d = Device_val(v_self);
  int n = (int)Wosize_val(v_at), idle = 0;
  while (!atomic_compare_exchange_weak(&d->scratch_lock, &idle, 1)) idle = 0;
  for (int i = 0; i < n; i++) {
    d->scratch_at[i] = (uint64_t)at(v_at, i);
    d->scratch_value[i] = (uint32_t)at(v_values, i);
  }
  d->scratch_n = n;
  atomic_store_explicit(&d->scratch_taken, 0, memory_order_relaxed);
  atomic_store_explicit(&d->scratch_ready, 1, memory_order_release);
  atomic_store_explicit(&d->scratch_lock, 0, memory_order_release);
  return Val_unit;
}

/* The value whose submission placed the last scratch writes, or 0. */
value caml_device_amd_scratch_taken(value v_self) {
  return Val_long(atomic_load_explicit(&Device_val(v_self)->scratch_taken,
                                       memory_order_acquire));
}

value caml_device_amd_poke32(value v_host, value v_off, value v_v) {
  uint8_t *p = Host_val(v_host);
  *(volatile uint32_t *)(p + Long_val(v_off)) = (uint32_t)Long_val(v_v);
  return Val_unit;
}

value caml_device_amd_zero(value v_host, value v_n) {
  memset(Host_val(v_host), 0, (size_t)Long_val(v_n));
  return Val_unit;
}

/* Makes [v] the value after the device's last one; the device is idle. A
   ring that released the last value is taken to have released v - 1, which
   the word now holds, and each slot gets low32(v - 1), as the writer's
   refresh keeps every slot within 2^31 values of the last. */
value caml_device_amd_renumber(value v_self, value v_v) {
  struct device_amd *d = Device_val(v_self);
  uint64_t prev = (uint64_t)Long_val(v_v) - 1;
  for (int i = 0; i < DEVICE_AMD_SLOTS; i++) {
    d->slots[2 * i] = (uint32_t)prev;
    d->slot_last[i] = prev;
  }
  for (int q = 0; q < DEVICE_AMD_QUEUES; q++)
    if (d->rings[q].released == d->last) d->rings[q].released = prev;
  d->last = prev;
  atomic_store_explicit(d->word, prev, memory_order_release);
  return Val_unit;
}
