/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills and the C entries, for the AMD suite. */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_edge.h"

/* The C entries. The representation matches Rig_amd_support.Edge. */

enum {
  part_queue,
  part_kind,
  part_launch,
  part_code,
  part_block,
  part_params,
  part_dst,
  part_src,
  part_copy,
  part_nafter,
  part_nwords,
  part_after
};

static intnat at(value a, int i) { return Long_val(Field(a, i)); }

/* The parts, their scratch and the waits in C, in one allocation the
   caller frees. */
static void *edge(value v_waits, value v_parts, struct rig_wait **wp,
                  struct rig_part **pp, void **sp) {
  int nwaits = (int)(Wosize_val(v_waits) / 2);
  int nparts = (int)Wosize_val(v_parts);
  size_t nafter = 0, nwords = 0;
  for (int i = 0; i < nparts; i++) {
    value p = Field(v_parts, i);
    nafter += (size_t)at(p, part_nafter);
    nwords += (size_t)at(p, part_nwords);
  }
  size_t size = nwaits * sizeof(struct rig_wait) +
                nparts * (sizeof(struct rig_part) + RIG_SCRATCH) +
                nafter * sizeof(int) + nwords * sizeof(uint32_t) + 1;
  char *mem = malloc(size);
  if (mem == NULL) caml_raise_out_of_memory();
  struct rig_wait *w = (struct rig_wait *)mem;
  struct rig_part *p = (struct rig_part *)(w + nwaits);
  uint8_t *scratch = (uint8_t *)(p + nparts);
  int *after = (int *)(scratch + nparts * RIG_SCRATCH);
  uint32_t *words = (uint32_t *)(after + nafter);
  for (int i = 0; i < nwaits; i++) {
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
    case RIG_COPY:
      p[i].copy.dst = (uint64_t)at(k, part_dst);
      p[i].copy.src = (uint64_t)at(k, part_src);
      p[i].copy.width = (uint64_t)at(k, part_copy);
      p[i].copy.rows = 1;
      p[i].copy.src_pitch = p[i].copy.dst_pitch = p[i].copy.width;
      break;
    case RIG_LAUNCH:
      p[i].launch.code = (uint64_t)at(k, part_code);
      p[i].launch.launch = (const void *)at(k, part_launch);
      p[i].launch.args.block = (uint32_t)at(k, part_block);
      p[i].launch.args.params = (uint32_t)at(k, part_params);
      p[i].launch.args.refs = (const struct rig_ref *)words;
      p[i].launch.args.nrefs = nw / 2;
      break;
    }
    for (int j = 0; j < na; j++) *after++ = (int)at(k, part_after + j);
    for (int j = 0; j < nw; j++)
      *words++ = (uint32_t)at(k, part_after + na + j);
  }
  *wp = w;
  *pp = p;
  *sp = scratch;
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
  void *scratch;
  void *mem = edge(Atom(0), v_parts, &w, &p, &scratch);
  int r = driver_of(v_self)->room((void *)Nativeint_val(v_self), 0, p,
                                  (int)Wosize_val(v_parts), scratch,
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
  void *scratch;
  int nslots = (int)Wosize_val(v_slots);
  uint64_t *slots = malloc((size_t)(nslots + 1) * sizeof *slots);
  if (slots == NULL) caml_raise_out_of_memory();
  for (int i = 0; i < nslots; i++) slots[i] = (uint64_t)at(v_slots, i);
  /* A submit may write its blocks' places: it gets a copy of [v_args]. */
  mlsize_t nargs = caml_string_length(v_args);
  uint8_t *args = malloc(nargs + 1);
  if (args == NULL) {
    free(slots);
    caml_raise_out_of_memory();
  }
  memcpy(args, String_val(v_args), nargs);
  void *mem = edge(v_waits, v_parts, &w, &p, &scratch);
  const char *failure = NULL;
  int r = driver_of(v_self)->submit(
      (void *)Nativeint_val(v_self), (uint64_t)Long_val(v_v), w,
      (int)(Wosize_val(v_waits) / 2), p, (int)Wosize_val(v_parts), scratch,
      args, slots, nslots, NULL, &failure);
  free(mem);
  free(args);
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
