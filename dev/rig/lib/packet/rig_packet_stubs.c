/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Rig_packet.load's writes into a C template. Rig_packet checked every
   bound. */

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#include "rig_packet.h"

/* A hole's fields, Rig_packet's tuple [hole]. */
enum { hole_at, hole_arg, hole_wide, hole_ops, hole_ks };

value caml_rig_packet_load(value v_at, value v_words, value v_holes) {
  struct rig_template *t = (struct rig_template *)Nativeint_val(v_at);
  t->nwords = (int)(caml_string_length(v_words) / 4);
  memcpy(t->words, String_val(v_words), 4 * (size_t)t->nwords);
  t->nholes = (int)Wosize_val(v_holes);
  for (int i = 0; i < t->nholes; i++) {
    value v = Field(v_holes, i);
    struct rig_hole *h = &t->holes[i];
    value ops = Field(v, hole_ops), ks = Field(v, hole_ks);
    h->at = (uint8_t)Long_val(Field(v, hole_at));
    h->arg = (uint8_t)Long_val(Field(v, hole_arg));
    h->wide = (uint8_t)Bool_val(Field(v, hole_wide));
    h->nops = (uint8_t)Wosize_val(ops);
    for (int j = 0; j < h->nops; j++) {
      h->op[j] = (uint8_t)Long_val(Field(ops, j));
      h->k[j] = (uint64_t)Int64_val(Field(ks, j));
    }
  }
  return Val_unit;
}
