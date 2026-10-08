/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Packet templates: a packet's 32-bit words, with holes a writer fills
   from its arguments each time it places the packet.

   Rig_packet.load writes a template from an OCaml packet; rig_fill places
   it. A hole is the word at index [at], or the two words from it, low
   first, when [wide]: argument [arg] after [nops] operations in order,
   each an addition of [k] modulo 2^64, a right shift by [k], or an or
   with [k]. */

#ifndef RIG_PACKET_H
#define RIG_PACKET_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RIG_TEMPLATE_WORDS 16
#define RIG_TEMPLATE_HOLES 6
#define RIG_HOLE_OPS 3
#define RIG_TEMPLATE_ARGS 3

enum { RIG_ADD, RIG_SHIFT, RIG_OR };

struct rig_hole {
  uint8_t at, arg, wide, nops;
  uint8_t op[RIG_HOLE_OPS];
  uint64_t k[RIG_HOLE_OPS];
};

struct rig_template {
  uint32_t words[RIG_TEMPLATE_WORDS];
  int nwords, nholes;
  struct rig_hole holes[RIG_TEMPLATE_HOLES];
};

/* Writes [t]'s words into [w], its holes filled from [args]: their
   count. */
static inline int rig_fill(const struct rig_template *t,
                           const uint64_t args[RIG_TEMPLATE_ARGS],
                           uint32_t *w) {
  memcpy(w, t->words, 4 * (size_t)t->nwords);
  for (int i = 0; i < t->nholes; i++) {
    const struct rig_hole *h = &t->holes[i];
    uint64_t x = args[h->arg];
    for (int j = 0; j < h->nops; j++) switch (h->op[j]) {
        case RIG_ADD: x += h->k[j]; break;
        case RIG_SHIFT: x >>= h->k[j]; break;
        default: x |= h->k[j]; break;
      }
    w[h->at] = (uint32_t)x;
    if (h->wide) w[h->at + 1] = (uint32_t)(x >> 32);
  }
  return t->nwords;
}

#endif
