/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Copies and casts.

   A copy moves bits, a block at a time through nx.array's nx_copy_block:
   rows of bytes by memcpy, a transposed source in 4x4 blocks, sub-byte
   rows as runs of whole bytes with compare-and-swap ends, other strides
   element by element. A cast is unstage ∘ stage: the stage brings
   the source's block to its carrier, in place where the source is already
   one in contiguous rows, and the unstage converts the carrier into the
   destination's dtype. A cast to the source's own dtype is a copy, NaN
   payloads kept. */

#include <caml/mlvalues.h>

#include "cpu.h"

/* Elements of a copy's block: at 64 Ki a block's call costs next to
   nothing beside its memcpy, and a job still has units to share. */
#define COPY_MOST (64 * 1024)

/* The widest form of a cast from [s] to [d]: its source, its carrier or its
   destination. */
static int64_t cast_most(int s, int d) {
  int w = nx_cpu_width(s), c = nx_cpu_width(nx_cpu_carrier(s));
  int x = nx_cpu_width(d);
  if (c > w) w = c;
  if (x > w) w = x;
  return NX_CPU_SLOT / w;
}

static void cast_block(const nx_cpu_block *b, void *ctx) {
  const nx_array *a = ctx;
  int s = a[1].dtype, c = nx_cpu_carrier(s), w = nx_cpu_width(c);
  /* A source that is its carrier in rows of contiguous elements is read in
     place. */
  if (s == c && b->s0[1] == 1) {
    nx_cpu_unstage(&a[0], b, 0, a[1].base + b->at[1] * w, b->s1[1] * w, c);
    return;
  }
  /* A destination that is the carrier, in rows of contiguous elements, is
     staged into. */
  if (a[0].dtype == c && b->s0[0] == 1) {
    nx_cpu_stage(&a[1], b, 1, a[0].base + b->at[0] * w, b->s1[0] * w);
    return;
  }
  _Alignas(64) uint8_t slot[NX_CPU_SLOT];
  nx_cpu_stage(&a[1], b, 1, slot, b->n0 * w);
  nx_cpu_unstage(&a[0], b, 0, slot, b->n0 * w, c);
}

static void copy_block(const nx_cpu_block *b, void *ctx) {
  const nx_array *a = ctx;
  nx_copy_block(a[0].base, b->at[0], b->s1[0], b->s0[0], a[1].base, b->at[1],
                b->s1[1], b->s0[1], b->n1, b->n0, a[0].bits);
}

/* Reads [vd], written, and [vs] of the dtypes [d] and [s] through the door,
   and walks them with [f]. */
static value walk(value vd, value vs, int d, int s, int64_t most,
                  nx_cpu_block_fn f) {
  nx_operand in[2] = {{vd, d, 1}, {vs, s, 0}};
  nx_array a[2];
  nx_loop l;
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  if (!(e = nx_coalesce(2, a, &l))) nx_cpu_walk(2, a, &l, most, f, a);
  nx_done(2, a);
  return Val_int(e);
}

value nx_cpu_copy(value vd, value vs) {
  int d = nx_array_dtype(vd);
  return walk(vd, vs, d, d, COPY_MOST, copy_block);
}

value nx_cpu_cast(value vd, value vs) {
  int d = nx_array_dtype(vd), s = nx_array_dtype(vs);
  if (d == s) return nx_cpu_copy(vd, vs);
  return walk(vd, vs, d, s, cast_most(s, d), cast_block);
}
