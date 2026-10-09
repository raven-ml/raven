/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Copies and casts.

   A copy moves bits, a block of planes at a time through nx.array's
   nx_copy_box: rows of bytes by memcpy, a transposed source in square
   blocks (on x86-64 through a buffer where its rows lie 4 KiB apart),
   sub-byte rows as runs of whole bytes with compare-and-swap ends, other
   strides element by element. A cast is unstage ∘ stage, plane by plane:
   the stage brings the source's block to its carrier, in place where the
   source is already one in contiguous rows, and the unstage converts the
   carrier into the destination's dtype. A cast to the source's own dtype
   is a copy, NaN payloads kept. */

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

/* A plane of a cast. */
static void cast_plane(const nx_cpu_block *b, const nx_array *a) {
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

/* A cast stages plane by plane: its slot holds one. */
static void cast_block(const nx_cpu_block *b, void *ctx) {
  if (b->n2 == 1) {
    cast_plane(b, ctx);
    return;
  }
  nx_cpu_block p = *b;
  p.n2 = 1;
  for (int64_t q = 0; q < b->n2; q++) {
    cast_plane(&p, ctx);
    for (int k = 0; k < 2; k++) p.at[k] += b->s2[k];
  }
}

#if defined(__x86_64__)
/* The bytes of a tile buffered on x86-64. There a load waits for an earlier
   store whose address has the same low 12 bits, and the rows of a tile
   whose input rows lie a multiple of 4 KiB apart all share theirs with the
   output's: such a tile is first copied row by row into a buffer, then
   transposed from it. A transposed 4096x4096 float32 copy on kimchi takes
   2.91 ms against 3.21; on the M1, which has no such wait, staging took it
   from 1.7 to 2.6 ms. */
#define TILE_BUFFER (32 * 1024)

/* Out of line, so that only a buffered tile's call takes its stack. */
static __attribute__((noinline)) void via_buffer(const nx_cpu_block *b,
                                                 const nx_array *a) {
  _Alignas(64) uint8_t tile[TILE_BUFFER];
  nx_copy_box(tile, a[1].base,
              &(nx_box){{1, b->n0, b->n1},
                        {0, b->at[1]},
                        {{0, b->n1, 1}, {0, b->s0[1], 1}}},
              a[0].bits);
  nx_copy_box(a[0].base, tile,
              &(nx_box){{1, b->n1, b->n0},
                        {b->at[0], 0},
                        {{0, b->s1[0], 1}, {0, 1, b->n1}}},
              a[0].bits);
}
#endif

static void copy_block(const nx_cpu_block *b, void *ctx) {
  const nx_array *a = ctx;
#if defined(__x86_64__)
  int64_t w = a[0].bits / 8;
  if (b->n2 == 1 && b->s0[0] == 1 && b->s1[1] == 1 && a[0].bits >= 8 &&
      b->s0[1] * w % 4096 == 0 && b->n0 * b->n1 * w <= TILE_BUFFER) {
    via_buffer(b, a);
    return;
  }
#endif
  nx_copy_box(a[0].base, a[1].base,
              &(nx_box){{b->n2, b->n1, b->n0},
                        {b->at[0], b->at[1]},
                        {{b->s2[0], b->s1[0], b->s0[0]},
                         {b->s2[1], b->s1[1], b->s0[1]}}},
              a[0].bits);
}

/* Reads [vd], written, and [vs] of the dtypes [d] and [s] through the door,
   and walks them with [f]. */
static value walk_operands(value vd, value vs, int d, int s, int64_t most,
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
  return walk_operands(vd, vs, d, d, COPY_MOST, copy_block);
}

value nx_cpu_cast(value vd, value vs) {
  int d = nx_array_dtype(vd), s = nx_array_dtype(vs);
  if (d == s) return nx_cpu_copy(vd, vs);
  return walk_operands(vd, vs, d, s, cast_most(s, d), cast_block);
}
