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
   carrier into the destination's dtype. A cast to the source's own dtype,
   and a bitcast, is a copy, NaN payloads kept. A transposed copy's square
   blocks store past the caches where its result gains (streams). */

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

/* The bytes of a piece of a row a plane casts through where the thread's
   kept scratch cannot be had. */
#define PIECE 1024

/* A plane staged into a slot of the carrier [c], then unstaged: the slot
   in the thread's kept scratch, which a thread's stack should not hold,
   else a piece of a row at a time through a small one. */
static void cast_slot(const nx_cpu_block *b, const nx_array *a, int c) {
  int64_t w = nx_cpu_width(c);
  uint8_t *slot = nx_cpu_take(NX_CPU_SLOT);
  if (slot != NULL) {
    nx_cpu_stage(&a[1], b, 1, slot, b->n0 * w);
    nx_cpu_unstage(&a[0], b, 0, slot, b->n0 * w, c);
    nx_cpu_give(slot);
    return;
  }
  _Alignas(64) uint8_t piece[PIECE];
  nx_cpu_block p = *b;
  p.n1 = 1;
  for (int64_t j = 0; j < b->n1; j++)
    for (int64_t i = 0; i < b->n0; i += PIECE / w) {
      p.n0 = b->n0 - i < PIECE / w ? b->n0 - i : PIECE / w;
      for (int k = 0; k < 2; k++)
        p.at[k] = b->at[k] + j * b->s1[k] + i * b->s0[k];
      nx_cpu_stage(&a[1], &p, 1, piece, p.n0 * w);
      nx_cpu_unstage(&a[0], &p, 0, piece, p.n0 * w, c);
    }
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
  cast_slot(b, a, c);
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

/* A copy's operands, destination first, and the copy of its boxes:
   nx_copy_box, or nx_stream_box past the caches. */
typedef struct {
  const nx_array *a;
  void (*box)(uint8_t *, const uint8_t *, const nx_box *, int);
} copying;

#if defined(__x86_64__)
/* The bytes of a tile buffered on x86-64. There a load waits for an earlier
   store whose address has the same low 12 bits, and the rows of a tile
   whose input rows lie a multiple of 4 KiB apart all share theirs with the
   output's: such a tile is first copied row by row into a buffer, then
   transposed from it. A transposed 4096x4096 float32 copy on kimchi takes
   2.91 ms against 3.21; on the M1, which has no such wait, staging took it
   from 1.7 to 2.6 ms. */
#define TILE_BUFFER (32 * 1024)

/* A tile through the thread's kept scratch, which a thread's stack should
   not hold: 0, or 1 where the scratch cannot be had, the tile then
   copied as any other. */
static int via_buffer(const nx_cpu_block *b, const copying *c) {
  const nx_array *a = c->a;
  uint8_t *tile = nx_cpu_take(TILE_BUFFER);
  if (tile == NULL) return 1;
  nx_copy_box(tile, a[1].base,
              &(nx_box){{1, b->n0, b->n1},
                        {0, b->at[1]},
                        {{0, b->n1, 1}, {0, b->s0[1], 1}}},
              a[0].bits);
  c->box(a[0].base, tile,
         &(nx_box){{1, b->n1, b->n0},
                   {b->at[0], 0},
                   {{0, b->s1[0], 1}, {0, 1, b->n1}}},
         a[0].bits);
  nx_cpu_give(tile);
  return 0;
}
#endif

static void copy_block(const nx_cpu_block *b, void *ctx) {
  const copying *c = ctx;
  const nx_array *a = c->a;
#if defined(__x86_64__)
  int64_t w = a[0].bits / 8;
  if (b->n2 == 1 && b->s0[0] == 1 && b->s1[1] == 1 && a[0].bits >= 8 &&
      b->s0[1] * w % 4096 == 0 && b->n0 * b->n1 * w <= TILE_BUFFER &&
      via_buffer(b, c) == 0)
    return;
#endif
  c->box(a[0].base, a[1].base,
         &(nx_box){{b->n2, b->n1, b->n0},
                   {b->at[0], b->at[1]},
                   {{b->s2[0], b->s1[0], b->s0[0]},
                    {b->s2[1], b->s1[1], b->s0[1]}}},
         a[0].bits);
}

/* Whether a copy of [bytes] on [threads] threads streams its square blocks
   (nx_stream_box): where its result exceeds what a cache keeps for its
   reader. On x86-64, past a core's L2 on one thread and past the largest
   cache on several: on Arrow Lake a transposed 4096x4096 float32 copy took
   5.34 ms streamed against 8.67 on one core and 1.83 against 2.70 on six,
   a float64 2048x2048 one 2.83 against 3.32 and 0.79 against 0.90, while
   results of 4 to 8 MiB took 0.95 to 1.04 times as long streamed on six.
   On arm64, past half of a core's L2 on one thread and past four times
   the largest cache on several: macOS reports the M1 Max's 12 MiB cluster
   L2, not its 48 MiB system cache, and on eight of its cores float32
   results of 4 to 16 MiB took 1.05 to 1.15 times as long streamed, of 32
   MiB 0.97 to 1.01, of 62 MiB 0.72 to 0.88. */
static int streams(int64_t bytes, int threads) {
  const nx_cpu_target *t = nx_cpu_table;
#if defined(__x86_64__)
  int64_t one = t->l2, several = t->llc;
#else
  int64_t one = t->l2 / 2, several = 4 * t->llc;
#endif
  return NX_STREAMS && bytes > (threads == 1 ? one : several);
}

/* Copies the loop [l] of the operands [a], destination first, on the
   threads the walk gives a tiled loop: only a transposed copy, which the
   walk tiles, streams. */
static void copy_walk(nx_array *a, const nx_loop *l) {
  int64_t n = 1;
  for (int i = 0; i < l->rank; i++) n *= l->extent[i];
  int64_t bytes = n * a[0].bits / 8;
  int threads = nx_cpu_threads(2 * bytes, nx_cpu_walk_cost(2 * bytes, 1, 1));
  copying c = {a, streams(bytes, threads) ? nx_stream_box : nx_copy_box};
  nx_cpu_walk(2, a, l, COPY_MOST, 1, copy_block, &c);
}

/* Casts the loop [l] of the operands [a], destination first. */
static void cast_walk(nx_array *a, const nx_loop *l) {
  nx_cpu_walk(2, a, l, cast_most(a[1].dtype, a[0].dtype), 1, cast_block, a);
}

void nx_cpu_copy_loop(uint8_t *dst, const uint8_t *src, int bits,
                      const nx_loop *l) {
  /* The walk and the block copy read the operands' bases and widths. */
  nx_array pair[2] = {{.base = dst, .bits = bits},
                      {.base = (uint8_t *)src, .bits = bits}};
  copy_walk(pair, l);
}

void nx_cpu_cast_loop(const nx_array *dst, const nx_array *src,
                      const nx_loop *l) {
  nx_array pair[2] = {*dst, *src};
  if (dst->dtype == src->dtype)
    nx_cpu_walk(2, pair, l, COPY_MOST, 1, copy_block, pair);
  else
    nx_cpu_walk(2, pair, l, cast_most(src->dtype, dst->dtype), 1, cast_block,
                pair);
}

/* Reads [vd], written, and [vs] of the dtypes [d] and [s] through the door,
   and walks them with [walk], whose element is [kind]'s (nx_prog.h's
   code) at the wider dtype. */
static value walk_operands(value vd, value vs, int d, int s, int kind,
                           void (*walk)(nx_array *, const nx_loop *)) {
  nx_operand in[2] = {{vd, d, 1}, {vs, s, 0}};
  nx_array a[2];
  nx_loop l;
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  if (!(e = nx_coalesce(2, a, &l))) {
    int wide = nx_cpu_width(d) > nx_cpu_width(s) ? d : s;
    double per = nx_cpu_kind_cost(1, kind, wide);
    int released = nx_cpu_release(nx_cpu_loop_cost(2, &l, per));
    walk(a, &l);
    nx_cpu_reacquire(released);
  }
  nx_done(2, a);
  return Val_int(e);
}

value nx_cpu_copy(value vd, value vs) {
  int d = nx_array_dtype(vd);
  return walk_operands(vd, vs, d, d, NX_OP1_COPY, copy_walk);
}

/* A bitcast copies the bits of a dtype of the destination's width. */
value nx_cpu_bitcast(value vd, value vs) {
  int d = nx_array_dtype(vd), s = nx_array_dtype(vs);
  if (nx_dtype_row_of(d).bits != nx_dtype_row_of(s).bits)
    return Val_int(NX_DTYPE);
  return walk_operands(vd, vs, d, s, NX_OP1_BITCAST, copy_walk);
}

value nx_cpu_cast(value vd, value vs) {
  int d = nx_array_dtype(vd), s = nx_array_dtype(vs);
  if (d == s) return nx_cpu_copy(vd, vs);
  return walk_operands(vd, vs, d, s, NX_OP1_CAST, cast_walk);
}
