/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Walks: how an elementwise loop is cut into blocks.

   Each element is computed from the operands' elements at its index alone,
   so the walk's order decides no bit, only how memory is read. Its rules,
   in order:

   1. An innermost axis of fewer than SHORT elements, as a small window's,
      would make a block of a few elements per row: the nearest outer axis
      of at least SHORT becomes the innermost. Byte-wide operands only: a
      sub-byte operand would lose its runs of whole bytes.
   2. Where operand 0 steps by one element along the innermost axis and an
      input steps less along an outer axis than along the innermost, as a
      transposed one does, that axis becomes the rows and blocks are tiles
      a few cache lines of the input wide, so that a tile reads each of the
      input's lines whole.
   3. Otherwise a block is a piece of a row, or whole rows of the innermost
      axis along the next.

   The units of the job are planes (indices of the axes outside the rows) ×
   blocks of rows × pieces of a row, and unit u's block is computed from u
   alone, so any thread runs any unit. Where a block is a whole plane of
   fewer than PLANES elements, a unit's block takes the planes that follow
   it along the next axis out too, up to PLANES elements: the kernel gets
   them in one call. */

#include "cpu.h"

#define SHORT 8

/* A tile is TILE_WIDE bytes of its widest operand wide along the rows: one
   128-byte line on the M1, eight 64-byte lines on x86-64. Along a row it
   reads across at most TILE_SPAN bytes of the input's address range. On
   kimchi a transposed float32 copy of 512x512 takes 12.4 us in tiles 512
   bytes wide and 13.5 in tiles 128 wide, of 4096x4096 3.21 ms and 3.41;
   on the M1 the 512x512 takes 11.7 us in tiles 512 bytes wide and 9.8 in
   tiles 128 wide. Of 4096x4096 on kimchi, 3.3 ms in tiles spanning 1 MiB
   and 4.0 in tiles spanning 2 MiB. */
#if defined(__x86_64__)
#define TILE_WIDE 512
#else
#define TILE_WIDE 128
#endif
#define TILE_SPAN (1 << 20)

/* A walk that moves elements one by one, through tiles or along strides,
   costs as much time as this many times its bytes moved by memcpy, and its
   job sizes its threads so: a transposed 512x512 float32 copy, tile by
   tile, takes 4.6 times memcpy's time on one kimchi core. */
#define STRIDED_COST 4

/* The elements of a unit of small planes. A unit's first block costs
   divisions, which a unit per plane of 2x2 windows, 52 elements, pays for
   every 52: kimchi copies the 2x2 windows of 32x16x26x26 float32 in 19.7
   us a plane to a unit, 15.5 at most 4096 elements to one. */
#define PLANES 4096

typedef struct {
  int n;
  nx_loop l;
  int64_t n0, n1;    /* a block's elements per row and rows, at most */
  int64_t pieces;    /* blocks along a row */
  int64_t rowblocks; /* blocks along the rows of a plane */
  int64_t n2;          /* planes a unit runs, along the innermost outer axis */
  int64_t planeblocks; /* units along that axis */
  nx_cpu_block_fn f;
  void *ctx;
} walk;

static void swap_axes(nx_loop *l, int n, int i, int j) {
  int64_t x = l->extent[i];
  l->extent[i] = l->extent[j];
  l->extent[j] = x;
  for (int k = 0; k < n; k++) {
    x = l->step[k][i];
    l->step[k][i] = l->step[k][j];
    l->step[k][j] = x;
  }
}

static int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }

/* The outer axis along which an input of [l] steps least, if it steps less
   there than along the innermost axis, where operand 0 steps by one
   element; -1 otherwise. An axis the input is broadcast along, of step 0,
   is not a transposed one. */
static int tile_axis(int n, const nx_loop *l) {
  int r = l->rank, t = -1;
  if (r < 2 || l->step[0][r - 1] != 1) return -1;
  for (int k = 1; k < n && t < 0; k++) {
    int64_t least = magnitude(l->step[k][r - 1]);
    for (int i = 0; i < r - 1; i++)
      if (l->step[k][i] != 0 && magnitude(l->step[k][i]) < least) {
        least = magnitude(l->step[k][i]);
        t = i;
      }
  }
  return t;
}

/* Unit u's block: its first plane, then the same block in each next plane
   along the innermost axis outside the rows that the unit runs. */
static void block_of(const walk *w, int64_t u, nx_cpu_block *b) {
  const nx_loop *l = &w->l;
  int r = l->rank;
  int64_t piece = u % w->pieces;
  u /= w->pieces;
  int64_t rows = u % w->rowblocks;
  u /= w->rowblocks;
  int64_t i0 = piece * w->n0, j0 = rows * w->n1;
  int64_t len = l->extent[r - 1], height = r > 1 ? l->extent[r - 2] : 1;
  b->n0 = len - i0 < w->n0 ? len - i0 : w->n0;
  b->n1 = height - j0 < w->n1 ? height - j0 : w->n1;
  for (int k = 0; k < w->n; k++) {
    b->s0[k] = l->step[k][r - 1];
    b->s1[k] = r > 1 ? l->step[k][r - 2] : 0;
    b->at[k] = l->first[k] + i0 * b->s0[k] + j0 * b->s1[k];
    b->s2[k] = r > 2 ? l->step[k][r - 3] : 0;
  }
  b->n2 = 1;
  if (r < 3) return;
  /* u is the plane: its index on each axis outside the rows, innermost
     first, the innermost in runs of n2. */
  int64_t x = u % w->planeblocks * w->n2, planes = l->extent[r - 3] - x;
  u /= w->planeblocks;
  b->n2 = planes < w->n2 ? planes : w->n2;
  for (int k = 0; k < w->n; k++) b->at[k] += x * l->step[k][r - 3];
  for (int i = r - 4; i >= 0; i--) {
    x = u % l->extent[i];
    u /= l->extent[i];
    for (int k = 0; k < w->n; k++) b->at[k] += x * l->step[k][i];
  }
}

static void run(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const walk *w = ctx;
  nx_cpu_block b;
  for (int64_t u = lo; u < hi; u++) {
    block_of(w, u, &b);
    w->f(&b, w->ctx);
  }
}

void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l, int64_t most,
                 nx_cpu_block_fn f, void *ctx) {
  walk w = {n, *l, 0, 0, 0, 0, 1, 1, f, ctx};
  nx_loop *wl = &w.l;
  int r = wl->rank, bits = 0, widest = 1, byte_wide = 1;
  if (wl->extent[0] == 0) return;
  for (int k = 0; k < n; k++) {
    bits += a[k].bits;
    if (a[k].bits < 8)
      byte_wide = 0;
    else if (a[k].bits / 8 > widest)
      widest = a[k].bits / 8;
  }
  int tiled = 0;
  if (byte_wide && r > 1) {
    if (wl->extent[r - 1] < SHORT)
      for (int i = r - 2; i >= 0; i--)
        if (wl->extent[i] >= SHORT) {
          swap_axes(wl, n, i, r - 1);
          break;
        }
    int t = tile_axis(n, wl);
    if (t >= 0) {
      swap_axes(wl, n, t, r - 2);
      tiled = 1;
    }
  }
  int64_t len = wl->extent[r - 1], height = r > 1 ? wl->extent[r - 2] : 1;
  if (tiled) {
    /* The tile reads its input's lines whole and writes long runs of the
       output, as long as its span allows. */
    int64_t stride = 1;
    for (int k = 1; k < n; k++) {
      int64_t b = magnitude(wl->step[k][r - 1]) * (a[k].bits / 8);
      if (b > stride) stride = b;
    }
    int64_t across = TILE_SPAN / stride > 1 ? TILE_SPAN / stride : 1;
    w.n1 = TILE_WIDE / widest < height ? TILE_WIDE / widest : height;
    if (across > most / w.n1) across = most / w.n1;
    w.n0 = across < len ? across : len;
  } else {
    w.n0 = len < most ? len : most;
    w.n1 = most / w.n0 < height ? most / w.n0 : height;
  }
  w.pieces = (len + w.n0 - 1) / w.n0;
  w.rowblocks = (height + w.n1 - 1) / w.n1;
  int64_t elements = 1;
  for (int i = 0; i < r; i++) elements *= wl->extent[i];
  int64_t planes = elements / (len * height);
  if (r > 2) {
    /* A block of a whole small plane runs with the next ones in its
       unit. */
    if (w.pieces == 1 && w.rowblocks == 1 && len * height < PLANES)
      w.n2 = PLANES / (len * height);
    if (w.n2 > wl->extent[r - 3]) w.n2 = wl->extent[r - 3];
    w.planeblocks = (wl->extent[r - 3] + w.n2 - 1) / w.n2;
    planes = planes / wl->extent[r - 3] * w.planeblocks;
  }
  int strided = tiled;
  for (int k = 0; k < n; k++) strided |= magnitude(wl->step[k][r - 1]) > 1;
  int64_t bytes = elements * bits / 8;
  nx_cpu_job(planes * w.rowblocks * w.pieces, bytes,
             strided ? STRIDED_COST * bytes : bytes, run, &w);
}
