/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Each family's attributes to launch records: which kernels a call runs,
   over which threadgroups, with which parameters. A pure function of the
   dtypes, the shapes and the strides: no association depends on the GPU's
   size or load. */

#define _GNU_SOURCE

#include "nx_metal.h"
#include "nx_dtype.h"

/* Contract */

/* The dense kernels' tiles, rows × columns (kernels.h), and the column
   of tiles that consecutive threadgroups take, as a power of two: 4 tiles
   reading one tile of b share it in the GPU's cache. */
enum { swizzle = 2 };
enum size { Large, Small, Wide, Checked };
static const uint32_t tile_rows[] = {NX_METAL_LARGE, NX_METAL_SMALL,
                                     NX_METAL_WIDE_M, NX_METAL_LARGE},
                      tile_cols[] = {NX_METAL_LARGE, NX_METAL_SMALL,
                                     NX_METAL_WIDE_N, NX_METAL_LARGE};

/* The steps of k the dense kernel stages for a dtype and tile. */
static uint32_t tile_k(int dt, enum size size) {
  if (size == Wide) return NX_METAL_BK_WIDE;
  return dt == NX_FLOAT16 || dt == NX_INT8 ? NX_METAL_BK_HALF : NX_METAL_BK;
}

/* A float product of fewer large tiles than this runs on small ones, which
   give the GPU's cores 4 times as many threadgroups; one of at most
   wide_rows rows runs on wide ones, which waste fewer of the matrix
   units' rows. */
enum { small_tiles = 128, wide_rows = 16 };

/* A product of fewer tiles than this splits along k into parts whose sums
   contract_combine adds, so that the GPU's cores have work and each
   threadgroup a shorter chain of steps: up to max_parts parts of at least
   min_part_k terms each, since shorter parts cost more in their partial
   sums than they save. A function of the shape, as every association
   is. */
enum { split_tiles = 256, max_parts = 16, min_part_k = 256 };

/* The combine's threads per threadgroup. */
enum { combine_threads = 256 };

static int dense_dtype(int dt) {
  return dt == NX_FLOAT32 || dt == NX_FLOAT16 || dt == NX_BFLOAT16;
}

/* The instances by a's and b's dtype (float32, float16, bfloat16) and
   order (a stored [m][k] or [k][m], b stored [k][n] or [n][k]). The small
   tiles read either order from one instance. */
static const int large[3][2][2] = {
    {{NX_METAL_contract_f32_nn, NX_METAL_contract_f32_nt},
     {NX_METAL_contract_f32_tn, NX_METAL_contract_f32_tt}},
    {{NX_METAL_contract_f16_nn, NX_METAL_contract_f16_nt},
     {NX_METAL_contract_f16_tn, NX_METAL_contract_f16_tt}},
    {{NX_METAL_contract_bf16_nn, NX_METAL_contract_bf16_nt},
     {NX_METAL_contract_bf16_tn, NX_METAL_contract_bf16_tt}}};
static const int wide[3][2][2] = {
    {{NX_METAL_contract_f32_wnn, NX_METAL_contract_f32_wnt},
     {NX_METAL_contract_f32_wtn, NX_METAL_contract_f32_wtt}},
    {{NX_METAL_contract_f16_wnn, NX_METAL_contract_f16_wnt},
     {NX_METAL_contract_f16_wtn, NX_METAL_contract_f16_wtt}},
    {{NX_METAL_contract_bf16_wnn, NX_METAL_contract_bf16_wnt},
     {NX_METAL_contract_bf16_wtn, NX_METAL_contract_bf16_wtt}}};
static const int small[3] = {NX_METAL_contract_f32_s, NX_METAL_contract_f16_s,
                             NX_METAL_contract_bf16_s};
static const int checked[3] = {NX_METAL_contract_f32_l, -1,
                               NX_METAL_contract_bf16_l};
static const int skinny[3][2] = {
    {NX_METAL_skinny_f32_n, NX_METAL_skinny_f32_t},
    {NX_METAL_skinny_f16_n, NX_METAL_skinny_f16_t},
    {NX_METAL_skinny_bf16_n, NX_METAL_skinny_bf16_t}};
static const int int8[2][2] = {
    {NX_METAL_contract_i8_nn, NX_METAL_contract_i8_nt},
    {NX_METAL_contract_i8_tn, NX_METAL_contract_i8_tt}};

/* The float dtype [dt]'s index in the tables above. */
static int dtype_index(int dt) {
  return dt == NX_FLOAT32 ? 0 : dt == NX_FLOAT16 ? 1 : 2;
}

static int fits32(int64_t x) { return x >= 0 && x <= UINT32_MAX; }

/* Whether every element of a rows × cols matrix with strides s (over the
   call's three axes) lies within 2^32 elements of the batch's first: the
   kernels index within a batch in 32 bits. */
static int spans32(const int64_t s[3], int64_t rows, int64_t cols) {
  return fits32(s[1]) && fits32(s[2]) &&
         (rows == 0 || cols == 0 ||
          fits32((rows - 1) * s[1] + (cols - 1) * s[2]));
}

static int integer(int dt) {
  return dt == NX_INT8 || dt == NX_UINT8 || dt == NX_INT16 ||
         dt == NX_UINT16 || dt == NX_INT32 || dt == NX_UINT32 ||
         dt == NX_INT64 || dt == NX_UINT64;
}

/* How many parts a product of [tiles] tiles splits into along k: floats
   only, and only a batch of one. */
static uint32_t split(const nx_metal_contract_in *c, uint64_t tiles,
                      int floats) {
  uint32_t parts = 1;
  if (!floats || c->batch != 1) return parts;
  while (tiles * parts < split_tiles && 2 * parts <= max_parts &&
         c->k % (2 * parts) == 0 && c->k / (2 * parts) >= min_part_k)
    parts *= 2;
  return parts;
}

/* The tiles of [size] that cover a product's outputs. */
static uint64_t tiles(const nx_metal_contract_in *c, enum size size) {
  uint64_t rows = tile_rows[size], cols = tile_cols[size];
  return (c->m + rows - 1) / rows * ((c->n + cols - 1) / cols) * c->batch;
}

/* Whether tiles of [size] leave at most an eighth of a product's rows
   empty. */
static int fills(const nx_metal_contract_in *c, enum size size) {
  uint32_t rows = tile_rows[size], covered = (c->m + rows - 1) / rows * rows;
  return 8 * (covered - c->m) <= covered;
}

/* The tile a product's dtype and shape pick: wide for few rows; large for
   products of whole large tiles, enough of them, and whole steps of k in
   each part; small for the other products of whole tiles. Past whole
   tiles, the largest of the dtype's tiles that leaves at most an eighth
   of the rows empty: checked large ones for float32 and bfloat16, given
   enough of them; small ones; wide ones for the half types, small ones
   for float32. Measured on the 1000 cube, checked large tiles run
   float32 and bfloat16 nt 0.92 of small ones and nn 1.04 and 0.99,
   float16 nn 1.06; on 48 rows, wide tiles run the half types 1.2
   to 1.7 times faster than small ones, and float32 nt 1.3 times slower.
   int8 runs on whole large tiles. */
static enum size tile_of(const nx_metal_contract_in *c, int dt, int floats) {
  if (!floats) return Large;
  if (c->m <= wide_rows) return Wide;
  uint32_t side = tile_rows[Large];
  int whole = c->m % side == 0 && c->n % side == 0;
  uint64_t large = tiles(c, Large);
  if (whole && large >= small_tiles &&
      c->k / split(c, large, floats) % tile_k(dt, Large) == 0)
    return Large;
  if (whole) return Small;
  if (dt != NX_FLOAT16 && large >= small_tiles && fills(c, Checked))
    return Checked;
  return dt == NX_FLOAT32 || fills(c, Small) ? Small : Wide;
}

/* [bytes] of the call's scratch after the [*used] bytes taken: their
   offset, on a 256-byte boundary. */
static uint64_t take(size_t *used, size_t bytes) {
  size_t at = (*used + 255) & ~(size_t)255;
  *used = at + bytes;
  return at;
}

/* Whether an operand's stored rows start on 16-byte boundaries: its
   address, row stride and, past one batch element, batch stride. */
static int aligned(uint64_t address, int64_t row, int64_t batch,
                   uint32_t count) {
  return address % 16 == 0 && row % 16 == 0 && (count == 1 || batch % 16 == 0);
}

/* An integer contraction on the SIMD units: any strides, operands widened
   to 64 bits. */
static int plan_integer(const nx_metal_contract_in *c,
                        const nx_metal_contract *p, nx_metal_records *r) {
  uint32_t tile = NX_METAL_INT_TILE;
  uint32_t groups[3] = {(c->n + tile - 1) / tile, (c->m + tile - 1) / tile,
                        c->batch};
  uint32_t threads[3] = {NX_METAL_INT_THREADS, 1, 1};
  int e = nx_metal_add(r, NX_METAL_contract_int, groups, threads, p,
                       sizeof *p, 4, 0);
  return e ? e : 1;
}

/* A float product of one row of a. */
static int plan_skinny(const nx_metal_contract_in *c,
                       const nx_metal_contract *p, nx_metal_records *r) {
  int b_t = (p->order & NX_METAL_B_T) != 0;
  uint32_t per = b_t ? NX_METAL_SKINNY_T : NX_METAL_SKINNY_N;
  uint32_t groups[3] = {(c->n + per - 1) / per, 1, c->batch};
  uint32_t threads[3] = {NX_METAL_THREADS, 1, 1};
  int e = nx_metal_add(r, skinny[dtype_index((int)p->dtype)][b_t], groups,
                       threads, p, sizeof *p, 4, 0);
  return e ? e : 1;
}

/* A product on the matrix units, each operand read where it lies, as
   p->order says: int8 into 32 bits if [i8], floats otherwise. */
static int plan_dense(const nx_metal_contract_in *c, nx_metal_contract *p,
                      int i8, nx_metal_records *r, size_t *scratch) {
  int floats = !i8;
  /* The tile the shape picks fixes the split, so every output's
     association is the shape's. */
  enum size size = tile_of(c, (int)p->dtype, floats);
  uint32_t rows = tile_rows[size], cols = tile_cols[size];
  uint32_t tiles_m = (c->m + rows - 1) / rows;
  uint32_t tiles_n = (c->n + cols - 1) / cols;
  p->swizzle = tiles_m >= 2 * (1 << swizzle) ? swizzle : 0;
  uint32_t column = 1u << p->swizzle;
  uint32_t groups[3] = {tiles_n * column, (tiles_m + column - 1) / column,
                        c->batch};
  uint32_t threads[3] = {NX_METAL_THREADS, 1, 1};
  int a_t = (p->order & NX_METAL_A_T) != 0;
  int b_t = (p->order & NX_METAL_B_T) != 0;
  int d = dtype_index((int)p->dtype);
  int entry = i8               ? int8[a_t][b_t]
              : size == Large ? large[d][a_t][b_t]
              : size == Small ? small[d]
              : size == Checked ? checked[d]
                              : wide[d][a_t][b_t];
  uint32_t parts = split(c, tiles(c, size), floats);
  if (parts == 1) {
    if (nx_metal_add(r, entry, groups, threads, p, sizeof *p, 4, 0))
      return NX_OUT_OF_MEMORY;
    return 1;
  }

  /* A product of few tiles splits along k into parts of equal length: a
     batch of float32 contractions, part q reading k from q · part_k,
     whose outputs contract_combine adds in order, then init. Integers sum
     in chunks within one threadgroup: they never split. */
  uint32_t part_k = c->k / parts;
  uint64_t count = (uint64_t)c->m * c->n;
  size_t used = 0, len = r->len;
  nx_metal_combine q = {
      .out = p->out, .parts = take(&used, parts * count * 4), .init = p->init,
      .init_batch = p->init_batch, .init_m = p->init_m,
      .init_n = p->init_n, .batch = 1, .m = c->m, .n = c->n, .split = parts,
      .init_dtype = p->init_dtype, .out_dtype = p->out_dtype};
  p->out = q.parts;
  p->a_batch = (int64_t)part_k * p->a_k;
  p->b_batch = (int64_t)part_k * p->b_k;
  p->batch = parts;
  p->k = part_k;
  p->init_dtype = NX_DTYPE_COUNT;
  p->out_dtype = NX_FLOAT32;
  groups[2] = parts;
  uint32_t cgroups[3] = {
      (uint32_t)((count + combine_threads - 1) / combine_threads), 1, 1};
  uint32_t cthreads[3] = {combine_threads, 1, 1};
  if (nx_metal_add(r, entry, groups, threads, p, sizeof *p, 4, 1u << 3) ||
      nx_metal_add(r, NX_METAL_contract_combine, cgroups, cthreads, &q,
                   sizeof q, 3, 1u << 1)) {
    r->len = len;
    return NX_OUT_OF_MEMORY;
  }
  *scratch = (used + 15) / 16 * 16;
  return 2;
}

int nx_metal_plan_contract(const nx_metal_contract_in *c,
                           const nx_metal_operand *a,
                           const nx_metal_operand *b,
                           const nx_metal_operand *out,
                           const nx_metal_operand *init,
                           nx_metal_records *r, size_t *scratch) {
  *scratch = 0;
  int ints = integer(a->dtype) && integer(c->acc) && integer(out->dtype) &&
             (!init || integer(init->dtype));
  int floats = c->acc == NX_FLOAT32 && dense_dtype(a->dtype) &&
               dense_dtype(out->dtype) && (!init || dense_dtype(init->dtype));
  if (a->dtype != b->dtype || !(ints || floats)) return NX_NOT_COMPUTED;
  const int64_t *as = a->strides, *bs = b->strides;
  const int64_t none[3] = {0, 0, 0}, *is = init ? init->strides : none;
  if (!spans32(as, c->m, c->k) || !spans32(bs, c->k, c->n) ||
      !spans32(is, c->m, c->n))
    return NX_NOT_COMPUTED;
  if (c->batch == 0 || c->m == 0 || c->n == 0) return 0;
  nx_metal_contract p = {
      .a = a->address, .b = b->address, .init = init ? init->address : 0,
      .out = out->address,
      .a_batch = as[0], .b_batch = bs[0], .init_batch = is[0],
      .a_m = (uint32_t)as[1], .a_k = (uint32_t)as[2],
      .b_k = (uint32_t)bs[1], .b_n = (uint32_t)bs[2],
      .init_m = (uint32_t)is[1], .init_n = (uint32_t)is[2],
      .batch = c->batch, .m = c->m, .n = c->n, .k = c->k,
      .init_dtype = init ? (uint32_t)init->dtype : NX_DTYPE_COUNT,
      .out_dtype = (uint32_t)out->dtype, .dtype = (uint32_t)a->dtype,
      .acc = (uint32_t)c->acc};
  /* The kernels read an operand with one axis of unit stride, or of one
     element; t names the operands whose other axis it is. */
  int a_t = as[2] != 1 && c->k > 1, b_t = bs[2] != 1 && c->n > 1;
  int ordered = !(a_t && as[1] != 1 && c->m > 1) &&
                !(b_t && bs[1] != 1 && c->k > 1);
  p.order = (a_t ? NX_METAL_A_T : 0) | (b_t ? NX_METAL_B_T : 0);
  /* int8 into 32 bits runs on the matrix units, exactly, in whole tiles
     whose stored rows start on 16-byte boundaries, which its 16-byte loads
     need; every other integer contraction on the SIMD units. */
  uint32_t side = tile_rows[Large];
  int i8 = ints && a->dtype == NX_INT8 &&
           (c->acc == NX_INT32 || c->acc == NX_UINT32) && ordered &&
           c->m % side == 0 && c->n % side == 0 &&
           c->k % NX_METAL_BK_HALF == 0 &&
           (c->k == 0 ||
            (aligned(a->address, a_t ? as[2] : as[1], as[0], c->batch) &&
             aligned(b->address, b_t ? bs[2] : bs[1], bs[0], c->batch)));
  if (ints && !i8) return plan_integer(c, &p, r);
  if (!ordered) return NX_NOT_COMPUTED;
  if (c->m == 1 && floats) return plan_skinny(c, &p, r);
  return plan_dense(c, &p, i8, r, scratch);
}
