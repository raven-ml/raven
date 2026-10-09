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

/* Appends a launch as nx_metal_add does: 0, or -2 if memory runs out. */
static int append(nx_metal_records *r, int entry, const uint32_t groups[3],
                  const uint32_t threads[3], const void *params,
                  uint32_t bytes, uint32_t addrs, uint32_t scratch) {
  return nx_metal_add(r, (uint32_t)entry, groups, threads, params, bytes,
                      addrs, scratch)
             ? -2
             : 0;
}

/* Contract */

/* The dense kernels' tiles, rows × columns (kernels.h), and the column
   of tiles that consecutive threadgroups take, as a power of two: 4 tiles
   reading one tile of b share it in the GPU's cache. */
enum { swizzle = 2 };
enum size { Large, Small, Wide };
static const uint32_t tile_rows[] = {NX_METAL_LARGE, NX_METAL_SMALL,
                                     NX_METAL_WIDE_M},
                      tile_cols[] = {NX_METAL_LARGE, NX_METAL_SMALL,
                                     NX_METAL_WIDE_N};

/* The steps of k the dense kernel stages for a dtype and tile. */
static uint32_t tile_k(int dt, enum size size) {
  if (size == Wide) return NX_METAL_BK_WIDE;
  return dt == NX_FLOAT16 || dt == NX_INT8 || dt == NX_UINT8
             ? NX_METAL_BK_HALF
             : NX_METAL_BK;
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

/* The instance for a's and b's dtype and order: NX_METAL_contract_<dt>_<a
   order><b order>, n where the operand's last axis has unit stride; its
   _s and _w twins, of small and wide tiles (floats only); and the _edge
   twin of a square tile's, which reads tiles that reach past m, n or k,
   as the wide tile's one instance does. */
static int dense_entry(int dt, int a_t, int b_t, enum size size, int edge) {
  int order = 2 * a_t + b_t, at = 5 * order + 2 * size + (size != Wide && edge);
  switch (dt) {
  case NX_FLOAT32: return NX_METAL_contract_f32_nn + at;
  case NX_FLOAT16: return NX_METAL_contract_f16_nn + at;
  case NX_BFLOAT16: return NX_METAL_contract_bf16_nn + at;
  case NX_INT8: return NX_METAL_contract_i8_nn + 2 * order + edge;
  default: return NX_METAL_contract_u8_nn + 2 * order + edge;
  }
}

/* The instance for a's and b's dtype and b's order. */
static int skinny_entry(int dt, int b_t) {
  int first = dt == NX_FLOAT32   ? NX_METAL_skinny_f32_n
              : dt == NX_FLOAT16 ? NX_METAL_skinny_f16_n
                                 : NX_METAL_skinny_bf16_n;
  return first + b_t;
}

static int fits32(int64_t x) { return x >= 0 && x <= UINT32_MAX; }

/* Whether every element of a rows × cols matrix with strides s (over the
   call's three axes) lies within 2^32 elements of the batch's first: the
   kernels index within a batch in 32 bits. */
static int spans32(const int64_t s[3], int64_t rows, int64_t cols) {
  return fits32(s[1]) && fits32(s[2]) &&
         (rows == 0 || cols == 0 || fits32((rows - 1) * s[1] + (cols - 1) * s[2]));
}

static int integer(int dt) {
  return dt == NX_INT8 || dt == NX_UINT8 || dt == NX_INT16 ||
         dt == NX_UINT16 || dt == NX_INT32 || dt == NX_UINT32 ||
         dt == NX_INT64 || dt == NX_UINT64;
}


/* An integer contraction: any strides, operands widened to the
   accumulator, 32 or 64 bits. */
static int plan_integer(const nx_metal_contract_in *c, nx_metal_contract *p,
                        nx_metal_records *r) {
  if (c->batch == 0 || c->m == 0 || c->n == 0) return 0;
  int wide = c->acc == NX_INT64 || c->acc == NX_UINT64;
  uint32_t tile = NX_METAL_INT_TILE;
  uint32_t groups[3] = {(c->n + tile - 1) / tile, (c->m + tile - 1) / tile,
                        c->batch};
  uint32_t threads[3] = {NX_METAL_INT_THREADS, 1, 1};
  int e = append(r, wide ? NX_METAL_contract_i64 : NX_METAL_contract_i32,
                 groups, threads, p, sizeof *p, 4, 0);
  return e ? e : 1;
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
  /* The matrix units take an operand with one axis of unit stride, or of
     one element; the order names which, the last axis where both may. */
  int a_t = as[2] != 1 && c->k > 1, b_t = bs[2] != 1 && c->n > 1;
  int ordered = !(a_t && as[1] != 1 && c->m > 1) && !(b_t && bs[1] != 1 && c->k > 1);
  /* int8 and uint8 into 32 bits run on the matrix units, exactly. */
  int bytes = (a->dtype == NX_INT8 || a->dtype == NX_UINT8) &&
              (c->acc == NX_INT32 || c->acc == NX_UINT32);
  if (ints && !(bytes && ordered))
    return plan_integer(c, &p, r);
  if (!ordered) return NX_NOT_COMPUTED;
  if (c->batch == 0 || c->m == 0 || c->n == 0) return 0;
  uint32_t threads[3] = {NX_METAL_THREADS, 1, 1};
  if (c->m == 1 && floats) {
    uint32_t per = b_t ? NX_METAL_SKINNY_T : NX_METAL_SKINNY_N;
    uint32_t groups[3] = {(c->n + per - 1) / per, 1, c->batch};
    int e = append(r, skinny_entry(a->dtype, b_t), groups, threads, &p,
                   sizeof p, 4, 0);
    return e ? e : 1;
  }
  uint32_t side = tile_rows[Large];
  uint64_t large =
      (uint64_t)((c->m + side - 1) / side) * ((c->n + side - 1) / side) * c->batch;
  enum size size = !floats                 ? Large
                   : c->m <= wide_rows     ? Wide
                   : large < small_tiles   ? Small
                                           : Large;
  uint32_t rows = tile_rows[size], cols = tile_cols[size];
  uint32_t tiles_m = (c->m + rows - 1) / rows;
  uint32_t tiles_n = (c->n + cols - 1) / cols;
  p.swizzle = tiles_m >= 2 * (1 << swizzle) ? swizzle : 0;
  uint32_t column = 1u << p.swizzle;
  uint32_t groups[3] = {tiles_n * column, (tiles_m + column - 1) / column,
                        c->batch};
  /* A product of few tiles splits along k into parts of equal length, a
     batch of contractions of float32 parts, which contract_combine adds
     in order. Integers sum in chunks within one threadgroup: they never
     split. */
  uint64_t tiles = (uint64_t)tiles_m * tiles_n * c->batch;
  uint32_t parts = 1;
  while (floats && c->batch == 1 && tiles * parts < split_tiles &&
         2 * parts <= max_parts && c->k % (2 * parts) == 0 &&
         c->k / (2 * parts) >= min_part_k)
    parts *= 2;
  uint32_t part_k = c->k / parts;
  /* Whole tiles read bytes 16 at a time: every tile's first byte, which
     the addresses, the row strides and the batch strides place, lies at a
     multiple of 16. */
  int unaligned = bytes && ((a->address | b->address) % 16 ||
                            (a_t ? as[2] : as[1]) % 16 ||
                            (b_t ? bs[2] : bs[1]) % 16 ||
                            (c->batch > 1 && (as[0] % 16 || bs[0] % 16)));
  int edge = c->m % rows || c->n % cols || part_k % tile_k(a->dtype, size) ||
             unaligned;
  int entry = dense_entry(a->dtype, a_t, b_t, size, edge);
  if (parts == 1) {
    int e = append(r, entry, groups, threads, &p, sizeof p, 4, 0);
    return e ? e : 1;
  }
  uint64_t count = (uint64_t)c->m * c->n;
  *scratch = (parts * count * 4 + 15) / 16 * 16;
  nx_metal_combine q = {
      .out = out->address, .parts = 0, .init = p.init,
      .init_batch = p.init_batch, .init_m = p.init_m, .init_n = p.init_n,
      .batch = 1, .m = c->m, .n = c->n, .split = parts,
      .init_dtype = p.init_dtype, .out_dtype = p.out_dtype};
  p.out = 0;
  p.a_batch = (int64_t)part_k * as[2];
  p.b_batch = (int64_t)part_k * bs[1];
  p.batch = parts;
  p.k = part_k;
  p.init_dtype = NX_DTYPE_COUNT;
  p.out_dtype = NX_FLOAT32;
  groups[2] = parts;
  int e = append(r, entry, groups, threads, &p, sizeof p, 4, 1u << 3);
  uint32_t cgroups[3] = {(uint32_t)((count + combine_threads - 1) / combine_threads), 1, 1};
  uint32_t cthreads[3] = {combine_threads, 1, 1};
  if (!e)
    e = append(r, NX_METAL_contract_combine, cgroups, cthreads, &q, sizeof q, 3,
               1u << 1);
  return e ? e : 2;
}
