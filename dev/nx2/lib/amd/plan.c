/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A call's launches, from its dtypes, shapes, attributes and the GPU's
   processor: never from the GPU's size, clocks or timing, so that a
   result's bits depend on its shape alone.

   A plan declines a call, and nx's expansion computes it, when an operand,
   init or y is complex or narrower than a byte; a float operand or init
   is wider than a float accumulator, which would round it before the sum;
   init is of the other kind than the accumulator; an integer meets a float
   accumulator or the reverse; the batch, free or contracting axes do not
   merge into one axis each; the batch is above 65,535 (the grid's z); or
   m, n or k is above 2^31 - 1. */

#include <string.h>

#include "nx_array.h"
#include "nx_amd.h"

/* The processors this library has a code object for. */
#define ARCH_GFX1201 1201

/* Records */

static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

/* Appends the launch of [kernel] over [groups] of [threads] work-items with
   [params]; -1 if memory runs out. */
static int add(nx_amd_records *out, int kernel, uint32_t gx, uint32_t gy,
               uint32_t gz, uint32_t threads, const void *params,
               uint32_t bytes, uint32_t addrs, uint32_t scratch) {
  const uint32_t groups[3] = {gx, gy, gz}, items[3] = {threads, 1, 1};
  return nx_amd_add(out, (uint32_t)kernel, groups, items, params, bytes, addrs,
                    scratch);
}

/* Contract */

/* The kernels by family and instance, from kernels.h's list. */
enum { F_ZERO, F_PACK, F_WMMA, F_SIMT, F_SKINNY };
enum { K_bf16, K_f16, K_s8 };
enum { ACC_f32, ACC_f64, ACC_i64 };
enum { S_column, S_across };
#define TILE_INDEX(name, ...) T_##name,
enum { NX_AMD_TILES(TILE_INDEX) T_COUNT };
#undef TILE_INDEX

/* An instance's arguments, by family: WMMA its operands' kind and its tile;
   SIMT its accumulator and its tile's side; SKINNY its accumulator and
   form. */
typedef struct {
  int family, kind, size;
} instance;

#define ARGS_ZERO() 0, 0
#define ARGS_PACK() 0, 0
#define ARGS_WMMA(kind, tile) K_##kind, T_##tile
#define ARGS_SIMT(acc, side) ACC_##acc, side
#define ARGS_SKINNY(acc, form) ACC_##acc, S_##form
#define INSTANCE(name, FAMILY, ...) {F_##FAMILY, ARGS_##FAMILY(__VA_ARGS__)},
static const instance instances[] = {NX_AMD_KERNELS(INSTANCE)};
#undef INSTANCE

static int find(int family, int kind, int size) {
  for (int i = 0; i < NX_AMD_KERNEL_COUNT; i++) {
    const instance *x = &instances[i];
    if (x->family == family && x->kind == kind && x->size == size) return i;
  }
  return -1;
}

typedef struct {
  int bm, bn, bkb, threads;
} tile;

#define TILE_ROW(name, bm, bn, bkb, wm, wn) \
  {bm, bn, bkb, (bm / wm) * (bn / wn) * 32},
static const tile tiles[] = {NX_AMD_TILES(TILE_ROW)};
#undef TILE_ROW

/* The workgroups of a tile over [batch] x [m] x [n] outputs. */
static uint64_t workgroups(int t, int64_t batch, int64_t m, int64_t n) {
  return batch * ceil_div(m, tiles[t].bm) * ceil_div(n, tiles[t].bn);
}

/* What the WMMA tiles cost on gfx1201: workgroups run in waves of [WAVE],
   and each tile's outputs per unit of time relative to the 128 x 128
   tile's, in percent. Measured on the R9700: the 128 x 128 tile's time
   steps every 96 workgroups, three to each of its 32 work-group
   processors; the 64 x 64 tile makes 76-83% of its outputs per unit of
   time at 2048 and 4096 cubed, bfloat16 and float16. The 16 x 64 tile's
   33 is Ada's: m <= 16 takes that tile alone, so its rate is never weighed.
   A GPU of another size runs the same tiles, with the same bits. */
#define WAVE 96
static const int efficiency[T_COUNT] = {[T_t128x128] = 100, [T_t64x64] = 80,
                                        [T_t16x64] = 33};

/* The WMMA tile of a product among those [kind] has an instance of: the
   one of least cost, waves times outputs per tile over its efficiency, by
   its shape alone; m <= 16 takes the 16-row tile. -1 if [kind] has none
   for the shape: the SIMT or skinny kernels sum it. */
static int wmma_tile(int kind, int64_t batch, int64_t m, int64_t n) {
  int best = -1;
  double least = -1;
  for (int t = 0; t < T_COUNT; t++) {
    if (find(F_WMMA, kind, t) < 0 || (m <= 16) != (t == T_t16x64)) continue;
    double cost = (double)ceil_div(workgroups(t, batch, m, n), WAVE) * tiles[t].bm *
                  tiles[t].bn / efficiency[t];
    if (least < 0 || cost < least) least = cost, best = t;
  }
  return best;
}

/* Merges the [count] axes [axes[k]] of each of [n] operands into one: its
   extent and each operand's stride. Answers 0 if they do not merge into
   one axis: the operands' extents differ, or an axis past extent 1 steps
   by other than the next such axis's stride times that axis's extent. No
   axis is extent 1, and stride 0. It reads the operands as they are:
   zeroing a descriptor per operand on the stack, some 2.5 KiB of stores a
   call, made the loads after it wait on Intel cores in half of all
   processes, those whose stack met the operands in the low 12 address
   bits (4K aliasing). */
static int merge(int n, const nx_amd_operand *const *ops,
                 const int (*axes)[NX_MAX_RANK], int count, int64_t *extent,
                 int64_t *strides) {
  int64_t e = 1;
  int found = 0;
  for (int i = 0; i < count; i++)
    for (int k = 1; k < n; k++)
      if (ops[k]->dim[axes[k][i]] != ops[0]->dim[axes[0][i]]) return 0;
  for (int k = 0; k < n; k++) strides[k] = 0;
  for (int i = 0; i < count; i++) {
    const int64_t d = ops[0]->dim[axes[0][i]];
    if (d == 0) {
      *extent = 0;
      for (int k = 0; k < n; k++) strides[k] = 0;
      return 1;
    }
    if (d == 1) continue;
    for (int k = 0; k < n; k++) {
      const int64_t s = ops[k]->dim[ops[k]->rank + axes[k][i]];
      if (found && strides[k] != s * d) return 0;
      strides[k] = s;
    }
    e *= d, found = 1;
  }
  *extent = e;
  return 1;
}

/* Whether operand rows along [contiguous] (stride 1) load as 16-byte
   vectors: every other stride a whole number of vectors, and the first
   element on a vector. */
static int vectors(uint64_t address, int64_t contiguous, int64_t lead,
                   int64_t batch, int bytes) {
  const int64_t per = 16 / bytes;
  return contiguous == 1 && lead % per == 0 && batch % per == 0 &&
         address % 16 == 0;
}

static int bytes_of(int dt) { return nx_dtype_row_of(dt).bits / 8; }

/* [bytes] of the call's scratch after the [*used] bytes taken: their
   offset, each piece on a 256-byte boundary. */
static size_t take(size_t *used, size_t bytes) {
  size_t at = (*used + 255) & ~(size_t)255;
  *used = at + bytes;
  return at;
}

/* Appends the launch that packs an operand of [batch] x [rows] x [k]
   elements of [dtype], at [address] with the strides [sz], [sr] and [sk],
   into scratch as elements of [to] with k contiguous and rows of whole
   vectors; sets [*operand] and [strides] to the packed copy's scratch
   offset and strides. -1 if memory runs out, or if its elements outnumber
   the work-items a grid holds. */
static int pack(nx_amd_records *out, size_t *used, const void **operand,
                int64_t strides[3], uint64_t address, int dtype, int to,
                int64_t batch, int64_t rows, int64_t k, int64_t sz, int64_t sr,
                int64_t sk) {
  const int es = bytes_of(to);
  const int64_t per = 16 / es, lead = (k + per - 1) / per * per;
  if ((uint64_t)(batch * rows * lead) > (uint64_t)UINT32_MAX * NX_CONTRACT_THREADS)
    return -1;
  const size_t at = take(used, (size_t)(batch * rows * lead * es));
  pack_params q = {(const void *)(uintptr_t)address, (void *)(uintptr_t)at,
                   {sz, sr, sk}, lead, (int32_t)batch, (int32_t)rows,
                   (int32_t)k, dtype, to, es};
  const uint64_t n = (uint64_t)(batch * rows * lead);
  *operand = (const void *)(uintptr_t)at;
  strides[0] = rows * lead, strides[1] = lead, strides[2] = 1;
  return add(out, NX_AMD_pack, (uint32_t)ceil_div(n, NX_CONTRACT_THREADS), 1,
             1, NX_CONTRACT_THREADS, &q, sizeof q, NX_PACK_ADDRS,
             NX_PACK_SCRATCH);
}

/* Whether [dt] is the SIMT accumulator [simt]'s own dtype: the dtype
   the SIMT and skinny kernels read. */
static int own(int simt, int dt) {
  switch (simt) {
  case ACC_f32: return dt == NX_FLOAT32;
  case ACC_f64: return dt == NX_FLOAT64;
  }
  return dt == NX_INT64 || dt == NX_UINT64;
}

static int own_dtype(int simt) {
  switch (simt) {
  case ACC_f32: return NX_FLOAT32;
  case ACC_f64: return NX_FLOAT64;
  }
  return NX_INT64;
}

static int is_f8(int dt) {
  return dt == NX_FLOAT8_E4M3FN || dt == NX_FLOAT8_E5M2;
}

/* Whether [dt] reads as a value the kernels sum: no complex, no sub-byte
   element. */
static int summable(int dt) {
  const nx_dtype_row r = nx_dtype_row_of(dt);
  return r.kind != NX_KIND_COMPLEX && r.bits >= 8;
}

static int is_int(int dt) {
  switch (dt) {
  case NX_INT64: case NX_UINT64: case NX_INT32: case NX_UINT32:
  case NX_INT16: case NX_UINT16: case NX_INT8: case NX_UINT8: case NX_BOOL:
    return 1;
  }
  return 0;
}

/* The split count of a grid of [grid] workgroups: doubled while the grid has
   fewer than [target] workgroups and each range keeps at least [k_min] of k,
   at most 16. The targets are constants of the processor, never a GPU's
   own count. The 16 x 64 tile's target of 128 is measured on the R9700;
   the other rules' targets and k_min (WMMA m > 16: 64 and 1024, SIMT: 64
   and 128, skinny: 256 and 1024) carry over from Ada's, as the R9700's
   runs at those shapes did not favour other values. */
static int split_count(uint64_t grid, uint64_t target, int64_t k,
                       int64_t k_min) {
  int s = 1;
  while (s < 16 && grid * s < target && k / (2 * s) >= k_min) s *= 2;
  return s;
}

int nx_amd_plan_contract(const nx_amd_contract_in *in,
                          const nx_amd_operand *ops, int arch,
                          nx_amd_records *out, size_t *scratch) {
  const nx_amd_operand *a = &ops[0], *b = &ops[1];
  const nx_amd_operand *init = in->init ? &ops[2] : NULL;
  const nx_amd_operand *y = &ops[in->init ? 3 : 2];
  *scratch = 0;
  if (arch != ARCH_GFX1201) return NX_NOT_COMPUTED;

  /* The axes of each group: batch pairs, a's free axes (m), b's (n), and
     the contracting pairs (k), with y's, and init's as y's. */
  _Static_assert(NX_MAX_RANK <= 32, "a rank's axes fit 32 bits");
  uint32_t used_a = 0, used_b = 0; /* bit i: axis i is grouped */
  int bat[4][NX_MAX_RANK], mm[3][NX_MAX_RANK], nn[3][NX_MAX_RANK],
      kk[2][NX_MAX_RANK];
  int nb = in->nbatch, nm = 0, nnn = 0, nk = in->ncontracting;
  for (int i = 0; i < nb; i++) {
    bat[0][i] = in->batch[i][0], bat[1][i] = in->batch[i][1];
    bat[2][i] = bat[3][i] = i;
    used_a |= 1u << in->batch[i][0], used_b |= 1u << in->batch[i][1];
  }
  for (int i = 0; i < nk; i++) {
    kk[0][i] = in->contracting[i][0], kk[1][i] = in->contracting[i][1];
    used_a |= 1u << in->contracting[i][0];
    used_b |= 1u << in->contracting[i][1];
  }
  for (int i = 0; i < a->rank; i++)
    if (!(used_a >> i & 1))
      mm[0][nm] = i, mm[1][nm] = mm[2][nm] = nb + nm, nm++;
  for (int i = 0; i < b->rank; i++)
    if (!(used_b >> i & 1))
      nn[0][nnn] = i, nn[1][nnn] = nn[2][nnn] = nb + nm + nnn, nnn++;
  if (y->rank != nb + nm + nnn) return NX_NOT_COMPUTED;

  const nx_amd_operand *gb[4] = {a, b, y, init ? init : y};
  const nx_amd_operand *gm[3] = {a, y, init ? init : y};
  const nx_amd_operand *gn[3] = {b, y, init ? init : y};
  const nx_amd_operand *gk[2] = {a, b};
  int64_t batch, m, n, k, sbat[4], sm[3], sn[3], sk[2];
  if (!merge(4, gb, bat, nb, &batch, sbat) || !merge(3, gm, mm, nm, &m, sm) ||
      !merge(3, gn, nn, nnn, &n, sn) || !merge(2, gk, kk, nk, &k, sk))
    return NX_NOT_COMPUTED;
  if (batch > 65535 || m > INT32_MAX || n > INT32_MAX || k > INT32_MAX)
    return NX_NOT_COMPUTED;
  if (batch * m * n == 0) return 0;
  const int acc = in->acc, float_acc = acc == NX_FLOAT32 || acc == NX_FLOAT64;
  if (!summable(a->dtype) || !summable(b->dtype) || !summable(y->dtype) ||
      (init && (!summable(init->dtype) || is_int(init->dtype) == float_acc)) ||
      (float_acc && (bytes_of(a->dtype) > bytes_of(acc) ||
                     bytes_of(b->dtype) > bytes_of(acc) ||
                     (init && bytes_of(init->dtype) > bytes_of(acc)))))
    return NX_NOT_COMPUTED;

  contract_params p;
  memset(&p, 0, sizeof p);
  p.a = (const void *)(uintptr_t)a->address;
  p.b = (const void *)(uintptr_t)b->address;
  p.init = init ? (const void *)(uintptr_t)init->address : NULL;
  p.y = (void *)(uintptr_t)y->address;
  p.sa[0] = sbat[0], p.sa[1] = sm[0], p.sa[2] = sk[0];
  p.sb[0] = sbat[1], p.sb[1] = sn[0], p.sb[2] = sk[1];
  p.sy[0] = sbat[2], p.sy[1] = sm[1], p.sy[2] = sn[1];
  p.si[0] = sbat[3], p.si[1] = sm[2], p.si[2] = sn[2];
  p.batch = (int32_t)batch, p.m = (int32_t)m, p.n = (int32_t)n,
  p.k = (int32_t)k;
  p.a_dtype = a->dtype, p.b_dtype = b->dtype, p.y_dtype = y->dtype;
  p.init_dtype = init ? init->dtype : y->dtype;
  p.acc_dtype = in->acc;

  /* The kernel, by the operands' and the accumulator's dtypes, then m.
     float8 operands decode exactly to bfloat16 and sum on its matrix unit,
     whose sums the suite checks against the error bound: the float8 unit's
     are unchecked. */
  const int at = a->dtype;
  const int bf16_like_a = at == NX_BFLOAT16 || is_f8(at);
  const int bf16_like_b = b->dtype == NX_BFLOAT16 || is_f8(b->dtype);
  const int f16 = at == NX_FLOAT16 && b->dtype == NX_FLOAT16;
  const int kind =
      acc != NX_FLOAT32 && acc != NX_INT32 ? -1
      : bf16_like_a && bf16_like_b ? (acc == NX_FLOAT32 ? K_bf16 : -1)
      : f16 ? (acc == NX_FLOAT32 ? K_f16 : -1)
      : at == NX_INT8 && b->dtype == NX_INT8 && acc == NX_INT32 ? K_s8 : -1;
  const int t = kind < 0 ? -1 : wmma_tile(kind, batch, m, n);
  const int wmma_kind = t < 0 ? -1 : kind;
  int simt = -1;
  if (acc == NX_FLOAT32 || acc == NX_FLOAT64) {
    if (is_int(at) || is_int(b->dtype) || bytes_of(at) == 0 ||
        bytes_of(b->dtype) == 0)
      return NX_NOT_COMPUTED;
    simt = acc == NX_FLOAT32 ? ACC_f32 : ACC_f64;
  } else if (acc == NX_INT32 || acc == NX_UINT32 || acc == NX_INT64 ||
             acc == NX_UINT64) {
    if (!is_int(at) || !is_int(b->dtype)) return NX_NOT_COMPUTED;
    simt = ACC_i64;
  } else
    return NX_NOT_COMPUTED;

  int kernel, splits, values, threads;
  uint32_t mask = 0;
  uint64_t gx;
  size_t used = 0;
  const size_t len = out->len;
  int records = 0;
  int e = 0; /* the last append's answer: NX_OUT_OF_MEMORY passes on */
  /* Whether a's and b's free axes are contiguous rather than k, and their
     rows along the contiguous axis 16-byte vectors. */
  int fa = sm[0] == 1 && sk[0] != 1, fb = sn[0] == 1 && sk[1] != 1;
  int va = vectors(a->address, fa ? sm[0] : sk[0], fa ? sk[0] : sm[0],
                   sbat[0], bytes_of(at));
  int vb = vectors(b->address, fb ? sn[0] : sk[1], fb ? sk[1] : sn[0],
                   sbat[1], bytes_of(b->dtype));
  if (wmma_kind >= 0) {
    /* The WMMA kernels read rows of k as whole vectors and never past a
       row's last: another operand, or one whose k ends inside a vector,
       is packed into scratch with k contiguous, its rows padded with
       zeros. */
    const int mes = wmma_kind == K_s8 ? 1 : 2, whole = k % (16 / mes) == 0;
    const int pack_a = fa || !va || is_f8(at) || !whole;
    const int pack_b = fb || !vb || is_f8(b->dtype) || !whole;
    const int to = wmma_kind == K_s8   ? NX_INT8
                   : wmma_kind == K_f16 ? NX_FLOAT16
                                        : NX_BFLOAT16;
    if (pack_a) {
      if ((e = pack(out, &used, &p.a, p.sa, a->address, at, to, batch, m, k,
                    sbat[0], sm[0], sk[0])) != 0)
        goto fail;
      records++;
    }
    if (pack_b) {
      if ((e = pack(out, &used, &p.b, p.sb, b->address, b->dtype, to, batch,
                    n, k, sbat[1], sn[0], sk[1])) != 0)
        goto fail;
      records++;
    }
    if (pack_a) mask |= NX_CONTRACT_SCRATCH_A;
    if (pack_b) mask |= NX_CONTRACT_SCRATCH_B;
    const tile *tl = &tiles[t];
    kernel = find(F_WMMA, wmma_kind, t);
    gx = ceil_div(m, tl->bm) * ceil_div(n, tl->bn);
    /* A split sum stores and reloads its partials: worth it to fill a
       GPU short of workgroups, or to stream a long k for a few rows. */
    splits = m <= 16 ? split_count(gx * batch, 128, k, 4 * tl->bkb / mes)
                     : split_count(gx * batch, 64, k, 1024);
    values = tl->bm * tl->bn / tl->threads, threads = tl->threads;
  } else {
    /* SIMT and skinny kernels read their accumulator's own dtype: an
       operand of another is packed into it, exactly, with k contiguous. */
    const int to = own_dtype(simt);
    if (!own(simt, at)) {
      if ((e = pack(out, &used, &p.a, p.sa, a->address, at, to, batch, m, k,
                    sbat[0], sm[0], sk[0])) != 0)
        goto fail;
      records++, fa = 0, va = 1, p.a_dtype = to;
      mask |= NX_CONTRACT_SCRATCH_A;
      sm[0] = p.sa[1], sk[0] = 1;
    }
    if (!own(simt, b->dtype)) {
      if ((e = pack(out, &used, &p.b, p.sb, b->address, b->dtype, to, batch,
                    n, k, sbat[1], sn[0], sk[1])) != 0)
        goto fail;
      records++, fb = 0, vb = 1, p.b_dtype = to;
      mask |= NX_CONTRACT_SCRATCH_B;
      sn[0] = p.sb[1], sk[1] = 1;
    }
  }
  if (wmma_kind < 0 && m <= 16) {
    const int nform = sn[0] == 1 && sk[1] != 1;
    kernel = find(F_SKINNY, simt, nform ? S_across : S_column);
    gx = nform ? ceil_div(n, 32) : ceil_div(n, 8);
    /* By the wave-per-column form's grid in both forms: the split, so the
       association, is the shape's. */
    splits = split_count(ceil_div(n, 8) * batch, 256, k, 1024);
    values = 2, threads = NX_CONTRACT_THREADS;
    p.aligned = (!fb && vb ? NX_CONTRACT_B_VECTORS : 0) |
                (!fa && va ? NX_CONTRACT_A_VECTORS : 0);
  } else if (wmma_kind < 0) {
    /* The SIMT tile of least cost among the accumulator's instances, waves
       times outputs over efficiency: the 64-wide tile computing half as
       fast as the 128-wide one on the R9700 (39-56% at 2048 and 4096
       cubed, float32). */
    static const int sides[2] = {128, 64}, eff[2] = {100, 50};
    int side = 64;
    double least = -1;
    for (int i = 0; i < 2; i++) {
      const int64_t q = sides[i];
      if (find(F_SIMT, simt, sides[i]) < 0) continue;
      const double cost =
          (double)ceil_div(batch * ceil_div(m, q) * ceil_div(n, q), WAVE) * q *
          q / eff[i];
      if (least < 0 || cost < least) least = cost, side = sides[i];
    }
    kernel = find(F_SIMT, simt, side);
    gx = ceil_div(m, side) * ceil_div(n, side);
    splits = split_count(gx * batch, 64, k, 128);
    values = side * side / NX_CONTRACT_THREADS, threads = NX_CONTRACT_THREADS;
    p.aligned = (va ? NX_CONTRACT_A_VECTORS : 0) |
                (vb ? NX_CONTRACT_B_VECTORS : 0);
  }
  if (gx > INT32_MAX) goto fail;
  p.splits = splits;

  /* Outputs stored 16 bytes at once: no init, y's j contiguous, its rows
     on 16-byte boundaries, and its dtype one the kernels store as they
     are, or bfloat16 or float16 from float32 (store8). */
  const int yd = y->dtype, yb = bytes_of(yd);
  const int wide = yd == NX_INT64 || yd == NX_UINT64;
  const int natural =
      wmma_kind == K_s8 ? yd == NX_INT32 || yd == NX_UINT32
      : simt == ACC_f32 ? yd == NX_FLOAT32 || yd == NX_BFLOAT16 || yd == NX_FLOAT16
      : simt == ACC_f64 ? yd == NX_FLOAT64
                        : wide && (acc == NX_INT64 || acc == NX_UINT64);
  if (!init && natural && sn[1] == 1 && sm[1] * yb % 16 == 0 &&
      sbat[2] * yb % 16 == 0 && y->address % 16 == 0)
    p.aligned |= NX_CONTRACT_Y_WHOLE;

  /* The split sum's partials and tickets, the tickets zeroed first. */
  if (splits > 1) {
    const int acc_bytes =
        wmma_kind < 0 && (simt == ACC_f64 || simt == ACC_i64) ? 8 : 4;
    const uint64_t tickets = (uint64_t)batch * gx;
    p.partials = (void *)(uintptr_t)take(
        &used, (size_t)tickets * splits * values * threads * acc_bytes);
    p.tickets = (uint32_t *)(uintptr_t)take(&used, tickets * 4);
    zero_params z = {p.tickets, tickets};
    if ((e = add(out, NX_AMD_zero_u32,
                 (uint32_t)ceil_div(tickets, NX_CONTRACT_THREADS), 1, 1,
                 NX_CONTRACT_THREADS, &z, sizeof z, NX_ZERO_ADDRS,
                 NX_ZERO_SCRATCH)) != 0)
      goto fail;
    records++;
    mask |= NX_CONTRACT_SCRATCH_SPLIT;
  }
  if ((e = add(out, kernel, (uint32_t)gx, (uint32_t)splits, (uint32_t)batch,
               (uint32_t)threads, &p, sizeof p, NX_CONTRACT_ADDRS, mask)) != 0)
    goto fail;
  *scratch = used;
  return records + 1;

fail:
  out->len = len;
  *scratch = 0;
  return e == NX_OUT_OF_MEMORY ? NX_OUT_OF_MEMORY : NX_NOT_COMPUTED;
}
