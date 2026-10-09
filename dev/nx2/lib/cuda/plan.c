/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A call's launches, from its dtypes, shapes, attributes and the GPU's
   architecture: never from the GPU's size, clocks or timing, so that a
   result's bits depend on its shape alone.

   A plan declines a call, and nx's expansion computes it, when an operand,
   init or y is complex or narrower than a byte; a float operand or init
   is wider than a float accumulator, which would round it before the sum;
   init is of the other kind than the accumulator; an integer meets a float
   accumulator or the reverse; the batch, free or contracting axes do not
   merge into one axis each; the batch is above 65,535 (the grid's z); or
   m, n or k is above 2^31 - 1. */

#include "nx_array.h"
#include "nx_cuda.h"

/* The architectures this library has a cubin for. */
#define ARCH_SM_89 89

/* Records */

static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

/* Appends the launch of [kernel] over [grid] with [params]: 0, or
   nx_cuda_add's failure. */
static int add(nx_cuda_records *out, int kernel, uint32_t gx, uint32_t gy,
               uint32_t gz, uint32_t threads, uint32_t shared,
               const void *params, uint32_t bytes, uint32_t addrs,
               uint32_t scratch) {
  const uint32_t grid[3] = {gx, gy, gz}, block[3] = {threads, 1, 1};
  return nx_cuda_add(out, (uint32_t)kernel, grid, block, shared, params,
                     bytes, addrs, scratch);
}

/* Contract */

/* The kernels by family and instance, from kernels.h's list, a table a
   family. The mma kinds are bits: an instance of kind any sums each. */
enum { K_bf16 = 1, K_f16 = 2, K_s8 = 4, K_any = 7 };
enum { A_k, A_m, A_n };
enum { ACC_f32, ACC_f64, ACC_i64 };
#define TILE_INDEX(name, ...) T_##name,
enum { NX_CUDA_TILES(TILE_INDEX) T_COUNT };
#undef TILE_INDEX

/* An mma instance by the kinds it sums, a's and b's contiguous axes and
   its tile; a SIMT one by its accumulator and side; a skinny one by its
   accumulator. */
typedef struct {
  int kernel, kinds, a, b, tile;
} mma_instance;
typedef struct {
  int kernel, sum, side;
} simt_instance;
typedef struct {
  int kernel, sum;
} skinny_instance;

#define NONE(...)
#define MMA_MMA(name, k, la, lb, t) \
  {NX_CUDA_##name, K_##k, A_##la, A_##lb, T_##t},
#define SIMT_SIMT(name, acc, side) {NX_CUDA_##name, ACC_##acc, side},
#define SKINNY_SKINNY(name, acc) {NX_CUDA_##name, ACC_##acc},
#define MMA_ZERO NONE
#define MMA_PACK NONE
#define MMA_SIMT NONE
#define MMA_SKINNY NONE
#define SIMT_ZERO NONE
#define SIMT_PACK NONE
#define SIMT_MMA NONE
#define SIMT_SKINNY NONE
#define SKINNY_ZERO NONE
#define SKINNY_PACK NONE
#define SKINNY_MMA NONE
#define SKINNY_SIMT NONE
#define MMA_ROW(name, FAMILY, ...) MMA_##FAMILY(name, __VA_ARGS__)
#define SIMT_ROW(name, FAMILY, ...) SIMT_##FAMILY(name, __VA_ARGS__)
#define SKINNY_ROW(name, FAMILY, ...) SKINNY_##FAMILY(name, __VA_ARGS__)
static const mma_instance mmas[] = {NX_CUDA_KERNELS(MMA_ROW)};
static const simt_instance simts[] = {NX_CUDA_KERNELS(SIMT_ROW)};
static const skinny_instance skinnies[] = {NX_CUDA_KERNELS(SKINNY_ROW)};
#define COUNT(xs) ((int)(sizeof xs / sizeof xs[0]))

/* The kernel of the first instance with the arguments given, or -1: an mma
   one that sums [kind]. */
static int find_mma(int kind, int a, int b, int t) {
  for (int i = 0; i < COUNT(mmas); i++)
    if ((mmas[i].kinds & kind) && mmas[i].a == a && mmas[i].b == b &&
        mmas[i].tile == t)
      return mmas[i].kernel;
  return -1;
}

static int find_simt(int sum, int side) {
  for (int i = 0; i < COUNT(simts); i++)
    if (simts[i].sum == sum && simts[i].side == side) return simts[i].kernel;
  return -1;
}

static int find_skinny(int sum) {
  for (int i = 0; i < COUNT(skinnies); i++)
    if (skinnies[i].sum == sum) return skinnies[i].kernel;
  return -1;
}

typedef struct {
  int bm, bn, bkb, threads, shared;
} tile;

#define TILE_ROW(name, bm, bn, bkb, wm, wn, s) \
  {bm, bn, bkb, (bm / wm) * (bn / wn) * 32, s * (bm + bn) * bkb},
static const tile tiles[] = {NX_CUDA_TILES(TILE_ROW)};
#undef TILE_ROW

/* What a tile of [bm] x [bn] outputs costs over [batch] x [m] x [n]
   outputs on sm_89: blocks run in waves of [WAVE], the GPU's 100 SMs, and
   a wave lasts the tile's outputs over its [efficiency], its outputs per
   unit of time relative to its family's fastest tile, in percent. Measured
   on kimchi's RTX 5000 Ada; a GPU of another size runs the same tiles, with
   the same bits. */
#define WAVE 100
static double cost(int64_t batch, int64_t m, int64_t n, int64_t bm,
                   int64_t bn, int efficiency) {
  const uint64_t blocks = batch * ceil_div(m, bm) * ceil_div(n, bn);
  return (double)ceil_div(blocks, WAVE) * bm * bn / efficiency;
}

/* The mma tiles' efficiency, against the 128 x 256 tile's. */
static const int efficiency[T_COUNT] = {[T_t128x128] = 90, [T_t128x256] = 100,
                                        [T_t64x64] = 60, [T_t16x64] = 30};

/* The mma tile of a product among those [kind] has an instance of with k
   contiguous: the one of least cost, by its shape alone; m <= 16 takes the
   16-row tile, which larger m take where it costs least. -1 if [kind] has
   none for the shape: the SIMT or skinny kernels sum it. */
static int mma_tile(int kind, int64_t batch, int64_t m, int64_t n) {
  int best = -1;
  double least = -1;
  for (int t = 0; t < T_COUNT; t++) {
    if (find_mma(kind, A_k, A_k, t) < 0 || (m <= 16 && t != T_t16x64))
      continue;
    const double c =
        cost(batch, m, n, tiles[t].bm, tiles[t].bn, efficiency[t]);
    if (least < 0 || c < least) least = c, best = t;
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
static int merge(int n, const nx_cuda_operand *const *ops,
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

/* Whether an operand of the strides [s] (batch, row, k) has its rows
   contiguous rather than its k. */
static int free_contiguous(const int64_t s[3]) { return s[1] == 1 && s[2] != 1; }

/* Whether the rows of the operand at [x], of the strides [s] and dtype [dt],
   load as 16-byte vectors along its contiguous axis. */
static int rows_vectors(const void *x, const int64_t s[3], int dt) {
  const int f = free_contiguous(s);
  return vectors((uint64_t)(uintptr_t)x, f ? s[1] : s[2], f ? s[2] : s[1],
                 s[0], bytes_of(dt));
}

/* [bytes] of the call's scratch after the [*used] bytes taken: their
   offset, each piece on a 256-byte boundary. */
static size_t take(size_t *used, size_t bytes) {
  size_t at = (*used + 255) & ~(size_t)255;
  *used = at + bytes;
  return at;
}

/* A plan under way: the contraction's parameters, the records it appends
   to and their count, its scratch so far and the mask of the parameters'
   addresses that are scratch. */
typedef struct {
  contract_params p;
  nx_cuda_records *out;
  int records;
  size_t used;
  uint32_t mask;
} plan;

/* Appends the launch that packs b if [side], else a, into scratch as
   elements of [to] with k contiguous and rows of whole vectors, and makes
   the parameters address the copy. 0, or add's failure. */
static int pack(plan *c, int side, int to) {
  contract_params *p = &c->p;
  const void **x = side ? &p->b : &p->a;
  int64_t *s = side ? p->sb : p->sa;
  const int64_t batch = p->batch, rows = side ? p->n : p->m, k = p->k;
  const int es = bytes_of(to);
  const int64_t per = 16 / es, lead = (k + per - 1) / per * per;
  const size_t at = take(&c->used, (size_t)(batch * rows * lead * es));
  pack_params q = {*x, (void *)(uintptr_t)at, {s[0], s[1], s[2]}, lead,
                   (int32_t)batch, (int32_t)rows, (int32_t)k,
                   side ? p->b_dtype : p->a_dtype, to, es};
  /* A thread a 16-byte vector. */
  const uint64_t n = (uint64_t)(batch * rows * lead) / per;
  const uint32_t grid = (uint32_t)(ceil_div(n, 256) < 65535 ? ceil_div(n, 256) : 65535);
  *x = (const void *)(uintptr_t)at;
  s[0] = rows * lead, s[1] = lead, s[2] = 1;
  c->records++;
  c->mask |= side ? NX_CONTRACT_SCRATCH_B : NX_CONTRACT_SCRATCH_A;
  return add(c->out, NX_CUDA_pack, grid ? grid : 1, 1, 1, 256, 0, &q, sizeof q,
             NX_PACK_ADDRS, NX_PACK_SCRATCH);
}

/* The dtype of the accumulator [sum]: what the SIMT kernels read, and the
   float skinny kernels. */
static int own_dtype(int sum) {
  switch (sum) {
  case ACC_f32: return NX_FLOAT32;
  case ACC_f64: return NX_FLOAT64;
  }
  return NX_INT64;
}

/* The integer dtype the skinny kernel reads integer operands [x] and [y]
   in: the narrowest that holds both exactly, or 64 bits, whose products
   wrap as the accumulator's sum does. */
static int common_int(int x, int y) {
  static const int types[2][4] = {{NX_UINT8, NX_UINT16, NX_UINT32, NX_INT64},
                                  {NX_INT8, NX_INT16, NX_INT32, NX_INT64}};
  const nx_dtype_row rx = nx_dtype_row_of(x), ry = nx_dtype_row_of(y);
  const int sx = rx.kind == NX_KIND_SIGNED, sy = ry.kind == NX_KIND_SIGNED;
  int bits = rx.bits > ry.bits ? rx.bits : ry.bits;
  /* A signed type holds an unsigned one's values with a bit to spare. */
  if (sx != sy && 2 * (sx ? ry.bits : rx.bits) > bits)
    bits = 2 * (sx ? ry.bits : rx.bits);
  return types[sx || sy][bits <= 8 ? 0 : bits <= 16 ? 1 : bits <= 32 ? 2 : 3];
}

/* Whether a SIMT or skinny kernel reads [dt] as [to] with no pack: the same
   bytes, bool as uint8, and either 64-bit integer as the other (a float64
   never meets an integer [to]: the plan declines that call). */
static int reads_as(int dt, int to) {
  return dt == to || (dt == NX_BOOL && to == NX_UINT8) ||
         (bytes_of(dt) == 8 && bytes_of(to) == 8);
}

/* Packs each operand the SIMT or skinny kernel cannot read as [to] into
   [to], exactly, with k contiguous: the kernel then reads both as [to].
   0, or add's failure. */
static int pack_into(plan *c, int to) {
  int e;
  if (!reads_as(c->p.a_dtype, to) && (e = pack(c, 0, to)) != 0) return e;
  if (!reads_as(c->p.b_dtype, to) && (e = pack(c, 1, to)) != 0) return e;
  c->p.a_dtype = c->p.b_dtype = to;
  return 0;
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

/* Whether the summable dtype [dt] is an integer or bool. */
static int is_int(int dt) {
  const enum nx_kind k = nx_dtype_row_of(dt).kind;
  return k == NX_KIND_SIGNED || k == NX_KIND_UNSIGNED || k == NX_KIND_BOOLEAN;
}

/* The split count of a grid of [grid] blocks: doubled while the grid has
   fewer than [target] blocks and each range keeps at least [k_min] of k,
   at most 16. The targets are constants measured on an architecture's
   GPUs, never a GPU's own count. */
static int split_count(uint64_t grid, uint64_t target, int64_t k,
                       int64_t k_min) {
  int s = 1;
  while (s < 16 && grid * s < target && k / (2 * s) >= k_min) s *= 2;
  return s;
}

/* Families */

/* The contraction kernel's launch: its [blocks] x [splits] x batch blocks
   of [threads] threads and [shared] dynamic shared bytes, and the [sums]
   partial sums a block holds when it splits, [sum_bytes] each. */
typedef struct {
  int kernel, splits, threads, shared, sum_bytes;
  uint64_t blocks, sums;
} launch;

/* Each family's launch, and the packs it takes, appended to [c]; answers
   0, NX_NOT_COMPUTED having appended nothing, or add's failure. */

/* The mma kernels, of [kind] on the tile [t]. An operand whose rows are
   not vectors is packed into scratch with k contiguous, as is a float8 one
   and one in a layout the tile has no instance of. */
static int plan_mma(plan *c, int kind, int t, launch *l) {
  const contract_params *p = &c->p;
  const tile *tl = &tiles[t];
  const int la = free_contiguous(p->sa) ? A_m : A_k;
  const int lb = free_contiguous(p->sb) ? A_n : A_k;
  const int pack_a = !rows_vectors(p->a, p->sa, p->a_dtype) ||
                     is_f8(p->a_dtype) ||
                     (la != A_k && find_mma(kind, la, A_k, t) < 0);
  const int pack_b =
      !rows_vectors(p->b, p->sb, p->b_dtype) || is_f8(p->b_dtype) ||
      (lb != A_k && find_mma(kind, pack_a ? A_k : la, lb, t) < 0);
  l->kernel = find_mma(kind, pack_a ? A_k : la, pack_b ? A_k : lb, t);
  l->blocks = ceil_div(p->m, tl->bm) * ceil_div(p->n, tl->bn);
  if (l->kernel < 0 || l->blocks > INT32_MAX) return NX_NOT_COMPUTED;
  const int to = kind == K_s8 ? NX_INT8 : kind == K_f16 ? NX_FLOAT16 : NX_BFLOAT16;
  int e;
  if (pack_a && (e = pack(c, 0, to)) != 0) return e;
  if (pack_b && (e = pack(c, 1, to)) != 0) return e;
  c->p.a_dtype = c->p.b_dtype = to;
  /* A split sum stores and reloads its partials: worth it to fill a GPU
     short of blocks, or to stream a long k for a few rows. */
  const int mes = kind == K_s8 ? 1 : 2;
  l->splits = p->m <= 16
                  ? split_count(l->blocks * p->batch, 256, p->k, 4 * tl->bkb / mes)
                  : split_count(l->blocks * p->batch, 64, p->k, 1024);
  l->threads = tl->threads, l->shared = tl->shared;
  l->sums = (uint64_t)tl->bm * tl->bn, l->sum_bytes = 4;
  return 0;
}

/* The SIMT kernels of the accumulator [sum], m > 16: the tile of least
   cost among its instances; on sm_89, the 64-wide tile computes 70% as fast
   as the 128-wide one. They read their accumulator's own dtype. */
static int plan_simt(plan *c, int sum, launch *l) {
  contract_params *p = &c->p;
  static const int sides[2] = {128, 64}, eff[2] = {100, 70};
  int side = 64;
  double least = -1;
  for (int i = 0; i < 2; i++) {
    if (find_simt(sum, sides[i]) < 0) continue;
    const double x = cost(p->batch, p->m, p->n, sides[i], sides[i], eff[i]);
    if (least < 0 || x < least) least = x, side = sides[i];
  }
  l->kernel = find_simt(sum, side);
  l->blocks = ceil_div(p->m, side) * ceil_div(p->n, side);
  if (l->kernel < 0 || l->blocks > INT32_MAX) return NX_NOT_COMPUTED;
  const int e = pack_into(c, own_dtype(sum));
  if (e != 0) return e;
  /* A SIMT block's k-tiles run one after another, each waiting on its
     loads: split while the grid has fewer than 256 blocks, down to 64 of k
     a range. */
  l->splits = split_count(l->blocks * p->batch, 256, p->k, 64);
  l->threads = 256, l->shared = 0;
  l->sums = (uint64_t)side * side, l->sum_bytes = sum == ACC_f32 ? 4 : 8;
  p->aligned = (rows_vectors(p->a, p->sa, p->a_dtype) ? NX_CONTRACT_A_VECTORS : 0) |
               (rows_vectors(p->b, p->sb, p->b_dtype) ? NX_CONTRACT_B_VECTORS : 0);
  return 0;
}

/* The skinny kernel of the accumulator [sum], m <= 16. It reads a float
   accumulator's own dtype, or one integer dtype both operands hold. */
static int plan_skinny(plan *c, int sum, launch *l) {
  contract_params *p = &c->p;
  l->kernel = find_skinny(sum);
  l->blocks = ceil_div(p->m, NX_SKINNY_ROWS) * ceil_div(p->n, 32);
  if (l->kernel < 0 || l->blocks > INT32_MAX) return NX_NOT_COMPUTED;
  const int to = sum == ACC_i64 ? common_int(p->a_dtype, p->b_dtype)
                                : own_dtype(sum);
  const int e = pack_into(c, to);
  if (e != 0) return e;
  /* By the columns alone: a row's sums are the same bits whatever rows come
     with it. */
  l->splits = split_count(ceil_div(p->n, 32) * p->batch, 64, p->k, 1024);
  /* A 32-bit integer accumulator sums in 32 bits here. */
  const int acc = p->acc_dtype, narrow = acc == NX_INT32 || acc == NX_UINT32;
  l->threads = 256, l->shared = 0;
  l->sums = 256, l->sum_bytes = sum == ACC_f32 || narrow ? 4 : 8;
  const int across = free_contiguous(p->sb);
  p->aligned =
      (across ? NX_CONTRACT_B_ACROSS : 0) |
      (!across && rows_vectors(p->b, p->sb, p->b_dtype) ? NX_CONTRACT_B_VECTORS : 0) |
      (!free_contiguous(p->sa) && rows_vectors(p->a, p->sa, p->a_dtype)
           ? NX_CONTRACT_A_VECTORS
           : 0);
  return 0;
}

int nx_cuda_plan_contract(const nx_cuda_contract_in *in,
                          const nx_cuda_operand *ops, int arch,
                          nx_cuda_records *out, size_t *scratch) {
  const nx_cuda_operand *a = &ops[0], *b = &ops[1];
  const nx_cuda_operand *init = in->init ? &ops[2] : NULL;
  const nx_cuda_operand *y = &ops[in->init ? 3 : 2];
  *scratch = 0;
  if (arch != ARCH_SM_89) return NX_NOT_COMPUTED;

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

  const nx_cuda_operand *gb[4] = {a, b, y, init ? init : y};
  const nx_cuda_operand *gm[3] = {a, y, init ? init : y};
  const nx_cuda_operand *gn[3] = {b, y, init ? init : y};
  const nx_cuda_operand *gk[2] = {a, b};
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

  plan c = {.out = out};
  contract_params *p = &c.p;
  p->a = (const void *)(uintptr_t)a->address;
  p->b = (const void *)(uintptr_t)b->address;
  p->init = init ? (const void *)(uintptr_t)init->address : NULL;
  p->y = (void *)(uintptr_t)y->address;
  p->sa[0] = sbat[0], p->sa[1] = sm[0], p->sa[2] = sk[0];
  p->sb[0] = sbat[1], p->sb[1] = sn[0], p->sb[2] = sk[1];
  p->sy[0] = sbat[2], p->sy[1] = sm[1], p->sy[2] = sn[1];
  p->si[0] = sbat[3], p->si[1] = sm[2], p->si[2] = sn[2];
  p->batch = (int32_t)batch, p->m = (int32_t)m, p->n = (int32_t)n,
  p->k = (int32_t)k;
  p->a_dtype = a->dtype, p->b_dtype = b->dtype, p->y_dtype = y->dtype;
  p->init_dtype = init ? init->dtype : y->dtype;
  p->acc_dtype = in->acc;

  /* The family, by the operands' and the accumulator's dtypes, then the
     shape. float8 operands decode exactly to bfloat16 and sum on its matrix
     unit: Ada's float8 unit keeps 13 bits of its sums, where the error
     bound needs each addition to err by at most 2u. */
  const int at = a->dtype, bt = b->dtype;
  const int bf16_like = (at == NX_BFLOAT16 || is_f8(at)) &&
                        (bt == NX_BFLOAT16 || is_f8(bt));
  const int kind =
      acc != NX_FLOAT32 && acc != NX_INT32 ? -1
      : bf16_like && acc == NX_FLOAT32 ? K_bf16
      : at == NX_FLOAT16 && bt == NX_FLOAT16 && acc == NX_FLOAT32 ? K_f16
      : at == NX_INT8 && bt == NX_INT8 && acc == NX_INT32 ? K_s8 : -1;
  const int t = kind < 0 ? -1 : mma_tile(kind, batch, m, n);
  int sum;
  if (float_acc) {
    if (is_int(at) || is_int(bt)) return NX_NOT_COMPUTED;
    sum = acc == NX_FLOAT32 ? ACC_f32 : ACC_f64;
  } else if (acc == NX_INT32 || acc == NX_UINT32 || acc == NX_INT64 ||
             acc == NX_UINT64) {
    if (!is_int(at) || !is_int(bt)) return NX_NOT_COMPUTED;
    sum = ACC_i64;
  } else
    return NX_NOT_COMPUTED;

  const size_t len = out->len;
  launch l;
  int e = t >= 0     ? plan_mma(&c, kind, t, &l)
          : m <= 16 ? plan_skinny(&c, sum, &l)
                    : plan_simt(&c, sum, &l);
  if (e != 0) goto fail;
  p->splits = l.splits;

  /* Outputs stored 16 bytes at once: no init, y's j contiguous, its rows
     on 16-byte boundaries, and its dtype one the kernels store as they
     are, or bfloat16 or float16 from float32 (store8, store_pair). */
  const int yd = y->dtype, yb = bytes_of(yd);
  const int wide = yd == NX_INT64 || yd == NX_UINT64;
  const int natural =
      t >= 0 && kind == K_s8 ? yd == NX_INT32 || yd == NX_UINT32
      : sum == ACC_f32 ? yd == NX_FLOAT32 || yd == NX_BFLOAT16 || yd == NX_FLOAT16
      : sum == ACC_f64 ? yd == NX_FLOAT64
                       : wide && (acc == NX_INT64 || acc == NX_UINT64);
  if (!init && natural && p->sy[2] == 1 && p->sy[1] * yb % 16 == 0 &&
      p->sy[0] * yb % 16 == 0 && y->address % 16 == 0)
    p->aligned |= NX_CONTRACT_Y_WHOLE;

  /* The split sum's partials and tickets, the tickets zeroed first. */
  if (l.splits > 1) {
    const uint64_t tickets = (uint64_t)batch * l.blocks;
    p->partials = (void *)(uintptr_t)take(
        &c.used, (size_t)tickets * l.splits * l.sums * l.sum_bytes);
    p->tickets = (uint32_t *)(uintptr_t)take(&c.used, tickets * 4);
    zero_params z = {p->tickets, tickets};
    if ((e = add(out, NX_CUDA_zero_u32, (uint32_t)ceil_div(tickets, 256), 1,
                 1, 256, 0, &z, sizeof z, NX_ZERO_ADDRS, NX_ZERO_SCRATCH)) != 0)
      goto fail;
    c.records++;
    c.mask |= NX_CONTRACT_SCRATCH_SPLIT;
  }
  if ((e = add(out, l.kernel, (uint32_t)l.blocks, (uint32_t)l.splits,
               (uint32_t)batch, (uint32_t)l.threads, (uint32_t)l.shared, p,
               sizeof *p, NX_CONTRACT_ADDRS, c.mask)) != 0)
    goto fail;
  *scratch = c.used;
  return c.records + 1;

fail:
  out->len = len;
  *scratch = 0;
  return e == NX_OUT_OF_MEMORY ? NX_OUT_OF_MEMORY : NX_NOT_COMPUTED;
}
