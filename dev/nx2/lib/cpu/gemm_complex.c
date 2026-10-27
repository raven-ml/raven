/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A complex contraction is one in its parts' dtype, of twice the columns
   and twice the products (the 1m method, Van Zee, "Implementing
   high-performance complex matrix multiplication via the 1m method", SIAM
   J. Sci. Comput. 2020, its variant for row-stored c). a's K complex
   numbers along the contraction are 2K reals, real and imaginary parts in
   turn, read in place where its numbers lie adjacent along k, or along
   each run of k where it is several; b's element (k, j) = x + iy is
   copied as the block

     row 2k:      … x   y …
     row 2k + 1:  … −y  x …

   so that column 2j sums re·x then im·(−y) for each k, output j's real
   part, and column 2j + 1 re·y then im·x, its imaginary part. Each part is
   a float contraction of 2K terms, whose bits the real contraction keeps
   at any layout and thread count. Where dst's or init's numbers lie apart
   along the columns, each part is a contraction of its own over b's
   columns of that part. */

#include <string.h>

#include "gemm.h"

/* Rows 2k and 2k + 1 of the block of [n] complex numbers at [x], [sx]
   parts apart: (x, y) into [even] and (−y, x) into [odd]. Adjacent
   numbers take a loop of their own, which vectorises. [even] may be [x]
   at [sx] 2, as a staged row is: each number is read before its place is
   written. */
#define BLOCK_ROWS(T, name)                                                \
  static void name(const T *x, int64_t sx, T *even, T *restrict odd,       \
                   int64_t n) {                                            \
    if (sx == 2) {                                                         \
      for (int64_t j = 0; j < 2 * n; j += 2) {                             \
        T re = x[j], im = x[j + 1];                                        \
        even[j] = re, even[j + 1] = im, odd[j] = -im, odd[j + 1] = re;     \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    for (int64_t j = 0; j < n; j++) {                                      \
      T re = x[j * sx], im = x[j * sx + 1];                                \
      even[2 * j] = re, even[2 * j + 1] = im;                              \
      odd[2 * j] = -im, odd[2 * j + 1] = re;                               \
    }                                                                      \
  }
BLOCK_ROWS(float, block_rows_f32)
BLOCK_ROWS(double, block_rows_f64)

/* The block's rows of parts [w] bytes wide. */
static void block_rows_of(int w, const uint8_t *x, int64_t sx,
                          uint8_t *even, uint8_t *odd, int64_t n) {
  if (w == 4)
    block_rows_f32((const float *)x, sx, (float *)even, (float *)odd, n);
  else
    block_rows_f64((const double *)x, sx, (double *)even, (double *)odd, n);
}

/* b's blocks: batch element e's rows 2k and 2k + 1 at
   d + (e·2K + 2k)·2N·w, w the parts' width. A unit is a row k of b of
   the complex problem [p], read piece by piece along its runs: in place
   where b is acc, staged into the odd row first otherwise. */
typedef struct {
  const problem *p;
  int w;
  uint8_t *d;
} blocks_job;

static void blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const blocks_job *j = ctx;
  const problem *p = j->p;
  const operand *q = &p->op[B];
  int64_t n = p->ext[COL], k = p->ext[CON], row = 2 * n * j->w;
  int w = j->w, staged = q->x->dtype != p->acc;
  for (int64_t u = lo; u < hi; u++) {
    uint8_t *even = j->d + 2 * u * row, *odd = even + row;
    for (int64_t j0 = 0, c; j0 < n; j0 += c) {
      int64_t st;
      c = piece(q, COL, j0, n - j0, &st);
      int64_t at = pos(p, B, u / k, 0, j0, u % k);
      uint8_t *e0 = even + 2 * j0 * w, *o0 = odd + 2 * j0 * w;
      if (staged) nx_cpu_stage_as(q->x, at, st, 0, c, 1, p->acc, o0, 0);
      else block_rows_of(w, q->x->base + at * 2 * w, 2 * st, e0, o0, c);
    }
    if (!staged) continue;
    memcpy(even, odd, (size_t)row);
    block_rows_of(w, even, 2, even, odd, n);
  }
}

/* [x]'s numbers read as their parts, of the dtype [r]. */
static nx_array as_parts(const nx_array *x, int r) {
  nx_array y = *x;
  y.dtype = r;
  y.bits = x->bits / 2;
  return y;
}

/* Operand [o] of the complex problem [p], rows along the axis [x1] of
   numbers along [x0], as [u]'s operand [y] of parts: in place, each step
   doubled, or under [copy] copied into acc in scratch (nx_cpu_take) as
   nx_gemm_pack_at reads it, each batch element's rows C-contiguous, one element
   where [o] steps 0 along the batch. Answers the copy, NULL in place or
   if allocation fails. */
static uint8_t *parts_of(const problem *p, int o, int x0, int x1, int copy,
                         nx_contract_view *u, nx_array *y) {
  const operand *q = &p->op[o];
  int r = part_of(p->acc), w = nx_cpu_width(r);
  int64_t n0 = p->ext[x0], n1 = p->ext[x1];
  if (!copy) {
    *y = as_parts(q->x, r);
    u->offset[o] = 2 * q->first;
    for (int x = BATCH; x <= CON; x++) u->stride[o][x] = 2 * q->st[x];
    return NULL;
  }
  int64_t ne = q->st[BATCH] == 0 ? 1 : p->ext[BATCH];
  uint8_t *d = nx_cpu_take(ne * n1 * n0 * 2 * w);
  if (d == NULL) return NULL;
  for (int64_t e = 0; e < ne; e++) {
    const int64_t c[4] = {e, 0, 0, 0};
    nx_gemm_pack_at(p, NULL, o, c, x0, x1, n0, n1, d + e * n1 * n0 * 2 * w, n0);
  }
  *y = (nx_array){.base = d, .dtype = r, .bits = 8 * w};
  u->offset[o] = 0;
  u->stride[o][BATCH] = q->st[BATCH] == 0 ? 0 : 2 * n1 * n0;
  u->stride[o][x1] = 2 * n0;
  u->stride[o][x0] = 2;
  return d;
}

/* The spans [s] of a, read in place, as parts: each step doubled, and
   along k, where its last run steps one number, that run twice as long
   and stepping one part. */
static void parts_spans(const operand *a, spans *s) {
  for (int x = ROW; x <= CON; x++) {
    if (a->sp[x] == NULL) continue;
    s[x] = *a->sp[x];
    for (int t = 0; t < s[x].n; t++) s[x].stride[t] *= 2;
  }
  if (a->sp[CON] == NULL) return;
  s[CON].extent[s[CON].n - 1] *= 2;
  s[CON].stride[s[CON].n - 1] = 1;
}

/* The complex contraction of [v] over [op] in [acc], a and b read along
   their spans [sp] where it is not NULL. */
int nx_gemm_complex_on(int acc, int order, const nx_contract_view *v,
                       const nx_array *const *op, const reads *sp) {
  int r = part_of(acc), w = nx_cpu_width(r);
  int64_t batch = v->extent[BATCH], m = v->extent[ROW], n = v->extent[COL];
  int64_t k = v->extent[CON];
  if (batch == 0 || m == 0 || n == 0) return 1;
  problem p = {.acc = acc, .w = nx_cpu_width(acc)};
  nx_gemm_lay_out(&p, v, op);
  for (int o = A; sp && o <= B; o++)
    for (int x = ROW; x <= CON; x++)
      if (sp->of[o][x].n > 1) p.op[o].sp[x] = &sp->of[o][x];
  nx_contract_view u = {.extent = {batch, m, 2 * n, 2 * k}};
  reads ps = {0};
  nx_array x[4];
  const nx_array *ops[4] = {&x[A], &x[B], op[INIT] ? &x[INIT] : NULL, &x[DST]};
  int copy_a = op[A]->dtype != acc || (step_of(&p.op[A], CON) != 1 && k > 1);
  int64_t eb = v->stride[B][BATCH] == 0 ? 1 : batch;
  uint8_t *ab = parts_of(&p, A, CON, ROW, copy_a, &u, &x[A]);
  if (!copy_a) parts_spans(&p.op[A], ps.of[A]);
  uint8_t *bb = nx_cpu_take(eb * 4 * k * n * w), *ib = NULL;
  /* An init of another dtype than acc, a result's of another dtype, is
     copied into acc's parts. */
  int copy_i = op[INIT] && op[INIT]->dtype != acc;
  if (op[INIT])
    ib = parts_of(&p, INIT, COL, ROW, copy_i, &u, &x[INIT]);
  parts_of(&p, DST, COL, ROW, 0, &u, &x[DST]);
  int done = 0;
  if ((copy_a && ab == NULL) || (copy_i && ib == NULL) || bb == NULL)
    goto out;
  u.stride[A][CON] = 1;
  blocks_job j = {.p = &p, .w = w, .d = bb};
  /* Each number of b read, two parts, and its block written, four. */
  int64_t bytes = eb * k * n * 6 * w;
  nx_cpu_job(eb * k, bytes, bytes, blocks, &j);
  x[B] = (nx_array){.base = bb, .dtype = r, .bits = 8 * w};
  u.stride[B][BATCH] = eb == 1 ? 0 : 4 * k * n;
  u.stride[B][CON] = 2 * n;
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[r];
  /* Each part's columns, in turn where dst's or init's numbers lie apart. */
  int apart = n > 1 && (u.stride[DST][COL] != 2 ||
                        (op[INIT] && u.stride[INIT][COL] != 2));
  if (!apart) {
    u.stride[B][COL] = 1;
    u.stride[DST][COL] = 1;
    if (op[INIT]) u.stride[INIT][COL] = 1;
    done = nx_gemm_contract_with(g, r, order, &u, ops, NULL, &ps);
    goto out;
  }
  u.extent[COL] = n;
  u.stride[B][COL] = 2;
  done = 1;
  for (int part = 0; part < 2 && done; part++) {
    nx_contract_view up = u;
    for (int o = B; o <= DST; o++) up.offset[o] += part;
    done = nx_gemm_contract_with(g, r, order, &up, ops, NULL, &ps);
  }
out:
  nx_cpu_give(ab);
  nx_cpu_give(ib);
  nx_cpu_give(bb);
  return done;
}
