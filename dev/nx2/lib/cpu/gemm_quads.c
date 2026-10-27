/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* An 8-bit a by a b of the same dtype into a 32-bit acc runs on the
   table's quads kernel where it has one: four consecutive elements along
   k are one 32-bit word, a quad, and the driver contracts words of K/4
   products each, which the kernel adds four products at a time. A word of
   a holds a row's four elements, a word of b a column's. Each operand is
   read in place where it lies adjacent along k with each word on a word's
   boundary, and copied into words otherwise, k padded with zeros to a
   multiple of 4: a as rows of words, b as rows of each quad's words. The
   sum is exact modulo 2^32, as acc's: an integer's bits do not depend on
   how its products are grouped. */

#include <stdlib.h>
#include <string.h>

#include "gemm.h"

/* b's words, copied: column i's word q at d + (i + q·pitch)·4. A unit is
   a column of a batch element where b is adjacent along k, else a quad of
   each of its columns, so that the reads run along the memory. */
typedef struct {
  const nx_array *b;
  const nx_contract_view *v;
  int64_t quads, pitch;
  uint8_t *d;
} quads_job;

/* Four rows of [n] bytes, each adjacent, interleaved into words: row t's
   byte i to d[4i + t]. */
static void interleave(const uint8_t *r0, const uint8_t *r1,
                       const uint8_t *r2, const uint8_t *r3, uint8_t *d,
                       int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    d[4 * i] = r0[i];
    d[4 * i + 1] = r1[i];
    d[4 * i + 2] = r2[i];
    d[4 * i + 3] = r3[i];
  }
}

/* A full quad of b's rows, each adjacent, interleaves; any other element
   moves alone. */
static void quad_words(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const quads_job *j = ctx;
  const int64_t *s = j->v->stride[B];
  int64_t n = j->v->extent[COL], k = j->v->extent[CON];
  int by_column = s[CON] == 1;
  int64_t per = by_column ? n : j->quads;
  for (int64_t u = lo; u < hi; u++) {
    int64_t e = u / per, i0 = 0, i1 = n, p0 = 0, p1 = 4 * j->quads;
    if (by_column) i0 = u % per, i1 = i0 + 1;
    else p0 = 4 * (u % per), p1 = p0 + 4;
    const uint8_t *x = j->b->base + j->v->offset[B] + e * s[BATCH];
    uint8_t *d = j->d + e * j->pitch * j->quads * 4;
    if (!by_column && s[COL] == 1 && p1 <= k) {
      const uint8_t *r = x + p0 * s[CON];
      interleave(r, r + s[CON], r + 2 * s[CON], r + 3 * s[CON],
                 d + p0 * j->pitch, n);
      continue;
    }
    for (int64_t i = i0; i < i1; i++)
      for (int64_t p = p0; p < p1; p++)
        d[(i + p / 4 * j->pitch) * 4 + p % 4] =
            p < k ? x[p * s[CON] + i * s[COL]] : 0;
  }
}

/* Whether operand [o] of [v] is read in place as words: adjacent along k,
   each word's first element on a word's boundary. */
static int words_in_place(const nx_contract_view *v, int o) {
  const int64_t *s = v->stride[o];
  int64_t k = v->extent[CON];
  if (s[CON] != 1 && k > 1) return 0;
  if (k % 4 != 0 || v->offset[o] % 4 != 0) return 0;
  for (int x = BATCH; x < CON; x++)
    if (s[x] % 4 != 0) return 0;
  return 1;
}

/* The words between the rows of a copy into words, words_of's or
   zip_rows': a row's [words], or a line more where that is a multiple of
   4 KiB. Rows a multiple of 4 KiB apart share their low 12 address bits,
   and with them the L1 sets of a sliver's steps. On one fudge core,
   contract-i8-i32-1024, whose b is copied into rows of 1024 words, took
   3.77 ms, and 3.68 with its rows a line further apart; the kernel alone,
   in cache, ran 273 G products a second, and 294. */
static int64_t pitch_of(int64_t words) {
  return words > 0 && words * 4 % 4096 == 0 ? words + 16 : words;
}

/* Operand [o] of [v] as [u]'s operand [y] of words of [acc]: in place, or
   under [copy] copied into memory of its own, a's rows of words and b's as
   quads_job lays them out, rows pitch_of apart, one batch element where
   [o] steps 0 along the batch. Answers the copy, NULL in place or if
   allocation fails. */
static uint8_t *words_of(const nx_contract_view *v, const nx_array *const *op,
                         int o, int acc, int copy, nx_contract_view *u,
                         nx_array *y) {
  const int64_t *s = v->stride[o];
  int64_t k = v->extent[CON], quads = ceil_div(k, 4);
  int64_t n = v->extent[o == A ? ROW : COL];
  *y = *op[o];
  y->dtype = acc;
  y->bits = 32;
  if (!copy) {
    u->offset[o] = v->offset[o] / 4;
    for (int x = BATCH; x < CON; x++) u->stride[o][x] = s[x] / 4;
    u->stride[o][CON] = 1;
    return NULL;
  }
  /* a's rows are of quads words, b's of n. */
  int64_t pitch = pitch_of(o == A ? quads : n);
  int64_t rows = o == A ? n : quads;
  int64_t ne = s[BATCH] == 0 ? 1 : v->extent[BATCH], words = rows * pitch;
  uint8_t *d = alloc(ne * words * 4);
  if (d == NULL) return NULL;
  y->base = d;
  u->offset[o] = 0;
  u->stride[o][BATCH] = s[BATCH] == 0 ? 0 : words;
  if (o == A) {
    for (int64_t e = 0; e < ne; e++)
      nx_cpu_stage_as(op[A], v->offset[A] + e * s[BATCH], s[CON], s[ROW], k,
                      n, op[A]->dtype, d + e * words * 4, 4 * pitch);
    for (int64_t r = 0; r < ne * n; r++)
      memset(d + r * pitch * 4 + k, 0, (size_t)(pitch * 4 - k));
    u->stride[A][ROW] = pitch;
    u->stride[A][CON] = 1;
    return d;
  }
  quads_job j = {.b = op[B], .v = v, .quads = quads, .pitch = pitch, .d = d};
  int64_t units = ne * (s[CON] == 1 ? n : quads);
  nx_cpu_job(units, ne * words * 8, ne * words * 8, quad_words, &j);
  u->stride[B][COL] = 1;
  u->stride[B][CON] = pitch;
  return d;
}

/* A product of few rows whose b's words lie adjacent along k: each
   output the dot of a's row and b's column, which reads b once, in the
   order it lies, where chain order's tiles would pad a's rows to MR and
   pack b's slivers. A unit is DOT_COLUMNS columns of a batch element,
   which stay in the L1 cache from one pair of rows to the next while k is
   at most 8192. The dots do not split k, as lane order's do: a product of
   at most DOT_COLUMNS rows and columns runs on one core whatever k. On
   the M1 Max, int8 rows by a 4096 × 4096 weight stored [n × k] take 50 µs
   at 1 row against chain order's 255, 595 at 32 against 710, and the
   same 1.2 ms at 64. */
#define QUAD_ROWS 32

typedef struct {
  const problem *p; /* a and b of words */
  nx_cpu_quads_dot dot;
  int64_t groups; /* of DOT_COLUMNS columns */
} quad_dots_job;

/* Operand [o]'s words from (e, i, j, 0). */
static const uint32_t *words_at(const problem *p, int o, int64_t e,
                                int64_t i, int64_t j) {
  return (const uint32_t *)p->op[o].x->base + pos(p, o, e, i, j, 0);
}

/* A product of few rows whose b lies adjacent along its columns, as a
   weight stored [k × n] does: each row of a times b's rows, read in place
   in the order they lie, where chain order would interleave all of b into
   words and pack it. A unit is a group of a batch element's columns, in
   multiples of AXPY_TILE, UNITS of them a thread, so that a busy core
   takes fewer: with one core busy elsewhere, one unit a thread took 1 row
   from memory in 250-400 µs against 225-275, 8 rows in 660-900 against
   435-585. Its rows' sums, at most AXPY_SUMS bytes, stay in L1.
   Below the target's quads_zip_rows, the axpy zips rows of b in
   registers for the rows of a, AXPY_STEP rows of b at a time; from there,
   AXPY_ZIPS bytes of b's rows are zipped into words once, by
   interleave(), and the quads kernel runs all rows over them, as chain
   order would without padding a's rows. A unit's rows lie a row of b
   apart, a stride no x86 hardware prefetcher follows (the L1's follow at
   most 2 KiB, the L2's stay within a 4 KiB page; Intel's optimization
   manual), so each axpy is passed the next step's rows to prefetch.

   On the M1 Max, int8 rows by a 4096 × 4096 weight stored [k × n] take,
   against chain order (from the system cache unless said): 1 row 142-149
   µs against 825-831, and 192-227 from memory against 880-920 (148 to
   read the weight); 8 rows 268-287 against 810-877; 32 rows 694-735
   against 1.31-1.35 ms; 64 1.25 ms against 2.0; 128 2.6 against 3.1.
   Steps of 16 rows ran
   1 row in 140 µs where 64 and 256 swung from 120 to 320. Splitting k
   across units as well read a cached weight 1.4 times faster but one
   from memory 1.15 times slower, and memory is where a model's weights
   lie. */
#define AXPY_ROWS 128
/* Past AXPY_FEW rows, the axpys take a product only where b's columns
   alone give each thread a unit and k is at least AXPY_DEEP, and past
   AXPY_MANY rows only where b is more than 1 / AXPY_HELD of the largest cache
   the host reports: chain order splits rows and copies b once for all of
   them, which few columns, a short k or a b that stays in cache repay.
   Through AXPY_FEW rows, units split rows too where the columns are too
   few. On the M1 Max (12 MiB of L2 a cluster), rows by b of n columns and
   k rows against chain order: 64 by n = k = 64 took 3.7 µs in bands
   against 5.2, and 6.5 by columns alone; 32 by n 4096, k 256 86 against
   55; 128 by n = k = 1024 181 against 158 and 64 by n = k = 1024 96
   against 87, b in cache; 128 by n = k = 2048 661 against 692. 16 by n =
   k = 64 took 3.08 µs in bands against 2.86: 65K products, which the
   bands' threads cost more than they save. 32 by n = k = 512 took 15.7
   µs at best against 22.3, but lost 7 of 8 rounds with a busy core: 8
   groups of columns give the 8 threads a unit each, no slack to
   rebalance; 32 by n = k = 1024 61 against 65. On the i9-9900K at 16
   threads (16 MiB of L3), with AVX2's axpys of several rows, by the
   axpys against chain order: 16 by n = k = 256 6.9 µs in bands against
   11.2, by n 64, k 4096 23.7 against 23.4; 32 by n = k = 1024 73 against
   97; 64 by n = k = 4096 2.10 ms against 3.22. Of those chain order
   takes: 32 by n 64, k 4096 43 µs against 32 and 64 by n = k = 64 4.4
   against 4.0, as the bounds mean; 32 by n 4096, k 256 91 against 114
   and 64 by n = k = 1024 136 against 152, which the axpys would win here
   and lose on the M1. */
#define AXPY_FEW 16
#define AXPY_MANY 32
/* The share of the largest cache that b passes past AXPY_MANY rows: on
   the M1 Max (12 MiB) 128 rows by b of 1 MiB lost to chain order and by
   4 MiB won. */
#define AXPY_HELD 8
#define AXPY_DEEP 512
#define AXPY_STEP 16
/* A unit's zipped rows and sums, 96 KiB, within the M1's 128 KiB L1. */
#define AXPY_ZIPS (64 * 1024)
#define AXPY_SUMS (32 * 1024)
#define AXPY_TILE 64 /* the axpy's columns at once */
_Static_assert(AXPY_SUMS / 4 / AXPY_ROWS >= AXPY_TILE,
               "a unit of the most rows holds a tile of columns");

/* Row [i] of element [e]'s outputs [j0, j0 + c): init, or 0, plus [y]. */
static void quad_row(const problem *p, int64_t e, int64_t i, int64_t j0,
                     int64_t c, const uint32_t *y) {
  uint32_t *d = (uint32_t *)at_r(p, e, i, j0), init[DOT_COLUMNS];
  int64_t sy = p->op[DST].st[COL];
  for (int64_t q0 = 0; q0 < c; q0 += DOT_COLUMNS) {
    int64_t cq = min64(DOT_COLUMNS, c - q0);
    memset(init, 0, sizeof init);
    if (p->op[INIT].x)
      stage_acc(p, INIT, pos(p, INIT, e, i, j0 + q0, 0), p->op[INIT].st[COL],
                0, cq, 1, (uint8_t *)init, cq);
    for (int64_t q = 0; q < cq; q++)
      d[(q0 + q) * sy] = init[q] + y[q0 + q];
  }
}

static void quad_dots(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const quad_dots_job *j = ctx;
  const problem *p = j->p;
  int64_t m = p->ext[ROW], n = p->ext[COL], k = p->ext[CON];
  int64_t lda = p->op[A].st[ROW], ldb = p->op[B].st[COL];
  _Alignas(16) uint32_t y[2][DOT_COLUMNS + 4];
  for (int64_t t = lo; t < hi; t++) {
    int64_t e = t / j->groups, j0 = t % j->groups * DOT_COLUMNS;
    int64_t c = min64(DOT_COLUMNS, n - j0), q;
    const uint32_t *b = words_at(p, B, e, 0, j0);
    for (int64_t i = 0; i < m; i += 2) {
      const uint32_t *a = words_at(p, A, e, i, 0);
      /* A last row alone is its own pair, a last column its own four. */
      int64_t pair = i + 1 < m ? lda : 0, ldy = DOT_COLUMNS + 4;
      for (q = 0; q + 4 <= c; q += 4)
        j->dot(a, pair, b + q * ldb, ldb, k, y[0] + q, ldy);
      for (; q < c; q++) j->dot(a, pair, b + q * ldb, 0, k, y[0] + q, ldy);
      quad_row(p, e, i, j0, c, y[0]);
      if (i + 1 < m) quad_row(p, e, i + 1, j0, c, y[1]);
    }
  }
}

/* [u]'s product of the words [op] in [acc] by quads' dots, the fewer of
   rows and columns as rows: a's and b's words both lie adjacent along k,
   so each output is the same dot with a and b swapped. */
static void quad_dots_on(nx_cpu_quads_dot dot, const nx_cpu_gemm *g, int acc,
                         const nx_contract_view *u,
                         const nx_array *const *op) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc), .g = g};
  nx_gemm_lay_out(&p, u, op);
  if (p.ext[COL] < p.ext[ROW]) nx_gemm_transpose(&p);
  int64_t batch = p.ext[BATCH], m = p.ext[ROW], n = p.ext[COL];
  int64_t k = p.ext[CON], groups = ceil_div(n, DOT_COLUMNS);
  quad_dots_job j = {.p = &p, .dot = dot, .groups = groups};
  int64_t bytes = batch * (m + n) * k * p.w;
  double cost = flop_cost(g->kernel, 2.0 * batch * m * n * k) + bytes;
  nx_cpu_job(batch * groups, bytes, cost, quad_dots, &j);
}

typedef struct {
  problem *p; /* a of words, b of its own elements */
  nx_cpu_quads_axpy axpy;
  int64_t k;            /* in elements */
  int64_t cols, groups; /* columns a group, groups */
  int64_t rows, bands;  /* rows a band, bands */
} quad_axpys_job;

/* The rows ahead of a zip that it prefetches, a line at a time, for the
   stride above AXPY_ROWS: on the i9-9900K 8 rows by a 4096 × 4096 weight
   took 579 µs at 16 rows ahead, 591 at 32 and 737 without; 128 rows 4.61
   ms against 4.81. On the M1 Max neither moved. */
#define ZIP_AHEAD 16
#define LINE_BYTES 64 /* x86's cache line, half the M1's */

/* Rows [q, q + kb) of b from column [j0] of element [e], zipped as
   interleave() does into rows of words of [c] columns at [d], [pitch]
   words apart, a row past k repeating the last, prefetching ZIP_AHEAD
   rows on. */
static void zip_rows(const problem *p, int64_t e, int64_t j0, int64_t c,
                     int64_t pitch, int64_t q, int64_t kb, uint32_t *d) {
  const uint8_t *b = p->op[B].x->base + pos(p, B, e, 0, j0, q);
  int64_t ldb = p->op[B].st[CON];
  for (int64_t w = 0; w < ceil_div(kb, 4); w++) {
    const uint8_t *r0 = b + 4 * w * ldb;
    int64_t ahead = 4 * w + ZIP_AHEAD, ahead_end = min64(ahead + 4, kb);
    for (int64_t r = ahead; r < ahead_end; r++)
      for (int64_t x = 0; x < c; x += LINE_BYTES)
        __builtin_prefetch(b + r * ldb + x);
    const uint8_t *r1 = 4 * w + 1 < kb ? r0 + ldb : r0;
    const uint8_t *r2 = 4 * w + 2 < kb ? r1 + ldb : r1;
    const uint8_t *r3 = 4 * w + 3 < kb ? r2 + ldb : r2;
    interleave(r0, r1, r2, r3, (uint8_t *)(d + w * pitch), c);
  }
}

/* A unit's sums [s] of [m] rows from [i0] and [c] columns from [j0] of
   element [e]: by the axpy below the target's quads_zip_rows, else by the
   quads kernel over b's rows zipped into [z], AXPY_ZIPS bytes. */
static void quad_unit(const quad_axpys_job *j, int64_t e, int64_t i0,
                      int64_t m, int64_t j0, int64_t c, uint32_t *s,
                      uint32_t *z) {
  const problem *p = j->p;
  int64_t ldb = p->op[B].st[CON];
  nx_cpu_micro km = p->g->kernel;
  if (m < nx_cpu_table->quads_zip_rows) {
    const uint8_t *a = (const uint8_t *)words_at(p, A, e, i0, 0);
    int64_t lda = 4 * p->op[A].st[ROW];
    for (int64_t q = 0; q < j->k; q += AXPY_STEP) {
      const uint8_t *b = p->op[B].x->base + pos(p, B, e, 0, j0, q);
      /* The next step's rows, or b's last AXPY_STEP, or all of b. */
      int64_t ahead = min64(q + AXPY_STEP, j->k - AXPY_STEP);
      if (ahead < 0) ahead = 0;
      const uint8_t *next = b + (ahead - q) * ldb;
      j->axpy(a + q, lda, m, b, ldb, min64(AXPY_STEP, j->k - q), c, s, c,
              next);
    }
    return;
  }
  int64_t zp = pitch_of(c), step = AXPY_ZIPS / zp / 16 * 16;
  for (int64_t q = 0; q < j->k; q += step) {
    int64_t kb = min64(step, j->k - q), words = ceil_div(kb, 4);
    zip_rows(p, e, j0, c, zp, q, kb, z);
    const uint32_t *a = words_at(p, A, e, i0, 0) + q / 4;
    for (int64_t jj = 0; jj < c; jj += km.nr)
      km.f(m, min64(km.nr, c - jj), words, a, p->op[A].st[ROW], 1, z + jj,
           zp, s + jj, c, NX_CPU_FROM_TILE);
  }
}

static void quad_axpys(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const quad_axpys_job *j = ctx;
  problem *p = j->p;
  int64_t m = p->ext[ROW], n = p->ext[COL];
  /* The sums in scratch, as a thread's stack should not hold 32 KiB. */
  uint32_t *s = (uint32_t *)scratch(p, AXPY_SUMS);
  if (s == NULL) return;
  /* A tile reads past its last column: a sliver's width beyond the zips. */
  uint32_t *z = NULL;
  if (j->rows >= nx_cpu_table->quads_zip_rows) {
    z = (uint32_t *)scratch(p, AXPY_ZIPS + 4 * p->g->kernel.nr);
    if (z == NULL) {
      nx_cpu_give(s);
      return;
    }
  }
  for (int64_t t = lo; t < hi; t++) {
    int64_t e = t / (j->groups * j->bands), g = t % (j->groups * j->bands);
    int64_t j0 = g / j->bands * j->cols, i0 = g % j->bands * j->rows;
    int64_t c = min64(j->cols, n - j0), r = min64(j->rows, m - i0);
    memset(s, 0, (size_t)(r * c * 4));
    quad_unit(j, e, i0, r, j0, c, s, z);
    for (int64_t i = 0; i < r; i++) quad_row(p, e, i0 + i, j0, c, s + i * c);
  }
  nx_cpu_give(z);
  nx_cpu_give(s);
}

/* The cost of quads' axpys over [v]'s extents, [k] its elements along k,
   and its bytes: each element's product counts as a word's does in the
   dots, the zips spending about three instructions on each SDOT. */
static double axpys_cost(const nx_cpu_gemm *g, const nx_contract_view *v,
                         int64_t k, int64_t *bytes) {
  int64_t batch = v->extent[BATCH], m = v->extent[ROW], n = v->extent[COL];
  *bytes = batch * (m + n) * k;
  return flop_cost(g->kernel, 2.0 * batch * m * n * k) + *bytes;
}

/* Whether quads' axpys take [v]'s product, [k] its elements along k
   (AXPY_FEW says when). */
static int axpys_fit(const nx_cpu_gemm *g, const nx_contract_view *v,
                     int64_t k) {
  int64_t m = v->extent[ROW], bytes;
  if (m > AXPY_ROWS) return 0;
  if (m <= AXPY_FEW) return 1;
  int64_t batch = v->extent[BATCH], n = v->extent[COL];
  if (k < AXPY_DEEP) return 0;
  if (m > AXPY_MANY && batch * n * k <= nx_cpu_llc / AXPY_HELD)
    return 0;
  double cost = axpys_cost(g, v, k, &bytes);
  int64_t threads = nx_cpu_threads_at(bytes, cost, NX_CPU_THREAD_BYTES);
  return batch * ceil_div(n, AXPY_TILE) >= threads;
}

/* [u]'s product in [acc] of a's words and b's [k] elements along k by
   quads' axpys: 0 if a worker's scratch could not be allocated. Where the
   groups of columns are fewer than the threads, units split rows too.
   Where the product zips b's rows (the target's quads_zip_rows), bands
   keep the quads kernel's MR rows, which zip as the whole would, unless
   whole tiles leave more than half the threads idle: on the M1 Max 2
   bands of 8 rows took 16 rows by n 64, k 4096 29.7 µs against 15 in 8
   bands of 2. */
static int quad_axpys_on(nx_cpu_quads_axpy axpy, const nx_cpu_gemm *g,
                         int acc, int64_t k, const nx_contract_view *u,
                         const nx_array *const *op) {
  problem p = {.acc = acc, .w = nx_cpu_width(acc), .g = g};
  nx_gemm_lay_out(&p, u, op);
  atomic_init(&p.failed, 0);
  int64_t batch = p.ext[BATCH], m = p.ext[ROW], n = p.ext[COL], bytes;
  double cost = axpys_cost(g, u, k, &bytes);
  int64_t share = ceil_div(n, units(bytes, cost, NX_CPU_THREAD_BYTES));
  int64_t cols = min64(ceil_div(share, AXPY_TILE),
                       AXPY_SUMS / 4 / m / AXPY_TILE) * AXPY_TILE;
  int64_t groups = ceil_div(n, cols), mr = g->kernel.mr;
  int64_t threads = nx_cpu_threads_at(bytes, cost, NX_CPU_THREAD_BYTES);
  int64_t bands = min64(m, ceil_div(threads, batch * groups));
  int64_t whole = min64(bands, m / mr);
  int zips = m >= nx_cpu_table->quads_zip_rows;
  if (zips && 2 * batch * groups * whole >= threads) bands = whole;
  int64_t rows = ceil_div(m, bands);
  quad_axpys_job j = {.p = &p, .axpy = axpy, .k = k, .cols = cols,
                      .groups = groups, .rows = rows,
                      .bands = ceil_div(m, rows)};
  nx_cpu_job(batch * groups * j.bands, bytes, cost, quad_axpys, &j);
  return !atomic_load(&p.failed);
}

/* The table's quads kernels for a contraction in [acc] of operands of
   [da] and [db], or NULL. */
const nx_cpu_gemm *nx_gemm_quads_for(int acc, int da, int db) {
  if (acc != NX_INT32 && acc != NX_UINT32) return NULL;
  if (da != db || (da != NX_INT8 && da != NX_UINT8)) return NULL;
  const nx_cpu_gemm *g = &nx_cpu_table->quads[da];
  return g->kernel.f ? g : NULL;
}

int nx_gemm_quads_on(const nx_cpu_gemm *g, int acc, const nx_contract_view *v,
                     const nx_array *const *op) {
  if (v->extent[BATCH] == 0 || v->extent[ROW] == 0 || v->extent[COL] == 0)
    return 1;
  nx_contract_view u = *v;
  u.extent[CON] = ceil_div(v->extent[CON], 4);
  int64_t m = v->extent[ROW], n = v->extent[COL];
  nx_cpu_quads_axpy axpy = nx_cpu_table->quads_axpy[op[A]->dtype];
  int by_rows = axpy && v->stride[B][CON] != 1 && v->stride[B][COL] == 1 &&
                axpys_fit(g, v, v->extent[CON]);
  nx_array x[2] = {*op[A], *op[B]};
  int copy_a = !words_in_place(v, A);
  int copy_b = !by_rows && !words_in_place(v, B);
  uint8_t *ab = words_of(v, op, A, acc, copy_a, &u, &x[A]), *bb = NULL;
  if (!by_rows) bb = words_of(v, op, B, acc, copy_b, &u, &x[B]);
  int done = 0;
  if ((copy_a && ab == NULL) || (copy_b && bb == NULL)) goto out;
  const nx_array *ops[4] = {&x[A], &x[B], op[INIT], op[DST]};
  if (by_rows) {
    done = quad_axpys_on(axpy, g, acc, v->extent[CON], &u, ops);
    goto out;
  }
  /* a's words lie adjacent along k; the dots need b's to as well. */
  nx_cpu_quads_dot dot = nx_cpu_table->quads_dot[op[A]->dtype];
  if (dot && u.stride[B][CON] == 1 && min64(m, n) <= QUAD_ROWS) {
    quad_dots_on(dot, g, acc, &u, ops);
    done = 1;
  } else
    done = nx_gemm_contract_with(g, acc, NX_CPU_CHAIN, &u, ops, NULL, NULL);
out:
  free(ab);
  free(bb);
  return done;
}
