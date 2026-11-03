/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Attention (Nx_kernel.S.attention): a softmax over each query row's
   visible keys of their scores, times their values, by blocks of keys
   with a running maximum, sum and accumulator per row (FlashAttention-2's
   forward, Dao 2023, arXiv 2307.08691, Algorithm 1), so that no score
   outlives its block.

   A unit is a lead index's key head, some of its group's query heads and
   some of their rows, at most ROWS rows in all, which share each block of
   keys and values: the units split the work, and the keys are never
   split, so the threads change no bit. A row's keys go in blocks of KEYS
   from key 0, up to its lead index's length; per block and row:

     visible   the pair program's output 0, past the keys the kernel hides
               itself: those past the length and those whose page table
               entry is out of range, which it never reads
     scores    the row's products with each key, added in increasing
               dimension, each fused (chain order), times the scale,
               rewritten by the program's output 1, -inf where hidden
     update    m' = max (m, scores); α = exp (m - m'); p = exp (s - m');
               l = α l + Σ p; acc = α acc + p v, the products added in
               chain order onto acc

   Units of many rows compute both products on the chain kernel's tiles
   (cpu.h, nx_cpu_kernel), over the keys staged transposed; units of few,
   each row against the keys as they lie and a gemv of the values. Each
   adds the same products in the same order. A row with no visible pair in
   a block leaves its state as it was, and a block no row of the unit sees
   is skipped before its keys are read. A row whose hidden pairs in a
   block hold a value that is not finite adds its visible pairs' products
   alone, in the same order: a hidden value never reaches o. Each row's
   arithmetic is a function of its own pairs and the blocks from key 0, so
   the units, the threads and the extents other than the keys' and their
   dimensions change no bit.

   Each unit classes its blocks first by the ranges of the mask's output
   0 over its box of pairs (Classes): a block the mask hides from every
   pair of the unit is skipped, one it shows to every pair is read without
   running the mask. A mask that reads no lead index, head or score is
   classed once a call per tile of rows and block, by its ranges, then
   where they leave the block mixed by running it at lead index 0 and
   head 0 (shared).

   Bound (nx_cpu.mli), to first order in u, the accumulator's unit
   roundoff (2^-24 or 2^-53), barring overflow and underflow, for a row of
   t keys in n blocks, scores s within [-b, b], m the final maximum; exp
   within 2 ulps, so its relative error is at most 5u (2 ulps of the
   correctly rounded value, 2.5 of the exact), and log within 1 ulp in
   float32 and 2 in float64, λu with λ 3 or 5:
   1. Exponents. p_j = exp (s_j - m_k) and α_i = exp (m_(i-1) - m_i) each
      round their difference once, u times its size. A key's weight in l
      and acc is p_j times the α of each later block, whose differences
      telescope to m - s_j, as m only rises: at most u (m - s_j) <= 2bu in
      all, and, weighted by the weights, at most u ln t (Σ w x <= ln t for
      w proportional to e^-x, x >= 0, one x 0).
   2. l. A key's term passes p_j's exp (5u), its block's sum (at most 7
      additions in its lane and 3 in the lanes' tree, 10u) and the addition
      into l (u); each later block multiplies l by α (exp 5u, product u)
      and adds into it (u): 7u. So l is within (7n + 9) u + u ln t
      relative.
   3. lse = m + log l: log errs by λu |log l| <= λu ln t, the addition by
      u |lse|: (7n + 9 + (λ + 1) ln t) u + u |lse|.
   4. o's weights, p_j Π α / l: the exponents' 2bu, p_j's exp (5u), each
      later block's α (exp 5u, acc's product u), l's (7n + 9) u + u ln t
      and the division's u: a relative (13n + 9 + 2b + ln t) u. The sums
      of products add γ(t + 1, 2u) M as nx.mli's chain does.
   The ε nx.mli reads covers 3 and 4 there:
   (13n + 9 + (λ + 2) ln t) u + u |lse|.

   At the end o = acc / l, rounded once to o's dtype, and lse = m + log l;
   a row of l = 0 has o = +0 and lse = -inf. Scores, weights, sums and
   accumulators are in the accumulator, lse's dtype, float32 or float64;
   narrower operands are widened on load. Values of the accumulator's dtype
   whose dimension steps one element are read in place; keys are staged
   transposed, or for few rows of float32 read in place where of unit
   steps. Each accumulator's units are attention_body.h's. */

#include <math.h>
#include <stdatomic.h>
#include <string.h>

#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_kinds.h"

/* The rows a unit holds at most, and the keys of a block. KEYS fixes the
   association: a row's blocks start at multiples of it. */
#define ROWS 64
#define KEYS 64

/* The door's operands, in order: o, lse, q, k, v, then the page table where
   paged and the lengths where given, then the loads. */
enum { O, LSE, Q, K, V };

/* A run of a block's keys that one step reaches, from key [j] of the
   block: [n] keys, key i's value at v + i·vs, read in place or staged, its
   key at position ka + i·kk of the keys' array. */
typedef struct {
  int j, n;
  const void *v;
  int64_t vs;
  int64_t ka, kk, va, vk;
} run;

/* Rows from which a unit's products run on the chain kernel's tiles; fewer
   take a gemv of each row. Both add in chain order. */
#define CHAIN_ROWS 12

/* A block's class for a unit's rows (Classes): its mask false at every
   pair, true at every pair, or either. */
enum { HIDDEN, SHOWN, MIXED };

typedef struct {
  const nx_array *a; /* the door's descriptors */
  const nx_array *table, *lengths;
  int rank, lead; /* q's rank, its lead axes */
  int64_t ext[NX_MAX_RANK];
  int64_t hq, hk, g, tq, tk, dk, dv;
  int64_t pages, page, max_pages;
  int dt, acc, odt, w; /* q's dtype, the accumulator's, o's and its
                          element's bytes */
  double scale;
  /* Units: heads of a group and rows of a head each holds. */
  int64_t hg, tr, chunks, tiles;
  /* The program, where an output is not the default: [mask] where output 0
     is not [true], [score] where output 1 is not the score itself. */
  const nx_cpu_prog *prog;
  nx_cpu_places pl;
  int mask, score;
  /* Keys and values read in place. */
  int k_in, v_in;
  /* The ranges of the mask's output 0 bound over the tiles' loop
     (Classes), else NULL, every block then mixed. */
  nx_cpu_prog *ranges;
  nx_cpu_places rpl;
  int64_t blocks;
  /* Classes of a tile's blocks where every lead index and head's are one
     (shared), else NULL. */
  uint8_t *cls;
  int64_t kd, vd; /* the keys' and values' steps along their dimension */
  nx_cpu_gemv gemv;
  nx_cpu_kernel pv;
  int nr;
  atomic_int failed;
} att;

/* Lead index [b]'s position in [x], whose first [lead] axes are [ext]'s. */
static int64_t lead_at(const nx_array *x, int lead, const int64_t *ext,
                       int64_t b) {
  int64_t at = x->offset;
  for (int i = lead - 1; i >= 0; i--) {
    at += b % ext[i] * x->dim[x->rank + i];
    b /= ext[i];
  }
  return at;
}

static inline int64_t step(const nx_array *x, int axis) {
  return x->dim[x->rank + axis];
}

/* The keys lead index [b] reads: its length within [0, tk]. */
static int64_t length_of(const att *c, int64_t b) {
  if (c->lengths == NULL) return c->tk;
  int64_t l =
      ((const int64_t *)c->lengths->base)[lead_at(c->lengths, c->lead, c->ext,
                                                  b)];
  return l < 0 ? 0 : l > c->tk ? c->tk : l;
}

static uint8_t *align64(uint8_t *p) {
  return p + (64 - (uintptr_t)p % 64) % 64;
}

/* The block of [nk] keys from [j0] of lead index [b] and key head [kh] as
   runs into [runs], each key's presence into [present]: a key whose page
   table entry is out of range is absent. Answers the runs' count. */
static int runs_of(const att *c, run *runs, int64_t b, int64_t kh,
                   int64_t j0, int nk, uint8_t *present) {
  const nx_array *k = &c->a[K], *v = &c->a[V];
  int n = 0;
  if (c->table == NULL) {
    memset(present, 1, (size_t)nk);
    int r = c->rank;
    int64_t ka = lead_at(k, c->lead, c->ext, b) + kh * step(k, r - 3) +
                 j0 * step(k, r - 2);
    int64_t va = lead_at(v, c->lead, c->ext, b) + kh * step(v, r - 3) +
                 j0 * step(v, r - 2);
    run *u = &runs[n++];
    u->j = 0;
    u->n = nk;
    u->ka = ka;
    u->kk = step(k, r - 2);
    u->va = va;
    u->vk = step(v, r - 2);
    return n;
  }
  const int64_t *t = (const int64_t *)c->table->base;
  int64_t ta = lead_at(c->table, c->lead, c->ext, b);
  int64_t ts = step(c->table, c->lead);
  for (int j = 0; j < nk;) {
    int64_t key = j0 + j, pi = key / c->page, at = key % c->page;
    int m = (int)(c->page - at < nk - j ? c->page - at : nk - j);
    int64_t e = t[ta + pi * ts];
    int in = e >= 0 && e < c->pages;
    memset(present + j, in, (size_t)m);
    if (in) {
      int64_t ka = k->offset + e * step(k, 0) + kh * step(k, 1) +
                   at * step(k, 2);
      int64_t va = v->offset + e * step(v, 0) + kh * step(v, 1) +
                   at * step(v, 2);
      run *u = &runs[n++];
      u->j = j;
      u->n = m;
      u->ka = ka;
      u->kk = step(k, 2);
      u->va = va;
      u->vk = step(v, 2);
    }
    j += m;
  }
  return n;
}

/* The scores of [r] rows at [q], row i's dk elements at q + i·dk, against
   [n] keys at [k], key j's elements at k + j·ks, into [s], row i's at
   s + i·KEYS, each added in increasing dimension from +0, each fused, as
   the chain kernel adds them: on x86-64 eight keys a gather of one
   dimension, which four rows share. */
#if defined(__AVX2__) && defined(__FMA__)
#define ROW_SCORES 1
#define GATHER(d) _mm256_i32gather_ps(kj + (d), idx, 4)
#define ROWS4(R)                                                        \
  static void scores##R(const float *q, int64_t dk, const float *kj,    \
                        __m256i idx, float *s) {                        \
    __m256 a[4];                                                        \
    for (int i = 0; i < R; i++) a[i] = _mm256_setzero_ps();             \
    _Pragma("GCC unroll 4") for (int64_t d = 0; d < dk; d++) {          \
      __m256 x = GATHER(d);                                             \
      _Pragma("GCC unroll 4") for (int i = 0; i < R; i++) a[i] =        \
          _mm256_fmadd_ps(_mm256_set1_ps(q[i * dk + d]), x, a[i]);      \
    }                                                                   \
    for (int i = 0; i < R; i++) _mm256_storeu_ps(s + i * KEYS, a[i]);   \
  }
ROWS4(1)
ROWS4(2)
ROWS4(3)
ROWS4(4)

static void row_scores(const float *q, int r, const float *k, int64_t ks,
                       int n, int64_t dk, float *s) {
  int j = 0;
  if (ks <= INT32_MAX / 8) {
    __m256i idx = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                     _mm256_set1_epi32((int32_t)ks));
    for (; j + 8 <= n; j += 8) {
      const float *kj = k + j * ks;
      int i = 0;
      for (; i + 4 <= r; i += 4) scores4(q + i * dk, dk, kj, idx, s + i * KEYS + j);
      if (r - i == 3) scores3(q + i * dk, dk, kj, idx, s + i * KEYS + j);
      else if (r - i == 2) scores2(q + i * dk, dk, kj, idx, s + i * KEYS + j);
      else if (r - i == 1) scores1(q + i * dk, dk, kj, idx, s + i * KEYS + j);
    }
  }
  for (; j < n; j++)
    for (int i = 0; i < r; i++) {
      float a = 0;
      for (int64_t d = 0; d < dk; d++)
        a = fmaf(q[i * dk + d], k[j * ks + d], a);
      s[i * KEYS + j] = a;
    }
}
#else
#define ROW_SCORES 0
#endif

/* The ranges of unit [u]'s first [nb] blocks into [lo] and [hi], its
   mask's least and greatest over each block's pairs: every block mixed
   where the ranges cannot run. */
static void ranges_of(const att *c, uint8_t *lo, uint8_t *hi, int64_t u,
                      int64_t nb) {
  int64_t p = u * c->blocks;
  memset(lo, 0, (size_t)nb);
  memset(hi, 1, (size_t)nb);
  nx_cpu_prog_terms(c->ranges, &c->rpl, 0, p, 1, nb, nb, 1, lo);
  nx_cpu_prog_terms(c->ranges, &c->rpl, 1, p, 1, nb, nb, 1, hi);
  if (nx_cpu_prog_failed(c->ranges)) {
    memset(lo, 0, (size_t)nb);
    memset(hi, 1, (size_t)nb);
  }
}

static int class_of(uint8_t lo, uint8_t hi) {
  return lo ? SHOWN : hi ? MIXED : HIDDEN;
}

/* The units, in each accumulator (attention_body.h). */
#define T float
#define DT NX_FLOAT32
#define WORD uint32_t
#define SWORD int32_t
#define EXPONENT 0x7f800000u
#define EXP nx_exp_f32
#define LOG nx_log_f32
#define FMA fmaf
#define ROWWISE ROW_SCORES
#define F(name) name##_f32
#include "attention_body.h"

#define T double
#define DT NX_FLOAT64
#define WORD uint64_t
#define SWORD int64_t
#define EXPONENT 0x7ff0000000000000u
#define EXP nx_exp_f64
#define LOG nx_log_f64
#define FMA fma
#define ROWWISE 0
#define F(name) name##_f64
#include "attention_body.h"

/* The shared classes of tiles [lo, hi): a tile's ranges, those of unit
   [t], at lead index 0 and head 0, then the mask at every pair of a block
   they leave mixed. */
static void shared(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  att *c = ctx;
  uint8_t vis[ROWS * KEYS];
  uint8_t *ends = c->ranges != NULL ? nx_cpu_take(2 * c->blocks) : NULL;
  for (int64_t t = lo; t < hi; t++) {
    int64_t i0 = t * c->tr;
    int nt = (int)(c->tq - i0 < c->tr ? c->tq - i0 : c->tr);
    if (ends != NULL) ranges_of(c, ends, ends + c->blocks, t, c->blocks);
    for (int64_t k = 0; k < c->blocks; k++) {
      int cls = ends != NULL ? class_of(ends[k], ends[c->blocks + k]) : MIXED;
      if (cls == MIXED) {
        int64_t j0 = k * KEYS;
        int nk = (int)(c->tk - j0 < KEYS ? c->tk - j0 : KEYS);
        nx_cpu_prog_terms(c->prog, &c->pl, 0, i0 * c->tk + j0, 1, nk, c->tk,
                          nt, vis);
        int n = 0;
        for (int j = 0; j < nt * nk; j++) n += vis[j] != 0;
        cls = n == 0 ? HIDDEN : n == nt * nk ? SHOWN : MIXED;
      }
      c->cls[t * c->blocks + k] = (uint8_t)cls;
    }
  }
  if (ends != NULL) nx_cpu_give(ends);
}

/* Whether the program's output 0 reads no lead index or head: no score,
   no coordinate before the rows' and no load that steps along such an
   axis. Its [nl] loads are at [loads]. */
static int shared_mask(const att *c, const nx_cpu_prog *g,
                       const nx_array *loads, int nl) {
  uint8_t axes[NX_MAX_RANK] = {0};
  uint8_t local[64], *reads = nl < 64 ? local : calloc((size_t)nl + 1, 1);
  if (reads == NULL) return 0;
  memset(reads, 0, (size_t)nl + 1);
  nx_cpu_prog_reads(g, 0, reads, axes);
  int shared = !reads[0];
  for (int a = 0; a < c->rank - 2; a++) shared &= !axes[a];
  for (int k = 0; k < nl; k++)
    for (int a = 0; reads[1 + k] && a < c->rank - 2; a++)
      shared &= loads[k].dim[a] <= 1 || loads[k].dim[loads[k].rank + a] == 0;
  if (reads != local) free(reads);
  return shared;
}

/* The chain order kernel of the accumulator [acc] that reads a in place
   and n columns of b in place: the table's, its other's, or the base
   table's. */
static const nx_cpu_micro *pv_kernel(int acc) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[acc];
  for (int i = 0; i < 2 && g != NULL; i++, g = g->other)
    if (g->kernel.f != NULL && g->kernel.a == NX_CPU_A_ELEMENTS &&
        g->kernel.b == NX_CPU_B_COLUMNS && g->enter == NULL)
      return &g->kernel;
  return &nx_cpu_base.gemm[acc].kernel;
}

/* Programs */

/* Whether output [o] of the program at [vp] is its default: [true] for
   output 0, operand 0 itself for output 1. */
static int is_default(value vp, int o) {
  value outs = Field(vp, NX_PROG_OUTS), nodes = Field(vp, NX_PROG_NODES);
  value d = Field(nodes, Int_val(Field(outs, o)));
  if (Is_long(d)) return 0;
  if (o == 1)
    return Tag_val(d) == NX_NODE_IN && Int_val(Field(d, NX_IN_OPERAND)) == 0;
  return Tag_val(d) == NX_NODE_CONST &&
         Int_val(Field(d, NX_CONST_DTYPE)) == NX_BOOL &&
         caml_string_length(Field(d, NX_CONST_BITS)) > 0 &&
         Byte_u(Field(d, NX_CONST_BITS), 0) != 0;
}

/* [g] bound to the [n] operands [ins] over a loop of [r] axes of extents
   [shape], with its places in [pl] (cpu.h, Places), [st] holding n + 2
   rows and [first] n + 1 entries: NX_OK, or nx_cpu_prog_bind's error. */
static int bind_over(nx_cpu_prog *g, const nx_array *ins, int n, int r,
                     const int64_t *shape, nx_cpu_places *pl,
                     int64_t (*st)[NX_MAX_RANK], int64_t *first) {
  int index, e = nx_cpu_prog_bind(g, ins, r, shape, &index);
  if (e) return e;
  /* The virtual operand, C-contiguous, and the operands, coalesced; the
     index steps as the virtual operand. */
  int64_t past = 1;
  for (int i = r - 1; i >= 0; i--) {
    pl->ext[i] = shape[i];
    st[0][i] = past;
    past *= shape[i];
    for (int k = 0; k < n; k++) st[1 + k][i] = ins[k].dim[r + i];
  }
  pl->rank = nx_coalesce_dims(1 + n, r, pl->ext, st);
  memcpy(pl->vs, st[0], sizeof(int64_t) * (size_t)pl->rank);
  memcpy(st[1 + n], st[0], sizeof(int64_t) * (size_t)pl->rank);
  for (int k = 0; k < n; k++) first[k] = ins[k].offset;
  first[n] = 0;
  pl->n = n + index;
  pl->step = st + 1;
  pl->first = first;
  return NX_OK;
}

/* An operand of [dt] holding the one element at [base] at every index of
   [r] axes of extents [shape]. */
static nx_array everywhere(const void *base, int dt, int r,
                           const int64_t *shape) {
  nx_array x = {.base = (uint8_t *)base,
                .dtype = dt,
                .bits = nx_dtype_row_of(dt).bits,
                .rank = r};
  for (int i = 0; i < r; i++) x.dim[i] = shape[i];
  return x;
}

/* The program [g] bound to the loads at [a], [nl] of them, over the pairs'
   loop of [c]: its operand 0 a stand-in, every element of which the
   kernel's tiles give (nx_cpu_prog_tile); and [c]'s places of the loads,
   [st] holding nl + 3 rows, [first] nl + 2 entries. */
static int bind(att *c, nx_cpu_prog *g, const nx_array *a, int nl,
                nx_array *ins, int64_t (*st)[NX_MAX_RANK], int64_t *first) {
  static const double zero = 0;
  int r = c->rank;
  int64_t shape[NX_MAX_RANK];
  memcpy(shape, c->ext, sizeof(int64_t) * (size_t)(r - 3));
  shape[r - 3] = c->hq;
  shape[r - 2] = c->tq;
  shape[r - 1] = c->tk;
  ins[0] = everywhere(&zero, c->acc, r, shape);
  for (int k = 0; k < nl; k++) ins[1 + k] = a[k];
  return bind_over(g, ins, nl + 1, r, shape, &c->pl, st, first);
}

/* Classes

   A unit reads the pairs of a box: its lead index, its group's heads
   [h0, h0 + nh), its rows [i0, i0 + nt) and, block by block, the keys
   [j0, j0 + 64). The ranges of the mask's output 0 over a block's box
   (Prog.ranges) are its least and greatest value there: where the least
   is true the mask shows every pair of the block, and the block is read
   without it; where the greatest is false it hides every pair, and the
   block is skipped. The unit hides keys past the length and keys of a
   table entry out of range itself; the box holds keys past the length,
   which only widens it.

   The ranges run over the tiles' loop [lead…; groups; tiles; blocks], a
   group being one unit's heads, hk · chunks of them, so that unit u's
   blocks are its positions u · blocks on. Each coordinate's bounds are a
   table along its axis. Each load the ranges read has its least and
   greatest element over each box in two tables of its dtype, along the
   axes where it steps: one pass over its elements, once a call. A load of
   fewer bits than a byte stands at its dtype's ends.

   A mask whose output 0 reads no lead index, head or score (shared_mask)
   has one class per tile and block for every unit, the ranges' where they
   decide it, else from the mask at every pair of the tile's rows at lead
   index 0 and head 0 (shared): a mask the ranges cannot follow, as one
   computed through floats, is then run once a call.

   Example: a causal mask k <= q + (len - tq), len = tq, over a unit of
   rows [64, 128) and the block of keys [192, 256) has its least,
   255 <= 64, and its greatest, 192 <= 127, false: hidden. */

/* The box of position [i] of the tiles' loop along its axis [a] in the
   pairs' loop: [*from, *from + *n). */
static void box_along(const att *c, int a, int64_t i, int64_t *from,
                      int64_t *n) {
  int r = c->rank;
  if (a < r - 3) {
    *from = i;
    *n = 1;
  } else if (a == r - 3) {
    int64_t ch = i % c->chunks;
    *from = i / c->chunks * c->g + ch * c->hg;
    *n = c->g - ch * c->hg < c->hg ? c->g - ch * c->hg : c->hg;
  } else if (a == r - 2) {
    *from = i * c->tr;
    *n = c->tq - *from < c->tr ? c->tq - *from : c->tr;
  } else {
    *from = i * KEYS;
    *n = c->tk - *from < KEYS ? c->tk - *from : KEYS;
  }
}

/* A load's table: its least and greatest elements over each box, at
   [lo] and [hi], [n] of [w] bytes each, C-contiguous over the extents
   [ext], 1 where the load does not step. Each starts on 8 bytes. */
typedef struct {
  const att *c;
  const nx_array *x;
  int w;
  int64_t n, ext[NX_MAX_RANK];
  uint8_t *lo, *hi;
} table;

/* Rows of a box staged at once. */
#define STAGED 16

static int64_t aligned8(int64_t b) { return (b + 7) / 8 * 8; }

/* The least and greatest of the [n] elements of the integer or boolean
   dtype [dt] at [v] into [lo] and [hi], with theirs unless [first]. A
   boolean's byte is true where it is not 0, which its least and greatest
   byte keep. */
#define EXTREMES(T)                                         \
  {                                                         \
    const T *e = (const T *)v;                              \
    T a = first ? e[0] : *(T *)lo, b = first ? e[0] : *(T *)hi; \
    for (int64_t j = 0; j < n; j++) {                       \
      a = e[j] < a ? e[j] : a;                              \
      b = e[j] > b ? e[j] : b;                              \
    }                                                       \
    *(T *)lo = a;                                           \
    *(T *)hi = b;                                           \
    return;                                                 \
  }

static void extremes(int dt, const uint8_t *v, int64_t n, int first,
                     uint8_t *lo, uint8_t *hi) {
  switch (dt) {
    case NX_INT64: EXTREMES(int64_t)
    case NX_UINT64: EXTREMES(uint64_t)
    case NX_INT32: EXTREMES(int32_t)
    case NX_UINT32: EXTREMES(uint32_t)
    case NX_INT16: EXTREMES(int16_t)
    case NX_UINT16: EXTREMES(uint16_t)
    case NX_INT8: EXTREMES(int8_t)
    default: EXTREMES(uint8_t)
  }
}

#undef EXTREMES

static void fill(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const table *t = ctx;
  const att *c = t->c;
  const nx_array *x = t->x;
  int r = c->rank;
  _Alignas(64) uint8_t v[STAGED * KEYS * 8];
  for (int64_t e = lo; e < hi; e++) {
    int64_t from[NX_MAX_RANK], n[NX_MAX_RANK], rest = e, at = x->offset;
    for (int a = r - 1; a >= 0; a--) {
      from[a] = 0;
      n[a] = 1;
      if (step(x, a) != 0 && x->dim[a] > 1)
        box_along(c, a, rest % t->ext[a], &from[a], &n[a]);
      rest /= t->ext[a];
      at += from[a] * step(x, a);
    }
    int64_t nk = n[r - 1];
    int first = 1;
    for (int64_t h = 0; h < n[r - 3]; h++)
      for (int64_t i = 0; i < n[r - 2]; i += STAGED, first = 0) {
        int64_t m = n[r - 2] - i < STAGED ? n[r - 2] - i : STAGED;
        nx_cpu_stage_as(x, at + h * step(x, r - 3) + i * step(x, r - 2),
                        step(x, r - 1), step(x, r - 2), nk, m, x->dtype, v,
                        nk);
        extremes(x->dtype, v, m * nk, first, t->lo + e * t->w,
                 t->hi + e * t->w);
      }
  }
}

/* The ranges [g] bound over the tiles' loop of [c], with the loads at
   [a], [nl] of them, read into tables in memory it takes: answers that
   memory, to free after the call's work, or NULL where host memory runs
   out, [g] then unbound. */
static void *bind_ranges(att *c, nx_cpu_prog *g, const nx_array *a, int nl) {
  static const int64_t zero[2] = {0, 0};
  /* int4's ends, then those of uint4 and bit, in a byte's low bits. */
  static const uint8_t ends[2][2] = {{0x88, 0x77}, {0x00, 0xFF}};
  int r = c->rank, n = 2 * (nl + 1) + 2 * r;
  int64_t shape[NX_MAX_RANK];
  memcpy(shape, c->ext, sizeof(int64_t) * (size_t)(r - 3));
  shape[r - 3] = c->hk * c->chunks;
  shape[r - 2] = c->tiles;
  shape[r - 1] = c->blocks;
  uint8_t *reads = calloc((size_t)n, 1), axes[NX_MAX_RANK];
  table *t = calloc((size_t)nl + 1, sizeof(table));
  if (reads == NULL || t == NULL) {
    free(reads);
    free(t);
    return NULL;
  }
  nx_cpu_prog_reads(g, 0, reads, axes);
  nx_cpu_prog_reads(g, 1, reads, axes);
  /* The memory: the operands, their places, the coordinates' bounds, then
     each read load's two tables. */
  int64_t coords = 0, bytes = 0;
  for (int i = 0; i < r; i++) coords += 2 * shape[i];
  for (int k = 0; k < nl; k++) {
    const nx_array *x = &a[k];
    nx_dtype_row row = nx_dtype_row_of(x->dtype);
    if (!(reads[2 * k + 2] || reads[2 * k + 3]) || row.bits < 8 ||
        row.kind == NX_KIND_FLOAT || row.kind == NX_KIND_COMPLEX)
      continue;
    t[k] = (table){.c = c, .x = x, .w = row.bits / 8, .n = 1};
    for (int i = 0; i < r; i++) {
      t[k].ext[i] = step(x, i) != 0 && x->dim[i] > 1 ? shape[i] : 1;
      t[k].n *= t[k].ext[i];
    }
    bytes += 2 * aligned8(t[k].n * t[k].w);
  }
  size_t size = (size_t)n * sizeof(nx_array) +
                (size_t)(n + 2) * NX_MAX_RANK * sizeof(int64_t) +
                (size_t)(n + 1 + coords) * sizeof(int64_t) + (size_t)bytes;
  uint8_t *mem = malloc(size);
  free(reads);
  if (mem == NULL) {
    free(t);
    return NULL;
  }
  nx_array *ins = (nx_array *)mem;
  int64_t(*st)[NX_MAX_RANK] = (int64_t(*)[NX_MAX_RANK])(ins + n);
  int64_t *first = (int64_t *)(st + n + 2), *coord = first + n + 1;
  uint8_t *tables = (uint8_t *)(coord + coords);
  /* The scores' bounds, of the accumulator's dtype, which the ranges never
     read, then each load's. */
  ins[0] = ins[1] = everywhere(zero, c->acc, r, shape);
  for (int k = 0; k < nl; k++) {
    nx_array *lo = &ins[2 * k + 2], *hi = lo + 1;
    int dt = a[k].dtype, sub = nx_dtype_row_of(dt).bits < 8;
    if (t[k].x == NULL) {
      const uint8_t *e = ends[dt != NX_INT4];
      *lo = everywhere(sub ? (const void *)e : zero, dt, r, shape);
      *hi = everywhere(sub ? (const void *)(e + 1) : zero, dt, r, shape);
      continue;
    }
    t[k].lo = tables;
    t[k].hi = tables + aligned8(t[k].n * t[k].w);
    tables += 2 * aligned8(t[k].n * t[k].w);
    *lo = everywhere(t[k].lo, dt, r, shape);
    *hi = everywhere(t[k].hi, dt, r, shape);
    for (int64_t s = 1, i = r - 1; i >= 0; i--) {
      lo->dim[r + i] = hi->dim[r + i] = t[k].ext[i] > 1 ? s : 0;
      s *= t[k].ext[i];
    }
    double b = nx_cpu_bytes(&a[k]);
    nx_cpu_job(t[k].n, (int64_t)b, NX_CPU_STRIDED_COST * b, fill, &t[k]);
  }
  free(t);
  /* The coordinates' bounds: a lead index its own, a group's heads, a
     tile's rows and a block's keys. */
  for (int i = 0; i < r; i++) {
    nx_array *lo = &ins[2 * (nl + 1) + 2 * i], *hi = lo + 1;
    for (int64_t j = 0; j < shape[i]; j++) {
      int64_t from, m;
      box_along(c, i, j, &from, &m);
      coord[j] = from;
      coord[shape[i] + j] = from + m - 1;
    }
    *lo = everywhere(coord, NX_INT64, r, shape);
    *hi = everywhere(coord + shape[i], NX_INT64, r, shape);
    lo->dim[r + i] = hi->dim[r + i] = 1;
    coord += 2 * shape[i];
  }
  if (bind_over(g, ins, n, r, shape, &c->rpl, st, first)) {
    free(mem);
    return NULL;
  }
  return mem;
}

/* The external */

/* What a call holds on the C heap: the copies of the program and of its
   ranges, and the tables of its loads. A custom block owns them, so that
   a raise of the door leaves them to its finalizer. */
typedef struct {
  nx_cpu_prog *g, *ranges;
  void *tables;
} held;

static void release(value owner) {
  held *h = Data_custom_val(owner);
  nx_cpu_prog_free(h->g);
  nx_cpu_prog_free(h->ranges);
  free(h->tables);
  h->g = NULL;
  h->ranges = NULL;
  h->tables = NULL;
}


static struct custom_operations held_ops = {
    "nx.cpu.attention",         release,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The door's operands from the roots of the call. */
typedef struct {
  const value *vo, *vlse, *vq, *vkv, *vlengths, *vloads;
  int dt, acc, odt, paged, lengths;
  const nx_cpu_prog *g;
} operands;

static int operand(void *ctx, int k, nx_operand *in) {
  const operands *o = ctx;
  int at = 5 + o->paged;
  if (k == O) *in = (nx_operand){*o->vo, o->odt, 1};
  else if (k == LSE) *in = (nx_operand){*o->vlse, o->acc, 1};
  else if (k == Q) *in = (nx_operand){*o->vq, o->dt, 0};
  else if (k == K || k == V)
    *in = (nx_operand){Field(*o->vkv, k - K), o->dt, 0};
  else if (o->paged && k == 5)
    *in = (nx_operand){Field(*o->vkv, 2), NX_INT64, 0};
  else if (o->lengths && k == at)
    *in = (nx_operand){Field(*o->vlengths, 0), NX_INT64, 0};
  else {
    int i = k - at - o->lengths;
    int dt = o->g ? nx_cpu_prog_in(o->g, 1 + i)
                  : nx_array_dtype(Field(*o->vloads, i));
    *in = (nx_operand){Field(*o->vloads, i), dt, 0};
  }
  return NX_OK;
}

static int same_shape(const nx_array *x, int r, const int64_t *s) {
  if (x->rank != r) return 0;
  for (int i = 0; i < r; i++)
    if (x->dim[i] != s[i]) return 0;
  return 1;
}

/* The extents of the call read into [c] from the descriptors [a], and
   whether they fit: NX_OK or NX_SHAPE. */
static int extents(att *c, const nx_array *a, int paged) {
  const nx_array *q = &a[Q], *k = &a[K], *v = &a[V];
  int r = q->rank;
  if (r < 3) return NX_SHAPE;
  c->rank = r;
  c->lead = r - 3;
  memcpy(c->ext, q->dim, sizeof(int64_t) * (size_t)(r - 3));
  c->hq = q->dim[r - 3];
  c->tq = q->dim[r - 2];
  c->dk = q->dim[r - 1];
  int64_t s[NX_MAX_RANK];
  memcpy(s, c->ext, sizeof(int64_t) * (size_t)c->lead);
  if (!paged) {
    if (k->rank != r || v->rank != r) return NX_SHAPE;
    c->hk = k->dim[r - 3];
    c->tk = k->dim[r - 2];
    c->dv = v->dim[r - 1];
    s[r - 3] = c->hk;
    s[r - 2] = c->tk;
    s[r - 1] = c->dk;
    if (!same_shape(k, r, s)) return NX_SHAPE;
    s[r - 1] = c->dv;
    if (!same_shape(v, r, s)) return NX_SHAPE;
  } else {
    const nx_array *t = c->table;
    if (k->rank != 4 || v->rank != 4 || t->rank != r - 2) return NX_SHAPE;
    c->pages = k->dim[0];
    c->hk = k->dim[1];
    c->page = k->dim[2];
    c->dv = v->dim[3];
    int64_t pk[4] = {c->pages, c->hk, c->page, c->dk};
    int64_t pv[4] = {c->pages, c->hk, c->page, c->dv};
    if (!same_shape(k, 4, pk) || !same_shape(v, 4, pv)) return NX_SHAPE;
    c->max_pages = t->dim[r - 3];
    s[r - 3] = c->max_pages;
    if (!same_shape(t, r - 2, s)) return NX_SHAPE;
    if (c->page > 0 && c->max_pages > NX_MAX_NUMEL / c->page) return NX_SHAPE;
    c->tk = c->max_pages * c->page;
  }
  if ((c->hk == 0 && c->hq != 0) || (c->hk > 0 && c->hq % c->hk != 0))
    return NX_SHAPE;
  c->g = c->hk == 0 ? 1 : c->hq / c->hk;
  if (c->lengths != NULL && !same_shape(c->lengths, c->lead, c->ext))
    return NX_SHAPE;
  s[r - 3] = c->hq;
  s[r - 2] = c->tq;
  s[r - 1] = c->dv;
  if (!same_shape(&a[O], r, s)) return NX_SHAPE;
  if (!same_shape(&a[LSE], r - 1, s)) return NX_SHAPE;
  return NX_OK;
}

/* Answers [e] after the door's claims end, raising Out_of_memory where
   [failed]. */
static value finish(value owner, int n, nx_array *a, int e, int failed) {
  nx_done(n, a);
  release(owner);
  if (failed) caml_raise_out_of_memory();
  return Val_int(e);
}

value nx_cpu_attention(value vp, value vranges, value vscale, value vo,
                       value vlse, value vlengths, value vq, value vkv,
                       value vloads) {
  CAMLparam5(vp, vranges, vscale, vo, vlse);
  CAMLxparam4(vlengths, vq, vkv, vloads);
  CAMLlocal2(owner, vr);
  /* q is float16, bfloat16, float32 or float64, the accumulator, lse's
     dtype, float32 or float64 and not narrower. */
  int dt = nx_array_dtype(vq), acc = nx_array_dtype(vlse);
  if (dt != NX_FLOAT16 && dt != NX_BFLOAT16 && dt != NX_FLOAT32 &&
      dt != NX_FLOAT64)
    CAMLreturn(Val_int(NX_DTYPE));
  if ((acc != NX_FLOAT32 && acc != NX_FLOAT64) ||
      (dt == NX_FLOAT64 && acc != NX_FLOAT64))
    CAMLreturn(Val_int(NX_DTYPE));
  int nl = (int)Wosize_val(vloads);
  int paged = Tag_val(vkv) == 1, lengths = Is_block(vlengths);
  int nins = (int)Wosize_val(Field(vp, NX_PROG_INS));
  if (nins != nl + 1 || Wosize_val(Field(vp, NX_PROG_OUTS)) != 2)
    CAMLreturn(Val_int(NX_SHAPE));
  att c = {0};
  c.mask = !is_default(vp, 0);
  c.score = !is_default(vp, 1);
  c.scale = Double_val(vscale);
  c.dt = dt;
  c.acc = acc;
  /* o is the accumulator's dtype or q's. */
  c.odt = nx_array_dtype(vo);
  if (c.odt != acc && c.odt != dt) CAMLreturn(Val_int(NX_DTYPE));
  c.w = nx_cpu_width(c.odt);
  /* What the call holds, owned by a custom block: the program's copy, the
     descriptors of more operands than the stack's table holds, the
     program's loads and the places' tables. */
  int n = 5 + paged + lengths + nl;
  owner = caml_alloc_custom(&held_ops, sizeof(held), 0, 1);
  *(held *)Data_custom_val(owner) = (held){NULL, NULL, NULL};
  held *h = Data_custom_val(owner);
  nx_cpu_prog *g = NULL;
  if (c.mask || c.score) {
    g = nx_cpu_prog_copy(&vp);
    if (g == NULL) caml_raise_out_of_memory();
    h->g = g;
    if (nx_cpu_prog_in(g, 0) != acc || nx_cpu_prog_out(g, 0) != NX_BOOL ||
        nx_cpu_prog_out(g, 1) != acc) {
      release(owner);
      CAMLreturn(Val_int(NX_DTYPE));
    }
    /* Visibility is decided before the scores exist: output 0 reads no
       score. */
    uint8_t axes[NX_MAX_RANK] = {0};
    uint8_t local[64], *reads = nins <= 64 ? local : calloc((size_t)nins, 1);
    if (reads == NULL) {
      release(owner);
      caml_raise_out_of_memory();
    }
    memset(reads, 0, (size_t)nins);
    nx_cpu_prog_reads(g, 0, reads, axes);
    int scored = reads[0];
    if (reads != local) free(reads);
    if (scored) {
      release(owner);
      CAMLreturn(Val_int(NX_SHAPE));
    }
  }
  /* The mask's ranges, where it has them. */
  nx_cpu_prog *rg = NULL;
  if (c.mask && Is_block(vranges)) {
    vr = Field(vranges, 0);
    rg = nx_cpu_prog_copy(&vr);
    if (rg == NULL) {
      release(owner);
      caml_raise_out_of_memory();
    }
    ((held *)Data_custom_val(owner))->ranges = rg;
  }

  size_t tables = (size_t)n * sizeof(nx_array) +
                  (size_t)(nl + 1) * sizeof(nx_array) +
                  (size_t)(nl + 3) * NX_MAX_RANK * sizeof(int64_t) +
                  (size_t)(nl + 2) * sizeof(int64_t);
  uint8_t *mem = n + nl > 64 ? nx_alloc_released(tables) : malloc(tables);
  if (mem == NULL) {
    release(owner);
    caml_raise_out_of_memory();
  }
  h->tables = mem;
  nx_array *a = (nx_array *)mem, *ins = a + n;
  int64_t(*st)[NX_MAX_RANK] = (int64_t(*)[NX_MAX_RANK])(ins + nl + 1);
  int64_t *first = (int64_t *)(st + nl + 3);
  operands ops = {&vo, &vlse, &vq, &vkv,   &vlengths, &vloads,
                  dt,  acc,   c.odt, paged, lengths,   g};
  int e = nx_read_each(n, operand, &ops, a);
  if (e) {
    release(owner);
    CAMLreturn(Val_int(e));
  }
  c.a = a;
  int at = 5 + paged;
  c.table = paged ? &a[5] : NULL;
  c.lengths = lengths ? &a[at] : NULL;
  const nx_array *loads = &a[at + lengths];
  if ((e = extents(&c, a, paged)))
    CAMLreturn(finish(owner, n, a, e, 0));
  {
    int64_t s[NX_MAX_RANK];
    memcpy(s, c.ext, sizeof(int64_t) * (size_t)c.lead);
    s[c.rank - 3] = c.hq;
    s[c.rank - 2] = c.tq;
    s[c.rank - 1] = c.tk;
    for (int i = 0; i < nl; i++)
      if (!same_shape(&loads[i], c.rank, s))
        CAMLreturn(finish(owner, n, a, NX_SHAPE, 0));
  }
  int64_t leads = 1;
  for (int i = 0; i < c.lead; i++) leads *= c.ext[i];
  int64_t rows = leads * c.hq * c.tq;
  if (rows == 0) CAMLreturn(finish(owner, n, a, NX_OK, 0));
  if (g != NULL && (e = bind(&c, g, loads, nl, ins, st, first)))
    CAMLreturn(finish(owner, n, a, e < 0 ? NX_OK : e, e < 0));
  c.prog = g;
  int r = c.rank;
  c.k_in = dt == acc && step(&a[K], paged ? 3 : r - 1) == 1;
  c.v_in = dt == acc && step(&a[V], paged ? 3 : r - 1) == 1;
  c.kd = step(&a[K], paged ? 3 : r - 1);
  c.vd = step(&a[V], paged ? 3 : r - 1);
  c.gemv = nx_cpu_table->gemv[acc];
  const nx_cpu_micro *mk = pv_kernel(acc);
  c.pv = mk->f;
  c.nr = mk->nr;
  /* Units of at most ROWS rows: a group's heads, then rows of each. */
  c.hg = c.g < ROWS ? c.g : ROWS;
  c.tr = ROWS / c.hg;
  c.chunks = (c.g + c.hg - 1) / c.hg;
  c.tiles = (c.tq + c.tr - 1) / c.tr;
  int64_t units = leads * c.hk * c.chunks * c.tiles;
  atomic_init(&c.failed, 0);
  /* Its cost: each pair's products at the core's speed, the operands'
     bytes moved once, and each unit's classes of its blocks. */
  c.blocks = (c.tk + KEYS - 1) / KEYS;
  double pairs = (double)rows * (double)c.tk;
  double cost = 2 * pairs * (double)(c.dk + c.dv) / nx_cpu_core_speed(acc);
  double bytes = 0;
  for (int i = 0; i < n; i++) bytes += nx_cpu_bytes(&a[i]);
  cost += NX_CPU_STRIDED_COST * bytes;
  int classed = rg != NULL && c.blocks > 0 &&
                nx_cpu_prog_ins(rg) == 2 * (nl + 1) + 2 * r &&
                nx_cpu_prog_outs(rg) == 2 &&
                nx_cpu_prog_out(rg, 0) == NX_BOOL &&
                nx_cpu_prog_out(rg, 1) == NX_BOOL;
  if (classed)
    cost += 2 * (double)units * (double)c.blocks * nx_cpu_prog_cost(rg);
  /* A mask every lead index and head share is classed once per tile and
     block, at most its pairs at one lead index and head run. */
  int share = g != NULL && c.mask && units > c.tiles && c.blocks > 0 &&
              shared_mask(&c, g, loads, nl);
  double pairs1 = (double)c.tq * (double)c.tk;
  if (share) cost += pairs1 * nx_cpu_prog_cost(g);
  int released = nx_cpu_release(cost);
  /* Ranges whose tables cannot be held leave every block mixed, and a
     shared table that cannot be held leaves each unit its ranges. */
  void *classes = classed ? bind_ranges(&c, rg, loads, nl) : NULL;
  c.ranges = classes != NULL ? rg : NULL;
  c.cls = share ? malloc((size_t)(c.tiles * c.blocks)) : NULL;
  if (c.cls != NULL)
    nx_cpu_job(c.tiles, (int64_t)pairs1, pairs1 * nx_cpu_prog_cost(g), shared,
               &c);
  nx_cpu_job(units, (int64_t)bytes, cost,
             acc == NX_FLOAT64 ? body_f64 : body_f32, &c);
  nx_cpu_reacquire(released);
  free(classes);
  free(c.cls);
  int failed = atomic_load(&c.failed) || (g != NULL && nx_cpu_prog_failed(g));
  CAMLreturn(finish(owner, n, a, NX_OK, failed));
}

value nx_cpu_attention_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_attention(argv[0], argv[1], argv[2], argv[3], argv[4],
                          argv[5], argv[6], argv[7], argv[8]);
}
