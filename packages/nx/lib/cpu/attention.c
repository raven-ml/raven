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

   A mask whose output 0 reads no lead index, head or score is classified
   once per tile of rows and block, at lead index 0 and head 0 (classes):
   a block it hides or shows whole is skipped or read without running it
   for each unit.

   At the end o = acc / l, rounded once to q's dtype, and lse = m + log l;
   a row of l = 0 has o = +0 and lse = -inf. Scores, weights, sums and
   accumulators are float32; float16 and bfloat16 operands are widened on
   load. Float32 values whose dimension steps one element are read in
   place; keys are staged transposed, or for few rows read in place where
   float32 of unit steps. */

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
  const float *v;
  int64_t vs;
  int64_t ka, kk, va, vk;
} run;

/* Rows from which a unit's products run on the chain kernel's tiles; fewer
   take a gemv of each row. Both add in chain order. */
#define CHAIN_ROWS 6

/* A block's class for every row of a tile under a mask that reads no lead
   or head (summary): its program's output 0 false at every pair, true at
   every pair, or both. */
enum { HIDDEN, SHOWN, MIXED };

typedef struct {
  const nx_array *a; /* the door's descriptors */
  const nx_array *table, *lengths;
  int rank, lead; /* q's rank, its lead axes */
  int64_t ext[NX_MAX_RANK];
  int64_t hq, hk, g, tq, tk, dk, dv;
  int64_t pages, page, max_pages;
  int dt, w; /* q's dtype and its element's bytes */
  float scale;
  /* Units: heads of a group and rows of a head each holds. */
  int64_t hg, tr, chunks, tiles;
  /* The program, where an output is not the default: [mask] where output 0
     is not [true], [score] where output 1 is not the score itself. */
  const nx_cpu_prog *prog;
  nx_cpu_places pl;
  int mask, score;
  /* Keys and values read in place. */
  int k_in, v_in;
  /* Classes of a tile's blocks (enum above), where every lead and head's
     visibility is one, else NULL. */
  const uint8_t *cls;
  int64_t blocks;
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

/* A worker's scratch, by unit. */
typedef struct {
  float *q;     /* ROWS × dk */
  float *s;     /* ROWS × KEYS: scores, then weights */
  float *t;     /* ROWS × KEYS: a score program's output */
  float *acc;   /* ROWS × dv */
  float *save;  /* ROWS × dv: rows the block leaves as they were */
  float *kt;    /* dk × KEYS: the keys transposed */
  float *vf;    /* KEYS × dv */
  float m[ROWS], l[ROWS];
  uint8_t *vis; /* ROWS × KEYS, row r's at r·nk */
  uint8_t sees[ROWS], hides[ROWS], picks[ROWS];
  uint8_t bad[KEYS]; /* a value not finite */
  run runs[KEYS];
} scratch;

static int64_t scratch_bytes(const att *c) {
  int64_t f = ROWS * c->dk + 2 * ROWS * KEYS + 2 * ROWS * c->dv +
              KEYS * c->dk + KEYS * c->dv;
  return (int64_t)sizeof(scratch) + 4 * f + ROWS * KEYS + 10 * 64;
}

static uint8_t *align64(uint8_t *p) {
  return p + (64 - (uintptr_t)p % 64) % 64;
}

static void lay_out(const att *c, scratch *x) {
  uint8_t *p = align64((uint8_t *)(x + 1));
#define TAKE(f, n)                         \
  x->f = (void *)p;                        \
  p = align64(p + (int64_t)(n) * sizeof(float));
  TAKE(q, ROWS * c->dk);
  TAKE(s, ROWS * KEYS);
  TAKE(t, ROWS * KEYS);
  TAKE(acc, ROWS * c->dv);
  TAKE(save, ROWS * c->dv);
  TAKE(kt, KEYS * c->dk);
  TAKE(vf, KEYS * c->dv);
#undef TAKE
  x->vis = p;
}

/* Whether any of the [n] float32 at [x] is not finite. */
static int any_bad(const float *x, int64_t n) {
  const uint32_t *b = (const uint32_t *)x;
  uint32_t e = 0x7f800000u, any = 0;
  for (int64_t i = 0; i < n; i++) any |= (b[i] & e) == e;
  return any != 0;
}

/* The block of [nk] keys from [j0] of lead index [b] and key head [kh] as
   runs into x->runs, each key's presence into [present]: a key whose page
   table entry is out of range is absent. Answers the runs' count. */
static int runs_of(const att *c, scratch *x, int64_t b, int64_t kh,
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
    run *u = &x->runs[n++];
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
      run *u = &x->runs[n++];
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

/* The values of the runs [x->runs], [n] of them: in place, or staged into
   vf at their place in the block. */
static void values_of(const att *c, scratch *x, int n) {
  const nx_array *v = &c->a[V];
  for (int g = 0; g < n; g++) {
    run *u = &x->runs[g];
    if (c->v_in) {
      u->v = (const float *)v->base + u->va;
      u->vs = u->vk;
    } else {
      nx_cpu_stage_as(v, u->va, c->vd, u->vk, c->dv, u->n, NX_FLOAT32,
                      (uint8_t *)(x->vf + u->j * c->dv), c->dv);
      u->v = x->vf + u->j * c->dv;
      u->vs = c->dv;
    }
  }
}

/* One row's scores against [n] keys at [k], key i's elements at k + i·ks,
   each added in increasing dimension from +0, each fused, as the chain
   kernel adds them: on x86-64 eight keys a gather of one dimension. */
#if defined(__AVX2__) && defined(__FMA__)
#define ROW_SCORES 1
static void row_scores(const float *q, const float *k, int64_t ks, int n,
                       int64_t dk, float *s) {
  int j = 0;
  if (ks <= INT32_MAX / 8) {
    __m256i idx = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                     _mm256_set1_epi32((int32_t)ks));
    for (; j + 8 <= n; j += 8) {
      const float *kj = k + j * ks;
      __m256 a = _mm256_setzero_ps();
      for (int64_t d = 0; d < dk; d++)
        a = _mm256_fmadd_ps(_mm256_set1_ps(q[d]),
                            _mm256_i32gather_ps(kj + d, idx, 4), a);
      _mm256_storeu_ps(s + j, a);
    }
  }
  for (; j < n; j++) {
    float a = 0;
    for (int64_t d = 0; d < dk; d++) a = fmaf(q[d], k[j * ks + d], a);
    s[j] = a;
  }
}
#else
#define ROW_SCORES 0
static void row_scores(const float *q, const float *k, int64_t ks, int n,
                       int64_t dk, float *s) {
  (void)q, (void)k, (void)ks, (void)n, (void)dk, (void)s;
}
#endif

/* Row r's update by its block's [nk] scores at [s], hidden ones -inf:
   its maximum, its weights into [s] and its sum, and its accumulator
   rescaled. A block whose visible scores are all -inf gives weights 0
   and leaves the state as it was. */
typedef float v8f __attribute__((vector_size(32)));
typedef int32_t v8i __attribute__((vector_size(32)));

static inline v8f load8(const float *p) {
  v8f v;
  memcpy(&v, p, sizeof v);
  return v;
}

/* The greatest of [m] and the [n] scores at [s], a NaN where one is. */
static float greatest(const float *s, int n, float m) {
  v8f m8 = {m, m, m, m, m, m, m, m};
  v8i nan = {0, 0, 0, 0, 0, 0, 0, 0};
  int j = 0;
  for (; j + 8 <= n; j += 8) {
    v8f y = load8(s + j);
    v8i gt = y > m8;
    m8 = (v8f)(((v8i)y & gt) | ((v8i)m8 & ~gt));
    nan |= y != y;
  }
  float mx = m;
  int any = 0;
  for (int i = 0; i < 8; i++) {
    mx = m8[i] > mx ? m8[i] : mx;
    any |= nan[i];
  }
  for (; j < n; j++) {
    mx = s[j] > mx ? s[j] : mx;
    any |= s[j] != s[j];
  }
  return any ? NAN : mx;
}

/* The [n] weights at [s] summed term j into lane j mod 8, the lanes then
   summed by halves: one order on every target. */
static float lanes_sum(const float *s, int n) {
  v8f l8 = {0, 0, 0, 0, 0, 0, 0, 0};
  int j = 0;
  for (; j + 8 <= n; j += 8) l8 += load8(s + j);
  float lane[8];
  memcpy(lane, &l8, sizeof lane);
  for (; j < n; j++) lane[j % 8] += s[j];
  for (int h = 4; h > 0; h /= 2)
    for (int i = 0; i < h; i++) lane[i] += lane[i + h];
  return lane[0];
}

static void update(const att *c, scratch *x, int r, int nk) {
  float *s = x->s + r * KEYS;
  float m = x->m[r], mx = greatest(s, nk, m);
  if (mx == -INFINITY) {
    memset(s, 0, sizeof(float) * (size_t)nk);
    return;
  }
  float alpha = nx_exp_f32(m - mx);
  for (int j = 0; j < nk; j++) s[j] = nx_exp_f32(s[j] - mx);
  x->l[r] = x->l[r] * alpha + lanes_sum(s, nk);
  x->m[r] = mx;
  if (alpha != 1.f) {
    float *acc = x->acc + r * c->dv;
    for (int64_t e = 0; e < c->dv; e++) acc[e] *= alpha;
  }
}

/* Row r's products of its visible pairs alone, as the kernel adds them:
   in increasing key, each fused into its sum. */
static void picked(const att *c, scratch *x, int r, int nr, int nk) {
  float *acc = x->acc + r * c->dv;
  const float *p = x->s + r * KEYS;
  const uint8_t *vis = x->vis + r * nk;
  for (int u = 0; u < nr; u++) {
    const run *g = &x->runs[u];
    for (int i = 0; i < g->n; i++) {
      if (!vis[g->j + i]) continue;
      float pj = p[g->j + i];
      const float *v = g->v + i * g->vs;
      for (int64_t e = 0; e < c->dv; e++) acc[e] = fmaf(pj, v[e], acc[e]);
    }
  }
}

/* The unit [u]. */
static void unit(att *c, scratch *x, int64_t u) {
  int64_t tile = u % c->tiles, chunk = u / c->tiles % c->chunks;
  int64_t kh = u / c->tiles / c->chunks % c->hk;
  int64_t b = u / c->tiles / c->chunks / c->hk;
  int64_t h0 = kh * c->g + chunk * c->hg, i0 = tile * c->tr;
  int nh = (int)(c->g - chunk * c->hg < c->hg ? c->g - chunk * c->hg : c->hg);
  int nt = (int)(c->tq - i0 < c->tr ? c->tq - i0 : c->tr);
  int rows = nh * nt, r = c->rank;
  int64_t dk = c->dk, dv = c->dv;
  const nx_array *q = &c->a[Q];
  /* The rows, head by head, staged into float32. */
  int64_t qa = lead_at(q, c->lead, c->ext, b) + i0 * step(q, r - 2);
  for (int hh = 0; hh < nh; hh++)
    nx_cpu_stage_as(q, qa + (h0 + hh) * step(q, r - 3), step(q, r - 1),
                    step(q, r - 2), dk, nt, NX_FLOAT32,
                    (uint8_t *)(x->q + hh * nt * dk), dk);
  for (int i = 0; i < rows; i++) {
    x->m[i] = -INFINITY;
    x->l[i] = 0;
  }
  memset(x->acc, 0, sizeof(float) * (size_t)(rows * dv));
  uint8_t present[KEYS];
  int64_t len = length_of(c, b);
  for (int64_t j0 = 0; j0 < len; j0 += KEYS) {
    int nk = (int)(len - j0 < KEYS ? len - j0 : KEYS);
    int cls = c->cls != NULL ? c->cls[tile * c->blocks + j0 / KEYS] : MIXED;
    if (cls == HIDDEN) continue;
    int mask = c->mask && cls == MIXED;
    int nr = runs_of(c, x, b, kh, j0, nk, present);
    if (nr == 0) continue;
    int all = 1;
    for (int j = 0; j < nk; j++) all &= present[j];
    /* Visibility: the mask program's output 0, head by head, of keys the
       kernel reads. */
    if (mask) {
      for (int hh = 0; hh < nh; hh++) {
        int64_t p = (((b * c->hq + h0 + hh) * c->tq + i0) * c->tk) + j0;
        nx_cpu_prog_terms(c->prog, &c->pl, 0, p, 1, nk, c->tk, nt,
                          x->vis + hh * nt * nk);
      }
      if (!all)
        for (int i = 0; i < rows; i++)
          for (int j = 0; j < nk; j++) x->vis[i * nk + j] &= present[j];
    } else if (!all)
      for (int i = 0; i < rows; i++)
        memcpy(x->vis + i * nk, present, (size_t)nk);
    int any = 0, hidden = 0;
    if (mask || !all)
      for (int i = 0; i < rows; i++) {
        const uint8_t *vi = x->vis + i * nk;
        int n = 0;
        for (int j = 0; j < nk; j++) n += vi[j] != 0;
        x->sees[i] = n > 0;
        x->hides[i] = n > 0 && n < nk;
        any |= n > 0;
        hidden |= x->hides[i];
      }
    else {
      memset(x->sees, 1, (size_t)rows);
      memset(x->hides, 0, (size_t)rows);
      any = 1;
    }
    if (!any) continue;
    /* Scores of the rows that see a key; an absent key's, 0 until the
       mask hides it. */
    if (!all || dk == 0)
      for (int i = 0; i < rows; i++)
        memset(x->s + i * KEYS, 0, sizeof(float) * (size_t)nk);
    /* Each score adds its products in increasing dimension from +0, each
       fused: on the chain kernel's tiles over the keys staged transposed,
       or, for few rows, each row against the keys in rows. */
    for (int g = 0; dk > 0 && g < nr; g++) {
      const run *v = &x->runs[g];
      if (rows < CHAIN_ROWS && ROW_SCORES) {
        const float *k;
        int64_t ks;
        if (c->k_in) {
          k = (const float *)c->a[K].base + v->ka;
          ks = v->kk;
        } else {
          nx_cpu_stage_as(&c->a[K], v->ka, c->kd, v->kk, dk, v->n, NX_FLOAT32,
                          (uint8_t *)x->kt, dk);
          k = x->kt;
          ks = dk;
        }
        for (int i = 0; i < rows; i++)
          if (x->sees[i])
            row_scores(x->q + i * dk, k, ks, v->n, dk, x->s + i * KEYS + v->j);
        continue;
      }
      nx_cpu_stage_as(&c->a[K], v->ka, v->kk, c->kd, v->n, dk, NX_FLOAT32,
                      (uint8_t *)(x->kt + v->j), KEYS);
      if (rows >= CHAIN_ROWS)
        for (int j = 0; j < v->n; j += c->nr) {
          int n = v->n - j < c->nr ? v->n - j : c->nr;
          c->pv(rows, n, dk, x->q, dk, 1, x->kt + v->j + j, KEYS,
                x->s + v->j + j, KEYS, NX_CPU_FROM_ZERO);
        }
      else
        for (int i = 0; i < rows; i++) {
          if (!x->sees[i]) continue;
          float *y = x->s + i * KEYS + v->j;
          memset(y, 0, sizeof(float) * (size_t)v->n);
          c->gemv(v->n, dk, x->kt + v->j, KEYS, x->q + i * dk, y);
        }
    }
    for (int i = 0; i < rows; i++) {
      float *s = x->s + i * KEYS;
      for (int j = 0; j < nk; j++) s[j] *= c->scale;
    }
    /* The score program's output 1 over the scores, head by head. */
    if (c->score)
      for (int hh = 0; hh < nh; hh++) {
        int64_t p = (((b * c->hq + h0 + hh) * c->tq + i0) * c->tk) + j0;
        float *s = x->s + hh * nt * KEYS;
        nx_cpu_prog_tile(c->prog, &c->pl, 1, p, nk, c->tk, nt,
                         (uint8_t *)x->t, 0, s, KEYS);
        for (int i = 0; i < nt; i++)
          if (x->sees[hh * nt + i])
            memcpy(s + i * KEYS, x->t + i * nk, sizeof(float) * (size_t)nk);
      }
    values_of(c, x, nr);
    if (mask || !all)
      for (int i = 0; i < rows; i++) {
        if (!x->hides[i]) continue;
        float *s = x->s + i * KEYS;
        const uint8_t *vi = x->vis + i * nk;
        for (int j = 0; j < nk; j++) s[j] = vi[j] ? s[j] : -INFINITY;
      }
    /* A value not finite at a key hidden from a row sends the row's
       products through its visible pairs alone. */
    int picking = 0;
    if (hidden) {
      for (int j = 0; j < nk; j++) x->bad[j] = 0;
      for (int g = 0; g < nr; g++) {
        const run *v = &x->runs[g];
        for (int j = 0; j < v->n; j++)
          x->bad[v->j + j] = (uint8_t)any_bad(v->v + j * v->vs, dv);
      }
      for (int i = 0; i < rows; i++) {
        x->picks[i] = 0;
        if (!x->hides[i]) continue;
        const uint8_t *vi = x->vis + i * nk;
        for (int j = 0; j < nk; j++) x->picks[i] |= !vi[j] && x->bad[j];
        picking |= x->picks[i];
      }
    }
    for (int i = 0; i < rows; i++) {
      if (x->sees[i]) update(c, x, i, nk);
      else memset(x->s + i * KEYS, 0, sizeof(float) * (size_t)nk);
    }
    if (rows < CHAIN_ROWS) {
      for (int i = 0; i < rows; i++) {
        if (!x->sees[i]) continue;
        if (picking && x->picks[i]) {
          picked(c, x, i, nr, nk);
          continue;
        }
        for (int g = 0; g < nr; g++) {
          const run *v = &x->runs[g];
          c->gemv(dv, v->n, v->v, v->vs, x->s + i * KEYS + v->j,
                  x->acc + i * dv);
        }
      }
      continue;
    }
    /* Rows the products must leave as they are: those that see no key,
       and those whose visible pairs are added apart. */
    for (int i = 0; i < rows; i++)
      if (!x->sees[i] || (picking && x->picks[i]))
        memcpy(x->save + i * dv, x->acc + i * dv, sizeof(float) * (size_t)dv);
    for (int64_t e = 0; e < dv; e += c->nr) {
      int n = (int)(dv - e < c->nr ? dv - e : c->nr);
      for (int g = 0; g < nr; g++) {
        const run *v = &x->runs[g];
        c->pv(rows, n, v->n, x->s + v->j, KEYS, 1, v->v + e, v->vs,
              x->acc + e, dv, NX_CPU_FROM_TILE);
      }
    }
    for (int i = 0; i < rows; i++) {
      int pick = picking && x->picks[i];
      if (!x->sees[i] || pick)
        memcpy(x->acc + i * dv, x->save + i * dv, sizeof(float) * (size_t)dv);
      if (pick) picked(c, x, i, nr, nk);
    }
  }
  /* o = acc / l, rounded once to q's dtype, and lse. */
  nx_cpu_run store = nx_cpu_convert(NX_FLOAT32, c->dt);
  const nx_array *o = &c->a[O], *lse = &c->a[LSE];
  for (int hh = 0; hh < nh; hh++)
    for (int i = 0; i < nt; i++) {
      int row = hh * nt + i;
      int64_t at = ((b * c->hq + h0 + hh) * c->tq + i0 + i);
      float *acc = x->acc + row * dv;
      float l = x->l[row], m = x->m[row];
      float y;
      if (l == 0) {
        memset(acc, 0, sizeof(float) * (size_t)dv);
        y = -INFINITY;
      } else {
        for (int64_t e = 0; e < dv; e++) acc[e] = acc[e] / l;
        y = m == INFINITY ? m : m + nx_log_f32(l);
      }
      store(acc, o->base + (o->offset + at * dv) * c->w, dv);
      ((float *)lse->base)[lse->offset + at] = y;
    }
}

/* The summary's classes of tiles [lo, hi), from the program at lead index
   and head 0. */
static void classes(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  att *c = ctx;
  uint8_t vis[ROWS * KEYS];
  uint8_t *cls = (uint8_t *)c->cls;
  for (int64_t t = lo; t < hi; t++) {
    int64_t i0 = t * c->tr;
    int nt = (int)(c->tq - i0 < c->tr ? c->tq - i0 : c->tr);
    for (int64_t k = 0; k < c->blocks; k++) {
      int64_t j0 = k * KEYS;
      int nk = (int)(c->tk - j0 < KEYS ? c->tk - j0 : KEYS);
      nx_cpu_prog_terms(c->prog, &c->pl, 0, i0 * c->tk + j0, 1, nk, c->tk, nt,
                        vis);
      int n = 0;
      for (int j = 0; j < nt * nk; j++) n += vis[j] != 0;
      cls[t * c->blocks + k] = n == 0 ? HIDDEN : n == nt * nk ? SHOWN : MIXED;
    }
  }
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

static void body(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  att *c = ctx;
  uint8_t *mem = nx_cpu_take(scratch_bytes(c));
  if (mem == NULL) {
    atomic_store(&c->failed, 1);
    return;
  }
  scratch *x = (scratch *)align64(mem);
  lay_out(c, x);
  for (int64_t u = lo; u < hi; u++) unit(c, x, u);
  nx_cpu_give(mem);
}

/* The chain order kernel that reads a in place and n columns of b in
   place: the table's, its other's, or the base table's. */
static const nx_cpu_micro *pv_kernel(void) {
  const nx_cpu_gemm *g = &nx_cpu_table->gemm[NX_FLOAT32];
  for (int i = 0; i < 2 && g != NULL; i++, g = g->other)
    if (g->kernel.f != NULL && g->kernel.a == NX_CPU_A_ELEMENTS &&
        g->kernel.b == NX_CPU_B_COLUMNS && g->enter == NULL)
      return &g->kernel;
  return &nx_cpu_base.gemm[NX_FLOAT32].kernel;
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

/* The program [g] bound to the loads at [a], [nl] of them, over the pairs'
   loop of [c]: its operand 0 a stand-in, every element of which the
   kernel's tiles give (nx_cpu_prog_tile); and [c]'s places of the loads
   (cpu.h, Places), [st] holding nl + 3 rows, [first] nl + 2 entries. */
static int bind(att *c, nx_cpu_prog *g, const nx_array *a, int nl,
                nx_array *ins, int64_t (*st)[NX_MAX_RANK], int64_t *first) {
  static const float zero = 0;
  int r = c->rank;
  int64_t shape[NX_MAX_RANK];
  memcpy(shape, c->ext, sizeof(int64_t) * (size_t)(r - 3));
  shape[r - 3] = c->hq;
  shape[r - 2] = c->tq;
  shape[r - 1] = c->tk;
  ins[0] = (nx_array){.base = (uint8_t *)&zero,
                      .dtype = NX_FLOAT32,
                      .bits = 32,
                      .rank = r};
  for (int i = 0; i < r; i++) {
    ins[0].dim[i] = shape[i];
    ins[0].dim[r + i] = 0;
  }
  for (int k = 0; k < nl; k++) ins[1 + k] = a[k];
  int index, e = nx_cpu_prog_bind(g, ins, r, shape, &index);
  if (e) return e;
  /* The virtual operand, C-contiguous, the stand-in and the loads,
     coalesced; the index steps as the virtual operand. */
  int64_t past = 1;
  for (int i = r - 1; i >= 0; i--) {
    c->pl.ext[i] = shape[i];
    st[0][i] = past;
    past *= shape[i];
    for (int k = 0; k <= nl; k++) st[1 + k][i] = ins[k].dim[r + i];
  }
  c->pl.rank = nx_coalesce_dims(2 + nl, r, c->pl.ext, st);
  memcpy(c->pl.vs, st[0], sizeof(int64_t) * (size_t)c->pl.rank);
  memcpy(st[2 + nl], st[0], sizeof(int64_t) * (size_t)c->pl.rank);
  for (int k = 0; k <= nl; k++) first[k] = ins[k].offset;
  first[nl + 1] = 0;
  c->pl.n = nl + 1 + index;
  c->pl.step = st + 1;
  c->pl.first = first;
  return NX_OK;
}

/* The external */

/* What a call holds on the C heap: the program's copy and the tables of
   its loads. A custom block owns them, so that a raise of the door leaves
   them to its finalizer. */
typedef struct {
  nx_cpu_prog *g;
  void *tables;
} held;

static void release(value owner) {
  held *h = Data_custom_val(owner);
  nx_cpu_prog_free(h->g);
  free(h->tables);
  h->g = NULL;
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
  int dt, paged, lengths;
  const nx_cpu_prog *g;
} operands;

static int operand(void *ctx, int k, nx_operand *in) {
  const operands *o = ctx;
  int at = 5 + o->paged;
  if (k == O) *in = (nx_operand){*o->vo, o->dt, 1};
  else if (k == LSE) *in = (nx_operand){*o->vlse, NX_FLOAT32, 1};
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

value nx_cpu_attention(value vp, value vscale, value vo, value vlse,
                       value vlengths, value vq, value vkv, value vloads) {
  CAMLparam5(vp, vscale, vo, vlse, vlengths);
  CAMLxparam3(vq, vkv, vloads);
  CAMLlocal1(owner);
  int dt = nx_array_dtype(vq);
  if (dt != NX_FLOAT16 && dt != NX_BFLOAT16 && dt != NX_FLOAT32)
    CAMLreturn(Val_int(NX_DTYPE));
  int nl = (int)Wosize_val(vloads);
  int paged = Tag_val(vkv) == 1, lengths = Is_block(vlengths);
  int nins = (int)Wosize_val(Field(vp, NX_PROG_INS));
  if (nins != nl + 1 || Wosize_val(Field(vp, NX_PROG_OUTS)) != 2)
    CAMLreturn(Val_int(NX_SHAPE));
  att c = {0};
  c.mask = !is_default(vp, 0);
  c.score = !is_default(vp, 1);
  c.scale = (float)Double_val(vscale);
  c.dt = dt;
  c.w = dt == NX_FLOAT32 ? 4 : 2;
  /* What the call holds, owned by a custom block: the program's copy, the
     descriptors of more operands than the stack's table holds, the
     program's loads and the places' tables. */
  int n = 5 + paged + lengths + nl;
  owner = caml_alloc_custom(&held_ops, sizeof(held), 0, 1);
  *(held *)Data_custom_val(owner) = (held){NULL, NULL};
  held *h = Data_custom_val(owner);
  nx_cpu_prog *g = NULL;
  if (c.mask || c.score) {
    g = nx_cpu_prog_copy(&vp);
    if (g == NULL) caml_raise_out_of_memory();
    h->g = g;
    if (nx_cpu_prog_in(g, 0) != NX_FLOAT32 ||
        nx_cpu_prog_out(g, 0) != NX_BOOL ||
        nx_cpu_prog_out(g, 1) != NX_FLOAT32) {
      release(owner);
      CAMLreturn(Val_int(NX_DTYPE));
    }
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
  operands ops = {&vo, &vlse, &vq, &vkv, &vlengths, &vloads,
                  dt,  paged, lengths, g};
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
  c.k_in = dt == NX_FLOAT32 && step(&a[K], paged ? 3 : r - 1) == 1;
  c.v_in = dt == NX_FLOAT32 && step(&a[V], paged ? 3 : r - 1) == 1;
  c.kd = step(&a[K], paged ? 3 : r - 1);
  c.vd = step(&a[V], paged ? 3 : r - 1);
  c.gemv = nx_cpu_table->gemv[NX_FLOAT32];
  const nx_cpu_micro *mk = pv_kernel();
  c.pv = mk->f;
  c.nr = mk->nr;
  /* Units of at most ROWS rows: a group's heads, then rows of each. */
  c.hg = c.g < ROWS ? c.g : ROWS;
  c.tr = ROWS / c.hg;
  c.chunks = (c.g + c.hg - 1) / c.hg;
  c.tiles = (c.tq + c.tr - 1) / c.tr;
  int64_t units = leads * c.hk * c.chunks * c.tiles;
  atomic_init(&c.failed, 0);
  /* A mask that every lead index and head share is classified once per
     tile and block; an allocation that fails leaves every block to be
     evaluated. */
  uint8_t *cls = NULL;
  c.blocks = (c.tk + KEYS - 1) / KEYS;
  if (g != NULL && c.mask && units > c.tiles && c.blocks > 0 &&
      shared_mask(&c, g, loads, nl))
    cls = malloc((size_t)(c.tiles * c.blocks));
  /* Its cost: each pair's products at the core's speed, and the operands'
     bytes moved once. */
  double pairs = (double)rows * (double)c.tk;
  double cost = 2 * pairs * (double)(c.dk + c.dv) / nx_cpu_core_speed(NX_FLOAT32);
  double bytes = 0;
  for (int i = 0; i < n; i++) bytes += nx_cpu_bytes(&a[i]);
  cost += NX_CPU_STRIDED_COST * bytes;
  int released = nx_cpu_release(cost);
  if (cls != NULL) {
    c.cls = cls;
    double pairs1 = (double)c.tq * (double)c.tk;
    nx_cpu_job(c.tiles, (int64_t)pairs1, pairs1 * nx_cpu_prog_cost(g),
               classes, &c);
  }
  nx_cpu_job(units, (int64_t)bytes, cost, body, &c);
  nx_cpu_reacquire(released);
  free(cls);
  int failed = atomic_load(&c.failed) || (g != NULL && nx_cpu_prog_failed(g));
  CAMLreturn(finish(owner, n, a, NX_OK, failed));
}

value nx_cpu_attention_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_attention(argv[0], argv[1], argv[2], argv[3], argv[4],
                          argv[5], argv[6], argv[7]);
}
