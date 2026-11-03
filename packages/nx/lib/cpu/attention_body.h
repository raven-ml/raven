/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* attention.c's units in one accumulator, included once per accumulator
   with T its element, DT its dtype code, WORD and SWORD the unsigned and
   signed integers of its width, EXPONENT its exponent's bits in WORD, EXP
   and LOG its kinds, FMA its fused multiply-add and ROWWISE 1 where
   row_scores serves it, which it undefines at its end. F(name) names the
   accumulator's instance. */

/* A worker's scratch, by unit. */
typedef struct {
  T *q;         /* ROWS × dk */
  T *s;         /* ROWS × KEYS: scores, then weights */
  T *t;         /* ROWS × KEYS: a score program's output */
  T *acc;       /* ROWS × dv */
  T *save;      /* ROWS × dv: rows the block leaves as they were */
  T *kt;        /* dk × KEYS: the keys transposed */
  T *vf;        /* KEYS × dv */
  T m[ROWS], l[ROWS];
  uint8_t *vis; /* ROWS × KEYS, row r's at r·nk */
  uint8_t *lo, *hi; /* blocks each: the mask's least and greatest */
  uint8_t sees[ROWS], hides[ROWS], picks[ROWS];
  uint8_t bad[KEYS]; /* a value not finite */
  run runs[KEYS];
} F(scratch);

static int64_t F(scratch_bytes)(const att *c) {
  int64_t f = ROWS * c->dk + 2 * ROWS * KEYS + 2 * ROWS * c->dv +
              KEYS * c->dk + KEYS * c->dv;
  int64_t classes = c->ranges != NULL ? 2 * c->blocks : 0;
  return (int64_t)sizeof(F(scratch)) + (int64_t)sizeof(T) * f + ROWS * KEYS +
         classes + 10 * 64;
}

static void F(lay_out)(const att *c, F(scratch) *x) {
  uint8_t *p = align64((uint8_t *)(x + 1));
#define TAKE(f, n)                         \
  x->f = (void *)p;                        \
  p = align64(p + (int64_t)(n) * sizeof(T));
  TAKE(q, ROWS * c->dk);
  TAKE(s, ROWS * KEYS);
  TAKE(t, ROWS * KEYS);
  TAKE(acc, ROWS * c->dv);
  TAKE(save, ROWS * c->dv);
  TAKE(kt, KEYS * c->dk);
  TAKE(vf, KEYS * c->dv);
#undef TAKE
  x->vis = p;
  x->lo = p + ROWS * KEYS;
  x->hi = x->lo + c->blocks;
}

/* Whether any of the [n] elements at [x] is not finite. */
static int F(any_bad)(const T *x, int64_t n) {
  const WORD *b = (const WORD *)x;
  WORD e = EXPONENT, any = 0;
  for (int64_t i = 0; i < n; i++) any |= (b[i] & e) == e;
  return any != 0;
}

/* The values of the runs [x->runs], [n] of them: in place, or staged into
   vf at their place in the block. */
static void F(values_of)(const att *c, F(scratch) *x, int n) {
  const nx_array *v = &c->a[V];
  for (int g = 0; g < n; g++) {
    run *u = &x->runs[g];
    if (c->v_in) {
      u->v = (const T *)v->base + u->va;
      u->vs = u->vk;
    } else {
      nx_cpu_stage_as(v, u->va, c->vd, u->vk, c->dv, u->n, DT,
                      (uint8_t *)(x->vf + u->j * c->dv), c->dv);
      u->v = x->vf + u->j * c->dv;
      u->vs = c->dv;
    }
  }
}

/* Row r's update by its block's [nk] scores at [s], hidden ones -inf:
   its maximum, its weights into [s] and its sum, and its accumulator
   rescaled. A block whose visible scores are all -inf gives weights 0
   and leaves the state as it was. Eight lanes of T, of the integers of
   T's width for comparisons. */
typedef T F(v8) __attribute__((vector_size(8 * sizeof(T))));
typedef SWORD F(v8i) __attribute__((vector_size(8 * sizeof(T))));

/* The greatest of [m] and the [n] scores at [s], a NaN where one is. */
static T F(greatest)(const T *s, int n, T m) {
  F(v8) m8 = {m, m, m, m, m, m, m, m};
  F(v8i) nan = {0, 0, 0, 0, 0, 0, 0, 0};
  int j = 0;
  for (; j + 8 <= n; j += 8) {
    F(v8) y;
    memcpy(&y, s + j, sizeof y);
    F(v8i) gt = y > m8;
    m8 = (F(v8))(((F(v8i))y & gt) | ((F(v8i))m8 & ~gt));
    nan |= y != y;
  }
  T mx = m;
  int any = 0;
  for (int i = 0; i < 8; i++) {
    mx = m8[i] > mx ? m8[i] : mx;
    any |= nan[i] != 0;
  }
  for (; j < n; j++) {
    mx = s[j] > mx ? s[j] : mx;
    any |= s[j] != s[j];
  }
  return any ? NAN : mx;
}

/* The [n] weights at [s] summed term j into lane j mod 8, the lanes then
   summed by halves: one order on every target. */
static T F(lanes_sum)(const T *s, int n) {
  F(v8) l8 = {0, 0, 0, 0, 0, 0, 0, 0};
  int j = 0;
  for (; j + 8 <= n; j += 8) {
    F(v8) y;
    memcpy(&y, s + j, sizeof y);
    l8 += y;
  }
  T lane[8];
  memcpy(lane, &l8, sizeof lane);
  for (; j < n; j++) lane[j % 8] += s[j];
  for (int h = 4; h > 0; h /= 2)
    for (int i = 0; i < h; i++) lane[i] += lane[i + h];
  return lane[0];
}

static void F(update)(const att *c, F(scratch) *x, int r, int nk) {
  T *s = x->s + r * KEYS;
  T m = x->m[r], mx = F(greatest)(s, nk, m);
  if (mx == -INFINITY) {
    memset(s, 0, sizeof(T) * (size_t)nk);
    return;
  }
  T alpha = EXP(m - mx);
  for (int j = 0; j < nk; j++) s[j] = EXP(s[j] - mx);
  x->l[r] = x->l[r] * alpha + F(lanes_sum)(s, nk);
  x->m[r] = mx;
  if (alpha != 1) {
    T *acc = x->acc + r * c->dv;
    for (int64_t e = 0; e < c->dv; e++) acc[e] *= alpha;
  }
}

/* Row r's products of its visible pairs alone, as the kernel adds them:
   in increasing key, each fused into its sum. */
static void F(picked)(const att *c, F(scratch) *x, int r, int nr, int nk) {
  T *acc = x->acc + r * c->dv;
  const T *p = x->s + r * KEYS;
  const uint8_t *vis = x->vis + r * nk;
  for (int u = 0; u < nr; u++) {
    const run *g = &x->runs[u];
    for (int i = 0; i < g->n; i++) {
      if (!vis[g->j + i]) continue;
      T pj = p[g->j + i];
      const T *v = (const T *)g->v + i * g->vs;
      for (int64_t e = 0; e < c->dv; e++) acc[e] = FMA(pj, v[e], acc[e]);
    }
  }
}

/* The unit [u]. */
static void F(unit)(att *c, F(scratch) *x, int64_t u) {
  int64_t tile = u % c->tiles, chunk = u / c->tiles % c->chunks;
  int64_t kh = u / c->tiles / c->chunks % c->hk;
  int64_t b = u / c->tiles / c->chunks / c->hk;
  int64_t h0 = kh * c->g + chunk * c->hg, i0 = tile * c->tr;
  int nh = (int)(c->g - chunk * c->hg < c->hg ? c->g - chunk * c->hg : c->hg);
  int nt = (int)(c->tq - i0 < c->tr ? c->tq - i0 : c->tr);
  int rows = nh * nt, r = c->rank;
  int64_t dk = c->dk, dv = c->dv;
  const T scale = (T)c->scale;
  const nx_array *q = &c->a[Q];
  /* The rows, head by head, staged into the accumulator. */
  int64_t qa = lead_at(q, c->lead, c->ext, b) + i0 * step(q, r - 2);
  for (int hh = 0; hh < nh; hh++)
    nx_cpu_stage_as(q, qa + (h0 + hh) * step(q, r - 3), step(q, r - 1),
                    step(q, r - 2), dk, nt, DT,
                    (uint8_t *)(x->q + hh * nt * dk), dk);
  for (int i = 0; i < rows; i++) {
    x->m[i] = -INFINITY;
    x->l[i] = 0;
  }
  memset(x->acc, 0, sizeof(T) * (size_t)(rows * dv));
  uint8_t present[KEYS];
  int64_t len = length_of(c, b);
  /* Classes (attention.c): the shared table's, else the unit's ranges',
     else mixed. */
  int ranged = c->cls == NULL && c->ranges != NULL && len > 0;
  if (ranged) ranges_of(c, x->lo, x->hi, u, (len + KEYS - 1) / KEYS);
  for (int64_t j0 = 0; j0 < len; j0 += KEYS) {
    int nk = (int)(len - j0 < KEYS ? len - j0 : KEYS);
    int64_t k = j0 / KEYS;
    int cls = c->cls != NULL ? c->cls[tile * c->blocks + k]
              : ranged       ? class_of(x->lo[k], x->hi[k])
                             : MIXED;
    if (cls == HIDDEN) continue;
    int mask = c->mask && cls == MIXED;
    int nr = runs_of(c, x->runs, b, kh, j0, nk, present);
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
        memset(x->s + i * KEYS, 0, sizeof(T) * (size_t)nk);
    /* Each score adds its products in increasing dimension from +0, each
       fused: on the chain kernel's tiles over the keys staged transposed,
       or, for few rows, each row against the keys in rows. */
    for (int g = 0; dk > 0 && g < nr; g++) {
      const run *v = &x->runs[g];
#if ROWWISE
      if (rows < CHAIN_ROWS) {
        const T *k;
        int64_t ks;
        if (c->k_in) {
          k = (const T *)c->a[K].base + v->ka;
          ks = v->kk;
        } else {
          nx_cpu_stage_as(&c->a[K], v->ka, c->kd, v->kk, dk, v->n, DT,
                          (uint8_t *)x->kt, dk);
          k = x->kt;
          ks = dk;
        }
        row_scores(x->q, rows, k, ks, v->n, dk, x->s + v->j);
        continue;
      }
#endif
      nx_cpu_stage_as(&c->a[K], v->ka, v->kk, c->kd, v->n, dk, DT,
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
          T *y = x->s + i * KEYS + v->j;
          memset(y, 0, sizeof(T) * (size_t)v->n);
          c->gemv(v->n, dk, x->kt + v->j, KEYS, x->q + i * dk, y);
        }
    }
    for (int i = 0; i < rows; i++) {
      T *s = x->s + i * KEYS;
      for (int j = 0; j < nk; j++) s[j] *= scale;
    }
    /* The score program's output 1 over the scores, head by head. */
    if (c->score)
      for (int hh = 0; hh < nh; hh++) {
        int64_t p = (((b * c->hq + h0 + hh) * c->tq + i0) * c->tk) + j0;
        T *s = x->s + hh * nt * KEYS;
        nx_cpu_prog_tile(c->prog, &c->pl, 1, p, nk, c->tk, nt,
                         (uint8_t *)x->t, 0, s, KEYS);
        for (int i = 0; i < nt; i++)
          if (x->sees[hh * nt + i])
            memcpy(s + i * KEYS, x->t + i * nk, sizeof(T) * (size_t)nk);
      }
    F(values_of)(c, x, nr);
    if (mask || !all)
      for (int i = 0; i < rows; i++) {
        if (!x->hides[i]) continue;
        T *s = x->s + i * KEYS;
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
          x->bad[v->j + j] =
              (uint8_t)F(any_bad)((const T *)v->v + j * v->vs, dv);
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
      if (x->sees[i]) F(update)(c, x, i, nk);
      else memset(x->s + i * KEYS, 0, sizeof(T) * (size_t)nk);
    }
    if (rows < CHAIN_ROWS) {
      for (int i = 0; i < rows; i++) {
        if (!x->sees[i]) continue;
        if (picking && x->picks[i]) {
          F(picked)(c, x, i, nr, nk);
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
        memcpy(x->save + i * dv, x->acc + i * dv, sizeof(T) * (size_t)dv);
    for (int64_t e = 0; e < dv; e += c->nr) {
      int n = (int)(dv - e < c->nr ? dv - e : c->nr);
      for (int g = 0; g < nr; g++) {
        const run *v = &x->runs[g];
        c->pv(rows, n, v->n, x->s + v->j, KEYS, 1, (const T *)v->v + e, v->vs,
              x->acc + e, dv, NX_CPU_FROM_TILE);
      }
    }
    for (int i = 0; i < rows; i++) {
      int pick = picking && x->picks[i];
      if (!x->sees[i] || pick)
        memcpy(x->acc + i * dv, x->save + i * dv, sizeof(T) * (size_t)dv);
      if (pick) F(picked)(c, x, i, nr, nk);
    }
  }
  /* o = acc / l, rounded once to o's dtype, and lse. */
  nx_cpu_run store = nx_cpu_convert(DT, c->odt);
  const nx_array *o = &c->a[O], *lse = &c->a[LSE];
  for (int hh = 0; hh < nh; hh++)
    for (int i = 0; i < nt; i++) {
      int row = hh * nt + i;
      int64_t at = ((b * c->hq + h0 + hh) * c->tq + i0 + i);
      T *acc = x->acc + row * dv;
      T l = x->l[row], m = x->m[row];
      T y;
      if (l == 0) {
        memset(acc, 0, sizeof(T) * (size_t)dv);
        y = -INFINITY;
      } else {
        for (int64_t e = 0; e < dv; e++) acc[e] = acc[e] / l;
        y = m == INFINITY ? m : m + LOG(l);
      }
      store(acc, o->base + (o->offset + at * dv) * c->w, dv);
      ((T *)lse->base)[lse->offset + at] = y;
    }
}

static void F(body)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  att *c = ctx;
  uint8_t *mem = nx_cpu_take(F(scratch_bytes)(c));
  if (mem == NULL) {
    atomic_store(&c->failed, 1);
    return;
  }
  F(scratch) *x = (F(scratch) *)align64(mem);
  F(lay_out)(c, x);
  for (int64_t u = lo; u < hi; u++) F(unit)(c, x, u);
  nx_cpu_give(mem);
}

#undef T
#undef DT
#undef WORD
#undef SWORD
#undef EXPONENT
#undef EXP
#undef LOG
#undef FMA
#undef ROWWISE
#undef F
