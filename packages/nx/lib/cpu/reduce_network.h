/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A network scan's fast pass over one slice by tiles. The network's results
   at a tile of PT terms aligned from the slice's start depend only on its
   terms and on two results of the network over the tiles' states at level
   NETWORK_P: the one before the tile and the one at it. In tile q, the
   result before its first state at level k < NETWORK_P, at index
   q 2^(P - k) - 1, odd, is the next level's at its half, and so on up to
   level P's at q - 1; the result at the tile's last state, index
   (q + 1) 2^(P - k) - 1, is level P's at q. Bottom-up, its terms' states
   join in pairs level by level, up to the tile's state at level NETWORK_P;
   top-down, each level's results follow from the next one's: an odd
   index's is the next level's result at its half, an even one's that
   level's result before joined with its own state. These are pass_'s
   joins of the same states in the same order, by the fast pass's
   arithmetic, so the bits are pass_'s; each level's joins run over its
   positions at once. Level k's states hold 2^k terms each, so a Moments
   join's weights are the same for the whole level. Results hold position
   0 for the result before the tile.

   The tiles' states at level NETWORK_P are all that ties tiles together,
   so where units are fewer than threads (network_in), the tiles' states
   are computed apart, then the network over them, then the tiles' results
   apart, each tile's states up again.

   reduce_float.h includes this file once per statistic, after defining
   N(name), which names name at the statistic and the type: N(tiles) is
   moments_tiles_f32; N_SIZE, the elements of a row a lane's state takes;
   N_FIELDS, the fields of a tile's results, its state's being two; and
   N_OUTS, the scan's outputs. The statistic gives at N, each inlined into
   the network's loops:
   - join(a), its join of two rows of states in arithmetic a, which
     stream_ takes; leaf, a row of terms' states; centers, the slices'
     centers; put, a term's outputs from its state, 1 where one is not
     finite;
   - for a tile: leaves, its terms' states from the terms at level 0; up,
     a level's states from the pairs below; top, a full tile's state at
     level NETWORK_P; seed, the first result of a slice's first tile at a
     level, its state; down, a level's results at the even positions; and
     results, the tile's outputs from its results, 1 where one is not
     finite. */
static void N(fast_leaf)(int64_t w, const T *x, const T *c, T *r) {
  N(leaf)(FAST, w, x, c, r);
}

/* Tile [t0]'s [L] terms, from [x] or [src], into level 0 of [scratch],
   their states of center [c] there and up through the levels, packed at
   states_at(k); a full tile's state at level NETWORK_P into [total]
   where it is not NULL. */
static void N(tile_up)(const T *x, int64_t xs,
                       const nx_cpu_terms *src, int64_t t0, int64_t L,
                       T c, T *scratch, T *total) {
  T *restrict sm = scratch, *restrict ss = sm + 2 * PT;
  if (src != NULL)
    for (int64_t i = 0, most = NX_CPU_NETWORK_TILE; i < L; i += most)
      src->fill(src->ctx, t0 + i, L - i < most ? L - i : most,
                (uint8_t *)(sm + i));
  else
    for (int64_t i = 0; i < L; i++) sm[i] = x[(t0 + i) * xs];
  N(leaves)(L, c, sm, ss);
  for (int k = 1; k <= NETWORK_P; k++) {
    int64_t a = states_at(k - 1), u = states_at(k);
    N(up)(k, L >> k, sm + a, ss + a, sm + u, ss + u);
  }
  if (total != NULL)
    N(top)(sm[states_at(NETWORK_P)], ss[states_at(NETWORK_P)], total);
}

/* Tile [t0]'s [L] outputs from tile_up_'s states and the network's
   results [before] the tile and [at] its last term, NULL in the first
   tile and a partial one, of center [c]: into [ym] and [yv], 1 where one
   is not finite. Results are packed at results_at(k), each of the N_FIELDS
   fields 2 PT + NETWORK_P + 1 after the one before. */
static int N(tile_down)(int64_t t0, int64_t L, T c, const T *before,
                        const T *at, T *ym, int64_t ms, T *yv,
                        int64_t vs, T *scratch) {
  T *sm = scratch, *ss = sm + 2 * PT, *r[N_FIELDS];
  for (int f = 0; f < N_FIELDS; f++)
    r[f] = ss + 2 * PT + f * (2 * PT + NETWORK_P + 1);
  for (int k = 0; before != NULL && k <= NETWORK_P; k++)
    for (int f = 0; f < N_FIELDS; f++) r[f][results_at(k)] = before[f];
  for (int f = 0; at != NULL && f < N_FIELDS; f++)
    r[f][results_at(NETWORK_P) + 1] = at[f];
  for (int k = NETWORK_P - 1; k >= 0; k--) {
    int64_t m = L >> k;
    const T *bm = sm + states_at(k), *bs = ss + states_at(k);
    T *u[N_FIELDS], *o[N_FIELDS];
    for (int f = 0; f < N_FIELDS; f++) {
      u[f] = r[f] + results_at(k + 1);
      o[f] = r[f] + results_at(k);
      S(odd)(m / 2, u[f] + 1, o[f] + 2);
    }
    int64_t i0 = 0;
    if (before == NULL && m > 0) {
      N(seed)(k, bm, bs, o);
      i0 = 1;
    }
    N(down)(k, i0, (m + 1) / 2, u, bm, bs, o);
  }
  for (int f = 0; f < N_FIELDS; f++) r[f]++;
  return N(results)(L, c, t0, r, ym, ms, yv, vs);
}

/* One slice, tile by tile: the streaming network over the tiles' states,
   its sets after tile_up_'s levels and tile_down_'s results, takes each
   full tile's state on to its result. */
static int N(tiles)(const T *x, int64_t xs, T *ym, int64_t ms, T *yv,
                    int64_t vs, int64_t n, const nx_cpu_terms *src,
                    T *scratch) {
  T *sets = scratch + 4 * PT + N_FIELDS * (2 * PT + NETWORK_P + 1);
  T *held[TILE_LEVELS], *in[TILE_LEVELS], *joined[TILE_LEVELS];
  const T *got[TILE_LEVELS];
  for (int k = 0; k < TILE_LEVELS; k++) {
    held[k] = sets + 3 * N_SIZE * k;
    in[k] = held[k] + N_SIZE;
    joined[k] = in[k] + N_SIZE;
  }
  T c, before[N_FIELDS];
  N(centers)(x, 0, 1, src, scratch, &c);
  int bad = 0;
  for (int64_t t0 = 0, q = 0; t0 < n; t0 += PT, q++) {
    int64_t L = n - t0 < PT ? n - t0 : PT;
    const T *at = NULL;
    N(tile_up)(x, xs, src, t0, L, c, scratch, L == PT ? in[0] : NULL);
    if (L == PT) {
      S(stream)(N(join)(FAST), 1, q, held, in, joined, got, NULL);
      at = got[0];
    }
    bad |= N(tile_down)(t0, L, c, q > 0 ? before : NULL, at, ym, ms,
                        yv, vs, scratch);
    for (int i = 0; at != NULL && i < N_FIELDS; i++) before[i] = at[i];
  }
  return bad;
}

/* One pass of arithmetic [a] over terms [lo, hi): 0, or 1 where the fast
   pass met a value not finite. Terms go in tiles of NX_CPU_NETWORK_TILE,
   and lanes not contiguous in [x], or in an output, through rows of the
   scratch, so that every loop over the lanes reads and writes contiguous
   elements. Terms that [src] computes go into those rows too. A range
   past a slice's start is a tile of PT terms, from the network's results
   [carry] before it and, where full, [top] at its last term. */
static int N(pass)(arith a, const T *x, int64_t xs, int64_t xl,
                   T *ym, int64_t ms, int64_t ml, T *yv, int64_t vs,
                   int64_t vl, int64_t lo, int64_t hi, int64_t w,
                   T *scratch, const nx_cpu_terms *src,
                   const T *carry, const T *top) {
  S(join) join = N(join)(a);
  int64_t size = N_SIZE * w, tile = NX_CPU_NETWORK_TILE;
  int bad = 0, levels = nx_cpu_network_levels(hi - lo);
  T c[NX_CPU_NETWORK_LANES];
  T *tx = scratch + 3 * levels * size, *tm = tx + tile * w,
    *tv = tm + tile * w;
  N(centers)(x, xl, w, src, tx, c);
  /* Computed terms are read from the tile's rows: xl is 0 then. */
  if (src != NULL) xl = 0;
  /* Level k's waiting, arriving and joined states, and its result. */
  T *held[64], *in[64], *joined[64];
  const T *got[64];
  for (int k = 0; k < levels; k++) {
    held[k] = scratch + 3 * k * size;
    in[k] = held[k] + size;
    joined[k] = in[k] + size;
  }
  for (int64_t t0 = lo; t0 < hi; t0 += tile) {
    int64_t tn = hi - t0 < tile ? hi - t0 : tile;
    if (src != NULL) src->fill(src->ctx, t0, tn, (uint8_t *)tx);
    else if (xl != 1) S(rows)(tn, w, x + t0 * xs, xs, xl, tx);
    for (int64_t i = 0; i < tn; i++) {
      int64_t t = t0 + i, p = t + NETWORK_AHEAD;
      if (p < hi && xl == 1) PREFETCH_ROW(x + p * xs, w, 0);
      if (p < hi && ml == 1) PREFETCH_ROW(ym + p * ms, w, 1);
      if (N_OUTS > 1 && p < hi && vl == 1) PREFETCH_ROW(yv + p * vs, w, 1);
      const T *r = top;
      if (top == NULL || t < hi - 1) {
        N(leaf)(a, w, xl == 1 ? x + t * xs : tx + i * w, c, in[0]);
        S(stream)(join, w, t - lo, held, in, joined, got, carry);
        r = got[0];
      }
      T *dm = ml == 1 ? ym + t * ms : tm + i * w;
      T *dv = N_OUTS > 1 && vl == 1 ? yv + t * vs : tv + i * w;
      bad |= N(put)(a, w, c, r, (T)t + 1, dm, dv);
    }
    if (ml != 1) S(unrows)(tn, w, tm, ym + t0 * ms, ms, ml);
    if (N_OUTS > 1 && vl != 1) S(unrows)(tn, w, tv, yv + t0 * vs, vs, vl);
  }
  return bad;
}

/* The scan (cpu.h): a slice alone runs its fast pass by tiles, slices
   side by side by pass_, and the exact one, if needed, by pass_. By a
   span, its step over tiles [q0, q1), each [L] terms from [t0], its
   state [st] at level NETWORK_P, the states before it at [st - size]. */
static void N(network)(const uint8_t *x_, int64_t xs, int64_t xl,
                       uint8_t *y_, int64_t ms, int64_t ml,
                       uint8_t *y1_, int64_t vs, int64_t vl,
                       int64_t n, int64_t w, uint8_t *scratch_,
                       const nx_cpu_terms *src, nx_cpu_span *span) {
  const T *x = (const T *)x_;
  T *ym = (T *)y_, *yv = (T *)y1_, *scratch = (T *)scratch_;
  if (span == NULL || span->step == NX_CPU_EXACT) {
    int exact = span != NULL;
    if (!exact)
      exact = w == 1 ? N(tiles)(x, xs, ym, ms, yv, vs, n, src, scratch)
                     : N(pass)(FAST, x, xs, xl, ym, ms, ml, yv, vs, vl,
                               0, n, w, scratch, src, NULL, NULL);
    if (exact)
      N(pass)(EXACT, x, xs, xl, ym, ms, ml, yv, vs, vl, 0, n, w,
              scratch, src, NULL, NULL);
    return;
  }
  int64_t size = N_SIZE * w;
  T *states = (T *)span->states, c[NX_CPU_NETWORK_LANES];
  if (span->step == NX_CPU_CARRIES) {
    S(carries)(N(join)(FAST), size, w, span->q1, states, scratch);
    return;
  }
  N(centers)(x, xl, w, src, scratch, c);
  int bad = 0;
  for (int64_t q = span->q0; q < span->q1; q++) {
    int64_t t0 = q * PT, L = n - t0 < PT ? n - t0 : PT;
    T *st = states + q * size;
    const T *before = q > 0 ? st - size : NULL, *at = L == PT ? st : NULL;
    if (span->step == NX_CPU_TOTALS && w == 1)
      N(tile_up)(x, xs, src, t0, PT, c[0], scratch, st);
    else if (span->step == NX_CPU_TOTALS)
      S(total)(N(join)(FAST), N(fast_leaf), size, x, xs, xl, c,
               t0, w, scratch, src, st);
    else if (w == 1) {
      N(tile_up)(x, xs, src, t0, L, c[0], scratch, NULL);
      bad |= N(tile_down)(t0, L, c[0], before, at, ym, ms, yv, vs,
                          scratch);
    } else
      bad |= N(pass)(FAST, x, xs, xl, ym, ms, ml, yv, vs, vl, t0,
                     t0 + L, w, scratch, src, before, at);
  }
  span->bad |= bad;
}
