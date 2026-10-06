/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_packed.c — the sub-byte family's reader, writer and kernels: copy,
   gather, pad, and the logical operations and reductions of bit.
   nx_c_packed.h states the storage and the rules every load and store keeps.
   The casts to and from bit live with the other casts, in nx_c_map.c. */

#include "nx_c_packed.h"

/* Reading */

void nx_c_packed_src_init(nx_c_packed_src *s, const nx_c_ndarray *a,
                          int bits) {
  s->base = (const uint8_t *)a->data;
  s->bits = bits;
  s->offset = a->offset;
  int nd = 0;
  for (int d = 0; d < a->ndim; d++) {
    if (a->shape[d] == 1) continue;
    if (nd > 0 && s->strides[nd - 1] == a->strides[d] * a->shape[d]) {
      s->shape[nd - 1] *= a->shape[d];
      s->strides[nd - 1] = a->strides[d];
      continue;
    }
    s->shape[nd] = a->shape[d];
    s->strides[nd] = a->strides[d];
    nd++;
  }
  if (nd == 0) {
    s->shape[0] = 1;
    s->strides[0] = 1;
    nd = 1;
  }
  s->ndim = nd;
}

/* The storage position of the first element of row r, the rows being the
   points of every axis but the last. */
static int64_t nx_c_packed_row(const nx_c_packed_src *s, int64_t r) {
  int64_t pos = s->offset;
  for (int d = s->ndim - 2; d >= 0; d--) {
    pos += (r % s->shape[d]) * s->strides[d];
    r /= s->shape[d];
  }
  return pos;
}

uint64_t nx_c_packed_read_any(const nx_c_packed_src *s, int64_t e, int k) {
  int bits = s->bits, last = s->ndim - 1;
  int64_t n_in = s->shape[last], s_in = s->strides[last];
  uint64_t w = 0;
  int filled = 0;
  while (filled < k) {
    int64_t c = e % n_in;
    int64_t pos = nx_c_packed_row(s, e / n_in) + c * s_in;
    int run = k - filled;
    if (n_in - c < run) run = (int)(n_in - c);
    uint64_t v;
    if (s_in == 1)
      v = nx_c_bits_load(s->base, pos * bits, run * bits);
    else if (s_in == 0)
      v = nx_c_packed_splat(nx_c_packed_get(s->base, pos, bits), bits, run);
    else if (s_in == -1)
      v = nx_c_packed_reverse(
          nx_c_bits_load(s->base, (pos - run + 1) * bits, run * bits), bits,
          run);
    else if (bits == 1) {
      v = 0;
      for (int j = 0; j < run; j++) {
        int64_t p = pos + j * s_in;
        v |= (uint64_t)((s->base[p >> 3] >> (p & 7)) & 1) << j;
      }
    } else {
      v = 0;
      for (int j = 0; j < run; j++)
        v |= (uint64_t)nx_c_packed_get(s->base, pos + j * s_in, bits)
             << (j * bits);
    }
    w |= v << (filled * bits);
    filled += run;
    e += run;
  }
  return w;
}

/* Writing */

/* One run of a destination: elements [first, first + n) of the storage at
   base, which are its elements [e0, e0 + n) in C order. */
typedef struct {
  uint8_t *base;
  int bits;
  int64_t first; /* bit */
  int64_t end;   /* bit */
  int64_t e0;
  bool zero_tail; /* end is the buffer's end: the bits past it are 0 */
  const nx_c_packed_filler *f;
} nx_c_packed_run;

/* Word q of the storage, the part of it the run covers. */
static void nx_c_packed_word(const nx_c_packed_run *r, int64_t q) {
  int64_t wlo = q * 64, whi = wlo + 64;
  int64_t lo = r->first > wlo ? r->first : wlo;
  int64_t hi = r->end < whi ? r->end : whi;
  uint64_t v = r->f->word(r->f->ctx, r->e0 + (lo - r->first) / r->bits,
                          (int)((hi - lo) / r->bits));
  if (lo == wlo && hi == whi) {
    nx_c_st64(r->base + q * 8, v);
    return;
  }
  int64_t top = hi;
  if (hi == r->end && r->zero_tail) top = (hi + 7) & ~(int64_t)7;
  nx_c_bits_store(r->base, lo, (int)(top - lo), v);
}

/* Words [qlo, qhi) of the run: the words it covers in part one at a time, the
   others together when the filler writes whole words. */
static void nx_c_packed_words(const nx_c_packed_run *r, int64_t qlo,
                              int64_t qhi) {
  if (qlo < qhi && qlo * 64 < r->first) nx_c_packed_word(r, qlo++);
  if (qlo < qhi && qhi * 64 > r->end) nx_c_packed_word(r, --qhi);
  if (qlo >= qhi) return;
  if (r->f->words) {
    int64_t e = r->e0 + (qlo * 64 - r->first) / r->bits;
    r->f->words(r->f->ctx, e, qhi - qlo, r->base + qlo * 8);
    return;
  }
  for (int64_t q = qlo; q < qhi; q++) nx_c_packed_word(r, q);
}

typedef struct {
  const nx_c_packed_run *run;
  int64_t q0;
} nx_c_packed_job;

static void nx_c_packed_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  (void)worker;
  const nx_c_packed_job *j = vctx;
  nx_c_packed_words(j->run, j->q0 + lo, j->q0 + hi);
}

nx_c_status nx_c_packed_write(const nx_c_ndarray *out, int bits,
                              const nx_c_packed_filler *f, int64_t bytes) {
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  if (total == 0) return NX_C_OK;
  for (int d = 0; d < out->ndim; d++)
    if (out->strides[d] == 0 && out->shape[d] > 1) return NX_C_ERR_OUT_ALIASED;

  nx_c_packed_src dst;
  nx_c_packed_src_init(&dst, out, bits);
  uint8_t *base = (uint8_t *)out->data;
  int last = dst.ndim - 1;
  int64_t n_in = dst.shape[last], s_in = dst.strides[last];

  /* One run: its words, on as many workers as the policy gives. */
  if (last == 0 && s_in == 1) {
    nx_c_packed_run r = {base,
                         bits,
                         out->offset * bits,
                         (out->offset + total) * bits,
                         0,
                         out->offset + total == out->length,
                         f};
    int64_t q0 = r.first >> 6, q1 = ((r.end - 1) >> 6) + 1;
    nx_c_packed_job j = {&r, q0};
    int nth = nx_c_threads_for(NX_C_COST_BANDWIDTH, total, 1, bytes);
    if (nth > q1 - q0) nth = (int)(q1 - q0);
    nx_c_parallel_for(nth, q1 - q0, bytes, nx_c_packed_body, &j, NULL);
    return NX_C_OK;
  }

  /* Rows that may share bytes: one worker, a row at a time. */
  for (int64_t row = 0, e = 0; e < total; row++, e += n_in) {
    int64_t pos = nx_c_packed_row(&dst, row);
    if (s_in == 1) {
      nx_c_packed_run r = {base,
                           bits,
                           pos * bits,
                           (pos + n_in) * bits,
                           e,
                           pos + n_in == out->length,
                           f};
      nx_c_packed_words(&r, r.first >> 6, ((r.end - 1) >> 6) + 1);
    } else {
      for (int64_t j = 0; j < n_in; j++)
        nx_c_packed_set(base, pos + j * s_in, bits,
                        (uint8_t)f->word(f->ctx, e + j, 1));
    }
  }
  return NX_C_OK;
}

/* Copy */

static uint64_t nx_c_packed_copy_fill(const void *ctx, int64_t e, int k) {
  return nx_c_packed_read((const nx_c_packed_src *)ctx, e, k);
}

/* Whole words of a copy: the source's bytes when it is one run starting on a
   byte, its words through the funnel shift when it starts inside one, and
   through the reader otherwise. */
static void nx_c_packed_copy_words(const void *ctx, int64_t e, int64_t n,
                                   uint8_t *dst) {
  const nx_c_packed_src *s = ctx;
  int per = 64 / s->bits;
  if (!nx_c_packed_dense(s)) {
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_packed_read(s, e + j * per, per));
    return;
  }
  int64_t bit = (s->offset + e) * s->bits;
  if ((bit & 7) == 0) {
    memcpy(dst, s->base + (bit >> 3), (size_t)(8 * n));
    return;
  }
  for (int64_t j = 0; j < n; j++)
    nx_c_st64(dst + 8 * j, nx_c_bits_load(s->base, bit + 64 * j, 64));
}

nx_c_status nx_c_packed_copy(const nx_c_ndarray *out, const nx_c_ndarray *in,
                             nx_c_dtype dt) {
  if (out->ndim != in->ndim) return NX_C_ERR_RANK_MISMATCH;
  for (int d = 0; d < out->ndim; d++)
    if (out->shape[d] != in->shape[d]) return NX_C_ERR_SHAPE;
  int bits = nx_c_packed_bits(dt);
  nx_c_packed_src src;
  nx_c_packed_src_init(&src, in, bits);
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  nx_c_packed_filler f = {nx_c_packed_copy_fill, nx_c_packed_copy_words, &src};
  return nx_c_packed_write(out, bits, &f, 2 * nx_c_dtype_bytes(dt, total));
}

/* Gather: element c of out is element c of data with its axis component
   replaced by indices' element c, or 0 for an index outside the axis.

   The output's dims, in C order, are walked as the byte gather walks them
   (nx_c_move.c): those of 1 dropped and neighbours merged where they compose
   in the data and the indices, never across axis. When the last walked dim
   follows axis, is dense in the data and holds the index constant, as the
   broadcast index of a take does, it is a run: one index load moves the whole
   run as its bits, by memcpy where they lie on whole bytes, so a take of rows
   of uint4 weights moves bytes as uint8's does. */

typedef struct {
  int ndim, bits;
  int64_t shape[NX_C_MAX_NDIM], ds[NX_C_MAX_NDIM], is[NX_C_MAX_NDIM];
  int64_t run, da, axis_len, d0, i0;
  const uint8_t *src;
  const int64_t *index;
} nx_c_packed_gather_ctx;

static void nx_c_packed_gather_plan(nx_c_packed_gather_ctx *g,
                                    const nx_c_ndarray *out,
                                    const nx_c_ndarray *data,
                                    const nx_c_ndarray *ix, int axis,
                                    int bits) {
  int nd = 0, walked_axis = -1;
  for (int d = 0; d < out->ndim; d++) {
    int64_t n = out->shape[d];
    int64_t ds = d == axis ? 0 : data->strides[d], is = ix->strides[d];
    if (d != axis && n == 1) continue;
    int last = nd - 1;
    if (d != axis && nd > 0 && last != walked_axis && g->ds[last] == ds * n &&
        g->is[last] == is * n) {
      g->shape[last] *= n;
      g->ds[last] = ds;
      g->is[last] = is;
      continue;
    }
    if (d == axis) walked_axis = nd;
    g->shape[nd] = n;
    g->ds[nd] = ds;
    g->is[nd] = is;
    nd++;
  }
  int last = nd - 1;
  g->run = 1;
  if (last > walked_axis && g->is[last] == 0 && g->ds[last] == 1) {
    g->run = g->shape[last];
    nd--;
  }
  g->ndim = nd;
  g->bits = bits;
  g->da = data->strides[axis];
  g->axis_len = data->shape[axis];
  g->d0 = data->offset;
  g->i0 = ix->offset;
  g->src = (const uint8_t *)data->data;
  g->index = (const int64_t *)ix->data;
}

/* The next walked position after coord, by an odometer, and the offsets d and
   i into the data and the indices that follow it. */
static inline void nx_c_packed_gather_step(const nx_c_packed_gather_ctx *g,
                                           int64_t *coord, int64_t *d,
                                           int64_t *i) {
  for (int dd = g->ndim - 1; dd >= 0; dd--) {
    if (++coord[dd] < g->shape[dd]) {
      *d += g->ds[dd];
      *i += g->is[dd];
      return;
    }
    *d -= (g->shape[dd] - 1) * g->ds[dd];
    *i -= (g->shape[dd] - 1) * g->is[dd];
    coord[dd] = 0;
  }
}

static uint64_t nx_c_packed_gather_fill(const void *vctx, int64_t e, int k) {
  const nx_c_packed_gather_ctx *g = vctx;
  int bits = g->bits;
  int64_t coord[NX_C_MAX_NDIM];
  int64_t pos = e, off = 0, d = g->d0, i = g->i0;
  if (g->run > 1) {
    pos = e / g->run;
    off = e % g->run;
  }
  /* The outermost walked dim holds what the others leave of pos: no
     division, so a gather of one walked dim divides nothing. */
  for (int dd = g->ndim - 1; dd >= 0; dd--) {
    coord[dd] = dd == 0 ? pos : pos % g->shape[dd];
    if (dd > 0) pos /= g->shape[dd];
    d += coord[dd] * g->ds[dd];
    i += coord[dd] * g->is[dd];
  }
  uint64_t w = 0;
  /* Elements one at a time, along the last walked dim's line and then the
     next, as the byte gather walks them. */
  if (g->run == 1) {
    int last = g->ndim - 1;
    const uint8_t *src = g->src;
    int64_t ds = g->ds[last], is = g->is[last], da = g->da, len = g->axis_len;
    for (int j = 0;;) {
      int m = k - j;
      if (g->shape[last] - coord[last] < m)
        m = (int)(g->shape[last] - coord[last]);
      const int64_t *index = g->index + i;
      for (int t = 0; t < m; t++) {
        int64_t x = index[t * is];
        if ((uint64_t)x < (uint64_t)len)
          w |= (uint64_t)nx_c_packed_get(src, d + t * ds + x * da, bits)
               << ((j + t) * bits);
      }
      j += m;
      if (j == k) return w;
      coord[last] += m - 1;
      d += (m - 1) * ds;
      i += (m - 1) * is;
      nx_c_packed_gather_step(g, coord, &d, &i);
    }
  }
  for (int filled = 0;;) {
    int n = k - filled;
    if (g->run - off < n) n = (int)(g->run - off);
    int64_t x = g->index[i];
    if ((uint64_t)x < (uint64_t)g->axis_len)
      w |= nx_c_bits_load(g->src, (d + x * g->da + off) * bits, n * bits)
           << (filled * bits);
    filled += n;
    if (filled == k) return w;
    off = 0;
    nx_c_packed_gather_step(g, coord, &d, &i);
  }
}

/* [nbits] bits of src from bit [from] written at bit [to] of dst, or zeros
   where src is NULL: bytes moved whole where the three are whole bytes, and
   otherwise a dst word, or the part of one, at a time. dst's words are the
   writer's own. */
static void nx_c_packed_move_bits(uint8_t *dst, int64_t to, const uint8_t *src,
                                  int64_t from, int64_t nbits) {
  if (((to | from | nbits) & 7) == 0) {
    if (src) memcpy(dst + (to >> 3), src + (from >> 3), (size_t)(nbits >> 3));
    else memset(dst + (to >> 3), 0, (size_t)(nbits >> 3));
    return;
  }
  while (nbits > 0) {
    int m = 64 - (int)(to & 63);
    if (m > nbits) m = (int)nbits;
    nx_c_bits_store(dst, to, m, src ? nx_c_bits_load(src, from, m) : 0);
    to += m;
    from += m;
    nbits -= m;
  }
}

/* Whole words of a gather from element e: run by run, each moved as its bits,
   so a take of whole rows moves their bytes. */
static void nx_c_packed_gather_words(const void *vctx, int64_t e, int64_t n,
                                     uint8_t *dst) {
  const nx_c_packed_gather_ctx *g = vctx;
  int bits = g->bits;
  if (g->run == 1) {
    int per = 64 / bits;
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_packed_gather_fill(g, e + j * per, per));
    return;
  }
  int64_t coord[NX_C_MAX_NDIM];
  int64_t pos = e / g->run, off = e % g->run, d = g->d0, i = g->i0;
  for (int dd = g->ndim - 1; dd >= 0; dd--) {
    coord[dd] = dd == 0 ? pos : pos % g->shape[dd];
    if (dd > 0) pos /= g->shape[dd];
    d += coord[dd] * g->ds[dd];
    i += coord[dd] * g->is[dd];
  }
  int64_t to = 0, end = n * 64;
  for (;;) {
    int64_t len = g->run - off;
    if (len > (end - to) / bits) len = (end - to) / bits;
    int64_t x = g->index[i];
    bool kept = (uint64_t)x < (uint64_t)g->axis_len;
    nx_c_packed_move_bits(dst, to, kept ? g->src : NULL,
                          (kept ? d + x * g->da + off : 0) * bits, len * bits);
    to += len * bits;
    if (to == end) return;
    off = 0;
    nx_c_packed_gather_step(g, coord, &d, &i);
  }
}

/* A gather into a vector has one dim, the axis, so no runs: it reads element
   by element, without the walk's bookkeeping, which costs a vector of random
   indices about a quarter of its time. */
typedef struct {
  const nx_c_ndarray *data;
  const nx_c_ndarray *indices;
  int bits;
} nx_c_packed_take_ctx;

static uint64_t nx_c_packed_take_fill(const void *vctx, int64_t e, int k) {
  const nx_c_packed_take_ctx *c = vctx;
  const nx_c_ndarray *data = c->data, *ix = c->indices;
  const int64_t *index = (const int64_t *)ix->data;
  int64_t n = data->shape[0];
  int bits = c->bits;
  uint64_t w = 0;
  for (int j = 0; j < k; j++) {
    int64_t i = index[ix->offset + (e + j) * ix->strides[0]];
    if ((uint64_t)i < (uint64_t)n)
      w |= (uint64_t)nx_c_packed_get(
               data->data, data->offset + i * data->strides[0], bits)
           << (j * bits);
  }
  return w;
}

nx_c_status nx_c_packed_gather(const nx_c_ndarray *out,
                               const nx_c_ndarray *data,
                               const nx_c_ndarray *indices, int axis,
                               nx_c_dtype dt) {
  if (axis < 0 || axis >= data->ndim) return NX_C_ERR_AXIS;
  if (data->ndim != indices->ndim || data->ndim != out->ndim)
    return NX_C_ERR_SHAPE;
  for (int d = 0; d < out->ndim; d++)
    if (out->shape[d] != indices->shape[d]) return NX_C_ERR_SHAPE;
  int bits = nx_c_packed_bits(dt);
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  if (out->ndim == 1) {
    nx_c_packed_take_ctx c = {data, indices, bits};
    nx_c_packed_filler f = {nx_c_packed_take_fill, NULL, &c};
    return nx_c_packed_write(
        out, bits, &f,
        total * (int64_t)sizeof(int64_t) + 2 * nx_c_dtype_bytes(dt, total));
  }
  nx_c_packed_gather_ctx g;
  nx_c_packed_gather_plan(&g, out, data, indices, axis, bits);
  int64_t bytes = total / g.run * (int64_t)sizeof(int64_t) +
                  2 * nx_c_dtype_bytes(dt, total);
  nx_c_packed_filler f = {nx_c_packed_gather_fill, nx_c_packed_gather_words,
                          &g};
  return nx_c_packed_write(out, bits, &f, bytes);
}

/* Logic */

typedef struct {
  nx_c_packed_src a, b;
  nx_c_bit_op op;
} nx_c_bit_logic_ctx;

static uint64_t nx_c_bit_logic_fill(const void *vctx, int64_t e, int k) {
  const nx_c_bit_logic_ctx *c = vctx;
  uint64_t a = nx_c_packed_read(&c->a, e, k), b = nx_c_packed_read(&c->b, e, k);
  switch (c->op) {
  case NX_C_BIT_AND: return a & b;
  case NX_C_BIT_OR: return a | b;
  case NX_C_BIT_XOR: return a ^ b;
  }
  return 0;
}

/* A word from element e of an operand that is one run or one element
   broadcast, and of one that may also be a run reversed. The second costs a
   branch the first does not pay. */
static inline uint64_t nx_c_bit_word(const nx_c_packed_src *s, int64_t e) {
  if (s->strides[0] == 0)
    return nx_c_packed_get(s->base, s->offset, 1) ? ~(uint64_t)0 : 0;
  return nx_c_bits_load(s->base, s->offset + e, 64);
}

static inline uint64_t nx_c_bit_word_any(const nx_c_packed_src *s,
                                         int64_t e) {
  if (s->strides[0] != -1) return nx_c_bit_word(s, e);
  return nx_c_packed_reverse(nx_c_bits_load(s->base, s->offset - e - 63, 64),
                             1, 64);
}

#define NX_C_BIT_LOGIC_WORDS(name, WORD, OP)                                   \
  static void name(const nx_c_bit_logic_ctx *c, int64_t e, int64_t n,          \
                   uint8_t *dst) {                                             \
    for (int64_t j = 0; j < n; j++) {                                          \
      uint64_t a = WORD(&c->a, e + 64 * j);                                    \
      uint64_t b = WORD(&c->b, e + 64 * j);                                    \
      nx_c_st64(dst + 8 * j, OP);                                              \
    }                                                                          \
  }
NX_C_BIT_LOGIC_WORDS(nx_c_bit_and_words, nx_c_bit_word, a & b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_or_words, nx_c_bit_word, a | b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_xor_words, nx_c_bit_word, a ^ b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_and_words_any, nx_c_bit_word_any, a & b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_or_words_any, nx_c_bit_word_any, a | b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_xor_words_any, nx_c_bit_word_any, a ^ b)
#undef NX_C_BIT_LOGIC_WORDS

/* Whole words of two operands each one run or one element broadcast, word by
   word, and of operands of which one is a run reversed likewise; any other
   layout through the reader. */
static void nx_c_bit_logic_words(const void *vctx, int64_t e, int64_t n,
                                 uint8_t *dst) {
  const nx_c_bit_logic_ctx *c = vctx;
  int64_t sa = c->a.strides[0], sb = c->b.strides[0];
  if (c->a.ndim != 1 || c->b.ndim != 1 || sa < -1 || sa > 1 || sb < -1 ||
      sb > 1) {
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_bit_logic_fill(c, e + 64 * j, 64));
    return;
  }
  bool reversed = sa == -1 || sb == -1;
  switch (c->op) {
  case NX_C_BIT_AND:
    (reversed ? nx_c_bit_and_words_any : nx_c_bit_and_words)(c, e, n, dst);
    return;
  case NX_C_BIT_OR:
    (reversed ? nx_c_bit_or_words_any : nx_c_bit_or_words)(c, e, n, dst);
    return;
  case NX_C_BIT_XOR:
    (reversed ? nx_c_bit_xor_words_any : nx_c_bit_xor_words)(c, e, n, dst);
    return;
  }
}

nx_c_status nx_c_bit_logic(nx_c_bit_op op, const nx_c_ndarray *out,
                           const nx_c_ndarray *a, const nx_c_ndarray *b) {
  if (out->ndim != a->ndim || out->ndim != b->ndim)
    return NX_C_ERR_RANK_MISMATCH;
  for (int d = 0; d < out->ndim; d++)
    if (out->shape[d] != a->shape[d] || out->shape[d] != b->shape[d])
      return NX_C_ERR_SHAPE;
  nx_c_bit_logic_ctx c;
  nx_c_packed_src_init(&c.a, a, 1);
  nx_c_packed_src_init(&c.b, b, 1);
  c.op = op;
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  nx_c_packed_filler f = {nx_c_bit_logic_fill, nx_c_bit_logic_words, &c};
  return nx_c_packed_write(out, 1, &f, 3 * ((total + 7) / 8));
}

/* Reduce: or (max, any) and and (min, all) of bit along axes. The view is
   split in two: the kept axes, in order, which number the outputs, and the
   reduced axes, which an output's elements range over. The order of the
   reduced axes does not change an or or an and, so they are sorted by
   decreasing stride and turned to positive strides, which makes the reduced
   block of a transposed or flipped view one run when its storage is.

   An output reads its block a run at a time and stops at its first decisive
   word: a set bit for or, a clear one for and. When the block is not one run
   but the kept axes are, reversed or not, as for the rows of [h; w] along
   axis 0, a word of outputs is reduced at once: the or (and) of the words of kept elements at
   each point of the block, stopping when every output is decided. */

typedef struct {
  nx_c_packed_src kept;    /* the kept axes, from the block's first element */
  nx_c_packed_src reduced; /* the reduced axes, from position 0 */
  int64_t size;            /* elements in a block */
  uint64_t flip;           /* 0 for or, ~0 for and: a decisive bit is 1 */
  bool across;             /* a word of outputs at once */
} nx_c_bit_reduce_ctx;

/* The storage position of element e of s. */
static int64_t nx_c_packed_pos(const nx_c_packed_src *s, int64_t e) {
  int last = s->ndim - 1;
  int64_t n_in = s->shape[last];
  return nx_c_packed_row(s, e / n_in) + (e % n_in) * s->strides[last];
}

/* Whether bits [bit, bit + n) of the storage at base hold a bit that is 1
   after xor with flip. Whole words are read 8 at a time between the partial
   ones at the ends. */
static bool nx_c_bits_decisive(const uint8_t *base, int64_t bit, int64_t n,
                               uint64_t flip) {
  int64_t end = bit + n;
  if (n == 0) return false;
  if (bit & 63) {
    int k = (int)(64 - (bit & 63) < n ? 64 - (bit & 63) : n);
    if ((nx_c_bits_load(base, bit, k) ^ flip) & nx_c_low_bits(k)) return true;
    bit += k;
  }
  for (; bit + 512 <= end; bit += 512) {
    const uint8_t *p = base + (bit >> 3);
    uint64_t any = 0;
    for (int j = 0; j < 8; j++) any |= nx_c_ld64(p + 8 * j) ^ flip;
    if (any) return true;
  }
  for (; bit + 64 <= end; bit += 64)
    if (nx_c_ld64(base + (bit >> 3)) ^ flip) return true;
  if (bit < end) {
    int k = (int)(end - bit);
    if ((nx_c_bits_load(base, bit, k) ^ flip) & nx_c_low_bits(k)) return true;
  }
  return false;
}

/* Whether the block of output o holds a decisive bit. */
static bool nx_c_bit_block(const nx_c_bit_reduce_ctx *c, int64_t o) {
  nx_c_packed_src r = c->reduced;
  r.offset = nx_c_packed_pos(&c->kept, o);
  if (nx_c_packed_dense(&r))
    return nx_c_bits_decisive(r.base, r.offset, c->size, c->flip);
  for (int64_t i = 0; i < c->size; i += 64) {
    int k = (int)(c->size - i < 64 ? c->size - i : 64);
    if ((nx_c_packed_read(&r, i, k) ^ c->flip) & nx_c_low_bits(k)) return true;
  }
  return false;
}

static uint64_t nx_c_bit_reduce_fill(const void *vctx, int64_t e, int k) {
  const nx_c_bit_reduce_ctx *c = vctx;
  uint64_t all = nx_c_low_bits(k), found = 0;
  if (!c->across) {
    for (int j = 0; j < k; j++)
      found |= (uint64_t)nx_c_bit_block(c, e + j) << j;
    return found ^ (c->flip & all);
  }
  /* The points of the block by an odometer over the reduced axes. */
  const nx_c_packed_src *r = &c->reduced;
  nx_c_packed_src kept = c->kept;
  int64_t at[NX_C_MAX_NDIM] = {0};
  for (int64_t i = 0; i < c->size && found != all; i++) {
    found |= (nx_c_packed_read(&kept, e, k) ^ c->flip) & all;
    for (int d = r->ndim - 1; d >= 0; d--) {
      kept.offset += r->strides[d];
      if (++at[d] < r->shape[d]) break;
      kept.offset -= at[d] * r->strides[d];
      at[d] = 0;
    }
  }
  return found ^ (c->flip & all);
}

nx_c_status nx_c_bit_reduce(nx_c_bit_op op, const nx_c_ndarray *out,
                            const nx_c_ndarray *in, const int *axes, int n) {
  bool reduced[NX_C_MAX_NDIM] = {false};
  for (int i = 0; i < n; i++) {
    if (axes[i] < 0 || axes[i] >= in->ndim || reduced[axes[i]])
      return NX_C_ERR_AXES;
    reduced[axes[i]] = true;
  }
  if (out->ndim != in->ndim - n) return NX_C_ERR_OUT_RANK;

  nx_c_ndarray kept = *in, block = *in;
  kept.ndim = 0;
  block.ndim = 0;
  block.offset = 0;
  int64_t size = 1;
  for (int d = 0; d < in->ndim; d++) {
    int64_t len = in->shape[d], stride = in->strides[d];
    if (!reduced[d]) {
      if (out->shape[kept.ndim] != len) return NX_C_ERR_SHAPE;
      kept.shape[kept.ndim] = len;
      kept.strides[kept.ndim++] = stride;
      continue;
    }
    /* An or or an and of copies is the element's, so a broadcast axis
       counts once; one of length 0 still empties the block. */
    if (stride == 0 && len > 0) continue;
    if (stride < 0) {
      kept.offset += (len - 1) * stride;
      stride = -stride;
    }
    int j = block.ndim++;
    for (; j > 0 && block.strides[j - 1] < stride; j--) {
      block.shape[j] = block.shape[j - 1];
      block.strides[j] = block.strides[j - 1];
    }
    block.shape[j] = len;
    block.strides[j] = stride;
    size *= len;
  }

  nx_c_bit_reduce_ctx c;
  nx_c_packed_src_init(&c.kept, &kept, 1);
  nx_c_packed_src_init(&c.reduced, &block, 1);
  c.size = size;
  c.flip = op == NX_C_BIT_AND ? ~(uint64_t)0 : 0;
  int64_t outputs = 1;
  for (int d = 0; d < out->ndim; d++) outputs *= out->shape[d];
  c.across = outputs > 1 && c.kept.ndim == 1 &&
             (c.kept.strides[0] == 1 || c.kept.strides[0] == -1) &&
             !nx_c_packed_dense(&c.reduced);
  nx_c_packed_filler f = {nx_c_bit_reduce_fill, NULL, &c};
  return nx_c_packed_write(out, 1, &f, (outputs * size + 7) / 8);
}

/* Pad: out's elements in C order are the pad value outside in's box and in's
   elements inside it. Each worker's words are composed as a stream of runs,
   the value over a border, in's row across the box, so a row of a padded
   image costs about a copy of its words. */

/* Whole words of storage written run after run: each run is appended to the
   bits of the last, and a word is stored once it is full. */
typedef struct {
  uint8_t *dst;
  uint64_t acc;
  int used; /* bits of acc that hold elements, < 64 */
} nx_c_bit_stream;

/* Appends the low n bits of v, 0 < n <= 64, whose bits above n are 0. The
   bits of v past a full word are v >> (64 - used), shifted in two steps so
   that a used of 0 shifts by 64 and keeps none. */
static inline void nx_c_stream_put(nx_c_bit_stream *st, uint64_t v, int n) {
  int used = st->used;
  st->acc |= v << used;
  st->used = used + n;
  if (st->used < 64) return;
  nx_c_st64(st->dst, st->acc);
  st->dst += 8;
  st->used -= 64;
  st->acc = (v >> 1) >> (63 - used);
}

/* Appends bits [bit, bit + n) of the storage at base. The bits' offset in
   their bytes and the stream's in its word stay the same across the run's
   words, so each word is one or two loads and one store. */
static void nx_c_stream_run(nx_c_bit_stream *st, const uint8_t *base,
                            int64_t bit, int64_t n) {
  const uint8_t *q = base + (bit >> 3);
  int sh = (int)(bit & 7), u = st->used;
  uint8_t *dst = st->dst;
  uint64_t acc = st->acc;
  int64_t words = n >> 6;
  for (int64_t j = 0; j < words; j++, q += 8, dst += 8) {
    uint64_t v = nx_c_ld64(q);
    if (sh) v = (v >> sh) | ((uint64_t)q[8] << (64 - sh));
    nx_c_st64(dst, acc | (v << u));
    acc = (v >> 1) >> (63 - u);
  }
  st->dst = dst;
  st->acc = acc;
  int rest = (int)(n & 63);
  if (rest)
    nx_c_stream_put(st, nx_c_bits_load(base, bit + (words << 6), rest), rest);
}

typedef struct {
  nx_c_packed_src in;
  int bits;
  int ndim; /* >= 1 */
  int64_t shape[NX_C_MAX_NDIM];  /* out's */
  int64_t before[NX_C_MAX_NDIM]; /* where in's box starts in out */
  int64_t inner[NX_C_MAX_NDIM];  /* in's shape */
  uint64_t value;                /* a word of the pad value */
} nx_c_packed_pad_ctx;

/* The runs below work on a copy of the stream, which stays in registers
   where the stream itself could alias the words it stores. */

static void nx_c_pad_value(const nx_c_packed_pad_ctx *p, int64_t n,
                           nx_c_bit_stream *st) {
  nx_c_bit_stream s = *st;
  int per = 64 / p->bits;
  for (; n > 0; n -= per) {
    int k = (int)(n < per ? n : per);
    nx_c_stream_put(&s, p->value & nx_c_low_bits(k * p->bits), k * p->bits);
  }
  *st = s;
}

static void nx_c_pad_input(const nx_c_packed_pad_ctx *p, int64_t ie, int64_t n,
                           nx_c_bit_stream *st) {
  nx_c_bit_stream s = *st;
  int per = 64 / p->bits;
  if (nx_c_packed_dense(&p->in)) {
    nx_c_stream_run(&s, p->in.base, (p->in.offset + ie) * p->bits,
                    n * p->bits);
  } else {
    for (; n > 0; n -= per, ie += per) {
      int k = (int)(n < per ? n : per);
      nx_c_stream_put(&s, nx_c_packed_read(&p->in, ie, k), k * p->bits);
    }
  }
  *st = s;
}

/* Elements [e, e + n) of out into st, a row at a time. */
static void nx_c_pad_emit(const nx_c_packed_pad_ctx *p, int64_t e, int64_t n,
                          nx_c_bit_stream *st) {
  int last = p->ndim - 1;
  int64_t w = p->shape[last], b = p->before[last], iw = p->inner[last];
  int64_t row = e / w, col = e % w;
  for (; n > 0; row++, col = 0) {
    int64_t end = col + n < w ? col + n : w;
    n -= end - col;

    /* The row of in this row reads, when it is inside the box. */
    bool inside = true;
    int64_t irow = 0, scale = 1, r = row;
    for (int d = last - 1; d >= 0; d--) {
      int64_t c = r % p->shape[d] - p->before[d];
      r /= p->shape[d];
      if (c < 0 || c >= p->inner[d]) inside = false;
      irow += c * scale;
      scale *= p->inner[d];
    }
    if (!inside) {
      nx_c_pad_value(p, end - col, st);
      continue;
    }

    int64_t lo = col > b ? col : b, hi = end < b + iw ? end : b + iw;
    if (lo >= hi) {
      nx_c_pad_value(p, end - col, st);
      continue;
    }
    nx_c_pad_value(p, lo - col, st);
    nx_c_pad_input(p, irow * iw + (lo - b), hi - lo, st);
    nx_c_pad_value(p, end - hi, st);
  }
}

static uint64_t nx_c_pad_word(const void *ctx, int64_t e, int k) {
  const nx_c_packed_pad_ctx *p = ctx;
  uint8_t word[8];
  nx_c_bit_stream st = {word, 0, 0};
  nx_c_pad_emit(p, e, k, &st);
  return st.dst == word ? st.acc : nx_c_ld64(word);
}

static void nx_c_pad_words(const void *ctx, int64_t e, int64_t n,
                           uint8_t *dst) {
  const nx_c_packed_pad_ctx *p = ctx;
  nx_c_bit_stream st = {dst, 0, 0};
  nx_c_pad_emit(p, e, n * (64 / p->bits), &st);
}

nx_c_status nx_c_packed_pad(const nx_c_ndarray *out, const nx_c_ndarray *in,
                            const nx_c_ndarray *value, const int64_t *before,
                            nx_c_dtype dt) {
  if (out->ndim != in->ndim) return NX_C_ERR_SHAPE;
  nx_c_packed_pad_ctx p;
  p.bits = nx_c_packed_bits(dt);
  p.ndim = out->ndim > 0 ? out->ndim : 1;
  p.shape[0] = 1;
  p.before[0] = 0;
  p.inner[0] = 1;
  for (int d = 0; d < out->ndim; d++) {
    p.shape[d] = out->shape[d];
    p.before[d] = before[d];
    p.inner[d] = in->shape[d];
    if (before[d] < 0 || before[d] + in->shape[d] > out->shape[d])
      return NX_C_ERR_SHAPE;
  }
  nx_c_packed_src_init(&p.in, in, p.bits);
  uint8_t v = nx_c_packed_get(value->data, value->offset, p.bits);
  p.value = nx_c_packed_splat(v, p.bits, 64 / p.bits);
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  nx_c_packed_filler f = {nx_c_pad_word, nx_c_pad_words, &p};
  return nx_c_packed_write(out, p.bits, &f, 2 * nx_c_dtype_bytes(dt, total));
}
