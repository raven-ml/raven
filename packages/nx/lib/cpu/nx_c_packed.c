/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_packed.c — the sub-byte family's reader, writer and kernels: copy,
   gather and the logical operations of bit. nx_c_packed.h states the storage
   and the rules every load and store keeps. The casts to and from bit live
   with the other casts, in nx_c_map.c. */

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
    else {
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
   byte, its words through the funnel shift when it starts inside one. */
static void nx_c_packed_copy_words(const void *ctx, int64_t e, int64_t n,
                                   uint8_t *dst) {
  const nx_c_packed_src *s = ctx;
  int per = 64 / s->bits;
  if (!nx_c_packed_dense(s)) {
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_packed_read_any(s, e + j * per, per));
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
   replaced by indices' element c, or 0 for an index outside the axis. */

typedef struct {
  const nx_c_ndarray *out;
  const nx_c_ndarray *data;
  const nx_c_ndarray *indices;
  int axis;
  int bits;
} nx_c_packed_gather_ctx;

static uint64_t nx_c_packed_gather_fill(const void *vctx, int64_t e, int k) {
  const nx_c_packed_gather_ctx *g = vctx;
  const nx_c_ndarray *data = g->data, *ix = g->indices, *out = g->out;
  const int64_t *index = (const int64_t *)ix->data;
  int64_t n = data->shape[g->axis];
  int bits = g->bits;
  uint64_t w = 0;
  if (out->ndim == 1) {
    for (int j = 0; j < k; j++) {
      int64_t i = index[ix->offset + (e + j) * ix->strides[0]];
      if ((uint64_t)i < (uint64_t)n)
        w |= (uint64_t)nx_c_packed_get(
                 data->data, data->offset + i * data->strides[0], bits)
             << (j * bits);
    }
    return w;
  }
  for (int j = 0; j < k; j++) {
    int64_t r = e + j, ioff = ix->offset, doff = data->offset, i = 0;
    int64_t coord[NX_C_MAX_NDIM];
    for (int d = out->ndim - 1; d >= 0; d--) {
      coord[d] = r % out->shape[d];
      r /= out->shape[d];
      ioff += coord[d] * ix->strides[d];
    }
    i = index[ioff];
    if ((uint64_t)i >= (uint64_t)n) continue;
    for (int d = 0; d < out->ndim; d++)
      doff += (d == g->axis ? i : coord[d]) * data->strides[d];
    w |= (uint64_t)nx_c_packed_get(data->data, doff, bits) << (j * bits);
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
  nx_c_packed_gather_ctx g = {out, data, indices, axis,
                              nx_c_packed_bits(dt)};
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  int64_t bytes = total * (int64_t)sizeof(int64_t) + 2 * nx_c_dtype_bytes(dt, total);
  nx_c_packed_filler f = {nx_c_packed_gather_fill, NULL, &g};
  return nx_c_packed_write(out, g.bits, &f, bytes);
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

/* An operand that is one run or one element broadcast: a word of it from
   element e. */
static inline uint64_t nx_c_bit_word(const nx_c_packed_src *s, int64_t e) {
  if (s->strides[0] == 0)
    return nx_c_packed_get(s->base, s->offset, 1) ? ~(uint64_t)0 : 0;
  return nx_c_bits_load(s->base, s->offset + e, 64);
}

#define NX_C_BIT_LOGIC_WORDS(name, OP)                                         \
  static void name(const nx_c_bit_logic_ctx *c, int64_t e, int64_t n,          \
                   uint8_t *dst) {                                             \
    for (int64_t j = 0; j < n; j++) {                                          \
      uint64_t a = nx_c_bit_word(&c->a, e + 64 * j);                           \
      uint64_t b = nx_c_bit_word(&c->b, e + 64 * j);                           \
      nx_c_st64(dst + 8 * j, OP);                                              \
    }                                                                          \
  }
NX_C_BIT_LOGIC_WORDS(nx_c_bit_and_words, a & b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_or_words, a | b)
NX_C_BIT_LOGIC_WORDS(nx_c_bit_xor_words, a ^ b)
#undef NX_C_BIT_LOGIC_WORDS

/* Whole words of two operands each one run or one element broadcast, word by
   word; any other layout element by element. */
static void nx_c_bit_logic_words(const void *vctx, int64_t e, int64_t n,
                                 uint8_t *dst) {
  const nx_c_bit_logic_ctx *c = vctx;
  if (!(c->a.ndim == 1 && c->b.ndim == 1 &&
        (c->a.strides[0] == 1 || c->a.strides[0] == 0) &&
        (c->b.strides[0] == 1 || c->b.strides[0] == 0))) {
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_bit_logic_fill(c, e + 64 * j, 64));
    return;
  }
  switch (c->op) {
  case NX_C_BIT_AND: nx_c_bit_and_words(c, e, n, dst); return;
  case NX_C_BIT_OR: nx_c_bit_or_words(c, e, n, dst); return;
  case NX_C_BIT_XOR: nx_c_bit_xor_words(c, e, n, dst); return;
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
