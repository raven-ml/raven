/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Sorts.

   Each slice along the axis sorts on its own, slices on the job's threads.
   A slice's elements become keys, unsigned integers in the order of
   nx_kinds.h's domains: a float's bits with every bit flipped where the
   sign is set and the sign flipped elsewhere, so -0 sits below +0; every
   NaN the greatest key, so NaNs are equal and last; a narrow float its
   float32 carrier's key; a signed integer its bits with the sign flipped; a
   complex number its real part's key, then its imaginary part's. A
   descending sort complements the keys, so that sorting them ascending,
   stably, gives the reversed order with equal elements in increasing
   position.

   Keys sort with their positions, stably: by insertion below SMALL
   elements; into k slots where a sort keeps k of many; otherwise by LSD
   radix of 8-bit digits, skipping a digit every key shares, from one
   pass that counts every digit. A complex128's two keys sort by the
   imaginary part's, then stably by the real part's. Values then move once,
   by position, their bits unchanged. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* Slices of fewer elements sort by insertion. */
#define SMALL 64

/* A sort that keeps at most KEEP elements, of a slice at least SPARE
   times longer, keeps them in slots. */
#define KEEP 64
#define SPARE 8

/* A single slice at least this long sorts on several threads. */
#define LONG (64 * 1024)

typedef struct {
  const nx_array *a; /* the values, the positions, the operand */
  int dt, w, kb;     /* the dtype, its bytes and its keys' */
  int descending;
  int64_t n, keep;   /* a slice's elements, and those it keeps */
  int64_t sx, sv, sp; /* steps along the axis */
  nx_loop l;          /* the slices: operands values, positions, x */
  uint64_t *scratch;  /* per worker: five arrays of n words */
} sorter;

/* Keys */

static inline uint64_t rank32(uint32_t b) {
  return b & 0x80000000u ? (uint32_t)~b : b | 0x80000000u;
}

static inline uint64_t rank64(uint64_t b) {
  return b & 0x8000000000000000ull ? ~b : b | 0x8000000000000000ull;
}

static inline uint64_t key32(float f) {
  return nx_float_nan(f) ? 0xFFFFFFFFull : rank32(nx_float_bits(f));
}

static inline uint64_t key64(double f) {
  return f != f ? ~0ull : rank64(nx_double_bits(f));
}

/* The key of the element at [p], and for a complex128 the real part's key
   into [hi]. */
static uint64_t key_of(const sorter *s, const uint8_t *p, uint64_t *hi) {
  switch (s->dt) {
    case NX_FLOAT32: return key32(*(const float *)p);
    case NX_FLOAT64: return key64(*(const double *)p);
    case NX_FLOAT16:
    case NX_BFLOAT16: return key32(nx_bits_to_float(s->dt, *(const uint16_t *)p));
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: return key32(nx_bits_to_float(s->dt, *p));
    case NX_INT64: return *(const uint64_t *)p ^ 0x8000000000000000ull;
    case NX_INT32: return *(const uint32_t *)p ^ 0x80000000u;
    case NX_INT16: return (uint16_t)(*(const uint16_t *)p ^ 0x8000u);
    case NX_INT8: return (uint8_t)(*p ^ 0x80u);
    case NX_UINT64: return *(const uint64_t *)p;
    case NX_UINT32: return *(const uint32_t *)p;
    case NX_UINT16: return *(const uint16_t *)p;
    case NX_COMPLEX64: {
      const float *c = (const float *)p;
      if (nx_float_nan(c[0]) || nx_float_nan(c[1])) return ~0ull;
      return key32(c[0]) << 32 | key32(c[1]);
    }
    case NX_COMPLEX128: {
      const double *c = (const double *)p;
      if (c[0] != c[0] || c[1] != c[1]) {
        *hi = ~0ull;
        return ~0ull;
      }
      *hi = key64(c[0]);
      return key64(c[1]);
    }
    default: return *p; /* uint8, bool */
  }
}

/* The bytes of [dt]'s keys. */
static int key_bytes(int dt) {
  switch (dt) {
    case NX_FLOAT16:
    case NX_BFLOAT16:
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: return 4;
    case NX_COMPLEX64:
    case NX_COMPLEX128: return 8;
    default: return nx_dtype_row_of(dt).bits / 8;
  }
}

/* Orders */

/* Sorts the [n] keys [k] with their positions [p] by insertion, stably. */
static void insertion(uint64_t *k, int64_t *p, int64_t n) {
  for (int64_t i = 1; i < n; i++) {
    uint64_t x = k[i];
    int64_t y = p[i], j = i;
    for (; j > 0 && k[j - 1] > x; j--) {
      k[j] = k[j - 1];
      p[j] = p[j - 1];
    }
    k[j] = x;
    p[j] = y;
  }
}

/* Keeps in [k] and [p] the [keep] least of the [n] keys [in], in order,
   the earlier position first among equal ones. */
static void slots(const uint64_t *in, int64_t n, int64_t keep, uint64_t *k,
                  int64_t *p) {
  int64_t used = 0;
  for (int64_t i = 0; i < n; i++) {
    uint64_t x = in[i];
    if (used == keep && x >= k[keep - 1]) continue;
    int64_t j = used < keep ? used++ : keep - 1;
    for (; j > 0 && k[j - 1] > x; j--) {
      k[j] = k[j - 1];
      p[j] = p[j - 1];
    }
    k[j] = x;
    p[j] = i;
  }
}

/* Sorts the [n] keys [k] of [kb] bytes with their positions [p] by LSD
   radix, stably, through the buffers [k2] and [p2]; the result lands back
   in [k] and [p]. */
static void radix(uint64_t *k, int64_t *p, uint64_t *k2, int64_t *p2,
                  int64_t n, int kb) {
  int64_t count[8][256];
  memset(count, 0, (size_t)kb * sizeof count[0]);
  for (int64_t i = 0; i < n; i++)
    for (int d = 0; d < kb; d++) count[d][(k[i] >> (8 * d)) & 255]++;
  uint64_t *ks = k, *kd = k2;
  int64_t *ps = p, *pd = p2;
  for (int d = 0; d < kb; d++) {
    int64_t *c = count[d], at = 0;
    if (c[(ks[0] >> (8 * d)) & 255] == n) continue;
    for (int b = 0; b < 256; b++) {
      int64_t x = c[b];
      c[b] = at;
      at += x;
    }
    for (int64_t i = 0; i < n; i++) {
      int64_t j = c[(ks[i] >> (8 * d)) & 255]++;
      kd[j] = ks[i];
      pd[j] = ps[i];
    }
    uint64_t *kt = ks;
    ks = kd;
    kd = kt;
    int64_t *pt = ps;
    ps = pd;
    pd = pt;
  }
  if (ks != k) {
    memcpy(k, ks, (size_t)n * sizeof *k);
    memcpy(p, ps, (size_t)n * sizeof *p);
  }
}

/* Sorts the [n] items [t], each a key of [kb] bytes, at most 4, above its
   position in the low 32 bits, by LSD radix over the key's digits through
   [t2]: half the bytes a pass moves of radix's. The result lands back in
   [t]. */
static void radix_packed(uint64_t *t, uint64_t *t2, int64_t n, int kb) {
  int64_t count[4][256];
  memset(count, 0, (size_t)kb * sizeof count[0]);
  for (int64_t i = 0; i < n; i++)
    for (int d = 0; d < kb; d++) count[d][(t[i] >> (32 + 8 * d)) & 255]++;
  uint64_t *ts = t, *td = t2;
  for (int d = 0; d < kb; d++) {
    int64_t *c = count[d], at = 0;
    int shift = 32 + 8 * d;
    if (c[(ts[0] >> shift) & 255] == n) continue;
    for (int b = 0; b < 256; b++) {
      int64_t x = c[b];
      c[b] = at;
      at += x;
    }
    for (int64_t i = 0; i < n; i++) td[c[(ts[i] >> shift) & 255]++] = ts[i];
    uint64_t *tt = ts;
    ts = td;
    td = tt;
  }
  if (ts != t) memcpy(t, ts, (size_t)n * sizeof *t);
}

/* A long slice's radix on the job's threads: each pass, every block of
   items counts its digits, the counts become each block's offsets, digit
   by digit and block by block, and every block moves its items to them.
   Blocks keep their items' order, so the pass stays stable. Keys of at
   most 4 bytes travel packed above their positions; longer ones beside
   them. */

typedef struct {
  const sorter *s;
  const uint8_t *x; /* the slice's first element */
  uint64_t *src, *dst;
  int64_t *psrc, *pdst; /* positions beside keys of 8 bytes, else NULL */
  int64_t n, blocks;
  int shift;
  int64_t (*count)[256];
  int64_t pv, pp; /* the slice's first value and position */
} spread;

static void block_range(const spread *p, int64_t b, int64_t *lo, int64_t *hi) {
  *lo = p->n * b / p->blocks;
  *hi = p->n * (b + 1) / p->blocks;
}

static void key_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  const sorter *s = p->s;
  uint64_t mask = s->kb == 8 ? ~0ull : (1ull << (8 * s->kb)) - 1, unused;
  uint64_t flip = s->descending ? mask : 0;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1;
    block_range(p, b, &i0, &i1);
    for (int64_t i = i0; i < i1; i++) {
      uint64_t k = key_of(s, p->x + i * s->sx * s->w, &unused) ^ flip;
      if (p->psrc == NULL) p->src[i] = k << 32 | (uint64_t)i;
      else {
        p->src[i] = k;
        p->psrc[i] = i;
      }
    }
  }
}

static void count_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1, *c = p->count[b];
    block_range(p, b, &i0, &i1);
    memset(c, 0, 256 * sizeof *c);
    for (int64_t i = i0; i < i1; i++) c[(p->src[i] >> p->shift) & 255]++;
  }
}

static void move_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1, at[256];
    block_range(p, b, &i0, &i1);
    memcpy(at, p->count[b], sizeof at);
    if (p->psrc == NULL)
      for (int64_t i = i0; i < i1; i++)
        p->dst[at[(p->src[i] >> p->shift) & 255]++] = p->src[i];
    else
      for (int64_t i = i0; i < i1; i++) {
        int64_t j = at[(p->src[i] >> p->shift) & 255]++;
        p->dst[j] = p->src[i];
        p->pdst[j] = p->psrc[i];
      }
  }
}

/* Writes the positions and values of the sorted items' blocks. */
static void emit_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  const sorter *s = p->s;
  int64_t *pos = (int64_t *)s->a[1].base + p->pp;
  uint8_t *v = s->a[0].base + p->pv * s->w;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1;
    block_range(p, b, &i0, &i1);
    if (i1 > s->keep) i1 = s->keep;
    for (int64_t i = i0; i < i1; i++) {
      int64_t q =
          p->psrc == NULL ? (int64_t)(p->src[i] & 0xFFFFFFFFu) : p->psrc[i];
      pos[i * s->sp] = q;
      memcpy(v + i * s->sv * s->w, p->x + q * s->sx * s->w, (size_t)s->w);
    }
  }
}

/* Sorts the one slice of [s] at [pv], [pp] and [px] through the scratch
   [t], four arrays of its length, on [blocks] blocks. */
static void sort_long(const sorter *s, int64_t pv, int64_t pp, int64_t px,
                      uint64_t *t, int64_t (*count)[256], int64_t blocks,
                      int64_t bytes) {
  int64_t n = s->n;
  int packed = s->kb <= 4;
  spread p = {.s = s,
              .x = s->a[2].base + px * s->w,
              .src = t,
              .dst = t + n,
              .psrc = packed ? NULL : (int64_t *)(t + 2 * n),
              .pdst = packed ? NULL : (int64_t *)(t + 3 * n),
              .n = n,
              .blocks = blocks,
              .count = count,
              .pv = pv,
              .pp = pp};
  nx_cpu_job(blocks, bytes, bytes, key_blocks, &p);
  for (int d = 0; d < s->kb; d++) {
    p.shift = (packed ? 32 : 0) + 8 * d;
    nx_cpu_job(blocks, bytes, bytes, count_blocks, &p);
    /* Offsets, digit by digit and block by block; a digit every key shares
       leaves the pass out. */
    int64_t at = 0, skip = 0;
    for (int g = 0; g < 256; g++) {
      int64_t all = 0;
      for (int64_t b = 0; b < blocks; b++) {
        int64_t c = count[b][g];
        count[b][g] = at;
        at += c;
        all += c;
      }
      skip |= all == n;
    }
    if (skip) continue;
    nx_cpu_job(blocks, bytes, bytes, move_blocks, &p);
    uint64_t *tt = p.src;
    p.src = p.dst;
    p.dst = tt;
    int64_t *pt = p.psrc;
    p.psrc = p.pdst;
    p.pdst = pt;
  }
  nx_cpu_job(blocks, bytes, bytes, emit_blocks, &p);
}

/* Slices */

static void sort_slice(const sorter *s, int64_t pv, int64_t pp, int64_t px,
                       uint64_t *scratch) {
  int64_t n = s->n;
  uint64_t *k = scratch, *k2 = scratch + n, *hi = scratch + 2 * n;
  int64_t *p = (int64_t *)(scratch + 3 * n), *p2 = (int64_t *)(scratch + 4 * n);
  const uint8_t *x = s->a[2].base + px * s->w;
  uint64_t mask = s->kb == 8 ? ~0ull : (1ull << (8 * s->kb)) - 1;
  uint64_t flip = s->descending ? mask : 0;
  for (int64_t i = 0; i < n; i++) {
    k[i] = key_of(s, x + i * s->sx * s->w, &hi[i]) ^ flip;
    hi[i] ^= flip;
    p[i] = i;
  }
  int64_t keep = s->keep;
  if (s->dt == NX_COMPLEX128) {
    /* The imaginary parts' order, then the real parts' over it. */
    radix(k, p, k2, p2, n, 8);
    for (int64_t i = 0; i < n; i++) k[i] = hi[p[i]];
    radix(k, p, k2, p2, n, 8);
  } else if (n < SMALL)
    insertion(k, p, n);
  else if (keep <= KEEP && keep * SPARE <= n) {
    memcpy(k2, k, (size_t)n * sizeof *k);
    slots(k2, n, keep, k, p);
  } else if (s->kb <= 4 && n <= UINT32_MAX) {
    for (int64_t i = 0; i < n; i++) k[i] = k[i] << 32 | (uint64_t)i;
    radix_packed(k, k2, n, s->kb);
    for (int64_t i = 0; i < keep; i++) p[i] = (int64_t)(k[i] & 0xFFFFFFFFu);
  } else
    radix(k, p, k2, p2, n, s->kb);
  int64_t *pos = (int64_t *)s->a[1].base + pp;
  for (int64_t i = 0; i < keep; i++) pos[i * s->sp] = p[i];
  uint8_t *v = s->a[0].base + pv * s->w;
  int64_t sv = s->sv, sx = s->sx;
  switch (s->w) {
#define MOVE(T)                                                              \
  for (int64_t i = 0; i < keep; i++)                                         \
    ((T *)v)[i * sv] = ((const T *)x)[p[i] * sx];                            \
  break
    case 1: MOVE(uint8_t);
    case 2: MOVE(uint16_t);
    case 4: MOVE(uint32_t);
    case 8: MOVE(uint64_t);
#undef MOVE
    default:
      for (int64_t i = 0; i < keep; i++)
        memcpy(v + i * sv * s->w, x + p[i] * sx * s->w, (size_t)s->w);
  }
}

static void sort_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  const sorter *s = ctx;
  const nx_loop *l = &s->l;
  uint64_t *scratch = s->scratch + (int64_t)worker * 5 * s->n;
  for (int64_t u = lo; u < hi; u++) {
    int64_t v = u, pv = l->first[0], pp = l->first[1], px = l->first[2];
    for (int i = l->rank - 1; i >= 0; i--) {
      int64_t x = v % l->extent[i];
      v /= l->extent[i];
      pv += x * l->step[0][i];
      pp += x * l->step[1][i];
      px += x * l->step[2][i];
    }
    sort_slice(s, pv, pp, px, scratch);
  }
}

value nx_cpu_sort(value vs, value vv, value vp, value vx) {
  CAMLparam4(vs, vv, vp, vx);
  const nx_spec_axis *sp = (const nx_spec_axis *)String_val(vs);
  int axis = sp->axis, descending = sp->combine;
  int64_t k = sp->k;
  int dt = nx_array_dtype(vx);
  if (nx_dtype_row_of(dt).bits < 8) CAMLreturn(Val_int(NX_DECLINED));
  int64_t xs[2 * NX_MAX_RANK], vs_[2 * NX_MAX_RANK], ps[2 * NX_MAX_RANK], off;
  int rx = nx_array_layout(vx, xs, &off), rv = nx_array_layout(vv, vs_, &off),
      rp = nx_array_layout(vp, ps, &off);
  if (axis >= rx || rv != rx || rp != rx) CAMLreturn(Val_int(NX_SHAPE));
  int64_t keep = k < 0 ? xs[axis] : k;
  if (keep > xs[axis]) CAMLreturn(Val_int(NX_SHAPE));
  for (int i = 0; i < rx; i++) {
    int64_t want = i == axis ? keep : xs[i];
    if (vs_[i] != want || ps[i] != want) CAMLreturn(Val_int(NX_SHAPE));
  }
  nx_operand in[3] = {{vv, dt, 1}, {vp, NX_INT64, 1}, {vx, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (e) CAMLreturn(Val_int(e));
  int64_t slices = 1;
  for (int i = 0; i < rx; i++)
    if (i != axis) slices *= xs[i];
  if (slices == 0 || keep == 0) {
    nx_done(3, a);
    CAMLreturn(Val_int(NX_OK));
  }
  sorter s = {.a = a,
              .dt = dt,
              .w = nx_dtype_row_of(dt).bits / 8,
              .kb = key_bytes(dt),
              .descending = descending,
              .n = xs[axis],
              .keep = keep,
              .sx = a[2].dim[rx + axis],
              .sv = a[0].dim[rx + axis],
              .sp = a[1].dim[rx + axis]};
  s.l.rank = rx;
  s.l.first[0] = a[0].offset;
  s.l.first[1] = a[1].offset;
  s.l.first[2] = a[2].offset;
  for (int i = 0; i < rx; i++) {
    s.l.extent[i] = i == axis ? 1 : xs[i];
    s.l.step[0][i] = a[0].dim[rx + i];
    s.l.step[1][i] = a[1].dim[rx + i];
    s.l.step[2][i] = a[2].dim[rx + i];
  }
  s.l.rank = nx_coalesce_dims(3, rx, s.l.extent, s.l.step);
  /* A slice's sort costs as long as copying its elements about eight
     times, a radix's passes. */
  int64_t bytes = slices * s.n * (s.w + 8);
  int64_t workers = nx_cpu_threads(bytes, 8 * bytes);
  /* One long slice sorts on the job's threads, a block of its items each;
     a complex128's two keys, and keeping a few, stay on one. */
  int slots = keep <= KEEP && keep * SPARE <= s.n;
  if (slices == 1 && workers > 1 && s.n >= LONG && dt != NX_COMPLEX128 &&
      !slots && s.n <= UINT32_MAX) {
    uint64_t *t = malloc(4 * (size_t)s.n * sizeof *t);
    int64_t(*count)[256] = malloc((size_t)workers * sizeof *count);
    if (t == NULL || count == NULL) {
      free(t);
      free(count);
      nx_done(3, a);
      caml_raise_out_of_memory();
    }
    sort_long(&s, s.l.first[0], s.l.first[1], s.l.first[2], t, count,
              workers, bytes);
    free(t);
    free(count);
    nx_done(3, a);
    CAMLreturn(Val_int(NX_OK));
  }
  if (workers > slices) workers = slices;
  s.scratch = malloc((size_t)workers * 5 * (size_t)s.n * sizeof(uint64_t));
  if (s.scratch == NULL) {
    nx_done(3, a);
    caml_raise_out_of_memory();
  }
  nx_cpu_job(slices, bytes, 8 * bytes, sort_units, &s);
  free(s.scratch);
  nx_done(3, a);
  CAMLreturn(Val_int(NX_OK));
}
