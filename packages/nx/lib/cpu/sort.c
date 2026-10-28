/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Sorts.

   Each slice along the axis sorts on its own, slices on the job's threads.
   A slice's elements become keys, unsigned integers in the order of
   prog.mli's Domains section: a float's bits with every bit flipped where the
   sign is set and the sign flipped elsewhere, so -0 sits below +0; every
   NaN the greatest key, so NaNs are equal and last; a narrow float its
   float32 carrier's key, float4's its code ranked as a float's bits; a
   signed integer its bits with the sign flipped; a complex number its real
   part's key, then its imaginary part's. A sub-byte slice's codes are read
   one per byte first. A descending sort complements the keys, so that
   sorting them ascending, stably, gives the reversed order with equal
   elements in increasing position.

   Keys sort with their positions, stably: by insertion below SMALL
   elements; into k slots where a sort keeps k of many; otherwise by LSD
   radix of 8-bit digits of each key less the slice's least, over the
   digits their span takes, skipping a digit every key shares, from one
   pass that counts every digit; a long slice's keys wider than 4 bytes
   take their top digit first. Keys that span at most 4 bytes travel above
   their positions in one word: integers of a narrow range sort as 32-bit
   keys whatever their width. A complex128's two keys sort by the
   imaginary part's, then stably by the real part's. Values then move
   once, by position, their bits unchanged. */

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

/* A long slice's buckets of at most this many items sort in the cache: an
   item's key and position, 16 bytes, through two buffers, 1 MiB a bucket,
   within the 3 MiB L2 of fudge's performance cores, where it was
   measured. */
#define BUCKET (32 * 1024)

/* A single slice at least this long sorts on several threads. */
#define LONG (64 * 1024)

/* A slice's scratch: two arrays of keys and two of positions. */
#define SCRATCH 4

/* What a sort takes, in copies of its elements' and positions' bytes
   (cpu.h, Calls and jobs, streaming; M1 Max and fudge, 2026-10-14): 75 for
   a float64 slice of 4096 (fudge), 100 for complex128 slices of 64 (M1),
   whose two keys sort apart. */
#define CALL_COST 128
#define COMPLEX_CALL_COST 256

typedef struct {
  const nx_array *a; /* the values, the positions, the operand */
  int dt, bits, w, kb; /* the dtype, its bits, its bytes and its keys' */
  int descending;
  int64_t n, keep;   /* a slice's elements, and those it keeps */
  int64_t sx, sv, sp; /* steps along the axis */
  nx_loop l;          /* the slices: operands values, positions, x */
  uint64_t *scratch;  /* per worker: SCRATCH arrays of n words */
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

/* A 4-bit float's code ranked as rank32 ranks a float32's bits. */
static inline uint64_t rank4(uint8_t b) {
  return b & 8 ? ~b & 15u : b | 8u;
}

/* The key of the element of dtype [dt] at [p], a sub-byte one's code in
   the low bits of the byte at [p], int4's sign-extended or not; a
   complex128's is its imaginary part's, real_key gives the other. */
static inline uint64_t key_of(int dt, const uint8_t *p) {
  switch (dt) {
    case NX_FLOAT32: return key32(*(const float *)p);
    case NX_FLOAT64: return key64(*(const double *)p);
    case NX_FLOAT16:
    case NX_BFLOAT16: return key32(nx_bits_to_float(dt, *(const uint16_t *)p));
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: return key32(nx_bits_to_float(dt, *p));
    case NX_INT64: return *(const uint64_t *)p ^ 0x8000000000000000ull;
    case NX_INT32: return *(const uint32_t *)p ^ 0x80000000u;
    case NX_INT16: return (uint16_t)(*(const uint16_t *)p ^ 0x8000u);
    case NX_INT8: return (uint8_t)(*p ^ 0x80u);
    case NX_INT4: return (*p ^ 8u) & 15u;
    case NX_FLOAT4_E2M1FN: return rank4(*p);
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
      return c[0] != c[0] || c[1] != c[1] ? ~0ull : key64(c[1]);
    }
    case NX_BOOL: return *p != 0; /* a boolean is true where not zero */
    default: return *p; /* uint8, uint4 and bit */
  }
}

/* A complex128's real part's key, which orders it before its imaginary
   part's. */
static uint64_t real_key(const double *c) {
  return c[0] != c[0] || c[1] != c[1] ? ~0ull : key64(c[0]);
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
    default: return nx_cpu_width(dt);
  }
}

/* Orders */

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

/* Sorts the [n] keys [k] of [kb] bytes with their positions [p] by
   nx_cpu_radix through the buffers [k2] and [p2]; the result lands back
   in [k] and [p]. */
static void radix(uint64_t *k, int64_t *p, uint64_t *k2, int64_t *p2,
                  int64_t n, int kb) {
  if (!nx_cpu_radix(k, p, k2, p2, n, kb, 0)) return;
  memcpy(k, k2, (size_t)n * sizeof *k);
  memcpy(p, p2, (size_t)n * sizeof *p);
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

/* A long slice's radix on the job's threads. Each block of items finds
   the span of its keys. Then each pass, every block counts its digits, the
   counts become each block's offsets, digit by digit and block by block,
   and every block moves its items to them; blocks keep their items'
   order, so the pass stays stable. Keys spanning at most 4 bytes travel
   packed above their positions, wider ones beside them.

   Wide keys take their top digit first, as an MSD radix does, where that
   leaves buckets of at most BUCKET items: each bucket's passes below the
   top then run in the cache. On fudge's six performance cores 1 Mi random
   int64 keys sort in 6.1 ms this way, in 7.0 by LSD passes alone. Packed
   keys, half the bytes, sorted 4% slower this way and keep LSD passes.

   Where the slice keeps every element, the positions' destination, and a
   wide key's values', serve as two of the passes' arrays: fresh pages from
   the system cost about as much as a pass over them. */

typedef struct {
  const sorter *s;
  const uint8_t *x; /* the slice's first element */
  uint64_t *src, *dst;
  int64_t *psrc, *pdst; /* positions beside wide keys, else NULL */
  int64_t n, blocks;
  double pass; /* each pass's share of the sort's cost (pass_cost) */
  int shift, digits;
  uint64_t least, flip; /* subtracted from, and xored into, each key */
  uint64_t (*span)[2];  /* each block's least and greatest key */
  int64_t (*count)[256];
  int64_t start[257]; /* each bucket's first item, after the first pass */
  int64_t *pos;       /* the positions' destination */
  uint8_t *v;         /* the values' destination */
} spread;

static void block_range(const spread *p, int64_t b, int64_t *lo, int64_t *hi) {
  *lo = p->n * b / p->blocks;
  *hi = p->n * (b + 1) / p->blocks;
}

static void span_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  const sorter *s = p->s;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1;
    block_range(p, b, &i0, &i1);
    uint64_t least = ~0ull, most = 0;
    for (int64_t i = i0; i < i1; i++) {
      uint64_t k = key_of(s->dt, p->x + i * s->sx * s->w) ^ p->flip;
      least = k < least ? k : least;
      most = k > most ? k : most;
    }
    p->span[b][0] = least;
    p->span[b][1] = most;
  }
}

static void key_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  const sorter *s = p->s;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1;
    block_range(p, b, &i0, &i1);
    for (int64_t i = i0; i < i1; i++) {
      uint64_t k = (key_of(s->dt, p->x + i * s->sx * s->w) ^ p->flip) - p->least;
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

/* Counts digit [d] of every block's items and makes the counts each
   block's offsets, digit by digit and block by block, each digit's first
   item in [start]; answers the most items of one digit. */
static int64_t count_digit(spread *p, int d, int64_t bytes) {
  p->shift = (p->psrc == NULL ? 32 : 0) + 8 * d;
  nx_cpu_job(p->blocks, bytes, p->pass, count_blocks, p);
  int64_t at = 0, most = 0;
  for (int g = 0; g < 256; g++) {
    int64_t first = at;
    for (int64_t b = 0; b < p->blocks; b++) {
      int64_t c = p->count[b][g];
      p->count[b][g] = at;
      at += c;
    }
    p->start[g] = first;
    most = at - first > most ? at - first : most;
  }
  p->start[256] = at;
  return most;
}

/* Moves every block's items to its offsets; the destination becomes the
   source. */
static void move(spread *p, int64_t bytes) {
  nx_cpu_job(p->blocks, bytes, p->pass, move_blocks, p);
  uint64_t *t = p->src;
  p->src = p->dst;
  p->dst = t;
  int64_t *q = p->psrc;
  p->psrc = p->pdst;
  p->pdst = q;
}

/* Sorts each bucket of the first pass's items by the digits below its
   own, in the cache, landing them back where the pass put them. */
static void bucket_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i = p->start[b], m = p->start[b + 1] - i;
    if (m < 2) continue;
    if (p->psrc == NULL) radix_packed(p->src + i, p->dst + i, m, p->digits);
    else radix(p->src + i, p->psrc + i, p->dst + i, p->pdst + i, m, p->digits);
  }
}

/* Writes the positions and values of the sorted items' blocks. An item
   may lie in the destination it is written to, at its own index. */
static void emit_blocks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const spread *p = ctx;
  const sorter *s = p->s;
  for (int64_t b = lo; b < hi; b++) {
    int64_t i0, i1;
    block_range(p, b, &i0, &i1);
    if (i1 > s->keep) i1 = s->keep;
    for (int64_t i = i0; i < i1; i++) {
      int64_t q =
          p->psrc == NULL ? (int64_t)(p->src[i] & 0xFFFFFFFFu) : p->psrc[i];
      p->pos[i * s->sp] = q;
      memcpy(p->v + i * s->sv * s->w, p->x + q * s->sx * s->w, (size_t)s->w);
    }
  }
}

/* [there], or n words of the system's, kept in [own] to be freed; NULL if
   memory runs out. */
static uint64_t *take(uint64_t *there, int64_t n, uint64_t **own, int *owned) {
  if (there != NULL) return there;
  uint64_t *t = malloc((size_t)n * sizeof *t);
  if (t != NULL) own[(*owned)++] = t;
  return t;
}

/* Distributes the items of [p] by their top digit, and, where every
   bucket fits BUCKET items, sorts each by the digits below, a job's unit
   each, and writes the result. Answers 0, having moved nothing, where a
   bucket is larger. */
static int msd(spread *p, int kb, int64_t bytes) {
  if (count_digit(p, kb - 1, bytes) > BUCKET) return 0;
  move(p, bytes);
  p->digits = kb - 1;
  nx_cpu_job(256, bytes, 2 * p->digits * p->pass, bucket_blocks, p);
  nx_cpu_job(p->blocks, bytes, p->pass, emit_blocks, p);
  return 1;
}

/* Runs the passes of [p], over [kb] bytes of keys, and writes the result. */
static void passes(spread *p, int kb, int64_t bytes) {
  nx_cpu_job(p->blocks, bytes, p->pass, key_blocks, p);
  /* Past 256 buckets of BUCKET items, some bucket is always larger. */
  if (p->psrc != NULL && p->n <= 256 * BUCKET && msd(p, kb, bytes)) return;
  /* A digit every key shares leaves its pass out. */
  for (int d = 0; d < kb; d++)
    if (count_digit(p, d, bytes) < p->n) move(p, bytes);
  nx_cpu_job(p->blocks, bytes, p->pass, emit_blocks, p);
}

/* A pass's share of the [cost] of a long slice's sort over keys of [kb]
   bytes: the passes over its items, at most the keys' span, the keys, a
   count and a move for each key byte, and the result's, each a like
   share. The MSD order's buckets sort their bytes below the top in the
   cache, two passes a byte. */
static double pass_cost(double cost, int kb) { return cost / (3 + 2 * kb); }

/* Sorts the one slice of [s], of [bytes] and [cost], on [blocks] blocks,
   each of its passes a share of the cost. Answers 0, or -1 where memory
   runs out. */
static int sort_long(const sorter *s, int64_t blocks, int64_t bytes,
                     double cost) {
  int64_t n = s->n;
  uint64_t mask = s->kb == 8 ? ~0ull : (1ull << (8 * s->kb)) - 1;
  spread p = {.s = s,
              .x = s->a[2].base + s->l.first[2] * s->w,
              .n = n,
              .blocks = blocks,
              .pass = pass_cost(cost, s->kb),
              .flip = s->descending ? mask : 0,
              .span = malloc((size_t)blocks * sizeof *p.span),
              .count = malloc((size_t)blocks * sizeof *p.count),
              .pos = (int64_t *)s->a[1].base + s->l.first[1],
              .v = s->a[0].base + s->l.first[0] * s->w};
  uint64_t *own[4];
  int owned = 0, ok = p.span != NULL && p.count != NULL;
  if (ok) {
    nx_cpu_job(blocks, bytes, p.pass, span_blocks, &p);
    uint64_t least = ~0ull, most = 0;
    for (int64_t b = 0; b < blocks; b++) {
      least = p.span[b][0] < least ? p.span[b][0] : least;
      most = p.span[b][1] > most ? p.span[b][1] : most;
    }
    p.least = least;
    int kb = nx_cpu_span_bytes(most - least);
    int whole = s->keep == n;
    uint64_t *in_pos = whole && s->sp == 1 ? (uint64_t *)p.pos : NULL;
    uint64_t *in_v = whole && s->sv == 1 && s->w == 8 ? (uint64_t *)p.v : NULL;
    if (kb <= 4) {
      p.src = take(in_pos, n, own, &owned);
      p.dst = take(NULL, n, own, &owned);
      ok = p.src != NULL && p.dst != NULL;
    } else {
      p.psrc = (int64_t *)take(in_pos, n, own, &owned);
      p.pdst = (int64_t *)take(NULL, n, own, &owned);
      p.src = take(NULL, n, own, &owned);
      p.dst = take(in_v, n, own, &owned);
      ok = p.psrc != NULL && p.pdst != NULL && p.src != NULL && p.dst != NULL;
    }
    if (ok) passes(&p, kb, bytes);
  }
  for (int i = 0; i < owned; i++) free(own[i]);
  free(p.span);
  free(p.count);
  return ok ? 0 : -1;
}

/* Slices */

/* Makes the keys of the slice at [x] into [k], xored with [flip], and
   their least and greatest into [span], the slice's elements [step] bytes
   apart. The common dtypes have loops of their own, where key_of's switch
   folds away and the loop vectorises; keys of at most 4 bytes find their
   span in 32 bits, which x86's base vectors compare and 64 bits they do
   not. Through key_of's switch, fudge's sort-f32-rows-512x512 ran 7%
   slower. */
static __attribute__((flatten)) void make_keys(const sorter *s,
                                               const uint8_t *x, int64_t step,
                                               uint64_t flip, uint64_t *k,
                                               uint64_t span[2]) {
  int64_t n = s->n;
#define KEYS(U, dt)                                                          \
  do {                                                                       \
    U least = (U)~0ull, most = 0;                                            \
    for (int64_t i = 0; i < n; i++) {                                        \
      U y = (U)(key_of(dt, x + i * step) ^ flip);                            \
      k[i] = y;                                                              \
      least = y < least ? y : least;                                         \
      most = y > most ? y : most;                                            \
    }                                                                        \
    span[0] = least;                                                         \
    span[1] = most;                                                          \
  } while (0)
  switch (s->dt) {
    case NX_FLOAT32: KEYS(uint32_t, NX_FLOAT32); break;
    case NX_INT32: KEYS(uint32_t, NX_INT32); break;
    case NX_FLOAT64: KEYS(uint64_t, NX_FLOAT64); break;
    case NX_INT64: KEYS(uint64_t, NX_INT64); break;
    default:
      if (s->kb <= 4) KEYS(uint32_t, s->dt);
      else KEYS(uint64_t, s->dt);
  }
#undef KEYS
}

/* Sorts the slice's keys [k], whose least and greatest are [span], by radix
   over their span, stably, through [k2] and [p2]: less the least, above
   their positions where they span at most 4 bytes. The kept positions land
   in [p]. */
static void by_span(const sorter *s, const uint64_t span[2], uint64_t *k,
                    int64_t *p, uint64_t *k2, int64_t *p2) {
  int64_t n = s->n, keep = s->keep;
  uint64_t least = span[0];
  int kb = nx_cpu_span_bytes(span[1] - least);
  if (kb > 4 || n > UINT32_MAX) {
    for (int64_t i = 0; i < n; i++) {
      k[i] -= least;
      p[i] = i;
    }
    radix(k, p, k2, p2, n, kb);
    return;
  }
  for (int64_t i = 0; i < n; i++) k[i] = (k[i] - least) << 32 | (uint64_t)i;
  radix_packed(k, k2, n, kb);
  for (int64_t i = 0; i < keep; i++) p[i] = (int64_t)(k[i] & 0xFFFFFFFFu);
}

static void sort_slice(const sorter *s, int64_t pv, int64_t pp, int64_t px,
                       uint64_t *scratch) {
  int64_t n = s->n;
  uint64_t *k = scratch, *k2 = scratch + n;
  int64_t *p = (int64_t *)(scratch + 2 * n), *p2 = (int64_t *)(scratch + 3 * n);
  const uint8_t *x = s->a[2].base + px * s->w;
  int64_t step = s->sx * s->w;
  /* A sub-byte slice's codes, a byte each, wait in k2 for their keys. */
  if (s->bits < 8) {
    uint8_t *c = (uint8_t *)k2;
    if (s->sx == 1) nx_sub_unpack_run(s->a[2].base, s->dt, px, c, n);
    else
      for (int64_t i = 0; i < n; i++)
        c[i] = (uint8_t)nx_sub_load(s->a[2].base, s->bits, px + i * s->sx);
    x = c;
    step = 1;
  }
  uint64_t mask = s->kb == 8 ? ~0ull : (1ull << (8 * s->kb)) - 1;
  uint64_t flip = s->descending ? mask : 0, span[2];
  make_keys(s, x, step, flip, k, span);
  int64_t keep = s->keep;
  int few = keep <= KEEP && keep * SPARE <= n;
  if (s->dt != NX_COMPLEX128 && n >= SMALL && !few)
    by_span(s, span, k, p, k2, p2);
  else {
    for (int64_t i = 0; i < n; i++) p[i] = i;
    if (s->dt == NX_COMPLEX128) {
      /* The imaginary parts' order, then the real parts' over it. */
      radix(k, p, k2, p2, n, 8);
      for (int64_t i = 0; i < n; i++)
        k[i] = real_key((const double *)(x + p[i] * s->sx * s->w)) ^ flip;
      radix(k, p, k2, p2, n, 8);
    } else if (n < SMALL)
      nx_cpu_insertion(k, p, n);
    else {
      memcpy(k2, k, (size_t)n * sizeof *k);
      slots(k2, n, keep, k, p);
    }
  }
  int64_t *pos = (int64_t *)s->a[1].base + pp;
  for (int64_t i = 0; i < keep; i++) pos[i * s->sp] = p[i];
  int64_t sv = s->sv, sx = s->sx;
  if (s->bits < 8) {
    for (int64_t i = 0; i < keep; i++)
      nx_sub_store(s->a[0].base, s->bits, pv + i * sv,
                   nx_sub_load(s->a[2].base, s->bits, px + p[i] * sx));
    return;
  }
  uint8_t *v = s->a[0].base + pv * s->w;
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
  uint64_t *scratch = s->scratch + (int64_t)worker * SCRATCH * s->n;
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

/* The external. [values] keeps the first elements of each slice, as many
   as its extent along the axis. */
value nx_cpu_sort(value vaxis, value vdescending, value vv, value vp,
                  value vx) {
  CAMLparam5(vaxis, vdescending, vv, vp, vx);
  intnat axis = Long_val(vaxis);
  int descending = Bool_val(vdescending);
  int dt = nx_array_dtype(vx);
  int64_t xd[2 * NX_MAX_RANK], vd[2 * NX_MAX_RANK], pd[2 * NX_MAX_RANK], off;
  int rank = nx_array_layout(vx, xd, &off), rv = nx_array_layout(vv, vd, &off),
      rp = nx_array_layout(vp, pd, &off);
  if (axis < 0 || axis >= rank || rv != rank || rp != rank ||
      vd[axis] > xd[axis])
    CAMLreturn(Val_int(NX_SHAPE));
  for (int i = 0; i < rank; i++)
    if (pd[i] != vd[i] || (i != axis && vd[i] != xd[i]))
      CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[3] = {{vv, dt, 1}, {vp, NX_INT64, 1}, {vx, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (e) CAMLreturn(Val_int(e));
  int rx = a[2].rank;
  const int64_t *xs = a[2].dim;
  int64_t keep = a[0].dim[axis];
  int64_t slices = 1;
  for (int i = 0; i < rx; i++)
    if (i != axis) slices *= xs[i];
  if (slices == 0 || keep == 0) {
    nx_done(3, a);
    CAMLreturn(Val_int(NX_OK));
  }
  sorter s = {.a = a,
              .dt = dt,
              .bits = nx_dtype_row_of(dt).bits,
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
  int64_t bytes = slices * s.n * (s.w + 8);
  int two_keys = dt == NX_COMPLEX64 || dt == NX_COMPLEX128;
  double cost = (two_keys ? COMPLEX_CALL_COST : CALL_COST) * (double)bytes;
  int released = nx_cpu_release(cost);
  int64_t workers = nx_cpu_threads(bytes, cost);
  /* One long slice sorts on the job's threads, a block of its items each;
     a complex128's two keys, a sub-byte slice, whose blocks would read
     their elements by byte, and keeping a few, stay on one. */
  int slots = keep <= KEEP && keep * SPARE <= s.n;
  if (slices == 1 && workers > 1 && s.n >= LONG && dt != NX_COMPLEX128 &&
      s.bits >= 8 && !slots && s.n <= UINT32_MAX) {
    int e = sort_long(&s, workers, bytes, cost);
    nx_cpu_reacquire(released);
    nx_done(3, a);
    if (e) caml_raise_out_of_memory();
    CAMLreturn(Val_int(NX_OK));
  }
  if (workers > slices) workers = slices;
  s.scratch =
      malloc((size_t)workers * SCRATCH * (size_t)s.n * sizeof(uint64_t));
  if (s.scratch == NULL) {
    nx_cpu_reacquire(released);
    nx_done(3, a);
    caml_raise_out_of_memory();
  }
  nx_cpu_job(slices, bytes, cost, sort_units, &s);
  free(s.scratch);
  nx_cpu_reacquire(released);
  nx_done(3, a);
  CAMLreturn(Val_int(NX_OK));
}
