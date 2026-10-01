/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Parquet's sequential decoders: the RLE and bit-packing hybrid, delta binary
   packing, the assembly of delta-encoded byte arrays, and PLAIN byte arrays.

   Each kernel reads the span [src + pos, src + pos + len), checks every read
   and write against the span and the arrays it is given, whatever its integer
   arguments, and returns the bytes of the span it consumed, or a negative
   error code. None allocates or raises, so all are [@@noalloc]. Parquet is
   little-endian, as the hosts nx runs on are. */

#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

enum {
  ERR_TRUNCATED = -1, /* the span ends inside a value */
  ERR_BOUND = -2,     /* a value is not below its bound */
  ERR_WIDTH = -3,     /* a bit width is wider than the values */
  ERR_HEADER = -4,    /* a delta header is malformed */
  ERR_COUNT = -5,     /* a delta header's count is not the expected one */
  ERR_DATA = -6,      /* the byte strings' data array is too small */
  ERR_CAPACITY = -7,  /* another output array is too small */
};

#define BYTES(v) ((uint8_t *)Caml_ba_data_val(v))
#define DIM(v) ((int64_t)Caml_ba_array_val(v)->dim[0])

/* [span(src, pos, len)] is true iff [pos, pos + len) lies in [src]. */
static inline int span(value src, int64_t pos, int64_t len)
{
  return pos >= 0 && len >= 0 && pos <= DIM(src) - len;
}

/* [room(at, n, cap)] is true iff [n] elements fit from index [at] in an array
   of [cap] elements. */
static inline int room(int64_t at, int64_t n, int64_t cap)
{
  return at >= 0 && n >= 0 && at <= cap && n <= cap - at;
}

/* The [w] bits, at most 64, at bit [bit] of [p], least significant first. The
   caller checked that they lie before [end]. */
static inline uint64_t bits_at(const uint8_t *p, const uint8_t *end,
                               uint64_t bit, int w)
{
  if (w == 0) return 0;
  const uint8_t *q = p + (bit >> 3);
  int s = (int)(bit & 7);
  uint64_t lo = 0;
  if (end - q >= 8) memcpy(&lo, q, 8);
  else
    for (int k = 0; k < end - q; k++) lo |= (uint64_t)q[k] << (8 * k);
  uint64_t v = lo >> s;
  if (s + w > 64) v |= (uint64_t)q[8] << (64 - s);
  return w == 64 ? v : v & ((UINT64_C(1) << w) - 1);
}

/* An unsigned LEB128 varint of at most [max] bytes, or -1. */
static int64_t varint(const uint8_t **p, const uint8_t *end, int max)
{
  uint64_t v = 0;
  for (int i = 0; i < max && *p < end; i++) {
    uint8_t c = *(*p)++;
    v |= (uint64_t)(c & 0x7f) << (7 * i);
    if (c < 0x80) return v > INT64_MAX ? -1 : (int64_t)v;
  }
  return -1;
}

static int zigzag(const uint8_t **p, const uint8_t *end, int64_t *v)
{
  const uint8_t *q = *p;
  uint64_t u = 0;
  for (int i = 0; i < 10 && q < end; i++) {
    uint8_t c = *q++;
    u |= (uint64_t)(c & 0x7f) << (7 * i);
    if (c < 0x80) {
      *p = q;
      *v = (int64_t)((u >> 1) ^ (~(u & 1) + 1));
      return 0;
    }
  }
  return ERR_HEADER;
}

/* The RLE and bit-packing hybrid */

static int64_t hybrid(const uint8_t *p, int64_t len, int w, int64_t n,
                      int wide, uint8_t *dst, int64_t at, int64_t bound)
{
  const uint8_t *start = p, *end = p + len;
  uint8_t *d8 = dst + at;
  int64_t *d64 = (int64_t *)dst + at;
  int64_t i = 0;
  if (w < 0 || w > 32) return ERR_WIDTH;
  while (i < n) {
    int64_t h = varint(&p, end, 5);
    if (h < 0) return ERR_TRUNCATED;
    if (h & 1) {
      int64_t bytes = (h >> 1) * w, count = (h >> 1) * 8;
      if (bytes > end - p) return ERR_TRUNCATED;
      if (count > n - i) count = n - i;
      for (int64_t k = 0; k < count; k++, i++) {
        uint64_t v = bits_at(p, end, (uint64_t)k * w, w);
        if (v >= (uint64_t)bound) return ERR_BOUND;
        if (wide) d64[i] = (int64_t)v;
        else d8[i] = (uint8_t)v;
      }
      p += bytes;
    } else {
      int64_t count = h >> 1, bytes = (w + 7) / 8;
      uint64_t v = 0;
      if (bytes > end - p) return ERR_TRUNCATED;
      for (int k = 0; k < bytes; k++) v |= (uint64_t)p[k] << (8 * k);
      p += bytes;
      if (count > n - i) count = n - i;
      if (count > 0 && v >= (uint64_t)bound) return ERR_BOUND;
      if (wide)
        for (int64_t k = 0; k < count; k++) d64[i + k] = (int64_t)v;
      else memset(d8 + i, (int)v, (size_t)count);
      i += count;
    }
  }
  return p - start;
}

/* [hybrid src pos len width n dst at bound] decodes [n] values of [width] bits
   into [dst] from index [at], an int64 or a uint8 array, each below [bound]. */
CAMLprim value talon_parquet_hybrid(value src, value pos, value len,
                                    value width, value n, value dst, value at,
                                    value bound)
{
  int wide = (Caml_ba_array_val(dst)->flags & CAML_BA_KIND_MASK) == CAML_BA_INT64;
  if (!span(src, Long_val(pos), Long_val(len))) return Val_long(ERR_TRUNCATED);
  if (!room(Long_val(at), Long_val(n), DIM(dst))) return Val_long(ERR_CAPACITY);
  return Val_long(hybrid(BYTES(src) + Long_val(pos), Long_val(len),
                         (int)Long_val(width), Long_val(n), wide, BYTES(dst),
                         Long_val(at), Long_val(bound)));
}

CAMLprim value talon_parquet_hybrid_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_hybrid(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6], argv[7]);
}

/* Delta binary packing */

static int64_t delta_binary_packed(const uint8_t *p, int64_t len, int64_t n,
                                   uint8_t *dst, int size)
{
  const uint8_t *start = p, *end = p + len;
  int64_t block = varint(&p, end, 5), minis = varint(&p, end, 5);
  int64_t total = varint(&p, end, 10);
  int64_t first;
  if (block < 0 || minis < 0 || total < 0 || zigzag(&p, end, &first))
    return ERR_HEADER;
  if (block == 0 || block % 128 || minis == 0 || block % minis ||
      (block / minis) % 32)
    return ERR_HEADER;
  if (total != n) return ERR_COUNT;
  int64_t per_mini = block / minis;
  uint64_t v = (uint64_t)first;
  for (int64_t i = 0; i < n;) {
    if (i == 0) {
      memcpy(dst, &v, size);
      i++;
      continue;
    }
    int64_t min;
    if (zigzag(&p, end, &min)) return ERR_HEADER;
    if (minis > end - p) return ERR_TRUNCATED;
    const uint8_t *widths = p;
    p += minis;
    for (int64_t m = 0; m < minis && i < n; m++) {
      int w = widths[m];
      if (w > 8 * size) return ERR_WIDTH;
      int64_t bytes = per_mini * w / 8;
      if (bytes > end - p) return ERR_TRUNCATED;
      for (int64_t k = 0; k < per_mini && i < n; k++, i++) {
        v += (uint64_t)min + bits_at(p, end, (uint64_t)k * w, w);
        memcpy(dst + i * size, &v, size);
      }
      p += bytes;
    }
  }
  return p - start;
}

/* [delta_binary_packed src pos len n dst at size] decodes [n] integers into
   [dst] from byte [at * size], each in [size] bytes, 4 or 8: an int32
   wraps as Parquet's 32-bit arithmetic does. */
CAMLprim value talon_parquet_delta_binary_packed(value src, value pos,
                                                 value len, value n,
                                                 value dst, value at,
                                                 value size)
{
  int64_t s = Long_val(size);
  int64_t bytes = (int64_t)caml_ba_byte_size(Caml_ba_array_val(dst));
  if (s != 4 && s != 8) return Val_long(ERR_CAPACITY);
  if (!span(src, Long_val(pos), Long_val(len))) return Val_long(ERR_TRUNCATED);
  if (!room(Long_val(at), Long_val(n), bytes / s)) return Val_long(ERR_CAPACITY);
  return Val_long(delta_binary_packed(BYTES(src) + Long_val(pos),
                                      Long_val(len), Long_val(n),
                                      BYTES(dst) + Long_val(at) * s, (int)s));
}

CAMLprim value talon_parquet_delta_binary_packed_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_delta_binary_packed(argv[0], argv[1], argv[2], argv[3],
                                           argv[4], argv[5], argv[6]);
}

/* Byte arrays */

/* [assemble src pos len n prefixes suffixes offsets at data] appends [n] byte
   strings to [data]: the [i]th is the first [prefixes.{i}] bytes of the one
   before it, then [suffixes.{i}] bytes of the span. Their ends go to
   [offsets] from index [at + 1]; [offsets.{at}] is where the first starts.
   The caller checked that each prefix fits the string before it. */
CAMLprim value talon_parquet_assemble(value src, value pos, value len, value n,
                                      value prefixes, value suffixes,
                                      value offsets, value at, value data)
{
  int64_t left = Long_val(len), count = Long_val(n), a = Long_val(at);
  const int64_t *pre = (int64_t *)Caml_ba_data_val(prefixes);
  const int64_t *suf = (int64_t *)Caml_ba_data_val(suffixes);
  int64_t *off = (int64_t *)Caml_ba_data_val(offsets);
  uint8_t *d = BYTES(data);
  int64_t cap = DIM(data), used = 0;
  if (!span(src, Long_val(pos), left)) return Val_long(ERR_TRUNCATED);
  if (!room(a, count + 1, DIM(offsets)) || count > DIM(prefixes) ||
      count > DIM(suffixes) || off[a] < 0 || off[a] > cap)
    return Val_long(ERR_CAPACITY);
  const uint8_t *p = BYTES(src) + Long_val(pos);
  int64_t end = off[a], prev = end;
  for (int64_t i = 0; i < count; i++) {
    int64_t pl = pre[i], sl = suf[i];
    if (pl < 0 || sl < 0 || pl > end - prev) return Val_long(ERR_BOUND);
    if (sl > left - used) return Val_long(ERR_TRUNCATED);
    if (pl + sl > cap - end) return Val_long(ERR_DATA);
    memcpy(d + end, d + prev, (size_t)pl);
    memcpy(d + end + pl, p + used, (size_t)sl);
    used += sl;
    prev = end;
    end += pl + sl;
    off[a + i + 1] = end;
  }
  return Val_long(used);
}

CAMLprim value talon_parquet_assemble_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_assemble(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6], argv[7], argv[8]);
}

/* [plain_byte_array src pos len n offsets at data] appends [n] PLAIN byte
   arrays, each a 4-byte length then its bytes, to [data], their ends to
   [offsets] from index [at + 1]. */
CAMLprim value talon_parquet_plain_byte_array(value src, value pos, value len,
                                              value n, value offsets,
                                              value at, value data)
{
  int64_t count = Long_val(n), a = Long_val(at);
  int64_t *off = (int64_t *)Caml_ba_data_val(offsets);
  uint8_t *d = BYTES(data);
  int64_t cap = DIM(data);
  if (!span(src, Long_val(pos), Long_val(len))) return Val_long(ERR_TRUNCATED);
  if (!room(a, count + 1, DIM(offsets)) || off[a] < 0 || off[a] > cap)
    return Val_long(ERR_CAPACITY);
  const uint8_t *start = BYTES(src) + Long_val(pos), *p = start;
  const uint8_t *end = start + Long_val(len);
  int64_t e = off[a];
  for (int64_t i = 0; i < count; i++) {
    uint32_t l;
    if (end - p < 4) return Val_long(ERR_TRUNCATED);
    memcpy(&l, p, 4);
    p += 4;
    if ((int64_t)l > end - p) return Val_long(ERR_TRUNCATED);
    if ((int64_t)l > cap - e) return Val_long(ERR_DATA);
    memcpy(d + e, p, l);
    p += l;
    e += l;
    off[a + i + 1] = e;
  }
  return Val_long(p - start);
}

CAMLprim value talon_parquet_plain_byte_array_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_plain_byte_array(argv[0], argv[1], argv[2], argv[3],
                                        argv[4], argv[5], argv[6]);
}
