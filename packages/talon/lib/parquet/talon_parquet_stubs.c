/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Parquet's sequential decoders: the RLE and bit-packing hybrid, into bytes,
   int64s or bits, the walk of a nested leaf's levels, PLAIN booleans, delta binary packing, the assembly of
   delta-encoded byte arrays, and PLAIN byte arrays; the gather of
   dictionary-encoded byte arrays and of their ends; and the spread of a
   page's values onto its rows.

   Each decoder reads the span [src + pos, src + pos + len), checks every read
   and write against the span and the arrays it is given, whatever its integer
   arguments, and returns the bytes of the span it consumed, or a negative
   error code. The gathers check their reads and writes alike and return 0 or
   the last end, or an error code. None allocates or raises, so all are
   [@@noalloc]. Parquet is little-endian, as the hosts nx runs on are. */

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
  ERR_ROW = -8,       /* a chunk's first entry does not start a row */
  ERR_REPEAT = -9,    /* an entry repeats a field that is not defined */
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

/* Bits

   A bit array holds element [i] at bit [i mod 8] of byte [i / 8], as Parquet
   packs booleans and levels and nx stores its bit dtype. A writer writes
   exactly the bits of its elements, so that pages can start inside a byte. */

/* [put(d, bit, v, n)] writes the [n] bits of [v], at most 64, least
   significant first, at bit [bit] of [d], and no other bit. 64 bits take
   one 8-byte store and, past a byte boundary, the bits of a ninth byte. */
static inline void put(uint8_t *d, uint64_t bit, uint64_t v, int n)
{
  if (n == 64) {
    uint8_t *q = d + (bit >> 3);
    int s = (int)(bit & 7);
    uint64_t lo;
    memcpy(&lo, q, 8);
    lo = s == 0 ? v : (lo & ((UINT64_C(1) << s) - 1)) | v << s;
    memcpy(q, &lo, 8);
    if (s > 0)
      q[8] = (uint8_t)((q[8] & ~((1u << s) - 1)) | (uint8_t)(v >> (64 - s)));
    return;
  }
  while (n > 0) {
    uint8_t *q = d + (bit >> 3);
    int s = (int)(bit & 7), k = 8 - s < n ? 8 - s : n;
    uint8_t mask = (uint8_t)(((1u << k) - 1) << s);
    *q = (uint8_t)((*q & ~mask) | (((uint8_t)v << s) & mask));
    v >>= k;
    bit += (uint64_t)k;
    n -= k;
  }
}

/* [ones(v)] is the number of set bits of [v]. */
static inline int ones(uint64_t v)
{
  v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
  v = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
  v = (v + (v >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
  return (int)((v * UINT64_C(0x0101010101010101)) >> 56);
}

/* [fill(d, bit, v, n)] writes [n] copies of the bit [v] from bit [bit]. */
static void fill(uint8_t *d, uint64_t bit, int v, int64_t n)
{
  uint64_t word = v ? ~UINT64_C(0) : 0;
  for (int64_t k = 0; k < n; k += 64)
    put(d, bit + (uint64_t)k, word, n - k < 64 ? (int)(n - k) : 64);
}

/* [blit(d, bit, p, end, n)] writes the first [n] bits of [p] at bit [bit] of
   [d]. The caller checked that they lie before [end]. */
static void blit(uint8_t *d, uint64_t bit, const uint8_t *p,
                 const uint8_t *end, int64_t n)
{
  for (int64_t k = 0; k < n; k += 64) {
    int m = n - k < 64 ? (int)(n - k) : 64;
    put(d, bit + (uint64_t)k, bits_at(p, end, (uint64_t)k, m), m);
  }
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

/* The element types a hybrid decodes into. Bits take values of at most one
   bit, which a bit-packed run already holds as bits. */
enum out { OUT_BYTES, OUT_INT64, OUT_BITS };

/* [group(p, w, dst, most)] writes the 8 values of [w] bits packed from [p]
   to [dst], and is the largest of them and [most]. The caller checked that 8
   bytes from each value's first byte lie in the span. For a constant [w], each
   value's byte and shift are constants. */
static inline uint64_t group(const uint8_t *p, int w, int64_t *dst,
                             uint64_t most)
{
  uint64_t mask = (UINT64_C(1) << w) - 1;
  for (int j = 0; j < 8; j++) {
    uint64_t lo;
    memcpy(&lo, p + j * w / 8, 8);
    uint64_t v = lo >> (j * w % 8) & mask;
    most = v > most ? v : most;
    dst[j] = (int64_t)v;
  }
  return most;
}

/* [unpack(p, end, w, count, out, dst, bound)] writes the [count] values of [w]
   bits, at most 32, packed from [p], to [dst] as [out] elements, and is
   [ERR_BOUND] if one is not below [bound], else 0. A value starts in the
   lowest 8 bits of its first byte, so one load of the eight bytes from there
   reads it whole, while those bytes lie before [end]. Dictionary indices, the
   int64 elements, unpack 8 at a time, [w] bytes, with [w] a constant. */
static int unpack(const uint8_t *p, const uint8_t *end, int w, int64_t count,
                  enum out out, uint8_t *dst, uint64_t bound)
{
  uint64_t mask = (UINT64_C(1) << w) - 1, most = 0, bit = 0;
  int64_t room = end - p - 8, k = 0;
  int64_t fast = room < 0 ? 0 : w == 0 ? count : room * 8 / w + 1;
  if (fast > count) fast = count;
  if (out == OUT_INT64) {
#define GROUPS(W)                                                     \
  case W:                                                             \
    for (; k + 8 <= fast; k += 8)                                     \
      most = group(p + k / 8 * W, W, (int64_t *)dst + k, most);      \
    break;
    switch (w) {
      GROUPS(1) GROUPS(2) GROUPS(3) GROUPS(4) GROUPS(5) GROUPS(6) GROUPS(7)
      GROUPS(8) GROUPS(9) GROUPS(10) GROUPS(11) GROUPS(12) GROUPS(13)
      GROUPS(14) GROUPS(15) GROUPS(16) GROUPS(17) GROUPS(18) GROUPS(19)
      GROUPS(20) GROUPS(21) GROUPS(22) GROUPS(23) GROUPS(24) GROUPS(25)
      GROUPS(26) GROUPS(27) GROUPS(28) GROUPS(29) GROUPS(30) GROUPS(31)
      GROUPS(32)
    }
#undef GROUPS
    bit = (uint64_t)(k * w);
  }
  for (; k < fast; k++, bit += (uint64_t)w) {
    uint64_t lo;
    memcpy(&lo, p + (bit >> 3), 8);
    uint64_t v = (lo >> (bit & 7)) & mask;
    most = v > most ? v : most;
    if (out == OUT_INT64) ((int64_t *)dst)[k] = (int64_t)v;
    else dst[k] = (uint8_t)v;
  }
  for (; k < count; k++, bit += (uint64_t)w) {
    uint64_t v = bits_at(p, end, bit, w);
    most = v > most ? v : most;
    if (out == OUT_INT64) ((int64_t *)dst)[k] = (int64_t)v;
    else dst[k] = (uint8_t)v;
  }
  return count > 0 && most >= bound ? ERR_BOUND : 0;
}

static int64_t hybrid(const uint8_t *p, int64_t len, int w, int64_t n,
                      enum out out, uint8_t *dst, int64_t at, int64_t bound)
{
  const uint8_t *start = p, *end = p + len;
  int64_t *d64 = (int64_t *)dst;
  int64_t i = 0;
  if (w < 0 || w > 32 || (out == OUT_BITS && w > 1)) return ERR_WIDTH;
  while (i < n) {
    int64_t h = varint(&p, end, 5);
    if (h < 0) return ERR_TRUNCATED;
    if (h & 1) {
      int64_t bytes = (h >> 1) * w, count = (h >> 1) * 8;
      if (bytes > end - p) return ERR_TRUNCATED;
      if (count > n - i) count = n - i;
      if (out == OUT_BITS) {
        if (w == 0) fill(dst, (uint64_t)(at + i), 0, count);
        else blit(dst, (uint64_t)(at + i), p, end, count);
        i += count;
      } else {
        uint8_t *to =
            out == OUT_INT64 ? (uint8_t *)(d64 + at + i) : dst + at + i;
        if (unpack(p, end, w, count, out, to, (uint64_t)bound) < 0)
          return ERR_BOUND;
        i += count;
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
      if (out == OUT_BITS) fill(dst, (uint64_t)(at + i), (int)v, count);
      else if (out == OUT_INT64)
        for (int64_t k = 0; k < count; k++) d64[at + i + k] = (int64_t)v;
      else memset(dst + at + i, (int)v, (size_t)count);
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
                         (int)Long_val(width), Long_val(n),
                         wide ? OUT_INT64 : OUT_BYTES, BYTES(dst),
                         Long_val(at), Long_val(bound)));
}

CAMLprim value talon_parquet_hybrid_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_hybrid(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6], argv[7]);
}

/* [hybrid_bits src pos len width n dst at] decodes [n] values of [width] bits,
   at most 1, into the bit array [dst] from bit [at]. */
CAMLprim value talon_parquet_hybrid_bits(value src, value pos, value len,
                                         value width, value n, value dst,
                                         value at)
{
  if (!span(src, Long_val(pos), Long_val(len))) return Val_long(ERR_TRUNCATED);
  if (!room(Long_val(at), Long_val(n), 8 * DIM(dst)))
    return Val_long(ERR_CAPACITY);
  return Val_long(hybrid(BYTES(src) + Long_val(pos), Long_val(len),
                         (int)Long_val(width), Long_val(n), OUT_BITS,
                         BYTES(dst), Long_val(at), 2));
}

CAMLprim value talon_parquet_hybrid_bits_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_hybrid_bits(argv[0], argv[1], argv[2], argv[3],
                                   argv[4], argv[5], argv[6]);
}

/* Levels

   A nested leaf's definition and repetition levels, an entry each, as uint8s
   or, past 255, int64s. [walk] reads the slots of one node of the column
   out of them: entry [i] opens a slot iff its repetition level is at most [r]
   and its definition level at least [d], the slot holds a value iff the
   definition level is at least [nonnull], and it adds an element to the
   slot's list iff its repetition level is at most [er] and its definition
   level at least [ed]. A leaf without one of the levels passes no array for
   it, and every level reads as zero.

   A valid file's first entry starts a row, and an entry that repeats the list
   defines it and follows a slot: [ERR_ROW] and [ERR_REPEAT] are a level stream
   no writer produces. */

#define LEVEL_KERNELS(T)                                                     \
  static int64_t count_##T(const T *a, int64_t n, int64_t v)                 \
  {                                                                          \
    int64_t c = 0;                                                           \
    for (int64_t i = 0; i < n; i++) c += (int64_t)a[i] == v;                 \
    return c;                                                                \
  }                                                                          \
                                                                             \
  static int64_t walk_##T(const T *def, const T *rep, int64_t n, int64_t r,  \
                          int64_t d, int64_t nonnull, int64_t er,            \
                          int64_t ed, uint8_t *valid, int64_t valid_bits,    \
                          int64_t *offsets, int64_t offsets_len)             \
  {                                                                          \
    int64_t slots = 0, elems = 0;                                            \
    if (n > 0 && rep && rep[0] != 0) return ERR_ROW;                         \
    for (int64_t i = 0; i < n; i++) {                                        \
      int64_t dl = def ? (int64_t)def[i] : 0;                                \
      int64_t rl = rep ? (int64_t)rep[i] : 0;                                \
      if (rl <= r && dl >= d) {                                              \
        if (offsets) {                                                       \
          if (slots >= offsets_len) return ERR_CAPACITY;                     \
          offsets[slots] = elems;                                            \
        }                                                                    \
        if (valid && dl >= nonnull) {                                        \
          if (slots >= valid_bits) return ERR_CAPACITY;                      \
          valid[slots >> 3] |= (uint8_t)(1u << (slots & 7));                 \
        }                                                                    \
        slots++;                                                             \
      }                                                                      \
      if (rl <= er) {                                                        \
        if (dl >= ed) {                                                      \
          if (slots == 0) return ERR_REPEAT;                                 \
          elems++;                                                           \
        } else if (rl == er)                                                 \
          return ERR_REPEAT;                                                 \
      }                                                                      \
    }                                                                        \
    if (offsets) {                                                           \
      if (slots >= offsets_len) return ERR_CAPACITY;                         \
      offsets[slots] = elems;                                                \
    }                                                                        \
    return slots;                                                            \
  }

LEVEL_KERNELS(uint8_t)
LEVEL_KERNELS(int64_t)

#define IS_WIDE(v) \
  ((Caml_ba_array_val(v)->flags & CAML_BA_KIND_MASK) == CAML_BA_INT64)

/* [level_count a at n v] is how many of the [n] levels from index [at] are
   [v]. */
CAMLprim value talon_parquet_level_count(value a, value at, value n, value v)
{
  int64_t f = Long_val(at), c = Long_val(n), x = Long_val(v);
  if (!room(f, c, DIM(a))) return Val_long(ERR_CAPACITY);
  return Val_long(IS_WIDE(a) ? count_int64_t((int64_t *)Caml_ba_data_val(a) + f, c, x)
                             : count_uint8_t(BYTES(a) + f, c, x));
}

/* [level_walk def rep n r d nonnull er ed valid offsets] is the number of slots
   of a node among the [n] entries, writing their bits to [valid] and their
   element counts so far to [offsets] when those are not empty. An empty [def]
   or [rep] is all zeros. */
CAMLprim value talon_parquet_level_walk(value def, value rep, value n, value r,
                                        value d, value nonnull, value er,
                                        value ed, value valid, value offsets)
{
  int64_t c = Long_val(n);
  if (c < 0 || (DIM(def) > 0 && c > DIM(def)) || (DIM(rep) > 0 && c > DIM(rep))
      || IS_WIDE(def) != IS_WIDE(rep))
    return Val_long(ERR_CAPACITY);
  void *ds = DIM(def) > 0 ? Caml_ba_data_val(def) : NULL;
  void *rs = DIM(rep) > 0 ? Caml_ba_data_val(rep) : NULL;
  uint8_t *v = DIM(valid) > 0 ? BYTES(valid) : NULL;
  int64_t *o = DIM(offsets) > 0 ? (int64_t *)Caml_ba_data_val(offsets) : NULL;
  if (IS_WIDE(def))
    return Val_long(walk_int64_t(ds, rs, c, Long_val(r), Long_val(d),
                                 Long_val(nonnull), Long_val(er), Long_val(ed),
                                 v, 8 * DIM(valid), o, DIM(offsets)));
  return Val_long(walk_uint8_t(ds, rs, c, Long_val(r), Long_val(d),
                               Long_val(nonnull), Long_val(er), Long_val(ed),
                               v, 8 * DIM(valid), o, DIM(offsets)));
}

CAMLprim value talon_parquet_level_walk_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_level_walk(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8], argv[9]);
}

/* [plain_bits src pos len n dst at] copies the [n] PLAIN booleans of the span,
   bit-packed from its first bit, into the bit array [dst] from bit [at]. */
CAMLprim value talon_parquet_plain_bits(value src, value pos, value len,
                                        value n, value dst, value at)
{
  int64_t count = Long_val(n), bytes = (count + 7) / 8;
  if (!span(src, Long_val(pos), Long_val(len))) return Val_long(ERR_TRUNCATED);
  if (!room(Long_val(at), count, 8 * DIM(dst))) return Val_long(ERR_CAPACITY);
  if (bytes > Long_val(len)) return Val_long(ERR_TRUNCATED);
  const uint8_t *p = BYTES(src) + Long_val(pos);
  blit(BYTES(dst), (uint64_t)Long_val(at), p, p + bytes, count);
  return Val_long(bytes);
}

CAMLprim value talon_parquet_plain_bits_byte(value *argv, int argc)
{
  (void)argc;
  return talon_parquet_plain_bits(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5]);
}

/* [count_bits b at n] is the number of set bits among the [n] bits of the bit
   array [b] from bit [at]. */
CAMLprim value talon_parquet_count_bits(value b, value at, value n)
{
  int64_t a = Long_val(at), count = Long_val(n), set = 0;
  if (!room(a, count, 8 * DIM(b))) return Val_long(ERR_CAPACITY);
  const uint8_t *p = BYTES(b) + a / 8, *end = BYTES(b) + DIM(b);
  for (int64_t k = 0; k < count; k += 64) {
    int m = count - k < 64 ? (int)(count - k) : 64;
    set += ones(bits_at(p, end, (uint64_t)(a % 8 + k), m));
  }
  return Val_long(set);
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

/* The spread

   A page's values are compacted: one per row its levels mark. The spread puts
   them at their rows and zero at the others, a 64-row word of the validity at
   a time: a word of valid rows copies a run, and a word of nulls writes
   zeros. */

/* [valid_word(v, base, m)] is the [m] bits of the bit array [v] from bit
   [base], a multiple of 8. */
static inline uint64_t valid_word(value v, int64_t base, int m)
{
  const uint8_t *p = BYTES(v) + base / 8;
  return bits_at(p, BYTES(v) + DIM(v), 0, m);
}

/* [spread_fixed valid src bits rows dst] puts the values of [src], each of
   [bits] bits, 1 or a multiple of 8 up to 128, at the rows the bit array
   [valid] marks in [dst], of [rows] values, and zero elsewhere; a bit
   destination's bits past its last row are zero too. [src] and [dst] do not
   overlap. It is the number of values it took. [src]'s bytes may hold a few
   bits past its last value, which this counts as values: the caller checks
   the number taken. */
CAMLprim value talon_parquet_spread_fixed(value valid, value src, value bits,
                                          value rows, value dst)
{
  int64_t n = Long_val(rows), b = Long_val(bits), j = 0;
  if (b != 1 && (b <= 0 || b % 8 || b > 128)) return Val_long(ERR_WIDTH);
  int64_t have = 8 * DIM(src) / b, k = b / 8;
  const uint8_t *s = BYTES(src), *s_end = s + DIM(src);
  uint8_t *d = BYTES(dst);
  if (n < 0 || 8 * DIM(valid) < n || n > 8 * DIM(dst) / b)
    return Val_long(ERR_CAPACITY);
  for (int64_t base = 0; base < n; base += 64) {
    int m = n - base < 64 ? (int)(n - base) : 64;
    uint64_t v = valid_word(valid, base, m);
    int set = ones(v);
    if (set > have - j) return Val_long(ERR_CAPACITY);
    if (b == 1) {
      /* The word's values, deposited at its set rows. [put] writes whole
         bytes, so the last byte's bits past the last row are written 0. */
      uint64_t w = bits_at(s, s_end, (uint64_t)j, set), out = w;
      j += set;
      if (set < m) {
        out = 0;
        for (int r = 0; r < m && w; r++)
          if (v >> r & 1) {
            out |= (w & 1) << r;
            w >>= 1;
          }
      }
      put(d, (uint64_t)base, out, (m + 7) & ~7);
    } else if (set == m) {
      memcpy(d + base * k, s + j * k, (size_t)(m * k));
      j += m;
    } else
      for (int r = 0; r < m; r++) {
        uint8_t *at = d + (base + r) * k;
        if (v >> r & 1) memcpy(at, s + j++ * k, (size_t)k);
        else memset(at, 0, (size_t)k);
      }
  }
  return Val_long(j);
}

/* [spread_offsets valid src rows dst] puts the byte strings whose ends are
   [src.{1}] onward at the rows the bit array [valid] marks: [dst.{r + 1}] is
   [dst.{r}] plus the length of row [r], [0] at a null, and [dst.{0}] is
   [src.{0}]. [src] and [dst] do not overlap. It is the number of strings it
   took. */
CAMLprim value talon_parquet_spread_offsets(value valid, value src,
                                            value rows, value dst)
{
  const int64_t *s = (const int64_t *)Caml_ba_data_val(src);
  int64_t *d = (int64_t *)Caml_ba_data_val(dst);
  int64_t n = Long_val(rows), have = DIM(src) - 1, j = 0;
  if (have < 0 || 8 * DIM(valid) < n || DIM(dst) < n + 1)
    return Val_long(ERR_CAPACITY);
  d[0] = s[0];
  for (int64_t base = 0; base < n; base += 64) {
    int m = n - base < 64 ? (int)(n - base) : 64;
    uint64_t v = valid_word(valid, base, m);
    if (ones(v) > have - j) return Val_long(ERR_CAPACITY);
    for (int r = 0; r < m; r++) {
      int64_t l = 0;
      if (v >> r & 1) {
        l = s[j + 1] - s[j];
        j++;
      }
      d[base + r + 1] = d[base + r] + l;
    }
  }
  return Val_long(j);
}

/* [gather_ends offsets idx ends] writes to [ends] from index 1 the ends of
   the byte arrays [idx.{0}] to [idx.{m - 1}], whose ends in their data are
   [offsets] from index 1, laid one after the other from [ends.{0} = 0], [m]
   being [idx]'s length. It is the last end, or an error code. */
CAMLprim value talon_parquet_gather_ends(value offsets, value idx, value ends)
{
  const int64_t *off = (const int64_t *)Caml_ba_data_val(offsets);
  const int64_t *ix = (const int64_t *)Caml_ba_data_val(idx);
  int64_t *e = (int64_t *)Caml_ba_data_val(ends);
  int64_t m = DIM(idx), n = DIM(offsets) > 0 ? DIM(offsets) - 1 : 0, sum = 0;
  if (DIM(ends) != m + 1) return Val_long(ERR_CAPACITY);
  e[0] = 0;
  for (int64_t i = 0; i < m; i++) {
    int64_t j = ix[i];
    if ((uint64_t)j >= (uint64_t)n) return Val_long(ERR_BOUND);
    int64_t l = off[j + 1] - off[j];
    /* With [-fwrapv], a sum past [INT64_MAX] is negative. */
    sum += l;
    if ((l | sum) < 0) return Val_long(ERR_DATA);
    e[i + 1] = sum;
  }
  return Val_long(sum);
}

/* [copy(o, d, l)] copies the [l] bytes from [d] to [o]. Up to 32 bytes, it
   copies the first and last words of the row's width, which overlap, rather
   than call [memcpy]: rows of text are mostly short. */
static inline void copy(uint8_t *o, const uint8_t *d, int64_t l)
{
  if (l > 32) {
    memcpy(o, d, (size_t)l);
    return;
  }
  if (l >= 16) {
    uint64_t a[2], b[2];
    memcpy(a, d, 16);
    memcpy(b, d + l - 16, 16);
    memcpy(o, a, 16);
    memcpy(o + l - 16, b, 16);
  }
  else if (l >= 8) {
    uint64_t a, b;
    memcpy(&a, d, 8);
    memcpy(&b, d + l - 8, 8);
    memcpy(o, &a, 8);
    memcpy(o + l - 8, &b, 8);
  }
  else if (l >= 4) {
    uint32_t a, b;
    memcpy(&a, d, 4);
    memcpy(&b, d + l - 4, 4);
    memcpy(o, &a, 4);
    memcpy(o + l - 4, &b, 4);
  }
  else if (l > 0) {
    o[0] = d[0];
    o[l / 2] = d[l / 2];
    o[l - 1] = d[l - 1];
  }
}

/* [gather_byte_arrays offsets data idx ends out] copies the byte arrays
   [idx.{0}] to [idx.{m - 1}] of [data], whose ends are [offsets] from index
   1, to [out], where their ends are [ends] from index 1, [m] being [idx]'s
   length. */
CAMLprim value talon_parquet_gather_byte_arrays(value offsets, value data,
                                                value idx, value ends,
                                                value out)
{
  const int64_t *off = (const int64_t *)Caml_ba_data_val(offsets);
  const int64_t *ix = (const int64_t *)Caml_ba_data_val(idx);
  const int64_t *e = (const int64_t *)Caml_ba_data_val(ends);
  const uint8_t *d = BYTES(data);
  uint8_t *o = BYTES(out);
  int64_t m = DIM(idx), n = DIM(offsets) > 0 ? DIM(offsets) - 1 : 0;
  if (DIM(ends) != m + 1 || e[0] != 0 || e[m] > DIM(out))
    return Val_long(ERR_CAPACITY);
  /* The offsets, checked once, cut the data into arrays that lie in it, of
     lengths 0 or more. */
  if (n > 0 && (off[0] < 0 || off[n] > DIM(data))) return Val_long(ERR_DATA);
  for (int64_t k = 0; k < n; k++)
    if (off[k + 1] < off[k]) return Val_long(ERR_DATA);
  /* Each row starts where the one before ends, from 0, and has a length of
     0 or more, so [at] is 0 or more, and [last] bounds its end. */
  int64_t last = e[m], at = 0;
  for (int64_t i = 0; i < m; i++) {
    int64_t j = ix[i];
    if ((uint64_t)j >= (uint64_t)n) return Val_long(ERR_BOUND);
    int64_t lo = off[j], l = off[j + 1] - lo, next = e[i + 1];
    if (next - at != l || next > last) return Val_long(ERR_CAPACITY);
    copy(o + at, d + lo, l);
    at = next;
  }
  return Val_long(0);
}
