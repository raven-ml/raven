/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The decoder of Zstandard compressed blocks, after RFC 8878, section 3.1.1.3:
  a literals section, Huffman-coded or not, then sequences coded with three
  interleaved FSE states over a bitstream read backward. The entropy tables
  and repeat offsets of a frame's blocks live in a [compress_zstd] context.
  ---------------------------------------------------------------------------*/

#include "compress_zstd.h"

#include <string.h>

#define BLOCK_MAX (128u * 1024u)
#define HUF_MAX_BITS 11u
#define HUF_MAX_SYMBOLS 256u
#define WEIGHTS_LOG 6u
#define LL_LOG 9u
#define ML_LOG 9u
#define OF_LOG 8u

static inline uint64_t load64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

static inline unsigned highbit(uint32_t v) {
  unsigned r = 0;
  while (v >>= 1)
    r++;
  return r;
}

/* Bitstreams read backward. The container holds the 8 bytes at [ptr];
   [consumed] bits of it, from its top, are read. A stream ends with a byte
   whose highest set bit marks its end. */

typedef struct {
  uint64_t container;
  unsigned consumed;
  const uint8_t *ptr;
  const uint8_t *start;
} bits_t;

static int bits_init(bits_t *b, const uint8_t *src, size_t len) {
  if (len == 0 || src[len - 1] == 0)
    return 0;
  b->start = src;
  if (len >= 8) {
    b->ptr = src + len - 8;
    b->container = load64(b->ptr);
    b->consumed = 8 - highbit(src[len - 1]);
  } else {
    b->ptr = src;
    b->container = 0;
    for (size_t i = 0; i < len; i++)
      b->container |= (uint64_t)src[i] << (8 * i);
    b->consumed = 8 - highbit(src[len - 1]) + (unsigned)(8 - len) * 8;
  }
  return 1;
}

static void bits_reload_slow(bits_t *b) {
  if (b->consumed > 64 || b->ptr == b->start)
    return;
  size_t n = b->consumed >> 3;
  if ((size_t)(b->ptr - b->start) < n)
    n = (size_t)(b->ptr - b->start);
  b->ptr -= n;
  b->consumed -= (unsigned)n * 8;
  b->container = load64(b->ptr);
}

/* Refills the container: at least 56 bits are then available, except near
   the start of the stream. */
static inline void bits_reload(bits_t *b) {
  if (b->ptr - b->start >= 8) {
    b->ptr -= b->consumed >> 3;
    b->consumed &= 7;
    b->container = load64(b->ptr);
  } else {
    bits_reload_slow(b);
  }
}

/* The next [n] bits, zeros past the start of the stream. */
static inline uint64_t bits_look(const bits_t *b, unsigned n) {
  uint64_t v = ((b->container << (b->consumed & 63)) >> 1) >> (63 - n);
  return v & (0 - (uint64_t)(b->consumed < 64));
}

static inline uint64_t bits_read(bits_t *b, unsigned n) {
  uint64_t v = bits_look(b, n);
  b->consumed += n;
  return v;
}

static inline int bits_ended(const bits_t *b) {
  return b->ptr == b->start && b->consumed == 64;
}

/* FSE tables */

/* The normalized distributions of RFC 8878, section 3.1.1.3.2.2. */
static const int16_t ll_default[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
static const int16_t ml_default[53] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  -1, -1, -1, -1, -1, -1, -1};
static const int16_t of_default[29] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, -1, -1, -1, -1, -1};

static int fse_build(compress_zstd_fse *t, const int16_t *counts,
                     unsigned symbols, unsigned log) {
  unsigned size = 1u << log;
  unsigned high = size;
  uint16_t next[64];
  for (unsigned s = 0; s < symbols; s++)
    if (counts[s] == -1) {
      t->table[--high].symbol = (uint8_t)s;
      next[s] = 1;
    }
  unsigned step = (size >> 1) + (size >> 3) + 3;
  unsigned mask = size - 1;
  unsigned pos = 0;
  for (unsigned s = 0; s < symbols; s++) {
    if (counts[s] <= 0)
      continue;
    next[s] = (uint16_t)counts[s];
    for (int i = 0; i < counts[s]; i++) {
      t->table[pos].symbol = (uint8_t)s;
      do
        pos = (pos + step) & mask;
      while (pos >= high);
    }
  }
  if (pos != 0)
    return 0;
  for (unsigned i = 0; i < size; i++) {
    unsigned state = next[t->table[i].symbol]++;
    unsigned bits = log - highbit(state);
    t->table[i].bits = (uint8_t)bits;
    t->table[i].base = (uint16_t)((state << bits) - size);
  }
  t->log = log;
  return 1;
}

static void fse_rle(compress_zstd_fse *t, uint8_t symbol) {
  t->table[0].symbol = symbol;
  t->table[0].bits = 0;
  t->table[0].base = 0;
  t->log = 0;
}

/* Reads the FSE table description at [src[*pos, end)] (RFC 8878, section
   4.1.1) for at most [symbols] symbols and an accuracy log of at most
   [max_log], and builds its table. */
static int fse_read(compress_zstd_fse *t, const uint8_t *src, size_t *pos,
                    size_t end, unsigned symbols, unsigned max_log) {
  size_t bit = *pos * 8, bit_end = end * 8;
#define READ(n, v)                                                            \
  do {                                                                        \
    if (bit + (n) > bit_end)                                                  \
      return 0;                                                               \
    v = 0;                                                                    \
    for (unsigned k = 0; k < (n); k++, bit++)                                 \
      v |= (unsigned)((src[bit >> 3] >> (bit & 7)) & 1u) << k;                \
  } while (0)
  unsigned log;
  READ(4, log);
  log += 5;
  if (log > max_log)
    return 0;
  int16_t counts[64];
  int remaining = (1 << log) + 1;
  unsigned s = 0;
  while (remaining > 1) {
    if (s >= symbols)
      return 0;
    unsigned nbits = highbit((uint32_t)remaining) + 1;
    unsigned lower = (1u << (nbits - 1)) - 1;
    unsigned threshold = (1u << nbits) - 1 - (unsigned)remaining;
    unsigned v;
    READ(nbits - 1, v);
    if (v >= threshold) {
      unsigned top;
      READ(1, top);
      v |= top << (nbits - 1);
      if (v > lower)
        v -= threshold;
    }
    int count = (int)v - 1;
    remaining -= count < 0 ? -count : count;
    counts[s++] = (int16_t)count;
    if (count == 0) {
      unsigned repeat;
      do {
        READ(2, repeat);
        if (s + repeat > symbols)
          return 0;
        for (unsigned k = 0; k < repeat; k++)
          counts[s++] = 0;
      } while (repeat == 3);
    }
  }
#undef READ
  if (remaining != 1)
    return 0;
  *pos = (bit + 7) / 8;
  return fse_build(t, counts, s, log);
}

/* Huffman tables */

/* Builds the table of [n] weights and the implied last one (RFC 8878, section
   4.2.1). */
static int huf_build(compress_zstd *z, const uint8_t *weights, unsigned n) {
  uint32_t sum = 0;
  for (unsigned s = 0; s < n; s++) {
    if (weights[s] > HUF_MAX_BITS)
      return 0;
    if (weights[s] > 0)
      sum += 1u << (weights[s] - 1);
  }
  if (sum == 0)
    return 0;
  unsigned max_bits = highbit(sum) + 1;
  uint32_t rest = (1u << max_bits) - sum;
  if (max_bits > HUF_MAX_BITS || (rest & (rest - 1)) != 0)
    return 0;
  uint8_t bits[HUF_MAX_SYMBOLS];
  for (unsigned s = 0; s < n; s++)
    bits[s] = (uint8_t)(weights[s] > 0 ? max_bits + 1 - weights[s] : 0);
  bits[n] = (uint8_t)(max_bits - highbit(rest));
  unsigned count[HUF_MAX_BITS + 2] = {0};
  for (unsigned s = 0; s <= n; s++)
    count[bits[s]]++;
  /* Codes of the most bits come first: their ranges in the table are the
     smallest. */
  uint32_t start[HUF_MAX_BITS + 2];
  start[max_bits] = 0;
  for (unsigned b = max_bits; b >= 1; b--)
    start[b - 1] = start[b] + count[b] * (1u << (max_bits - b));
  if (start[0] != 1u << max_bits)
    return 0;
  for (unsigned s = 0; s <= n; s++) {
    if (bits[s] == 0)
      continue;
    uint32_t length = 1u << (max_bits - bits[s]);
    for (uint32_t i = 0; i < length; i++) {
      z->huf[start[bits[s]] + i].symbol = (uint8_t)s;
      z->huf[start[bits[s]] + i].bits = bits[s];
    }
    start[bits[s]] += length;
  }
  z->huf_bits = max_bits;
  z->huf_valid = 1;
  return 1;
}

static int huf_read(compress_zstd *z, const uint8_t *src, size_t *pos,
                    size_t end) {
  uint8_t weights[HUF_MAX_SYMBOLS];
  unsigned n = 0;
  if (*pos >= end)
    return 0;
  unsigned header = src[(*pos)++];
  if (header >= 128) {
    n = header - 127;
    if ((n + 1) / 2 > end - *pos)
      return 0;
    for (unsigned i = 0; i < n; i++)
      weights[i] = (uint8_t)(i % 2 == 0 ? src[*pos + i / 2] >> 4
                                        : src[*pos + i / 2] & 15);
    *pos += (n + 1) / 2;
  } else {
    if (header == 0 || header > end - *pos)
      return 0;
    size_t table_end = *pos + header;
    compress_zstd_fse t;
    size_t p = *pos;
    if (!fse_read(&t, src, &p, table_end, HUF_MAX_BITS + 1, WEIGHTS_LOG))
      return 0;
    bits_t b;
    if (!bits_init(&b, src + p, table_end - p))
      return 0;
    unsigned s1 = (unsigned)bits_read(&b, t.log);
    unsigned s2 = (unsigned)bits_read(&b, t.log);
    /* Two states take turns until the stream would run out: each state then
       gives its last symbol. */
    for (;;) {
      if (n + 2 > HUF_MAX_SYMBOLS - 1)
        return 0;
      weights[n++] = t.table[s1].symbol;
      bits_reload(&b);
      s1 = t.table[s1].base + (unsigned)bits_read(&b, t.table[s1].bits);
      if (b.consumed > 64) {
        weights[n++] = t.table[s2].symbol;
        break;
      }
      weights[n++] = t.table[s2].symbol;
      bits_reload(&b);
      s2 = t.table[s2].base + (unsigned)bits_read(&b, t.table[s2].bits);
      if (b.consumed > 64) {
        weights[n++] = t.table[s1].symbol;
        break;
      }
    }
    *pos = table_end;
  }
  return huf_build(z, weights, n);
}

/* Decodes [n[i]] symbols of each of the [streams] Huffman streams
   [src[i]] into [out[i]]; each must be read exactly. The streams are decoded
   in turns, so that their symbols decode in parallel. */
static int huf_streams(const compress_zstd *z, unsigned streams,
                       const uint8_t *const *src, const size_t *len,
                       uint8_t *const *out, const size_t *n) {
  bits_t b[4];
  size_t done[4] = {0, 0, 0, 0};
  unsigned max_bits = z->huf_bits;
  for (unsigned i = 0; i < streams; i++)
    if (!bits_init(&b[i], src[i], len[i]))
      return 0;
  /* After a reload away from a stream's start, 56 bits are available: four
     symbols of at most 11 bits. */
  for (;;) {
    int fast = 1;
    for (unsigned i = 0; i < streams; i++) {
      bits_reload(&b[i]);
      fast &= b[i].ptr != b[i].start && done[i] + 4 <= n[i];
    }
    if (!fast)
      break;
    for (unsigned k = 0; k < 4; k++)
      for (unsigned i = 0; i < streams; i++) {
        compress_zstd_huf e =
            z->huf[((b[i].container << b[i].consumed) >> 1) >> (63 - max_bits)];
        out[i][done[i] + k] = e.symbol;
        b[i].consumed += e.bits;
      }
    for (unsigned i = 0; i < streams; i++)
      done[i] += 4;
  }
  for (unsigned i = 0; i < streams; i++) {
    for (; done[i] < n[i]; done[i]++) {
      bits_reload(&b[i]);
      compress_zstd_huf e = z->huf[bits_look(&b[i], max_bits)];
      out[i][done[i]] = e.symbol;
      b[i].consumed += e.bits;
    }
    bits_reload(&b[i]);
    if (!bits_ended(&b[i]))
      return 0;
  }
  return 1;
}

/* Literals */

static int literals(compress_zstd *z, const uint8_t *src, size_t *pos,
                    size_t end, const uint8_t **lit, size_t *lit_len) {
  size_t p = *pos;
  if (p >= end)
    return COMPRESS_ZSTD_TRUNCATED;
  unsigned type = src[p] & 3, format = (src[p] >> 2) & 3;
  size_t regenerated, compressed = 0;
  if (type < 2) {
    unsigned header = format == 1 ? 2 : format == 3 ? 3 : 1;
    if (end - p < header)
      return COMPRESS_ZSTD_TRUNCATED;
    uint32_t h = src[p] | (header > 1 ? (uint32_t)src[p + 1] << 8 : 0) |
                 (header > 2 ? (uint32_t)src[p + 2] << 16 : 0);
    regenerated = h >> (header == 1 ? 3 : 4);
    p += header;
    if (regenerated > BLOCK_MAX)
      return COMPRESS_ZSTD_LITERALS;
    if (type == 0) {
      if (end - p < regenerated)
        return COMPRESS_ZSTD_TRUNCATED;
      *lit = src + p;
      p += regenerated;
    } else {
      if (p >= end)
        return COMPRESS_ZSTD_TRUNCATED;
      memset(z->literals, src[p++], regenerated);
      *lit = z->literals;
    }
    *lit_len = regenerated;
    *pos = p;
    return COMPRESS_ZSTD_OK;
  }
  unsigned header = format < 2 ? 3 : format == 2 ? 4 : 5;
  unsigned width = format < 2 ? 10 : format == 2 ? 14 : 18;
  if (end - p < header)
    return COMPRESS_ZSTD_TRUNCATED;
  uint64_t h = 0;
  for (unsigned i = 0; i < header; i++)
    h |= (uint64_t)src[p + i] << (8 * i);
  h >>= 4;
  regenerated = (size_t)(h & ((1u << width) - 1));
  compressed = (size_t)(h >> width);
  unsigned streams = format == 0 ? 1 : 4;
  p += header;
  if (regenerated > BLOCK_MAX || compressed > end - p)
    return COMPRESS_ZSTD_LITERALS;
  size_t stop = p + compressed;
  if (type == 2) {
    if (!huf_read(z, src, &p, stop))
      return COMPRESS_ZSTD_HUFFMAN;
  } else if (!z->huf_valid) {
    return COMPRESS_ZSTD_HUFFMAN;
  }
  const uint8_t *starts[4];
  size_t sizes[4];
  uint8_t *outs[4];
  size_t counts[4];
  if (streams == 1) {
    starts[0] = src + p;
    sizes[0] = stop - p;
    outs[0] = z->literals;
    counts[0] = regenerated;
  } else {
    if (stop - p < 6)
      return COMPRESS_ZSTD_LITERALS;
    size_t total = 0;
    for (unsigned i = 0; i < 3; i++) {
      sizes[i] = src[p + 2 * i] | ((size_t)src[p + 2 * i + 1] << 8);
      total += sizes[i];
    }
    p += 6;
    if (total > stop - p)
      return COMPRESS_ZSTD_LITERALS;
    sizes[3] = stop - p - total;
    size_t segment = (regenerated + 3) / 4;
    if (segment * 3 > regenerated)
      return COMPRESS_ZSTD_LITERALS;
    for (unsigned i = 0; i < 4; i++) {
      starts[i] = src + p;
      p += sizes[i];
      outs[i] = z->literals + i * segment;
      counts[i] = i < 3 ? segment : regenerated - 3 * segment;
    }
  }
  if (!huf_streams(z, streams, starts, sizes, outs, counts))
    return COMPRESS_ZSTD_LITERALS;
  *lit = z->literals;
  *lit_len = regenerated;
  *pos = stop;
  return COMPRESS_ZSTD_OK;
}

/* Sequences */

static const uint32_t ll_base[36] = {
    0,  1,  2,   3,   4,   5,    6,    7,    8,    9,     10,    11,
    12, 13, 14,  15,  16,  18,   20,   22,   24,   28,    32,    40,
    48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
static const uint8_t ll_bits[36] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,
                                    0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  3,  3,
                                    4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const uint32_t ml_base[53] = {
    3,  4,  5,  6,  7,  8,  9,  10,  11,  12,   13,   14,   15,   16,
    17, 18, 19, 20, 21, 22, 23, 24,  25,  26,   27,   28,   29,   30,
    31, 32, 33, 34, 35, 37, 39, 41,  43,  47,   51,   59,   67,   83,
    99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};
static const uint8_t ml_bits[53] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

/* Selects the table of [mode] and gives its states the values of their
   symbols. */
static int table_for(compress_zstd_fse *t, int *valid, unsigned mode,
                     const uint8_t *src, size_t *pos, size_t end,
                     const int16_t *defaults, unsigned default_symbols,
                     unsigned default_log, unsigned symbols, unsigned max_log,
                     const uint32_t *baselines, const uint8_t *extras) {
  switch (mode) {
  case 0:
    fse_build(t, defaults, default_symbols, default_log);
    break;
  case 1:
    if (*pos >= end || src[*pos] >= symbols)
      return 0;
    fse_rle(t, src[(*pos)++]);
    break;
  case 2:
    if (!fse_read(t, src, pos, end, symbols, max_log))
      return 0;
    break;
  default:
    return *valid;
  }
  for (unsigned i = 0; i < (1u << t->log); i++) {
    unsigned s = t->table[i].symbol;
    t->table[i].baseline = baselines == NULL ? 1u << s : baselines[s];
    t->table[i].extra = extras == NULL ? (uint8_t)s : extras[s];
  }
  *valid = 1;
  return 1;
}

/* Copies [n] bytes from [offset] back. Wide copies may write up to 15 bytes
   past the match, which [room] holds and are not yet produced. */
static inline void copy_match(uint8_t *d, size_t offset, size_t n,
                              size_t room) {
  const uint8_t *s = d - offset;
  if (n + 16 > room) {
    for (size_t k = 0; k < n; k++)
      d[k] = s[k];
    return;
  }
  if (offset < 8) {
    /* The data repeats every [offset] bytes, hence every [step >= 8]. */
    size_t step = offset;
    while (step < 8)
      step += offset;
    for (size_t k = 0; k < step; k++)
      d[k] = s[k];
    s = d;
    d += step;
    if (n <= step)
      return;
    n -= step;
  }
  for (size_t k = 0; k < n; k += 8)
    memcpy(d + k, s + k, 8);
}

int compress_zstd_block(compress_zstd *z, const uint8_t *src, size_t pos,
                        size_t end, uint8_t *dst, size_t hist, size_t *out,
                        size_t dst_end) {
  const uint8_t *lit;
  size_t lit_len;
  int status = literals(z, src, &pos, end, &lit, &lit_len);
  if (status != COMPRESS_ZSTD_OK)
    return status;
  if (pos >= end)
    return COMPRESS_ZSTD_TRUNCATED;
  size_t count = src[pos++];
  if (count >= 128) {
    if (count < 255) {
      if (pos >= end)
        return COMPRESS_ZSTD_TRUNCATED;
      count = ((count - 128) << 8) + src[pos++];
    } else {
      if (end - pos < 2)
        return COMPRESS_ZSTD_TRUNCATED;
      count = src[pos] + ((size_t)src[pos + 1] << 8) + 0x7f00;
      pos += 2;
    }
  }
  size_t op = *out;
  const uint8_t *lit_end = lit + lit_len;
  /* Literals may be read 16 bytes at a time up to here. */
  const uint8_t *lit_limit =
      lit == z->literals ? z->literals + sizeof(z->literals) : src + end;
  if (count > 0) {
    if (pos >= end)
      return COMPRESS_ZSTD_TRUNCATED;
    unsigned modes = src[pos++];
    if ((modes & 3) != 0)
      return COMPRESS_ZSTD_SEQUENCES;
    if (!table_for(&z->ll, &z->ll_valid, modes >> 6, src, &pos, end,
                   ll_default, 36, 6, 36, LL_LOG, ll_base, ll_bits) ||
        !table_for(&z->of, &z->of_valid, (modes >> 4) & 3, src, &pos, end,
                   of_default, 29, 5, 32, OF_LOG, NULL, NULL) ||
        !table_for(&z->ml, &z->ml_valid, (modes >> 2) & 3, src, &pos, end,
                   ml_default, 53, 6, 53, ML_LOG, ml_base, ml_bits))
      return COMPRESS_ZSTD_SEQUENCES;
    bits_t b;
    if (!bits_init(&b, src + pos, end - pos))
      return COMPRESS_ZSTD_SEQUENCES;
    unsigned ll_state = (unsigned)bits_read(&b, z->ll.log);
    unsigned of_state = (unsigned)bits_read(&b, z->of.log);
    unsigned ml_state = (unsigned)bits_read(&b, z->ml.log);
    for (size_t i = 0; i < count; i++) {
      const compress_zstd_fse_entry *ll = &z->ll.table[ll_state];
      const compress_zstd_fse_entry *of = &z->of.table[of_state];
      const compress_zstd_fse_entry *ml = &z->ml.table[ml_state];
      /* A reload leaves at least 57 bits. A sequence reads at most 31 + 16 +
         16 bits of values and 9 + 9 + 8 of states: one reload is enough
         unless its values are long. */
      int long_values = of->extra + ml->extra + ll->extra > 31;
      bits_reload(&b);
      size_t offset = of->baseline + bits_read(&b, of->extra);
      if (long_values)
        bits_reload(&b);
      size_t match = ml->baseline + bits_read(&b, ml->extra);
      size_t length = ll->baseline + bits_read(&b, ll->extra);
      if (offset > 3) {
        offset -= 3;
        z->rep[2] = z->rep[1];
        z->rep[1] = z->rep[0];
        z->rep[0] = offset;
      } else {
        unsigned index = (unsigned)offset - (length == 0 ? 0 : 1);
        if (index == 0) {
          offset = z->rep[0];
        } else {
          offset = index == 3 ? z->rep[0] - 1 : z->rep[index];
          if (index != 1)
            z->rep[2] = z->rep[1];
          z->rep[1] = z->rep[0];
          z->rep[0] = offset;
        }
      }
      if (i + 1 < count) {
        if (long_values)
          bits_reload(&b);
        ll_state = ll->base + (unsigned)bits_read(&b, ll->bits);
        ml_state = ml->base + (unsigned)bits_read(&b, ml->bits);
        of_state = of->base + (unsigned)bits_read(&b, of->bits);
      }
      if (b.consumed > 64 || length > (size_t)(lit_end - lit))
        return COMPRESS_ZSTD_SEQUENCES;
      if (length + match > dst_end - op)
        return COMPRESS_ZSTD_TOO_LONG;
      if (length <= 16 && lit_limit - lit >= 16 && dst_end - op >= 16)
        memcpy(dst + op, lit, 16);
      else
        memcpy(dst + op, lit, length);
      lit += length;
      op += length;
      if (offset == 0)
        return COMPRESS_ZSTD_ZERO_OFFSET;
      if (offset > op - hist)
        return COMPRESS_ZSTD_OFFSET;
      copy_match(dst + op, offset, match, dst_end - op);
      op += match;
    }
    bits_reload(&b);
    if (!bits_ended(&b))
      return COMPRESS_ZSTD_SEQUENCES;
  } else if (pos != end) {
    return COMPRESS_ZSTD_SEQUENCES;
  }
  size_t rest = (size_t)(lit_end - lit);
  if (rest > dst_end - op)
    return COMPRESS_ZSTD_TOO_LONG;
  memcpy(dst + op, lit, rest);
  *out = op + rest;
  return COMPRESS_ZSTD_OK;
}

void compress_zstd_reset(compress_zstd *z) {
  z->huf_valid = 0;
  z->ll_valid = z->of_valid = z->ml_valid = 0;
  z->rep[0] = 1;
  z->rep[1] = 4;
  z->rep[2] = 8;
}
