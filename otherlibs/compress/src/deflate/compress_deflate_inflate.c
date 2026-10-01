/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  A resumable decoder of RFC 1951 deflate data. Codes decode through
  two-level tables: a root table indexed by the next 10 (literal/length) or 8
  (distance) bits, whose entries either hold a symbol or point at a subtable
  for longer codes. Entries of length and distance symbols hold their base
  and extra bit count, so a match decodes without further lookups.
  ---------------------------------------------------------------------------*/

#include "compress_deflate.h"

#include <string.h>

#define LIT_ROOT 10u
#define DIST_ROOT 8u
#define CODELEN_ROOT 7u

/* An entry: bits 0-7 the length of its code, 8-12 its extra bits (a subtable's
   index bits for [SUB]), 13-15 its kind, 16-31 its value: a literal byte, a
   base length or distance, a code length symbol or a subtable's offset. */
enum { LITERAL, BASE, END, SUB, INVALID };

#define ENTRY(kind, extra, value)                                             \
  (((uint32_t)(value) << 16) | ((uint32_t)(kind) << 13) |                     \
   ((uint32_t)(extra) << 8))
#define E_LEN(e) ((e) & 0xffu)
#define E_EXTRA(e) (((e) >> 8) & 0x1fu)
#define E_KIND(e) (((e) >> 13) & 7u)
#define E_VALUE(e) ((e) >> 16)

enum { MODE_HEADER, MODE_STORED, MODE_CODES, MODE_DONE };

static uint32_t lit_template[288];
static uint32_t dist_template[32];
static uint32_t codelen_template[19];
static uint32_t fixed_lit[1u << LIT_ROOT];
static uint32_t fixed_dist[1u << DIST_ROOT];

static unsigned reverse(unsigned code, unsigned len) {
  unsigned r = 0;
  for (unsigned i = 0; i < len; i++, code >>= 1)
    r = (r << 1) | (code & 1u);
  return r;
}

/* Fills [table] for the canonical code of [lengths]. Returns 0 if the code is
   over-subscribed or incomplete. An incomplete code is accepted when [codes]
   is 0 and it is empty or a single code of length 1, as zlib does. */
static int build(uint32_t *table, size_t capacity, unsigned root,
                 const uint8_t *lengths, unsigned n, const uint32_t *template,
                 int codes) {
  unsigned count[16] = {0};
  for (unsigned s = 0; s < n; s++)
    count[lengths[s]]++;
  int left = 1;
  unsigned max = 0;
  for (unsigned len = 1; len <= 15; len++) {
    left = (left << 1) - (int)count[len];
    if (left < 0)
      return 0;
    if (count[len] != 0)
      max = len;
  }
  if (left > 0 && (codes || max > 1))
    return 0;
  for (size_t i = 0; i < (1u << root); i++)
    table[i] = ENTRY(INVALID, 0, 0) | root;

  unsigned offset[16];
  uint16_t sorted[288];
  uint16_t code_of[288];
  offset[1] = 0;
  for (unsigned len = 1; len < 15; len++)
    offset[len + 1] = offset[len] + count[len];
  for (unsigned s = 0; s < n; s++)
    if (lengths[s] != 0)
      sorted[offset[lengths[s]]++] = (uint16_t)s;
  unsigned total = 0;
  unsigned code = 0;
  for (unsigned len = 1; len <= 15; len++, code <<= 1)
    for (unsigned c = 0; c < count[len]; c++)
      code_of[total++] = (uint16_t)code++;

  size_t next = 1u << root;
  for (unsigned k = 0; k < total; k++) {
    unsigned s = sorted[k];
    unsigned len = lengths[s];
    unsigned c = code_of[k];
    unsigned rev = reverse(c, len);
    uint32_t e = template[s] | len;
    if (len <= root) {
      for (size_t i = rev; i < (1u << root); i += 1u << len)
        table[i] = e;
      continue;
    }
    /* Codes longer than [root] that share their first [root] bits are
       consecutive in canonical order: the last of them is the longest. */
    unsigned prefix = c >> (len - root);
    unsigned r = rev & ((1u << root) - 1u);
    if (E_KIND(table[r]) != SUB) {
      unsigned j = k;
      while (j + 1 < total &&
             (unsigned)(code_of[j + 1] >> (lengths[sorted[j + 1]] - root)) ==
                 prefix)
        j++;
      unsigned bits = lengths[sorted[j]] - root;
      if (next + (1u << bits) > capacity)
        return 0;
      table[r] = ENTRY(SUB, bits, next) | root;
      for (size_t i = 0; i < (1u << bits); i++)
        table[next + i] = ENTRY(INVALID, 0, 0) | (root + bits);
      next += 1u << bits;
    }
    size_t base = E_VALUE(table[r]);
    unsigned bits = E_EXTRA(table[r]);
    for (size_t i = rev >> root; i < (1u << bits); i += 1u << (len - root))
      table[base + i] = e;
  }
  return 1;
}

void compress_inflate_init(void) {
  static const uint16_t length_base[29] = {
      3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
      31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const uint8_t length_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                           1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                           4, 4, 4, 4, 5, 5, 5, 5, 0};
  static const uint16_t dist_base[30] = {
      1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
      33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
      1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
  static const uint8_t dist_extra[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                         4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                         9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  for (unsigned s = 0; s < 256; s++)
    lit_template[s] = ENTRY(LITERAL, 0, s);
  lit_template[256] = ENTRY(END, 0, 0);
  for (unsigned s = 257; s < 286; s++)
    lit_template[s] = ENTRY(BASE, length_extra[s - 257], length_base[s - 257]);
  lit_template[286] = lit_template[287] = ENTRY(INVALID, 0, 0);
  for (unsigned s = 0; s < 30; s++)
    dist_template[s] = ENTRY(BASE, dist_extra[s], dist_base[s]);
  dist_template[30] = dist_template[31] = ENTRY(INVALID, 0, 0);
  for (unsigned s = 0; s < 19; s++)
    codelen_template[s] = ENTRY(LITERAL, 0, s);

  uint8_t lengths[288];
  memset(lengths, 8, 144);
  memset(lengths + 144, 9, 112);
  memset(lengths + 256, 7, 24);
  memset(lengths + 280, 8, 8);
  build(fixed_lit, 1u << LIT_ROOT, LIT_ROOT, lengths, 288, lit_template, 0);
  memset(lengths, 5, 32);
  build(fixed_dist, 1u << DIST_ROOT, DIST_ROOT, lengths, 32, dist_template, 0);
}

void compress_inflate_reset(compress_inflate *s) {
  s->mode = MODE_HEADER;
  s->last = 0;
  s->bits = 0;
  s->nbits = 0;
  s->stored = 0;
  s->copy_length = 0;
  s->copy_distance = 0;
  s->lit = fixed_lit;
  s->dist = fixed_dist;
}

static inline uint64_t load64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

/* Copies a match of [n] bytes at [distance] behind [d], which has [room]
   bytes after it. Wide copies may write up to 15 bytes past the match, which
   are inside [room] and not yet produced. */
static inline void copy_match(uint8_t *d, size_t distance, size_t n,
                              size_t room) {
  const uint8_t *s = d - distance;
  if (n + 16 > room) {
    for (size_t k = 0; k < n; k++)
      d[k] = s[k];
    return;
  }
  if (distance < 8) {
    /* The data repeats every [distance] bytes, hence every [step >= 8]. */
    size_t step = distance;
    while (step < 8)
      step += distance;
    for (size_t k = 0; k < step; k++)
      d[k] = s[k];
    if (n <= step)
      return;
    s = d;
    d += step;
    n -= step;
  }
  for (size_t k = 0; k < n; k += 8)
    memcpy(d + k, s + k, 8);
}

/* The bit buffer holds [nbits] valid bits. Bits above them are zero or the
   stream's next bits, so refills may OR the same bytes again. */
#define DROP(n) (bits >>= (n), nbits -= (n))

#define NEED(n)                                                               \
  while (nbits < (n)) {                                                       \
    if (pos == end)                                                           \
      goto more;                                                              \
    bits |= (uint64_t)src[pos++] << nbits;                                    \
    nbits += 8;                                                               \
  }

#define REFILL()                                                              \
  do {                                                                        \
    if (end - pos >= 8) {                                                     \
      bits |= load64(src + pos) << nbits;                                     \
      pos += (63 - nbits) >> 3;                                               \
      nbits |= 56;                                                            \
    } else {                                                                  \
      while (nbits < 56 && pos < end) {                                       \
        bits |= (uint64_t)src[pos++] << nbits;                                \
        nbits += 8;                                                           \
      }                                                                       \
    }                                                                         \
  } while (0)

#define LOOKUP(e, table, root)                                                \
  do {                                                                        \
    e = (table)[bits & ((1u << (root)) - 1u)];                                \
    if (E_KIND(e) == SUB)                                                     \
      e = (table)[E_VALUE(e) +                                                \
                  ((bits >> (root)) & ((1u << E_EXTRA(e)) - 1u))];            \
  } while (0)

int compress_inflate_run(compress_inflate *s, compress_inflate_io *io) {
  static const uint8_t order[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                    11, 4,  12, 3, 13, 2, 14, 1, 15};
  const uint8_t *src = io->src;
  size_t pos = io->src_pos;
  size_t end = io->src_end;
  uint8_t *dst = io->dst;
  size_t out = io->dst_pos;
  size_t out_end = io->dst_end;
  size_t hist = io->dst_hist;
  uint64_t bits = s->bits;
  unsigned nbits = s->nbits;
  size_t save_pos;
  uint64_t save_bits;
  unsigned save_nbits;
  int status;

  for (;;) {
    save_pos = pos;
    save_bits = bits;
    save_nbits = nbits;
    switch (s->mode) {
    case MODE_DONE:
      status = COMPRESS_INFLATE_END;
      goto leave;

    case MODE_HEADER: {
      NEED(3);
      s->last = (int)(bits & 1u);
      unsigned type = (bits >> 1) & 3u;
      DROP(3);
      if (type == 0) {
        DROP(nbits & 7u);
        pos -= nbits >> 3;
        bits = 0;
        nbits = 0;
        if (end - pos < 4)
          goto more;
        unsigned len = src[pos] | ((unsigned)src[pos + 1] << 8);
        unsigned nlen = src[pos + 2] | ((unsigned)src[pos + 3] << 8);
        if (len != (~nlen & 0xffffu)) {
          status = COMPRESS_INFLATE_STORED_LENGTH;
          goto fail;
        }
        pos += 4;
        s->stored = len;
        s->mode = MODE_STORED;
      } else if (type == 1) {
        s->lit = fixed_lit;
        s->dist = fixed_dist;
        s->mode = MODE_CODES;
      } else if (type == 2) {
        uint8_t lengths[286 + 30];
        uint32_t codelen[1u << CODELEN_ROOT];
        NEED(14);
        unsigned nlit = 257 + (unsigned)(bits & 31u);
        unsigned ndist = 1 + (unsigned)((bits >> 5) & 31u);
        unsigned ncode = 4 + (unsigned)((bits >> 10) & 15u);
        DROP(14);
        if (nlit > 286 || ndist > 30) {
          status = COMPRESS_INFLATE_CODE_COUNTS;
          goto fail;
        }
        memset(lengths, 0, 19);
        for (unsigned i = 0; i < ncode; i++) {
          NEED(3);
          lengths[order[i]] = (uint8_t)(bits & 7u);
          DROP(3);
        }
        if (!build(codelen, 1u << CODELEN_ROOT, CODELEN_ROOT, lengths, 19,
                   codelen_template, 1)) {
          status = COMPRESS_INFLATE_HUFFMAN;
          goto fail;
        }
        unsigned n = nlit + ndist;
        for (unsigned i = 0; i < n;) {
          uint32_t e;
          for (;;) {
            e = codelen[bits & ((1u << CODELEN_ROOT) - 1u)];
            if (E_LEN(e) <= nbits)
              break;
            if (pos == end)
              goto more;
            bits |= (uint64_t)src[pos++] << nbits;
            nbits += 8;
          }
          DROP(E_LEN(e));
          unsigned symbol = E_VALUE(e);
          if (symbol < 16) {
            lengths[i++] = (uint8_t)symbol;
            continue;
          }
          unsigned repeat;
          uint8_t value = 0;
          if (symbol == 16) {
            if (i == 0) {
              status = COMPRESS_INFLATE_REPEAT;
              goto fail;
            }
            NEED(2);
            repeat = 3 + (unsigned)(bits & 3u);
            DROP(2);
            value = lengths[i - 1];
          } else if (symbol == 17) {
            NEED(3);
            repeat = 3 + (unsigned)(bits & 7u);
            DROP(3);
          } else {
            NEED(7);
            repeat = 11 + (unsigned)(bits & 127u);
            DROP(7);
          }
          if (repeat > n - i) {
            status = COMPRESS_INFLATE_REPEAT;
            goto fail;
          }
          memset(lengths + i, value, repeat);
          i += repeat;
        }
        if (lengths[256] == 0) {
          status = COMPRESS_INFLATE_NO_END_CODE;
          goto fail;
        }
        if (!build(s->lit_table, COMPRESS_INFLATE_LIT_ENOUGH, LIT_ROOT,
                   lengths, nlit, lit_template, 0) ||
            !build(s->dist_table, COMPRESS_INFLATE_DIST_ENOUGH, DIST_ROOT,
                   lengths + nlit, ndist, dist_template, 0)) {
          status = COMPRESS_INFLATE_HUFFMAN;
          goto fail;
        }
        s->lit = s->lit_table;
        s->dist = s->dist_table;
        s->mode = MODE_CODES;
      } else {
        status = COMPRESS_INFLATE_BLOCK_TYPE;
        goto fail;
      }
      break;
    }

    case MODE_STORED: {
      size_t n = s->stored;
      if (n > end - pos)
        n = end - pos;
      if (n > out_end - out)
        n = out_end - out;
      memcpy(dst + out, src + pos, n);
      pos += n;
      out += n;
      s->stored -= n;
      if (s->stored == 0) {
        s->mode = s->last ? MODE_DONE : MODE_HEADER;
        break;
      }
      if (out == out_end) {
        status = COMPRESS_INFLATE_OUTPUT;
        goto leave;
      }
      save_pos = pos;
      goto more;
    }

    case MODE_CODES:
      for (;;) {
        if (s->copy_length != 0) {
          size_t n = s->copy_length;
          if (n > out_end - out)
            n = out_end - out;
          copy_match(dst + out, s->copy_distance, n, out_end - out);
          out += n;
          s->copy_length -= n;
          if (s->copy_length != 0) {
            status = COMPRESS_INFLATE_OUTPUT;
            goto leave;
          }
        }
        save_pos = pos;
        save_bits = bits;
        save_nbits = nbits;
        if (end - pos >= 8 && out_end - out >= 258 + 16) {
          /* Far from both ends: a refill holds at least 56 bits, a whole
             literal/length and distance pair (48 bits) or three literals,
             and the output holds the longest match. */
          bits |= load64(src + pos) << nbits;
          pos += (63 - nbits) >> 3;
          nbits |= 56;
          uint32_t e;
          LOOKUP(e, s->lit, LIT_ROOT);
          unsigned kind = E_KIND(e);
          if (kind == LITERAL) {
            /* Two more literals fit in the refill; anything else waits for
               the next one. */
            DROP(E_LEN(e));
            dst[out++] = (uint8_t)E_VALUE(e);
            for (int k = 0; k < 2; k++) {
              LOOKUP(e, s->lit, LIT_ROOT);
              if (E_KIND(e) != LITERAL)
                break;
              DROP(E_LEN(e));
              dst[out++] = (uint8_t)E_VALUE(e);
            }
            continue;
          }
          if (kind == END) {
            DROP(E_LEN(e));
            s->mode = s->last ? MODE_DONE : MODE_HEADER;
            break;
          }
          if (kind != BASE) {
            status = COMPRESS_INFLATE_SYMBOL;
            goto fail;
          }
          unsigned len = E_LEN(e);
          size_t length =
              E_VALUE(e) + (size_t)((bits >> len) & ((1u << E_EXTRA(e)) - 1u));
          DROP(len + E_EXTRA(e));
          LOOKUP(e, s->dist, DIST_ROOT);
          if (E_KIND(e) != BASE) {
            status = COMPRESS_INFLATE_SYMBOL;
            goto fail;
          }
          len = E_LEN(e);
          size_t distance =
              E_VALUE(e) + (size_t)((bits >> len) & ((1u << E_EXTRA(e)) - 1u));
          if (distance > out - hist) {
            status = COMPRESS_INFLATE_DISTANCE;
            goto fail;
          }
          DROP(len + E_EXTRA(e));
          copy_match(dst + out, distance, length, out_end - out);
          out += length;
          continue;
        }
        if (nbits < 48)
          REFILL();
        uint32_t e;
        LOOKUP(e, s->lit, LIT_ROOT);
        unsigned len = E_LEN(e);
        if (len > nbits)
          goto more;
        unsigned kind = E_KIND(e);
        if (kind == LITERAL) {
          if (out == out_end) {
            status = COMPRESS_INFLATE_OUTPUT;
            goto leave;
          }
          DROP(len);
          dst[out++] = (uint8_t)E_VALUE(e);
          continue;
        }
        if (kind == END) {
          DROP(len);
          s->mode = s->last ? MODE_DONE : MODE_HEADER;
          break;
        }
        if (kind != BASE) {
          status = COMPRESS_INFLATE_SYMBOL;
          goto fail;
        }
        unsigned total = len + E_EXTRA(e);
        if (total > nbits)
          goto more;
        size_t length =
            E_VALUE(e) + (size_t)((bits >> len) & ((1u << E_EXTRA(e)) - 1u));
        DROP(total);
        if (nbits < 28)
          REFILL();
        LOOKUP(e, s->dist, DIST_ROOT);
        len = E_LEN(e);
        total = len + E_EXTRA(e);
        if (total > nbits)
          goto more;
        if (E_KIND(e) != BASE) {
          status = COMPRESS_INFLATE_SYMBOL;
          goto fail;
        }
        size_t distance =
            E_VALUE(e) + (size_t)((bits >> len) & ((1u << E_EXTRA(e)) - 1u));
        if (distance > out - hist) {
          status = COMPRESS_INFLATE_DISTANCE;
          goto fail;
        }
        DROP(total);
        size_t n = length;
        if (n > out_end - out)
          n = out_end - out;
        copy_match(dst + out, distance, n, out_end - out);
        out += n;
        if (n < length) {
          s->copy_length = length - n;
          s->copy_distance = distance;
          status = COMPRESS_INFLATE_OUTPUT;
          goto leave;
        }
      }
      break;
    }
  }

more:
  status = io->final ? COMPRESS_INFLATE_TRUNCATED : COMPRESS_INFLATE_INPUT;
fail:
  pos = save_pos;
  bits = save_bits;
  nbits = save_nbits;
leave:
  /* Whole bytes in the bit buffer were read ahead: give them back. */
  pos -= nbits >> 3;
  nbits &= 7u;
  bits &= ((uint64_t)1 << nbits) - 1u;
  s->bits = bits;
  s->nbits = nbits;
  io->src_pos = pos;
  io->dst_pos = out;
  return status;
}
