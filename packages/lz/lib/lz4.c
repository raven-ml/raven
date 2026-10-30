/* LZ4 blocks and frames, whole-buffer only.

   The formats are lz4's "LZ4 Block Format Description" and "LZ4 Frame
   Format Description" 1.6.4 (doc/ of https://github.com/lz4/lz4). Their
   constants are the specifications', named below.

   A block is a run of sequences, each a token, literals and a match:

     token      literal length   literals   offset   match length
     LLLL MMMM  (LLLL = 15:      ...        2 bytes  (MMMM = 15:
                 255* then <255)            LE       255* then <255)

   The last sequence holds literals only. A match copies [MMMM + 4] bytes
   from [offset] bytes back, and may overlap the bytes it writes.

   A frame is a header, blocks each prefixed by their size (the high bit
   marks a stored block), an end mark of four zero bytes, and an optional
   XXH32 of the content:

     magic  FLG BD [content size] [dict id] HC  (size block [xxh32])*  0  [xxh32]

   Blocks of a frame are linked (a match reaches up to 64 KiB back into
   earlier blocks of the same frame) or independent.

   An error is at the first byte of what is wrong. A sequence is two
   codes: its literals, which begin at the token, and its match, which
   begins at the offset field. A bad offset, and a match passing the
   output's limit, are at the offset field; literals passing it, a
   sequence the input or the block cuts, and a last sequence breaking the
   end-of-block rules, at the token. A frame's field is at its first byte:
   the block size, a checksum, the content size, the dictionary ID; a
   byte that begins no frame is at that byte. */

#include "lz4.h"

#include <string.h>

#include "checksum.h"

/* The block format's constants. */
#define MIN_MATCH 4      /* a match token's length 0 means 4 bytes */
#define LAST_LITERALS 5  /* a block ends with at least 5 literals */
#define MATCH_MARGIN 12  /* the last match starts 12 bytes before the end */

/* The frame format's constants. */
#define FRAME_MAGIC 0x184D2204u
#define SKIP_MAGIC 0x184D2A50u /* to 0x184D2A5F */
#define SKIP_MASK 0xFFFFFFF0u
#define LEGACY_MAGIC 0x184C2102u
#define FLG_VERSION 0x40u /* bits 7-6 are 01 */
#define FLG_INDEPENDENT 0x20u
#define FLG_BLOCK_CHECKSUM 0x10u
#define FLG_CONTENT_SIZE 0x08u
#define FLG_CONTENT_CHECKSUM 0x04u
#define FLG_RESERVED 0x02u
#define FLG_DICT_ID 0x01u
#define BD_RESERVED 0x8Fu
#define STORED 0x80000000u /* a block size's flag for a stored block */

/* Bytes */

static inline uint32_t le16(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static inline uint32_t le32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

static inline uint64_t le64(const uint8_t *p) {
  return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
}

static inline void put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static inline void put64(uint8_t *p, uint64_t v) {
  put32(p, (uint32_t)v);
  put32(p + 4, (uint32_t)(v >> 32));
}

static inline void copy8(uint8_t *d, const uint8_t *s) { memcpy(d, s, 8); }
static inline void copy16(uint8_t *d, const uint8_t *s) { memcpy(d, s, 16); }

/* Block decoding */

/* The fast loop decodes whole sequences while each buffer has this many
   bytes left, so that literals of up to 14 bytes are copied as 16 and a
   match as 8- or 16-byte stores past its end. A sequence that does not
   fit in the slack is left to the exact loop. */
#define IN_SLACK 32
#define OUT_SLACK 32

/* How a block's decoding ended: every sequence decoded; the input ended
   before the block did; a code asked for output past [oend]; or a code
   is not the format. */
enum outcome { DECODED, ENDS_EARLY, OVERFLOWS, MALFORMED };

/* A block to decode: [src, src + len) into [op, oend), matches reaching
   back to [lo] at the furthest. [cut] says the input ends inside the
   block: [src + len] is not its end, so reaching it is ENDS_EARLY and the
   end-of-block rules do not apply. On return [op] is the end of the
   output; on failure [at] is the offset in [src] of the code that failed,
   and [msg] names a malformation. */
struct block {
  const uint8_t *src;
  size_t len;
  bool cut;
  uint8_t *lo;
  uint8_t *op;
  uint8_t *oend;
  size_t at;
  const char *msg;
  struct told *told;
};

/* The call's work (lz.h), and the input taken and output written when it
   was last reported. */
struct told {
  struct lz_work *work;
  const uint8_t *in;
  uint8_t *out;
};

/* [tell t ip op] reports the work up to [ip] and [op]. */
static void tell(struct told *t, const uint8_t *ip, uint8_t *op) {
  lz_did(t->work, (size_t)(ip - t->in) + (size_t)(op - t->out));
  t->in = ip;
  t->out = op;
}

/* [tell_ahead t ip op n] reports the work up to [ip] and [op] before a
   copy whose work is [n] bytes, and releases first if the copy would
   reach the threshold (lz.h). */
static void tell_ahead(struct told *t, const uint8_t *ip, uint8_t *op,
                       size_t n) {
  tell(t, ip, op);
  lz_ahead(t->work, n);
}

/* For an offset under 8, the smallest multiple of it that is 8 or more:
   once 8 bytes of the match are written, copying 8 bytes from that far
   back continues the pattern. */
static const uint8_t period8[8] = {0, 8, 8, 9, 8, 10, 12, 14};

/* For an offset under 8, a 1 in every byte whose index is a multiple of
   it. */
static const uint64_t spread[8] = {
    0,
    0x0101010101010101ull,
    0x0001000100010001ull,
    0x0001000001000001ull,
    0x0000000100000001ull,
    0x0000010000000001ull,
    0x0001000000000001ull,
    0x0100000000000001ull,
};

/* Copies the match of [len] bytes at [off] back, writing up to 31 bytes
   past its end. Each store's bytes are read only once written: a load
   that overlaps a store still in flight stalls. */
static inline void wide_match(uint8_t *op, size_t off, size_t len) {
  const uint8_t *m = op - off;
  if (off >= 16) {
    copy16(op, m);
    for (size_t i = 16; i < len; i += 16) copy16(op + i, m + i);
  } else if (off >= 8) {
    for (size_t i = 0; i < len; i += 8) copy8(op + i, m + i);
  } else {
    /* The first 8 bytes: the pattern's [off] bytes, repeated by a
       multiplication whose factor has a 1 every [off] bytes. */
    uint64_t pattern = le64(m) & (((uint64_t)1 << (8 * off)) - 1);
    put64(op, pattern * spread[off]);
    size_t step = period8[off];
    for (size_t i = 8; i < len; i += 8) copy8(op + i, op + i - step);
  }
}

static enum outcome fail(struct block *b, enum outcome o, const uint8_t *at,
                         const char *msg) {
  b->at = (size_t)(at - b->src);
  b->msg = msg;
  return o;
}

/* Where decoding stands: the next sequence's token, the output's end so
   far, and the length of the last match, 0 before the first. */
struct cursor {
  const uint8_t *ip;
  uint8_t *op;
  uint64_t last_match;
};

/* Decodes whole sequences while both buffers have their slack, and stops
   before the first sequence that may not fit: the exact loop decodes it.
   It also stops once the output reaches [span], before [oend], so that
   its work is reported (lz.h); a sequence may run past [span], and
   reports before a copy of LZ_WORK_SPAN or more. Kept out
   of line, so that the exact loop's calls do not take its registers. */
__attribute__((noinline)) static enum outcome fast_sequences(struct block *b,
                                                             struct cursor *c,
                                                             uint8_t *span) {
  const uint8_t *ip = c->ip;
  const uint8_t *const iend = b->src + b->len;
  uint8_t *op = c->op;
  uint8_t *const oend = b->oend;
  uint8_t *const lo = b->lo;
  uint64_t last_match = c->last_match;
  unsigned s;

  while (iend - ip >= IN_SLACK && span - op >= OUT_SLACK) {
    const uint8_t *const seq = ip;
    uint8_t *const out = op;
    unsigned token = *ip++;
    uint64_t lit = token >> 4;
    if (lit == 15) {
      /* A length past the room leaves the rest of its bytes unread: the
         exact loop refuses it. */
      do {
        if (ip == iend) goto exact;
        s = *ip++;
        lit += s;
      } while (s == 255 && lit <= (uint64_t)(oend - op));
      if ((uint64_t)(iend - ip) < lit + IN_SLACK ||
          (uint64_t)(oend - op) < lit + OUT_SLACK)
        goto exact;
      if (lit >= LZ_WORK_SPAN) tell_ahead(b->told, ip, op, 2 * lit);
      for (uint64_t i = 0; i < lit; i += 16) copy16(op + i, ip + i);
    } else {
      copy16(op, ip);
    }
    op += lit;
    ip += lit;

    const uint8_t *const at_offset = ip;
    size_t off = le16(ip);
    ip += 2;
    if (off == 0) return fail(b, MALFORMED, at_offset, "invalid zero offset");
    if (off > (size_t)(op - lo))
      return fail(b, MALFORMED, at_offset, "invalid offset too far back");
    uint64_t len = token & 15;
    if (len == 15) {
      do {
        if (ip == iend) goto exact;
        s = *ip++;
        len += s;
      } while (s == 255 && len + MIN_MATCH <= (uint64_t)(oend - op));
    }
    len += MIN_MATCH;
    if ((uint64_t)(oend - op) < len + 16) goto exact;
    if (len >= LZ_WORK_SPAN) tell_ahead(b->told, ip, op, len);
    wide_match(op, off, len);
    op += len;
    last_match = len;
    continue;

  exact:
    /* Undo the sequence: the exact loop decodes it again. */
    ip = seq;
    op = out;
    break;
  }

  c->ip = ip;
  c->op = op;
  c->last_match = last_match;
  return DECODED;
}

/* The token of the last sequence in [p, end), whole sequences that the
   fast loop decoded. Only a block that ends after a match reads it: the
   fast loop does not keep it. */
static const uint8_t *last_token(const uint8_t *p, const uint8_t *end) {
  const uint8_t *token = p;
  while (p < end) {
    token = p;
    unsigned t = *p++;
    size_t lit = t >> 4;
    if (lit == 15) {
      while (p < end) {
        unsigned s = *p++;
        lit += s;
        if (s != 255) break;
      }
    }
    if (lit + 2 > (size_t)(end - p)) break;
    p += lit + 2;
    if ((t & 15) == 15) {
      while (p < end && *p++ == 255) {
      }
    }
  }
  return token;
}

static enum outcome block_decode(struct block *b) {
  /* The fast loop in spans of LZ_WORK_SPAN of output, each reported. A
     span it leaves short has a sequence for the exact loop. */
  struct cursor c = {b->src, b->op, 0};
  for (;;) {
    uint8_t *span = (size_t)(b->oend - c.op) > LZ_WORK_SPAN + OUT_SLACK
                        ? c.op + LZ_WORK_SPAN + OUT_SLACK
                        : b->oend;
    if (fast_sequences(b, &c, span) == MALFORMED) return MALFORMED;
    if (span == b->oend || span - c.op >= OUT_SLACK) break;
    tell(b->told, c.ip, c.op);
  }

  const uint8_t *ip = c.ip;
  const uint8_t *const iend = b->src + b->len;
  uint8_t *op = c.op;
  uint8_t *const oend = b->oend;
  uint8_t *const lo = b->lo;
  uint64_t last_match = c.last_match;
  unsigned s;

  /* The token of the last sequence this loop decoded. A block that ends
     after a match has no last literals, and that sequence breaks the
     end-of-block rules. */
  const uint8_t *prev = NULL;
  for (;;) {
    const uint8_t *const seq = ip;
    if (ip == iend)
      return fail(b, ENDS_EARLY, prev ? prev : last_token(b->src, ip),
                  "invalid end of block");
    unsigned token = *ip++;
    uint64_t lit = token >> 4;
    /* The room before the input: a length only grows with its bytes, so
       one that passes the room overflows however many of its bytes, or
       of its literals, follow. Its scan stops there, which bounds a
       hostile length's to the room over 255 bytes. */
    if (lit == 15) {
      do {
        if (ip == iend) return fail(b, ENDS_EARLY, seq, NULL);
        s = *ip++;
        lit += s;
      } while (s == 255 && lit <= (uint64_t)(oend - op));
    }
    if (lit > (uint64_t)(oend - op)) return fail(b, OVERFLOWS, seq, NULL);
    if (lit > (uint64_t)(iend - ip)) return fail(b, ENDS_EARLY, seq, NULL);
    if (lit >= LZ_WORK_SPAN) tell_ahead(b->told, ip, op, 2 * lit);
    memcpy(op, ip, lit);
    op += lit;
    ip += lit;

    if (ip == iend) {
      if (b->cut) return fail(b, ENDS_EARLY, seq, NULL);
      /* The last sequence. If a match came before, the block's end rules
         hold: at least 5 literals end it, and the last match starts at
         least 12 bytes before its end. */
      if (last_match != 0 &&
          (lit < LAST_LITERALS || lit + last_match < MATCH_MARGIN))
        return fail(b, MALFORMED, seq, "invalid end of block");
      b->op = op;
      return DECODED;
    }

    if (iend - ip < 2) return fail(b, ENDS_EARLY, seq, NULL);
    const uint8_t *const at_offset = ip;
    size_t off = le16(ip);
    ip += 2;
    /* The offset before the match length: a bad offset is final whatever
       follows. */
    if (off == 0) return fail(b, MALFORMED, at_offset, "invalid zero offset");
    if (off > (size_t)(op - lo))
      return fail(b, MALFORMED, at_offset, "invalid offset too far back");
    uint64_t len = token & 15;
    if (len == 15) {
      do {
        if (ip == iend) return fail(b, ENDS_EARLY, seq, NULL);
        s = *ip++;
        len += s;
      } while (s == 255 && len + MIN_MATCH <= (uint64_t)(oend - op));
    }
    len += MIN_MATCH;
    if (len > (uint64_t)(oend - op)) return fail(b, OVERFLOWS, at_offset, NULL);
    if (len >= LZ_WORK_SPAN) tell_ahead(b->told, ip, op, len);
    /* A match longer than its offset repeats its first [off] bytes. Each
       copy is of the bytes from [m] to [d], which end where it begins, so
       copies never overlap and each doubles the bytes written. */
    const uint8_t *m = op - off;
    uint8_t *const end = op + len;
    for (uint8_t *d = op; d < end;) {
      size_t n = (size_t)(d - m) < (size_t)(end - d) ? (size_t)(d - m)
                                                     : (size_t)(end - d);
      memcpy(d, m, n);
      d += n;
    }
    op = end;
    last_match = len;
    prev = seq;
  }
}

enum lz_status lz_lz4_block_decode(struct lz_io *io, struct lz_error *err) {
  struct told t = {io->work, io->src, io->dst};
  struct block b = {.src = io->src,
                    .len = io->src_len,
                    .lo = io->dst,
                    .op = io->dst,
                    .oend = io->dst + io->dst_cap,
                    .told = &t};
  switch (block_decode(&b)) {
  case DECODED:
    io->consumed = io->src_len;
    io->produced = (size_t)(b.op - io->dst);
    return LZ_DONE;
  case ENDS_EARLY:
    /* The input is the block: it ends inside it. */
    return lz_fail(err, LZ_TRUNCATED, (int64_t)io->src_len, NULL);
  case OVERFLOWS:
    return lz_fail(err, LZ_TOO_LONG, (int64_t)b.at, NULL);
  case MALFORMED:
    break;
  }
  return lz_fail(err, LZ_MALFORMED, (int64_t)b.at, b.msg);
}

/* Block encoding */

/* Matches are found through a hash table of the last position seen with
   each 5-byte prefix's hash: 2^13 entries of 16 bits. Hashing a fifth byte
   spreads the positions of short repeats, such as words, over more
   entries. An entry holds a
   position modulo 2^16, and the candidate is the position that many bytes
   back, under 64 KiB: a stale entry names some earlier position, which
   the comparison of 4 bytes then rejects or accepts as a true match. The
   table thus serves inputs of any size. */
#define HASH_LOG 13

/* The search steps one byte at a time, then a byte more for each 2^6
   positions without a match, so that incompressible input is crossed
   quickly. */
#define SKIP_LOG 6

/* An input shorter than this is literals only: a block's first match
   follows at least one literal and starts 12 bytes before its end. */
#define MIN_INPUT (MATCH_MARGIN + 1)

/* The 5 bytes at [p], shifted to the top of a 64-bit word and multiplied
   by lz4's prime for 5 bytes: the product's top bits mix all 5. */
static inline uint32_t hash(const uint8_t *p) {
  return (uint32_t)(((le64(p) << 24) * 889523592379ull) >> (64 - HASH_LOG));
}

/* The index of the first byte that differs in two 8-byte words read from
   memory, given their XOR, which is not 0. */
static inline size_t first_difference(uint64_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return (size_t)(__builtin_clzll(x) >> 3);
#else
  return (size_t)(__builtin_ctzll(x) >> 3);
#endif
}

/* The number of bytes from [p] and [q] that agree, up to [limit]. The
   first 8 are compared apart: most matches end within them. */
static inline size_t agree(const uint8_t *p, const uint8_t *q,
                           const uint8_t *limit) {
  const uint8_t *const start = p;
  uint64_t a, b;
  if (limit - p >= 8) {
    memcpy(&a, p, 8);
    memcpy(&b, q, 8);
    if (a != b) return first_difference(a ^ b);
    p += 8;
    q += 8;
  }
  while (limit - p >= 8) {
    memcpy(&a, p, 8);
    memcpy(&b, q, 8);
    if (a != b) return (size_t)(p - start) + first_difference(a ^ b);
    p += 8;
    q += 8;
  }
  while (p < limit && *p == *q) {
    p++;
    q++;
  }
  return (size_t)(p - start);
}

/* The bytes a length of [n] takes after its token's 15. */
static inline size_t length_bytes(size_t n) {
  return n < 15 ? 0 : (n - 15) / 255 + 1;
}

static inline uint8_t *put_length(uint8_t *op, size_t n) {
  if (n < 15) return op;
  size_t full = (n - 15) / 255;
  memset(op, 255, full);
  op[full] = (uint8_t)(n - 15 - 255 * full);
  return op + full + 1;
}

/* A sequence's token, offset and match length take fewer bytes than its
   match covers, so a block of n bytes is at most its literals with a token
   and length bytes for each run: n + n / 255 + 2. The bound is lz4's,
   which leaves a margin. */
size_t lz_lz4_block_bound(size_t n) { return n + n / 255 + 16; }

/* Writes [n] literals from [p], each followed in the input by at least 5
   more bytes: up to 16 are copied as 8 or 16, which stays inside the
   input, and inside the output when [room] says it holds 16 bytes. */
static inline uint8_t *put_literals(uint8_t *op, const uint8_t *p, size_t n,
                                    bool room) {
  if (room && n <= 8)
    copy8(op, p);
  else if (room && n <= 16)
    copy16(op, p);
  else
    memcpy(op, p, n);
  return op + n;
}

#define TABLE_BYTES (sizeof(uint16_t) << HASH_LOG)

/* Writes the block of [src, src + n) to [dst, dst + cap) and returns its
   length, or 0 if it does not fit. A block is never empty. [table] has
   TABLE_BYTES, and may be NULL when [n] is under MIN_INPUT. */
static size_t block_encode(const uint8_t *const src, size_t n,
                           uint8_t *const dst, size_t cap,
                           uint16_t *const table) {
  const uint8_t *const iend = src + n;
  const uint8_t *anchor = src;
  uint8_t *op = dst;
  uint8_t *const oend = dst + cap;

  if (n >= MIN_INPUT) {
    /* The last position a match may start at, and where it must end. */
    const uint8_t *const mflimit = iend - MATCH_MARGIN;
    const uint8_t *const mlimit = iend - LAST_LITERALS;
    memset(table, 0, TABLE_BYTES);

    const uint8_t *ip = src + 1;
    for (;;) {
      /* Find a match: [ip] at [d] bytes after an equal 4 bytes. */
      const uint8_t *next = ip;
      unsigned attempts = 1u << SKIP_LOG;
      uint32_t h = hash(next);
      size_t d;
      do {
        ip = next;
        size_t step = attempts++ >> SKIP_LOG;
        if (mflimit - ip < (ptrdiff_t)step) goto last;
        next = ip + step;
        uint16_t cur = (uint16_t)(ip - src);
        d = (uint16_t)(cur - table[h]);
        table[h] = cur;
        h = hash(next);
      } while (d == 0 || memcmp(ip - d, ip, 4) != 0);

      /* Extend it backwards over the pending literals. */
      const uint8_t *m = ip - d;
      while (ip > anchor && m > src && ip[-1] == m[-1]) {
        ip--;
        m--;
      }

      for (;;) {
        size_t len = MIN_MATCH + agree(ip + MIN_MATCH, m + MIN_MATCH, mlimit);
        size_t lit = (size_t)(ip - anchor);
        if ((size_t)(oend - op) <
            1 + length_bytes(lit) + lit + 2 + length_bytes(len - MIN_MATCH))
          return 0;

        uint8_t *token = op++;
        size_t lc = lit < 15 ? lit : 15;
        size_t mc = len - MIN_MATCH < 15 ? len - MIN_MATCH : 15;
        *token = (uint8_t)(lc << 4 | mc);
        op = put_length(op, lit);
        op = put_literals(op, anchor, lit, oend - op >= 16);
        op[0] = (uint8_t)d;
        op[1] = (uint8_t)(d >> 8);
        op = put_length(op + 2, len - MIN_MATCH);

        ip += len;
        anchor = ip;
        if (ip > mflimit) goto last;

        /* Index a position inside the match, then try the next one
           directly: matches often follow each other. */
        table[hash(ip - 2)] = (uint16_t)(ip - 2 - src);
        uint32_t hn = hash(ip);
        uint16_t cur = (uint16_t)(ip - src);
        d = (uint16_t)(cur - table[hn]);
        table[hn] = cur;
        if (d == 0 || memcmp(ip - d, ip, 4) != 0) break;
        m = ip - d;
      }
      ip++;
    }
  }

last:;
  size_t lit = (size_t)(iend - anchor);
  if ((size_t)(oend - op) < 1 + length_bytes(lit) + lit) return 0;
  *op++ = (uint8_t)((lit < 15 ? lit : 15) << 4);
  op = put_length(op, lit);
  memcpy(op, anchor, lit);
  return (size_t)(op + lit - dst);
}


/* The hash table for [n] input bytes, NULL when it needs none; [*ok] is
   false when it needs one and the allocation is refused. */
static uint16_t *table_for(size_t n, bool *ok) {
  if (n < MIN_INPUT) {
    *ok = true;
    return NULL;
  }
  uint16_t *t = lz_alloc(TABLE_BYTES);
  *ok = t != NULL;
  return t;
}

enum lz_status lz_lz4_block_encode(struct lz_io *io) {
  lz_did(io->work, io->src_len);
  bool ok;
  uint16_t *table = table_for(io->src_len, &ok);
  if (!ok) return LZ_ERROR;
  size_t n = block_encode(io->src, io->src_len, io->dst, io->dst_cap, table);
  lz_free(table);
  if (n == 0) return LZ_NEED_OUTPUT;
  io->consumed = io->src_len;
  io->produced = n;
  return LZ_DONE;
}


/* Frame decoding */

static inline size_t max_block(unsigned id) { return (size_t)1 << (8 + 2 * id); }

/* Whether the [n] bytes at [p], at most 4, begin the magic number of a
   frame, a skippable frame or a legacy frame: a cut magic number is
   truncated, any other byte begins no frame. */
static bool begins_magic(const uint8_t *p, size_t n) {
  static const uint32_t magic[3] = {FRAME_MAGIC, SKIP_MAGIC, LEGACY_MAGIC};
  static const uint32_t mask[3] = {0xFFFFFFFFu, SKIP_MASK, 0xFFFFFFFFu};
  for (int j = 0; j < 3; j++) {
    bool same = true;
    for (size_t i = 0; i < n; i++)
      same &= ((p[i] ^ (magic[j] >> (8 * i))) & (mask[j] >> (8 * i)) & 0xFF) == 0;
    if (same) return true;
  }
  return false;
}

/* Decodes the frame whose descriptor starts at [*pos] (after its magic
   number) into [dst] from [*out], advancing both past it. */
static enum lz_status frame_decode(const uint8_t *src, size_t n, size_t *pos,
                                   uint8_t *dst, size_t cap, size_t *out,
                                   struct told *t, struct lz_error *err) {
  size_t q = *pos;
  if (n - q < 1) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
  unsigned flg = src[q];
  if ((flg & 0xC0u) != FLG_VERSION)
    return lz_fail(err, LZ_MALFORMED, (int64_t)q, "invalid version");
  if (flg & FLG_RESERVED)
    return lz_fail(err, LZ_MALFORMED, (int64_t)q, "reserved bit set");
  if (n - q < 2) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
  unsigned bd = src[q + 1];
  if (bd & BD_RESERVED)
    return lz_fail(err, LZ_MALFORMED, (int64_t)q + 1, "reserved bit set");
  unsigned id = bd >> 4;
  if (id < 4)
    return lz_fail(err, LZ_MALFORMED, (int64_t)q + 1,
                   "invalid block maximum size");
  size_t sized = flg & FLG_CONTENT_SIZE ? 8 : 0;
  size_t header = 3 + sized + (flg & FLG_DICT_ID ? 4 : 0);
  if (n - q < header) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
  if (((lz_xxh32(0, src + q, header - 1) >> 8) & 0xFF) != src[q + header - 1])
    return lz_fail(err, LZ_MALFORMED, (int64_t)(q + header - 1),
                   "incorrect header checksum");
  if (flg & FLG_DICT_ID)
    return lz_fail(err, LZ_UNSUPPORTED, (int64_t)(q + 2 + sized), "dictionary");
  const int64_t size_at = (int64_t)q + 2;
  uint64_t size = sized ? le64(src + q + 2) : 0;
  q += header;

  size_t max = max_block(id);
  bool block_sums = flg & FLG_BLOCK_CHECKSUM;
  bool content_sum = flg & FLG_CONTENT_CHECKSUM;
  struct lz_xxh32 h;
  if (content_sum) lz_xxh32_init(&h, 0);
  uint8_t *const first = dst + *out;
  uint8_t *op = first;

  for (;;) {
    if (n - q < 4) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
    uint32_t word = le32(src + q);
    size_t at_size = q;
    q += 4;
    if (word == 0) break;
    tell(t, src + q, op);
    size_t len = word & ~STORED;
    if (len > max)
      return lz_fail(err, LZ_MALFORMED, (int64_t)at_size, "invalid block size");

    /* The output this block may produce, and what passing it means: the
       destination's end, too long at the code that asks; the frame's
       maximum block size, malformed there; or the content size the
       header states, malformed at that field. A tie is malformed. */
    size_t limit = (size_t)(dst + cap - op);
    int why = LZ_TOO_LONG;
    const char *msg = NULL;
    if (max <= limit) {
      limit = max;
      why = LZ_MALFORMED;
      msg = "block too large";
    }
    bool content_limit = false;
    if (sized && size - (uint64_t)(op - first) <= limit) {
      limit = (size_t)(size - (uint64_t)(op - first));
      why = LZ_MALFORMED;
      msg = "incorrect content size";
      content_limit = true;
    }

    /* A block the input cuts is decoded as far as the input goes: an
       error before the cut is final, and the cut is truncated. */
    bool cut = n - q < len;
    size_t avail = cut ? n - q : len;
    size_t produced;
    if (word & STORED) {
      if (len > limit)
        return lz_fail(err, why, content_limit ? size_at : (int64_t)at_size,
                       msg);
      if (cut) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
      if (len >= LZ_WORK_SPAN) tell_ahead(t, src + q, op, 2 * len);
      memcpy(op, src + q, len);
      produced = len;
    } else {
      struct block b = {.src = src + q,
                        .len = avail,
                        .cut = cut,
                        .lo = flg & FLG_INDEPENDENT ? op : first,
                        .op = op,
                        .oend = op + limit,
                        .told = t};
      switch (block_decode(&b)) {
      case DECODED:
        break;
      case ENDS_EARLY:
        if (cut) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
        return lz_fail(err, LZ_MALFORMED, (int64_t)(q + b.at),
                       b.msg ? b.msg : "block ends inside a sequence");
      case OVERFLOWS:
        return lz_fail(err, why,
                       content_limit ? size_at : (int64_t)(q + b.at), msg);
      case MALFORMED:
        return lz_fail(err, LZ_MALFORMED, (int64_t)(q + b.at), b.msg);
      }
      produced = (size_t)(b.op - op);
    }
    q += len;
    if (block_sums) {
      if (n - q < 4) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
      if (lz_xxh32(0, src + q - len, len) != le32(src + q))
        return lz_fail(err, LZ_MALFORMED, (int64_t)q, "incorrect block checksum");
      q += 4;
    }
    if (content_sum) lz_xxh32_update(&h, op, produced);
    op += produced;
  }

  if (sized && (uint64_t)(op - first) != size)
    return lz_fail(err, LZ_MALFORMED, size_at, "incorrect content size");
  if (content_sum) {
    if (n - q < 4) return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
    if (lz_xxh32_digest(&h) != le32(src + q))
      return lz_fail(err, LZ_MALFORMED, (int64_t)q, "incorrect content checksum");
    q += 4;
  }
  *pos = q;
  *out = (size_t)(op - dst);
  return LZ_DONE;
}

/* Frames are read to the input's end: wherever bytes remain after a
   frame, the next begins, and a byte that begins no frame is malformed at
   its offset. */
enum lz_status lz_lz4_frame_decode(struct lz_io *io, struct lz_error *err) {
  const uint8_t *const src = io->src;
  size_t n = io->src_len, pos = 0, out = 0;
  struct told t = {io->work, src, io->dst};
  do {
    /* An empty or skippable frame makes no report of its own. */
    tell(&t, src + pos, io->dst + out);
    const uint8_t *p = src + pos;
    size_t rest = n - pos;
    if (rest < 4) {
      if (begins_magic(p, rest))
        return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
      return lz_fail(err, LZ_MALFORMED, (int64_t)pos, "unknown frame magic");
    }
    uint32_t magic = le32(p);
    if ((magic & SKIP_MASK) == SKIP_MAGIC) {
      if (rest < 8 || le32(p + 4) > rest - 8)
        return lz_fail(err, LZ_TRUNCATED, (int64_t)n, NULL);
      pos += 8 + (size_t)le32(p + 4);
      continue;
    }
    if (magic == LEGACY_MAGIC)
      return lz_fail(err, LZ_UNSUPPORTED, (int64_t)pos, "legacy frame");
    if (magic != FRAME_MAGIC)
      return lz_fail(err, LZ_MALFORMED, (int64_t)pos, "unknown frame magic");
    pos += 4;
    enum lz_status s =
        frame_decode(src, n, &pos, io->dst, io->dst_cap, &out, &t, err);
    if (s != LZ_DONE) return s;
  } while (pos < n);
  io->consumed = n;
  io->produced = out;
  return LZ_DONE;
}

/* Frame encoding */

/* The frame encoder writes blocks of up to 4 MiB, the format's largest,
   each independent, with the content size and the content's XXH32, and
   declares the smallest maximum block size that holds the input. */
#define BLOCK_ID_MAX 7
#define HEADER_SIZE 15 /* magic, FLG, BD, content size, HC */
#define FOOTER_SIZE 8  /* end mark, content checksum */

size_t lz_lz4_frame_bound(size_t n) {
  size_t block = max_block(BLOCK_ID_MAX);
  return HEADER_SIZE + n + 4 * ((n + block - 1) / block) + FOOTER_SIZE;
}

static enum lz_status frame_encode(struct lz_io *io, uint16_t *table) {
  const uint8_t *const src = io->src;
  size_t n = io->src_len;
  uint8_t *const dst = io->dst;
  uint8_t *const oend = dst + io->dst_cap;
  if (io->dst_cap < HEADER_SIZE) return LZ_NEED_OUTPUT;

  unsigned id = 4;
  while (id < BLOCK_ID_MAX && n > max_block(id)) id++;
  size_t block = max_block(id);
  put32(dst, FRAME_MAGIC);
  dst[4] = FLG_VERSION | FLG_INDEPENDENT | FLG_CONTENT_SIZE |
           FLG_CONTENT_CHECKSUM;
  dst[5] = (uint8_t)(id << 4);
  put64(dst + 6, n);
  dst[14] = (uint8_t)(lz_xxh32(0, dst + 4, 10) >> 8);
  uint8_t *op = dst + HEADER_SIZE;

  struct lz_xxh32 h;
  lz_xxh32_init(&h, 0);
  for (size_t p = 0; p < n;) {
    size_t len = n - p < block ? n - p : block;
    size_t room = (size_t)(oend - op);
    if (room < 4) return LZ_NEED_OUTPUT;
    lz_xxh32_update(&h, src + p, len);
    /* A block is stored when compressing it saves nothing. */
    size_t cap = room - 4 < len - 1 ? room - 4 : len - 1;
    size_t c = block_encode(src + p, len, op + 4, cap, table);
    if (c != 0) {
      put32(op, (uint32_t)c);
    } else {
      if (room - 4 < len) return LZ_NEED_OUTPUT;
      put32(op, (uint32_t)len | STORED);
      memcpy(op + 4, src + p, len);
      c = len;
    }
    op += 4 + c;
    p += len;
  }

  if (oend - op < FOOTER_SIZE) return LZ_NEED_OUTPUT;
  put32(op, 0);
  put32(op + 4, lz_xxh32_digest(&h));
  io->consumed = n;
  io->produced = (size_t)(op + FOOTER_SIZE - dst);
  return LZ_DONE;
}

enum lz_status lz_lz4_frame_encode(struct lz_io *io) {
  lz_did(io->work, io->src_len);
  bool ok;
  uint16_t *table = table_for(io->src_len, &ok);
  if (!ok) return LZ_ERROR;
  enum lz_status s = frame_encode(io, table);
  lz_free(table);
  return s;
}
