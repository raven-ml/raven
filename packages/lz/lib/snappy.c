/* Snappy blocks: a varint preamble stating the decoded length, then a
   sequence of elements, each a literal or a copy from earlier output
   (format_description.txt in google/snappy).

   The decoder runs a fast loop while the input holds a whole element and
   64 more bytes and the output has 64 bytes of room: it moves literals and
   copies in 16-byte stores that may run past the element's end but never
   past the preamble's length. Near either end, and for the elements the
   fast loop declines (a literal longer than 60 bytes, an offset that is
   zero or before the start), one element is decoded with every bound
   checked, and the fast loop resumes.

   An error is at the first byte of what is wrong: the preamble's first
   byte for a preamble past 32 bits or above the destination, an element's
   tag for an element that is not the format.

   The encoder follows Google's snappy 1.2 with its multiplicative hash:
   64 KiB fragments, each with a fresh hash table of 4-byte prefixes, a
   scan that skips further the longer it finds no match, and snappy's
   choice of literal and copy encodings. Its tables come from [lz_alloc],
   so that the call's C stack stays small on any thread. */

#include <string.h>

#include "snappy.h"

/* Format constants (format_description.txt) */

/* The two low bits of a tag byte. */
enum { LITERAL = 0, COPY_1 = 1, COPY_2 = 2, COPY_4 = 3 };

/* A preamble is the varint of a 32-bit length: at most 5 bytes, the last
   holding bits 28 to 31. */
#define PREAMBLE_MAX 5
#define PREAMBLE_LAST_MAX 0x0f

/* A literal up to this length keeps its length minus one in the tag; a
   longer one keeps it in the 1 to 4 bytes after the tag. */
#define SHORT_LITERAL 60

/* Loads */

static inline uint32_t le32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap32(v);
#endif
  return v;
}

static inline uint64_t le64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

static inline void store32(uint8_t *p, uint32_t v) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap32(v);
#endif
  memcpy(p, &v, 4);
}

static inline void copy16(uint8_t *dst, const uint8_t *src) {
  uint8_t t[16];
  memcpy(t, src, 16);
  memcpy(dst, t, 16);
}

/* Decoding */

/* [tags[t]] describes the element that tag byte [t] starts: bits 0-7 its
   length, 0 for a literal whose length follows the tag; bits 8-10 the
   high bits of a 1-byte-offset copy's offset; bits 11-13 the bytes after
   the tag, before the literal's data or the next element: a copy's
   offset, or a long literal's length minus one in the 1 to 4 bytes that
   tags 60 to 63 name. */
#define TAG(t)                                                             \
  ((t) % 4 == LITERAL                                                      \
       ? ((t) / 4 < SHORT_LITERAL ? (t) / 4 + 1 : ((t) / 4 % 4 + 1) << 11) \
   : (t) % 4 == COPY_1 ? (4 + ((t) / 4 & 7)) | ((t) / 32) << 8 | 1 << 11  \
   : (t) % 4 == COPY_2 ? ((t) / 4 + 1) | 2 << 11                          \
                       : ((t) / 4 + 1) | 4 << 11)
#define TAG4(t) TAG(t), TAG(t + 1), TAG(t + 2), TAG(t + 3)
#define TAG16(t) TAG4(t), TAG4(t + 4), TAG4(t + 8), TAG4(t + 12)
#define TAG64(t) TAG16(t), TAG16(t + 16), TAG16(t + 32), TAG16(t + 48)

static const uint16_t tags[256] = {TAG64(0), TAG64(64), TAG64(128),
                                   TAG64(192)};

/* The bytes of a 4-byte load that [n] bytes after a tag hold. */
static const uint32_t trailer_mask[5] = {0, 0xff, 0xffff, 0xffffff,
                                         0xffffffff};

/* The fast loop's slack: a tag and 64 bytes after it, read as a 4-byte
   trailer or as a literal of up to 60 bytes in 16-byte loads; and 64
   bytes of output, written by a literal or copy in 16-byte stores. */
#define IN_SLACK 65
#define OUT_SLACK 64

/* Copies [len] (1 to 64) bytes from [src] to [dst] in 16-byte steps, which
   write up to 64 bytes. The source is at least 16 bytes before [dst] or
   apart from it. */
static inline void copy64(uint8_t *dst, const uint8_t *src, size_t len) {
  copy16(dst, src);
  if (len <= 16) return;
  copy16(dst + 16, src + 16);
  if (len <= 32) return;
  copy16(dst + 32, src + 32);
  copy16(dst + 48, src + 48);
}

/* [period[d]] is the smallest multiple of [d] that is at least 16. */
static const uint8_t period[16] = {0,  16, 16, 18, 16, 20, 18, 21,
                                   16, 18, 20, 22, 24, 26, 28, 30};

/* Writes the [len] (1 to 64) bytes of a copy from [off] (1 to 15) back,
   where the copy overlaps its own output: the first 16 bytes repeat the
   [off]-byte pattern, and each further 16 bytes copy from a whole number
   of patterns back, at least 16, so that each load reads finished
   output. Writes up to 64 bytes. */
static inline void expand(uint8_t *op, size_t off, size_t len) {
  const uint8_t *from = op - off;
  for (int k = 0; k < 16; k++) op[k] = from[k];
  if (len <= 16) return;
  size_t p = period[off];
  copy16(op + 16, op + 16 - p);
  if (len <= 32) return;
  copy16(op + 32, op + 32 - p);
  copy16(op + 48, op + 48 - p);
}

static enum lz_status malformed(struct lz_error *err, const uint8_t *src,
                                const uint8_t *at, const char *msg) {
  return lz_fail(err, LZ_MALFORMED, at - src, msg);
}

enum lz_status lz_snappy_decode(struct lz_io *io, struct lz_error *err) {
  const uint8_t *src = io->src, *ip = src, *ip_end = src + io->src_len;

  /* The preamble. */
  uint64_t n = 0;
  for (int k = 0;; k++) {
    if (ip == ip_end) return lz_fail(err, LZ_TRUNCATED, io->src_len, NULL);
    uint8_t b = *ip;
    if (k == PREAMBLE_MAX - 1 && b > PREAMBLE_LAST_MAX)
      return malformed(err, src, src, "invalid preamble");
    n |= (uint64_t)(b & 0x7f) << (7 * k);
    ip++;
    if (b < 0x80) break;
  }
  if (n > io->dst_cap) return lz_fail(err, LZ_TOO_LONG, 0, NULL);

  uint8_t *dst = io->dst, *op = dst, *op_end = dst + n;
  for (;;) {
    /* Elements with the slack around them, bounds unchecked. */
    while (ip_end - ip >= IN_SLACK && op_end - op >= OUT_SLACK) {
      uint32_t tag = ip[0], e = tags[tag];
      size_t len = e & 0xff;
      if (tag % 4 == LITERAL) {
        if (len == 0) break;
        copy64(op, ip + 1, len);
        ip += 1 + len;
        op += len;
        continue;
      }
      size_t extra = e >> 11;
      size_t off = (e & 0x700) + (le32(ip + 1) & trailer_mask[extra]);
      if (off - 1 >= (size_t)(op - dst)) break;
      ip += 1 + extra;
      if (off >= 16)
        copy64(op, op - off, len);
      else
        expand(op, off, len);
      op += len;
    }
    if (ip == ip_end) break;

    /* One element, every bound checked. Each element writes at least one
       byte, so none fits once the output holds the preamble's length. */
    const uint8_t *at = ip;
    uint32_t tag = *ip++, e = tags[tag];
    size_t len = e & 0xff, extra = e >> 11, room = op_end - op;
    if (room == 0 || len > room)
      return malformed(err, src, at, "output longer than the preamble");
    /* The bytes after the tag, low byte first: a long literal's length
       minus one, or a copy's offset, which starts from the tag's bits.
       Each byte only adds to the value the earlier ones give, so once a
       literal passes the room, or an offset the output, no later byte
       makes the element valid: it is malformed, at its tag, even when the
       input ends inside it. */
    bool is_copy = tag % 4 != LITERAL;
    size_t produced = op - dst;
    if (is_copy && (produced == 0 || (e & 0x700) > produced))
      return malformed(err, src, at, "copy offset too far back");
    uint32_t trailer = 0;
    for (size_t k = 0; k < extra; k++, ip++) {
      if (ip == ip_end) return lz_fail(err, LZ_TRUNCATED, io->src_len, NULL);
      trailer |= (uint32_t)*ip << (8 * k);
      if (len == 0 && (size_t)trailer + 1 > room)
        return malformed(err, src, at, "output longer than the preamble");
      if (is_copy && (e & 0x700) + (size_t)trailer > produced)
        return malformed(err, src, at, "copy offset too far back");
    }

    if (tag % 4 == LITERAL) {
      if (len == 0) len = (size_t)trailer + 1;
      if (len > (size_t)(ip_end - ip))
        return lz_fail(err, LZ_TRUNCATED, io->src_len, NULL);
      memcpy(op, ip, len);
      ip += len;
      op += len;
      continue;
    }

    size_t off = (e & 0x700) + trailer;
    if (off == 0) return malformed(err, src, at, "invalid copy offset");
    const uint8_t *from = op - off;
    for (size_t k = 0; k < len; k++) op[k] = from[k];
    op += len;
  }

  if (op != op_end) return lz_fail(err, LZ_TRUNCATED, io->src_len, NULL);
  io->consumed = io->src_len;
  io->produced = n;
  return LZ_DONE;
}

/* Encoding */

/* The encoder's constants are snappy 1.2's (snappy.h, snappy.cc). Input is
   compressed in fragments of 64 KiB, so that a hash table entry and every
   offset fit 16 bits. */
#define FRAGMENT (1 << 16)
#define TABLE_MIN_BITS 8
#define TABLE_MAX_BITS 15
#define HASH_MULTIPLIER 0x1e35a7bdu

/* The scan stops this many bytes before the fragment's end, so that its
   loads and the 16-byte literal copies stay inside the fragment. */
#define INPUT_MARGIN 15

/* After 32 positions without a match the scan steps 2 bytes, after 32
   more bytes 3, and so on: [skip >> SKIP_SHIFT] is the step. */
#define SKIP_START 32
#define SKIP_SHIFT 5

/* The largest output of a fragment of [n] bytes, with the 15 bytes past
   it that a 16-byte literal store may write. A literal of [k] bytes costs
   at most [k + 3] and a copy of [m >= 4] at most [m - 1]; every literal
   but the last is followed by a copy, and costs [k + 1] unless [k > 60],
   so each literal and copy cost at most 2 bytes more than they cover, and
   only when they cover 65 bytes or more. */
static size_t fragment_max(size_t n) { return n + n / 32 + 3 + 16; }

/* A position's table index: the top 15 bits of its 4 bytes times the
   multiplier, masked to the table's size. */
static inline uint32_t hash(uint32_t v, int bits) {
  return (v * HASH_MULTIPLIER) >> (32 - TABLE_MAX_BITS) & ((1u << bits) - 1);
}

/* The table's size for a fragment of [n] bytes: [n] rounded up to a power
   of two, within the bounds. */
static int table_bits(size_t n) {
  int bits = TABLE_MIN_BITS;
  while (bits < TABLE_MAX_BITS && ((size_t)1 << bits) < n) bits++;
  return bits;
}

/* The length of the common prefix of [a] and [b], at most [end - b]. Sets
   [*after] to the 8 bytes that follow the prefix in [b] when 8 remain. A
   prefix under 8 bytes takes [*after] from loads that do not wait for the
   comparison, which keeps them off the encoder's chain of dependent
   loads: match, its end, the next hash, the next candidate. */
static inline size_t match_length(const uint8_t *a, const uint8_t *b,
                                  const uint8_t *end, uint64_t *after) {
  const uint8_t *start = b;
  if (end - b >= 16) {
    uint64_t lo = le64(b), hi = le64(b + 4), x = le64(a) ^ lo;
    if (x != 0) {
      int k = __builtin_ctzll(x) >> 3;
      *after = k < 4 ? lo >> (8 * k) : hi >> (8 * (k - 4));
      return k;
    }
  }
  while (end - b >= 8) {
    uint64_t x = le64(a) ^ le64(b);
    if (x != 0) {
      b += __builtin_ctzll(x) >> 3;
      if (end - b >= 8) *after = le64(b);
      return (size_t)(b - start);
    }
    a += 8;
    b += 8;
  }
  while (b < end && *a == *b) {
    a++;
    b++;
  }
  if (end - b >= 8) *after = le64(b);
  return (size_t)(b - start);
}

/* Writes a literal of the [len] bytes at [lit]: its length minus one in
   the tag up to 60 bytes, else in as few bytes after the tag as hold it.
   With [wide], it moves the bytes in 16-byte steps, which may read and
   write up to 15 bytes past them. */
static inline uint8_t *literal(uint8_t *op, const uint8_t *lit, size_t len,
                               bool wide) {
  size_t m = len - 1;
  if (m < SHORT_LITERAL) {
    *op++ = (uint8_t)(m << 2 | LITERAL);
  } else {
    int k = 1;
    while (m >> (8 * k) != 0) k++;
    *op++ = (uint8_t)((SHORT_LITERAL - 1 + k) << 2 | LITERAL);
    for (int i = 0; i < k; i++) *op++ = (uint8_t)(m >> (8 * i));
  }
  if (!wide) {
    memcpy(op, lit, len);
    return op + len;
  }
  for (size_t i = 0; i < len; i += 16) copy16(op + i, lit + i);
  return op + len;
}

/* Writes one copy element of 4 to 64 bytes from [off] (below 64 KiB)
   back: the 2-byte form when its 3 length bits and 11 offset bits hold
   them, else the 3-byte form. The 4-byte store writes up to 2 bytes past
   the element. */
static inline uint8_t *copy_element(uint8_t *op, size_t off, size_t len) {
  if (len <= 11 && off < 2048) {
    store32(op, (uint32_t)(COPY_1 | (len - 4) << 2 | (off >> 8) << 5 |
                           (off & 0xff) << 8));
    return op + 2;
  }
  store32(op, (uint32_t)(COPY_2 | (len - 1) << 2 | off << 8));
  return op + 3;
}

/* Writes a copy of [len >= 4] bytes as elements of at most 64. snappy 1.2
   cuts 64 bytes while 68 or more remain and 60 when 65 to 67 do, so that
   the last element keeps at least 4. */
static inline uint8_t *copy(uint8_t *op, size_t off, size_t len) {
  while (len > 64) {
    size_t piece = len >= 68 ? 64 : 60;
    op = copy_element(op, off, piece);
    len -= piece;
  }
  return copy_element(op, off, len);
}

/* Returns the position [table] holds for [bytes], the 4 bytes at [p],
   and records [p] in its place. A slot never written holds the
   fragment's start, which the caller's comparison of bytes rejects or
   accepts as any other candidate. */
static inline const uint8_t *look(uint16_t *table, int bits,
                                  const uint8_t *in, const uint8_t *p,
                                  uint32_t bytes) {
  uint16_t *slot = &table[hash(bytes, bits)];
  const uint8_t *candidate = in + *slot;
  *slot = (uint16_t)(p - in);
  return candidate;
}

/* Probes positions from [*pp], whose 4 bytes are [bytes], until one
   matches its candidate, and returns the candidate with [*pp] at the
   position; or returns NULL when the next step would pass [limit]. The
   first 32 probes step one byte (the step is [skip >> SKIP_SHIFT], and
   [skip] runs from 32 to 63 meanwhile): with 32 bytes before [limit],
   they need no test of it. */
static inline const uint8_t *scan(uint16_t *table, int bits,
                                  const uint8_t *in, const uint8_t *limit,
                                  const uint8_t **pp, uint32_t bytes) {
  const uint8_t *p = *pp, *candidate;
  uint32_t skip = SKIP_START;
  if (limit - p >= SKIP_START) {
    for (int i = 0; i < SKIP_START; i++) {
      candidate = look(table, bits, in, p, bytes);
      if (le32(candidate) == bytes) {
        *pp = p;
        return candidate;
      }
      bytes = le32(++p);
    }
    skip = 2 * SKIP_START;
  }
  for (;;) {
    const uint8_t *next = p + (skip >> SKIP_SHIFT);
    skip += skip >> SKIP_SHIFT;
    if (next > limit) return NULL;
    candidate = look(table, bits, in, p, bytes);
    if (le32(candidate) == bytes) {
      *pp = p;
      return candidate;
    }
    p = next;
    bytes = le32(p);
  }
}

/* Compresses the [n] (1 to FRAGMENT) bytes at [in] to [op], which has
   room for [fragment_max n] bytes, with a zeroed table of [1 << bits]
   slots.

   The scan looks each position up in the table, which maps the hash of
   4 bytes to the last position looked at with that hash, and records the
   position there. When the candidate holds the same 4 bytes, the bytes
   since the last element become a literal, and the match, extended as
   far as the bytes agree, a copy. The position at the copy's end is
   looked up at once, so that consecutive copies need no literal between
   them, and the position before it is recorded. Without a match, the
   scan steps further the longer it has gone without one. Within
   [INPUT_MARGIN] bytes of the end it stops, and the rest is a literal. */
static uint8_t *fragment(const uint8_t *in, size_t n, uint8_t *op,
                         uint16_t *table, int bits) {
  const uint8_t *end = in + n, *lit = in;
  if (n < INPUT_MARGIN) return literal(op, in, n, false);

  /* [bytes] holds the 4 bytes at [p]. */
  const uint8_t *limit = end - INPUT_MARGIN, *p = in + 1;
  uint32_t bytes = le32(p);
  for (;;) {
    const uint8_t *candidate = scan(table, bits, in, limit, &p, bytes);
    if (!candidate) return literal(op, lit, end - lit, false);
    op = literal(op, lit, p - lit, true);

    uint64_t after = 0;
    for (;;) {
      size_t len = 4 + match_length(candidate + 4, p + 4, end, &after);
      op = copy(op, p - candidate, len);
      p += len;
      lit = p;
      if (p >= limit)
        return lit < end ? literal(op, lit, end - lit, false) : op;
      look(table, bits, in, p - 1, le32(p - 1));
      bytes = (uint32_t)after;
      candidate = look(table, bits, in, p, bytes);
      if (le32(candidate) != bytes) break;
    }
    p++;
    bytes = (uint32_t)(after >> 8);
  }
}

size_t lz_snappy_bound(size_t n) { return 32 + n + n / 6; }

enum lz_status lz_snappy_encode(struct lz_io *io) {
  const uint8_t *src = io->src;
  size_t n = io->src_len;
  uint8_t *op = io->dst, *op_end = io->dst + io->dst_cap;

  for (uint64_t v = n;; v >>= 7) {
    if (op == op_end) return LZ_NEED_OUTPUT;
    *op++ = (uint8_t)(v < 0x80 ? v : (v & 0x7f) | 0x80);
    if (v < 0x80) break;
  }

  /* One table serves every fragment, sized for the largest. A fragment
     goes straight to the destination when it has room for the fragment's
     largest output; otherwise through [scratch], and is copied if it
     fits. */
  size_t first = n < FRAGMENT ? n : FRAGMENT;
  uint16_t *table = NULL;
  uint8_t *scratch = NULL;
  enum lz_status status = LZ_DONE;
  if (n > 0 && !(table = lz_alloc(sizeof(uint16_t) << table_bits(first))))
    return LZ_ERROR;
  for (size_t at = 0; at < n; at += FRAGMENT) {
    size_t len = n - at < FRAGMENT ? n - at : FRAGMENT;
    int bits = table_bits(len);
    memset(table, 0, sizeof(uint16_t) << bits);
    if ((size_t)(op_end - op) >= fragment_max(len)) {
      op = fragment(src + at, len, op, table, bits);
      continue;
    }
    if (!scratch && !(scratch = lz_alloc(fragment_max(first)))) {
      status = LZ_ERROR;
      break;
    }
    size_t k = fragment(src + at, len, scratch, table, bits) - scratch;
    if (k > (size_t)(op_end - op)) {
      status = LZ_NEED_OUTPUT;
      break;
    }
    memcpy(op, scratch, k);
    op += k;
  }
  if (table) lz_free(table);
  if (scratch) lz_free(scratch);
  if (status != LZ_DONE) return status;

  io->consumed = n;
  io->produced = op - io->dst;
  return LZ_DONE;
}
