/* The literals section (RFC 8878 section 3.1.1.3.1) and its Huffman
   coding (section 4.2).

   Huffman codes are at most 11 bits, so every table has 2^11 cells, read
   with the next 11 bits of the stream: a code of n bits fills the
   2^(11 - n) cells it prefixes. A single-symbol cell is a 16-bit
   [symbol << 8 | length]. A double-symbol cell, built from the single
   ones, also decodes the symbol that follows when both codes fit in 11
   bits: its 32 bits are [s1 | s2 << 8 | length << 16 | count << 24],
   which writes both bytes at once and advances by [count]. Double
   symbols cost a larger table and pay off on large, compressible
   sections.

   In a whole-buffer decode the literals go to the destination past the
   block's output; near its end the output may run into them, and
   zstd_dec_seq.c then keeps its wide copies off those still unread. */

#include "zstd_dec.h"

#define HUF_LOG 11
#define HUF_CELLS (1 << HUF_LOG)

enum { HUF_NONE, HUF_X1, HUF_X2 };

static const char header_past[] = "Literals_Section_Header past the block";
static const char regen_big[] = "Regenerated_Size above Block_Maximum_Size";
static const char regen_four[] = "Regenerated_Size short of four streams";
static const char raw_past[] = "Raw_Literals_Block past the block";
static const char rle_past[] = "RLE_Literals_Block past the block";
static const char csize_past[] = "Compressed_Size past the block";
static const char csize_small[] = "Compressed_Size short of its streams";
static const char no_tree[] = "Treeless_Literals_Block with no Huffman tree";
static const char tree_past[] = "Huffman_Tree_Description past Compressed_Size";
static const char big_weight[] = "Huffman Weight above 11";
static const char no_code[] = "Huffman weights complete to no prefix code";
static const char weights_zero[] = "Huffman weights' bitstream ends in a 0 byte";
static const char weights_inexact[] = "Huffman weights' bitstream not read exactly";
static const char jump_past[] = "Jump_Table past Total_Streams_Size";
static const char stream_zero[] = "Huffman-coded stream ends in a 0 byte";
static const char stream_inexact[] = "Huffman-coded stream not read exactly";

/* [refused f at msg] sets [f] to a malformed input at [at] and is 0, the
   size or count of nothing read. */
static size_t refused(struct lz_zstd_fail *f, int64_t at, const char *msg) {
  lz_zstd_fail(f, LZ_MALFORMED, at, msg);
  return 0;
}

/* Weights */

/* Section 4.2.1.2: the weights of up to 255 symbols, FSE-coded in the [n]
   bytes at [s] by two states that take turns over one table of accuracy
   log at most 6. The stream ends when a state's update reads past its
   start: the other state's symbol is the last weight. It is the number
   of weights, or -1 with [f] set at an offset from [s]. */
static int fse_weights(uint8_t w[256], const uint8_t *s, size_t n,
                       struct lz_zstd_fail *f) {
  int16_t norm[256];
  unsigned max = 11, log;
  size_t used = lz_fse_ncount(norm, &max, &log, 6, s, n, f);
  if (used == 0) return -1;
  struct lz_zstd_cell t[64];
  lz_fse_build(t, norm, max, log, NULL, NULL);

  struct lz_bits b;
  if (!lz_bits_init(&b, s + used, n - used))
    return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)n - 1, weights_zero);
  size_t st[2];
  st[0] = lz_bits_read(&b, log);
  st[1] = lz_bits_read(&b, log);
  lz_bits_reload(&b);
  if (b.used > 64)
    return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)used, weights_inexact);

  int count = 0;
  for (int k = 0;; k ^= 1) {
    if (count == 255) break;
    const struct lz_zstd_cell *c = &t[st[k]];
    w[count++] = (uint8_t)c->base;
    st[k] = c->next + lz_bits_read(&b, c->bits);
    lz_bits_reload(&b);
    if (b.used > 64) {
      if (count == 255) break;
      w[count++] = (uint8_t)t[st[k ^ 1]].base;
      return count;
    }
  }
  return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)used, weights_inexact);
}

/* Section 4.2.1: the tree description at [s], FSE-coded or as 4-bit
   fields, into the single-symbol table. The last symbol's weight
   completes the weights to a power of two, which gives the longest code.
   It is the description's size, or 0 with [f] set at an offset from
   [s]: a weight above 11 at its byte, a description that builds no
   prefix code at its first. */
static size_t huf_table(uint16_t *t, const uint8_t *s, size_t n,
                        struct lz_zstd_fail *f) {
  uint8_t w[256];
  int count;
  size_t size;
  unsigned head = s[0];
  if (head < 128) {
    size = 1 + head;
    if (size > n) return refused(f, 0, tree_past);
    count = fse_weights(w, s + 1, head, f);
    if (count < 0) {
      f->at += 1;
      return 0;
    }
  } else {
    count = (int)head - 127;
    size = 1 + ((size_t)count + 1) / 2;
    if (size > n) return refused(f, 0, tree_past);
    for (int i = 0; i < count; i++) {
      w[i] = (uint8_t)(i & 1 ? s[1 + i / 2] & 15 : s[1 + i / 2] >> 4);
      if (w[i] > HUF_LOG) return refused(f, 1 + i / 2, big_weight);
    }
  }

  uint32_t rank[HUF_LOG + 2] = {0};
  uint32_t total = 0;
  for (int i = 0; i < count; i++) {
    rank[w[i]]++;
    total += (1u << w[i]) >> 1;
  }
  if (total == 0) return refused(f, 0, no_code);
  unsigned bits = lz_high32(total) + 1;
  uint32_t rest = (1u << bits) - total;
  if (bits > HUF_LOG || (rest & (rest - 1))) return refused(f, 0, no_code);
  w[count] = (uint8_t)(lz_high32(rest) + 1);
  rank[w[count]]++;
  count++;

  /* Codes are given in increasing weight, then symbol, from the all-zero
     code: the cells of weight w start after those of every lower weight. */
  unsigned shift = HUF_LOG - bits;
  uint32_t pos[HUF_LOG + 2];
  uint32_t at = 0;
  for (unsigned k = 1; k <= bits; k++) {
    pos[k] = at;
    at += rank[k] << (k - 1 + shift);
  }
  for (int sym = 0; sym < count; sym++) {
    unsigned k = w[sym];
    if (k == 0) continue;
    uint32_t len = 1u << (k - 1 + shift);
    uint16_t cell = (uint16_t)(sym << 8 | (bits + 1 - k));
    uint16_t *p = t + pos[k];
    pos[k] += len;
    if (len >= 4) {
      uint64_t four = cell * 0x0001000100010001ull;
      for (uint32_t i = 0; i < len; i += 4) memcpy(p + i, &four, 8);
    } else {
      for (uint32_t i = 0; i < len; i++) p[i] = cell;
    }
  }
  return size;
}

/* The double-symbol table. After a code of n1 bits at cell i, the cell's
   last 11 - n1 bits begin the next code, which fits when its length n2 is
   at most 11 - n1. The run of a symbol of length n1 starts at a multiple
   of 2^(11 - n1), so its j-th cell continues with the code at j << n1
   whatever the symbol: one row per length serves every symbol of that
   length. */
static void huf_double(uint32_t *t2, const uint16_t *t1) {
  uint32_t rows[HUF_CELLS];
  bool built[HUF_LOG + 1] = {false};
  for (uint32_t i = 0; i < HUF_CELLS;) {
    uint32_t c1 = t1[i];
    uint32_t n1 = c1 & 0xFF;
    uint32_t len = HUF_CELLS >> n1;
    uint32_t *row = rows + len;
    if (!built[n1]) {
      built[n1] = true;
      for (uint32_t j = 0; j < len; j++) {
        uint32_t c2 = t1[j << n1];
        uint32_t n = n1 + (c2 & 0xFF);
        row[j] = n <= HUF_LOG ? (c2 >> 8) << 8 | n << 16 | 2u << 24
                              : n1 << 16 | 1u << 24;
      }
    }
    uint32_t s1 = c1 >> 8;
    for (uint32_t j = 0; j < len; j++) t2[i + j] = s1 | row[j];
    i += len;
  }
}

/* Streams with the checked reader */

/* [one_x1 t b op n] decodes [n] symbols with the single-symbol table. */
LZ_INLINE void one_x1(const uint16_t *t, struct lz_bits *b, uint8_t *op,
                      size_t n) {
  while (n > 0) {
    lz_bits_reload(b);
    size_t k = n < 4 ? n : 4;
    for (size_t i = 0; i < k; i++) {
      uint32_t c = t[(b->c << (b->used & 63)) >> (64 - HUF_LOG)];
      b->used += c & 0xFF;
      *op++ = (uint8_t)(c >> 8);
    }
    n -= k;
  }
}

/* [one_x2 t2 t1 b op n] decodes [n] symbols two at a time where they fit,
   the last one alone. */
LZ_INLINE void one_x2(const uint32_t *t2, const uint16_t *t1,
                      struct lz_bits *b, uint8_t *op, size_t n) {
  uint8_t *end = op + n;
  while (end - op >= 2) {
    lz_bits_reload(b);
    for (int i = 0; i < 4 && end - op >= 2; i++) {
      uint32_t c = t2[(b->c << (b->used & 63)) >> (64 - HUF_LOG)];
      memcpy(op, &c, 2);
      b->used += (c >> 16) & 0xFF;
      op += c >> 24;
    }
  }
  if (op < end) one_x1(t1, b, op, 1);
}

/* Lanes

   Four streams decode as four lanes. A lane keeps its unread bits at the
   top of a word and, under them, one set bit, its marker: shifting a
   code's bits out moves the marker up, so the zeros below the marker
   count the bits taken since the word was loaded. A refill steps the
   lane's pointer back over the whole bytes among them and loads the word
   there; at least 56 unread bits then sit above the marker (55 on a
   lane's first word, whose end mark may fill its last byte), room for
   five codes of 11 bits.

   A burst takes five codes from each lane, then refills it. Bursts run
   unchecked while every lane has the room for one: 7 bytes below its
   pointer, the most a refill steps back, and 5 bytes of output, 10 for a
   double-symbol table, whose cells write two bytes. The lanes then
   finish with the checked reader, which takes the stream's exact end. */

struct lane {
  uint64_t w;
  const uint8_t *p;
};

/* [lane_start l s n] starts [l] on the [n] bytes at [s], at least 8, whose
   last is not 0: the end mark and the zeros above it are taken. */
LZ_INLINE void lane_start(struct lane *l, const uint8_t *s, size_t n) {
  unsigned mark = 8 - lz_high32(s[n - 1]);
  l->p = s + n - 8;
  l->w = (lz_le64(l->p) | 1) << mark;
}

LZ_INLINE void lane_refill(struct lane *l) {
  unsigned taken = (unsigned)__builtin_ctzll(l->w);
  l->p -= taken >> 3;
  l->w = (lz_le64(l->p) | 1) << (taken & 7);
}

/* [lane_bits l s] is the checked reader where [l] stands on the stream at
   [s]. */
LZ_INLINE struct lz_bits lane_bits(const struct lane *l, const uint8_t *s) {
  struct lz_bits b;
  b.start = s;
  b.p = l->p;
  b.c = lz_le64(l->p);
  b.used = (unsigned)__builtin_ctzll(l->w);
  return b;
}

LZ_INLINE void burst_x1(const uint16_t *t, struct lane *l, uint8_t *o) {
  uint64_t w = l->w;
  for (int j = 0; j < 5; j++) {
    unsigned cell = t[w >> (64 - HUF_LOG)];
    o[j] = (uint8_t)(cell >> 8);
    w <<= cell & 63;
  }
  l->w = w;
  lane_refill(l);
}

LZ_INLINE uint8_t *burst_x2(const uint32_t *t, struct lane *l, uint8_t *o) {
  uint64_t w = l->w;
  for (int j = 0; j < 5; j++) {
    uint32_t cell = t[w >> (64 - HUF_LOG)];
    memcpy(o, &cell, 2);
    w <<= (cell >> 16) & 63;
    o += cell >> 24;
  }
  l->w = w;
  lane_refill(l);
  return o;
}

/* [bursts l o s e out] is the number of bursts every lane has room for,
   lane [k] standing on the stream at [s[k]] and writing [out] bytes a
   burst at most to [o[k]], below [e[k]]. */
LZ_INLINE size_t bursts(const struct lane l[4], uint8_t *const o[4],
                        const uint8_t *const s[4], uint8_t *const e[4],
                        size_t out) {
  size_t n = SIZE_MAX;
  for (int k = 0; k < 4; k++) {
    size_t in = (size_t)(l[k].p - s[k]) / 7;
    size_t room = (size_t)(e[k] - o[k]) / out;
    if (in < n) n = in;
    if (room < n) n = room;
  }
  return n;
}

/* Section 3.1.1.3.1.6: four streams, of sizes from the jump table at [s],
   each regenerating a quarter of the [n] literals, rounded up, but the
   last. [x2] selects the double-symbol table. Every stream must end where
   its bits do. It is false with [f] set at an offset from [s]. */
LZ_INLINE bool four_body(const struct lz_zstd_dec *st, bool x2, uint8_t *out,
                         size_t n, const uint8_t *s, size_t len,
                         struct lz_zstd_fail *f) {
  size_t size[4];
  size[0] = lz_le16(s);
  size[1] = lz_le16(s + 2);
  size[2] = lz_le16(s + 4);
  size_t three = size[0] + size[1] + size[2];
  if (three + 6 >= len) return refused(f, 0, jump_past);
  size[3] = len - 6 - three;

  const uint8_t *first[4];
  uint8_t *o[4], *end[4];
  size_t seg = (n + 3) / 4;
  first[0] = s + 6;
  for (int k = 1; k < 4; k++) first[k] = first[k - 1] + size[k - 1];
  for (int k = 0; k < 4; k++) {
    o[k] = out + (size_t)k * seg;
    end[k] = k == 3 ? out + n : o[k] + seg;
  }
  struct lz_bits b[4];
  for (int k = 0; k < 4; k++)
    if (!lz_bits_init(&b[k], first[k], size[k]))
      return refused(f, first[k] + size[k] - 1 - s, stream_zero);

  if (size[0] >= 8 && size[1] >= 8 && size[2] >= 8 && size[3] >= 8) {
    struct lane l[4];
    for (int k = 0; k < 4; k++) lane_start(&l[k], first[k], size[k]);
    if (x2) {
      for (size_t m; (m = bursts(l, o, first, end, 10)) > 0;)
        for (; m > 0; m--)
          for (int k = 0; k < 4; k++) o[k] = burst_x2(st->huf2, &l[k], o[k]);
    } else {
      for (size_t m; (m = bursts(l, o, first, end, 5)) > 0;)
        for (; m > 0; m--)
          for (int k = 0; k < 4; k++) {
            burst_x1(st->huf1, &l[k], o[k]);
            o[k] += 5;
          }
    }
    for (int k = 0; k < 4; k++) b[k] = lane_bits(&l[k], first[k]);
  }

  for (int k = 0; k < 4; k++) {
    size_t left = (size_t)(end[k] - o[k]);
    if (x2)
      one_x2(st->huf2, st->huf1, &b[k], o[k], left);
    else
      one_x1(st->huf1, &b[k], o[k], left);
    if (!lz_bits_done(&b[k])) return refused(f, first[k] - s, stream_inexact);
  }
  return true;
}

static bool four_base(const struct lz_zstd_dec *st, bool x2, uint8_t *out,
                      size_t n, const uint8_t *s, size_t len,
                      struct lz_zstd_fail *f) {
  return four_body(st, x2, out, n, s, len, f);
}

#if LZ_ZSTD_BMI2
LZ_TARGET_BMI2 static bool four_bmi2(const struct lz_zstd_dec *st, bool x2,
                                     uint8_t *out, size_t n, const uint8_t *s,
                                     size_t len, struct lz_zstd_fail *f) {
  return four_body(st, x2, out, n, s, len, f);
}
#endif

static bool four_streams(const struct lz_zstd_dec *st, bool x2, uint8_t *out,
                         size_t n, const uint8_t *s, size_t len,
                         struct lz_zstd_fail *f) {
#if LZ_ZSTD_BMI2
  if (st->path == LZ_ZSTD_WITH_BMI2)
    return four_bmi2(st, x2, out, n, s, len, f);
#endif
  return four_base(st, x2, out, n, s, len, f);
}

/* Section 3.1.1.3.1.6: one stream; as four_streams. */
static bool one_stream(const struct lz_zstd_dec *st, uint8_t *out, size_t n,
                       const uint8_t *s, size_t len, struct lz_zstd_fail *f) {
  struct lz_bits b;
  if (!lz_bits_init(&b, s, len))
    return refused(f, (int64_t)len - 1, stream_zero);
  one_x1(st->huf1, &b, out, n);
  if (!lz_bits_done(&b)) return refused(f, 0, stream_inexact);
  return true;
}

/* Double symbols pay for their table on sections of a few KiB that code
   their literals in well under 8 bits each. */
static bool want_x2(size_t n, size_t len) {
  return n >= 4096 && len * 8 < n * 7;
}

/* [place st o size l] puts [l] where [size] decoded literals go: a
   stream's buffer, or the destination at the first byte the block's wide
   copies cannot write, or as near it as they fit before the end. There
   the next blocks' output soon lands on the lines the literals brought
   into the cache. It is false when the destination cannot hold them, nor
   then the block's output. */
static bool place(const struct lz_zstd_dec *st, const struct lz_zstd_out *o,
                  size_t size, struct lz_zstd_lits *l) {
  l->len = size;
  l->dest = !st->stream;
  l->tight = false;
  if (st->stream) {
    l->p = st->lits;
    return true;
  }
  size_t room = (size_t)(o->end - o->op);
  if (size > room) return false;
  size_t at = lz_zstd_reach(o) + LZ_ZSTD_SLACK;
  if (at > room - size) at = room - size;
  l->p = o->op + at;
  l->tight = room - at - size < LZ_ZSTD_SLACK;
  return true;
}

/* Section 3.1.1.3.1.1: the header, then raw, repeated or Huffman-coded
   literals. A field the format refuses is at its first byte: the
   Regenerated_Size at the header's first, the Compressed_Size at the
   byte holding its low bits. */
long lz_zstd_literals(struct lz_zstd_dec *st, const uint8_t *s, size_t n,
                      const struct lz_zstd_out *o, struct lz_zstd_lits *l,
                      struct lz_zstd_fail *f) {
  if (n == 0) return lz_zstd_fail(f, LZ_MALFORMED, 0, header_past);
  unsigned type = s[0] & 3;
  unsigned format = (s[0] >> 2) & 3;

  if (type < 2) {
    size_t head = format == 1 ? 2 : format == 3 ? 3 : 1;
    if (n < head) return lz_zstd_fail(f, LZ_MALFORMED, 0, header_past);
    size_t size = format == 1   ? lz_le16(s) >> 4
                  : format == 3 ? lz_le24(s) >> 4
                                : (size_t)s[0] >> 3;
    if (size > st->block_max)
      return lz_zstd_fail(f, LZ_MALFORMED, 0, regen_big);
    if (type == 1 && n == head)
      return lz_zstd_fail(f, LZ_MALFORMED, 0, rle_past);
    if (type == 0 && n - head < size)
      return lz_zstd_fail(f, LZ_MALFORMED, 0, raw_past);
    /* Raw literals are read in place when the block holds the wide
       copies' slack after them. */
    if (type == 0 && n - head - size >= LZ_ZSTD_SLACK) {
      l->p = s + head;
      l->len = size;
      l->dest = false;
      l->tight = false;
      return (long)(head + size);
    }
    if (!place(st, o, size, l)) return lz_zstd_fail(f, LZ_TOO_LONG, 0, NULL);
    uint8_t *buf = (uint8_t *)l->p;
    if (type == 1) {
      memset(buf, s[head], size);
      return (long)(head + 1);
    }
    memcpy(buf, s + head, size);
    return (long)(head + size);
  }

  /* Size_Format 0 is one stream, the others four. */
  static const uint8_t heads[4] = {3, 3, 4, 5};
  static const uint8_t bits[4] = {10, 10, 14, 18};
  size_t head = heads[format];
  if (n < head) return lz_zstd_fail(f, LZ_MALFORMED, 0, header_past);
  uint64_t h = format < 2 ? lz_le24(s) : lz_le32(s);
  if (format == 3) h |= (uint64_t)s[4] << 32;
  size_t mask = ((size_t)1 << bits[format]) - 1;
  size_t size = (size_t)(h >> 4) & mask;
  size_t csize = (size_t)(h >> (4 + bits[format])) & mask;
  int64_t csize_at = format < 2 ? 1 : 2;
  bool four = format != 0;
  size_t min = four ? 10 : 1;
  if (size > st->block_max)
    return lz_zstd_fail(f, LZ_MALFORMED, 0, regen_big);
  if (four && 3 * ((size + 3) / 4) > size)
    return lz_zstd_fail(f, LZ_MALFORMED, 0, regen_four);
  if (csize > n - head)
    return lz_zstd_fail(f, LZ_MALFORMED, csize_at, csize_past);
  if (csize < min)
    return lz_zstd_fail(f, LZ_MALFORMED, csize_at, csize_small);

  const uint8_t *c = s + head;
  bool x2;
  if (type == 2) {
    size_t tree = huf_table(st->huf1, c, csize, f);
    if (tree == 0) {
      f->at += (int64_t)head;
      return -1;
    }
    c += tree;
    csize -= tree;
    if (csize < min)
      return lz_zstd_fail(f, LZ_MALFORMED, csize_at, csize_small);
    x2 = four && want_x2(size, csize);
    if (x2) huf_double(st->huf2, st->huf1);
    st->huf = x2 ? HUF_X2 : HUF_X1;
  } else {
    if (st->huf == HUF_NONE) return lz_zstd_fail(f, LZ_MALFORMED, 0, no_tree);
    x2 = st->huf == HUF_X2;
  }

  if (!place(st, o, size, l)) return lz_zstd_fail(f, LZ_TOO_LONG, 0, NULL);
  uint8_t *buf = (uint8_t *)l->p;
  bool fine = four ? four_streams(st, x2, buf, size, c, csize, f)
                   : one_stream(st, buf, size, c, csize, f);
  if (!fine) {
    f->at += c - s;
    return -1;
  }
  return (long)(c - s + csize);
}
