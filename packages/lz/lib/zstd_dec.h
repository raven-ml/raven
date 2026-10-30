/* What the units of the Zstandard decoder share: little-endian loads, the
   backward bit reader of RFC 8878's entropy-coded streams, the output a
   block is written to, and the entries of each unit.

     zstd_decode.c    frames, blocks, the window, the stream's stages
     zstd_dec_lit.c   the literals section: Huffman tables and streams
     zstd_dec_fse.c   FSE table descriptions and decoding tables
     zstd_dec_seq.c   the sequences section: decoding and execution

   Every offset an entry reports in an error is relative to the start of
   the block content it was given, and is the first byte of what is
   wrong: a field's first byte, a section's or a table description's
   first byte when it runs past the block, a bitstream's first byte when
   it is not read exactly to its start, the byte holding a sequence's
   first bit when the sequence breaks a rule. An element that has no byte
   in the block is reported at the block's end, which zstd_decode.c moves
   to the block's last byte. */

#ifndef LZ_ZSTD_DEC_H
#define LZ_ZSTD_DEC_H

#include <string.h>

#include "zstd.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the zstd decoder reads little-endian words natively"
#endif

#if defined(__GNUC__)
#define LZ_LIKELY(x) __builtin_expect(!!(x), 1)
#define LZ_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define LZ_INLINE static inline __attribute__((always_inline))
#define LZ_NOINLINE __attribute__((noinline))
#else
#define LZ_NOINLINE
#define LZ_LIKELY(x) (x)
#define LZ_UNLIKELY(x) (x)
#define LZ_INLINE static inline
#endif

/* The hot loops are compiled twice on x86-64: as the baseline allows, and
   for BMI1, BMI2 and LZCNT, whose shifts and bit counts set no flags and
   take any register. A decoder's [path], set at its init from the CPU,
   picks one (zstd_dec_paths.h). */
#if defined(__x86_64__) && defined(__GNUC__)
#define LZ_ZSTD_BMI2 1
#define LZ_TARGET_BMI2 __attribute__((target("bmi,bmi2,lzcnt")))
#else
#define LZ_ZSTD_BMI2 0
#endif

enum { LZ_ZSTD_GENERIC, LZ_ZSTD_WITH_BMI2 };

static inline uint16_t lz_le16(const uint8_t *p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return v;
}

static inline uint32_t lz_le24(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16;
}

static inline uint32_t lz_le32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}

static inline uint64_t lz_le64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

/* [lz_high32 x] is the index of the highest set bit of [x], which is not
   0. */
static inline unsigned lz_high32(uint32_t x) {
  return 31u - (unsigned)__builtin_clz(x);
}

/* Backward bit reader

   RFC 8878 section 4.1 and 4.2.2: a stream is written forward and read
   from its last byte to its first; the highest set bit of the last byte
   marks where the data begins. [c] holds the 8 bytes at [p]; [used] counts
   its bits already read, from the top. Reading never moves [p] below
   [start] nor loads past the stream's end: a stream shorter than 8 bytes
   sits in [c] whole, its missing high bytes counted as read. A read past
   the stream's start returns bits of no meaning and leaves [used] above
   64; the stream was read exactly when [p == start] and [used == 64]. */

struct lz_bits {
  uint64_t c;
  unsigned used;
  const uint8_t *p;
  const uint8_t *start;
};

/* [lz_bits_init b s n] starts reading the [n] bytes at [s]. It is false
   when the stream is empty or its last byte holds no end mark. */
static inline bool lz_bits_init(struct lz_bits *b, const uint8_t *s,
                                size_t n) {
  if (n == 0 || s[n - 1] == 0) return false;
  unsigned mark = 8 - lz_high32(s[n - 1]);
  b->start = s;
  if (n >= 8) {
    b->p = s + n - 8;
    b->c = lz_le64(b->p);
    b->used = mark;
    return true;
  }
  uint64_t c = 0;
  for (size_t i = 0; i < n; i++) c |= (uint64_t)s[i] << (8 * i);
  b->p = s;
  b->c = c;
  b->used = mark + 8 * (unsigned)(8 - n);
  return true;
}

/* [lz_bits_read b n] reads [n] bits, 0 to 56 of them, as a number whose
   first bit read is the highest. */
LZ_INLINE uint64_t lz_bits_read(struct lz_bits *b, unsigned n) {
  uint64_t v = (b->c >> ((64 - b->used - n) & 63)) & ((1ull << n) - 1);
  b->used += n;
  return v;
}

/* [lz_bits_reload b] moves the window down so that it holds at least 56
   unread bits, or all that are left. */
LZ_INLINE void lz_bits_reload(struct lz_bits *b) {
  if (b->p == b->start) return;
  size_t n = b->used >> 3;
  size_t room = (size_t)(b->p - b->start);
  if (LZ_UNLIKELY(n > room)) n = room;
  b->p -= n;
  b->used -= 8 * (unsigned)n;
  b->c = lz_le64(b->p);
}

static inline bool lz_bits_done(const struct lz_bits *b) {
  return b->p == b->start && b->used == 64;
}

/* [lz_bits_byte b] is the offset from [b->start] of the byte holding the
   next bit to read, or 0 once the stream is read to its start. */
static inline size_t lz_bits_byte(const struct lz_bits *b) {
  int64_t left = 8 * (int64_t)(b->p - b->start) + 64 - (int64_t)b->used;
  return left > 0 ? (size_t)(left - 1) >> 3 : 0;
}

/* Output

   A block is written at [op]. The frame's output in the current segment
   starts at [base]; with a streaming window that has wrapped, the [ext]
   bytes before [ext_end] precede [base] in the frame. No write passes
   [end], the end of the buffer, and a block's output never passes
   [op + block_max]. */

struct lz_zstd_out {
  uint8_t *base;
  uint8_t *op;
  uint8_t *end;
  const uint8_t *ext_end;
  size_t ext;
  uint64_t window;
  size_t block_max;
};

/* Errors */

/* A block's verdict: an error kind of lz.h with its offset from the
   block's content, negative in its header, and its message. */
struct lz_zstd_fail {
  int kind;
  int64_t at;
  const char *msg;
};

static inline int lz_zstd_fail(struct lz_zstd_fail *f, int kind, int64_t at,
                               const char *msg) {
  f->kind = kind;
  f->at = at;
  f->msg = msg;
  return -1;
}

/* FSE (zstd_dec_fse.c) */

/* [lz_fse_ncount norm max log log_max s n f] reads the table description
   at [s], of at most [n] bytes, into [norm]: the normalised count of each
   symbol up to [*max], which it lowers to the last present symbol, and
   its accuracy log, at most [log_max]. It is the description's size, or
   0 if the description is malformed, with [f] set at an offset from
   [s]: the field that breaks a rule, or 0 when the description runs past
   the [n] bytes. */
size_t lz_fse_ncount(int16_t norm[256], unsigned *max, unsigned *log,
                     unsigned log_max, const uint8_t *s, size_t n,
                     struct lz_zstd_fail *f);

/* [lz_fse_build t norm max log base extra] fills the [1 << log] cells of
   [t] from a valid description. Symbol [s] decodes as [base[s]] and
   [extra[s]] extra bits; with [base] NULL, as [s] itself and no extra
   bits. */
void lz_fse_build(struct lz_zstd_cell *t, const int16_t *norm, unsigned max,
                  unsigned log, const uint32_t *base, const uint8_t *extra);

/* Literals (zstd_dec_lit.c) */

/* A block's literals: [len] bytes at [p], in the destination past the
   block's output with [dest], else in the block or a stream's buffer.
   LZ_ZSTD_SLACK readable bytes follow them unless [tight]. */
struct lz_zstd_lits {
  const uint8_t *p;
  size_t len;
  bool dest;
  bool tight;
};

/* [lz_zstd_reach o] is how far past [o->op] a sequence copied wide may
   end: LZ_ZSTD_SLACK short of the buffer's end, since its copies write
   that far past it, and within the block's maximum. */
static inline size_t lz_zstd_reach(const struct lz_zstd_out *o) {
  size_t room = (size_t)(o->end - o->op);
  size_t fast = room > LZ_ZSTD_SLACK ? room - LZ_ZSTD_SLACK : 0;
  return fast < o->block_max ? fast : o->block_max;
}

/* [lz_zstd_literals st s n o l f] decodes the literals section at the
   start of the [n] bytes of block content [s] into [l]: in place, in a
   stream's buffer, or in [o]'s destination. It is the section's size, or
   -1 with [f] set. */
long lz_zstd_literals(struct lz_zstd_dec *st, const uint8_t *s, size_t n,
                      const struct lz_zstd_out *o, struct lz_zstd_lits *l,
                      struct lz_zstd_fail *f);

/* Sequences (zstd_dec_seq.c) */

/* [lz_zstd_sequences st s n at l o f] decodes the sequences section [s],
   of [n] bytes, which starts [at] bytes into the block, and executes it
   with the literals [l] into [o]. It is the block's output size, or -1
   with [f] set. */
long lz_zstd_sequences(struct lz_zstd_dec *st, const uint8_t *s, size_t n,
                       size_t at, const struct lz_zstd_lits *l,
                       const struct lz_zstd_out *o, struct lz_zstd_fail *f);

#endif
