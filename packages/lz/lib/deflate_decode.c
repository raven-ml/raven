/* Deflate decoding: RFC 1951's blocks inside the zlib (RFC 1950) and gzip
   (RFC 1952) wrappers, one kernel for both drivers of lz.h.

   [lz_deflate_decode] parses a member as a state machine over [mode] and
   returns when the input runs out, the destination fills or the member
   ends:

     START -> ZLIB | GZIP_ID .. GZIP_HCRC -> BLOCK -+-> STORED -> COPY ----+
                                              ^     +-> TABLE -> PRECODE  |
                                              |     |     -> LENS -> HUFF |
                                              |     +-> HUFF <-> MATCH    |
                                              +---------------------------+
              BLOCK after the last block -> CHECK -> ISIZE -> START

   Huffman-coded data runs through two loops:

     fast   while the input holds FAST_IN bytes and the destination
            FAST_OUT: it fills the bit buffer 8 bytes at a time without
            bounds checks and copies matches with wide stores that may run
            past the match into the destination's room. inflate_fast.h
            holds it, compiled once for any CPU and once with BMI2.
     slow   one symbol at a time, pulling single bytes as the symbol needs
            them, so that it stops exactly where the input or the
            destination ends.

   The whole-buffer driver's destination is the window: matches read the
   output itself. A streaming decoder writes into its buffer (lz.h), whose
   bytes before the output are the window. Once a step wraps to the
   buffer's start, a match reaching before it reads the window's earlier
   segment, which ends at [ext_end], in place: the fast loop copies it
   with the same wide stores, whose reads past the segment's end stay in
   the buffer's slack.

   The bit buffer holds [bitsleft] bits, the stream's next bit lowest.
   Outside the fast loop the bits above [bitsleft] are zero and every byte
   in the buffer counts as consumed. Returning LZ_DONE or LZ_NEED_OUTPUT
   hands back the whole bytes this call pulled but did not use, so that
   [consumed] ends where the member ends. LZ_NEED_INPUT keeps them: they
   are a part of the symbol or field the next call completes, so no call
   takes a byte after the stream's last bit.

   Errors carry zlib's messages and follow its refusal rules (inflate.c,
   inftrees.c). An error's [at] is the byte holding the first bit of what
   is wrong: a Huffman code with its extra bits, a code length item, a bit
   field, or a byte-aligned field (a magic, a stored length's complement,
   a checksum, ISIZE). A code set is wrong at the length item that proves
   it so. In the whole-buffer decode, output that would pass the
   destination's end is LZ_TOO_LONG at the code that asks for it. */

#include "deflate.h"

#include <string.h>

#include "checksum.h"

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

/* Codes */

#define LITLEN_BITS 11
#define DIST_BITS 8
#define PRECODE_BITS 7

/* The longest code (RFC 1951, 3.2.7). */
#define MAX_CODE 15

/* RFC 1951, 3.2.5: the base and extra bits of length symbols 257 to 285
   and of distance symbols 0 to 29. */
static const uint16_t length_base[29] = {
    3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const uint8_t length_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                         1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t dist_base[30] = {
    1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
    33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t dist_extra[30] = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,
                                       4, 4, 5, 5, 6, 6, 7,  7,  8,  8,
                                       9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/* RFC 1951, 3.2.7: the order in which a dynamic block lists the code
   length code's lengths. */
static const uint8_t precode_order[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                          11, 4,  12, 3, 13, 2, 14, 1, 15};

/* A table entry is one word. An entry in a subtable describes its whole
   code, root bits included, so that a symbol's entry reads the same bits
   whichever table holds it.

     bits 0-5    the bits the symbol takes, its code and its extra bits;
                 for a subtable pointer, the root's bits
     bit 6       E_LIT: a literal
     bit 7       E_X: neither a literal, a length nor a distance; its
                 kind is in bits 12-15
     bits 8-11   the code's length, where its extra bits begin; for a
                 subtable pointer, the subtable's index bits
     bits 12-15  0, or an E_X entry's kind
     bits 16-31  the value: a literal, a length or distance base, a code
                 length symbol, or a subtable's first index

   The low byte of an entry that is not E_X is the count of bits under
   its symbol and the next byte the count of its code's, so that each
   serves as a shift or extraction count as it stands. */
#define E_LIT 0x40u
#define E_X 0x80u
#define K_SUB 1u
#define K_END 2u
#define K_BAD 3u /* a symbol no valid stream sends */

#define TAKES(e) ((e) & 63)
#define CODE_LEN(e) (((e) >> 8) & 15)
#define KIND(e) (((e) >> 12) & 15)
#define VALUE(e) ((e) >> 16)

static uint32_t sym_entry(uint32_t flags, unsigned value, unsigned len,
                          unsigned extra) {
  return flags | (len + extra) | len << 8 | (uint32_t)value << 16;
}

static uint32_t x_entry(unsigned kind, unsigned value, unsigned takes,
                        unsigned aux) {
  return E_X | takes | aux << 8 | kind << 12 | (uint32_t)value << 16;
}

static uint32_t litlen_entry(unsigned sym, unsigned len) {
  if (sym < 256) return sym_entry(E_LIT, sym, len, 0);
  if (sym == 256) return x_entry(K_END, 0, len, 0);
  if (sym > 285) return x_entry(K_BAD, 0, len, 0);
  unsigned i = sym - 257;
  return sym_entry(0, length_base[i], len, length_extra[i]);
}

static uint32_t dist_entry(unsigned sym, unsigned len) {
  if (sym > 29) return x_entry(K_BAD, 0, len, 0);
  return sym_entry(0, dist_base[sym], len, dist_extra[sym]);
}

enum code { PRECODE, LITLEN, DIST };

static uint32_t entry(enum code kind, unsigned sym, unsigned len) {
  switch (kind) {
  case LITLEN:
    return litlen_entry(sym, len);
  case DIST:
    return dist_entry(sym, len);
  default:
    return sym_entry(0, sym, len, 0);
  }
}

static unsigned reverse(unsigned code, unsigned len) {
  unsigned r = 0;
  for (; len > 0; len--, code >>= 1) r = r << 1 | (code & 1);
  return r;
}

/* [build t enough root lens n kind] fills [t] with the decode table of
   the canonical code (RFC 1951, 3.2.2) whose lengths are lens[0..n), and
   is false when zlib refuses the code. [*root] is the root's bits; for
   the literal/length code it shrinks to the longest code's, so that a
   block of short codes fills a smaller root. A code is accepted when it is
   complete; or when it has no symbol at all; or, for the literal/length
   and distance codes, when it has a single symbol of length 1, which is
   then code 0 while code 1 is an error. An empty code decodes every
   symbol as an error, except the code length code, whose empty code zlib
   reads as 1-bit codes for length 0. */
static bool build(uint32_t *t, unsigned enough, unsigned *root_bits,
                  const uint8_t *lens, unsigned n, enum code kind) {
  unsigned root = *root_bits;
  unsigned count[MAX_CODE + 1] = {0};
  for (unsigned s = 0; s < n; s++) count[lens[s]]++;

  int left = 1;
  unsigned max = 0;
  for (unsigned len = 1; len <= MAX_CODE; len++) {
    left = 2 * left - (int)count[len];
    if (left < 0) return false;
    if (count[len]) max = len;
  }
  if (kind == LITLEN && max > 0 && max < root) root = max;
  *root_bits = root;

  if (max == 0 || left > 0) {
    uint32_t bad = x_entry(K_BAD, 0, 1, 0);
    uint32_t zero = bad, one = bad;
    if (max == 0 && kind == PRECODE) zero = one = entry(PRECODE, 0, 1);
    if (max > 0) {
      if (kind == PRECODE || max != 1) return false;
      unsigned s = 0;
      while (lens[s] != 1) s++;
      zero = entry(kind, s, 1);
    }
    for (unsigned i = 0; i < 1u << root; i++) t[i] = (i & 1) ? one : zero;
    return true;
  }

  /* The symbols in canonical order: by length, then by value. */
  uint16_t sorted[288];
  unsigned start[MAX_CODE + 2];
  start[1] = 0;
  for (unsigned len = 1; len <= MAX_CODE; len++)
    start[len + 1] = start[len] + count[len];
  for (unsigned s = 0; s < n; s++)
    if (lens[s]) sorted[start[lens[s]]++] = (uint16_t)s;
  unsigned m = start[MAX_CODE + 1];

  /* Codes no longer than the root take one entry each in a table of
     2^len entries, which doubles, by copying its first half, as the
     length grows: a copy repeats each entry for the next index bit. */
  unsigned code = 0, prev = lens[sorted[0]], k = 0;
  unsigned size = 1u << (prev < root ? prev : root);
  for (; k < m && lens[sorted[k]] <= root; k++) {
    unsigned s = sorted[k], len = lens[s];
    if (k > 0) code = (code + 1) << (len - prev);
    prev = len;
    for (; size < 1u << len; size *= 2) memcpy(t + size, t, size * sizeof *t);
    t[reverse(code, len)] = entry(kind, s, len);
  }
  for (; size < 1u << root; size *= 2) memcpy(t + size, t, size * sizeof *t);

  unsigned next = 1u << root;
  unsigned prefix = ~0u, sub = 0, sub_bits = 0;
  for (; k < m; k++) {
    unsigned s = sorted[k], len = lens[s];
    if (k > 0) code = (code + 1) << (len - prev);
    prev = len;
    unsigned rev = reverse(code, len);

    /* A code longer than the root goes to the subtable of its first
       [root] bits. The codes sharing them follow each other in canonical
       order and fill the subtable's codespace: it is as wide as the
       longest of them needs. */
    if ((rev & ((1u << root) - 1)) != prefix) {
      prefix = rev & ((1u << root) - 1);
      unsigned space = 1u << (MAX_CODE - root), longest = len;
      for (unsigned j = k; j < m && space > 0; j++) {
        unsigned l = lens[sorted[j]];
        if (space < 1u << (MAX_CODE - l)) return false;
        space -= 1u << (MAX_CODE - l);
        longest = l;
      }
      sub_bits = longest - root;
      sub = next;
      next += 1u << sub_bits;
      if (next > enough) return false;
      t[prefix] = x_entry(K_SUB, sub, root, sub_bits);
    }
    uint32_t e = entry(kind, s, len);
    for (unsigned i = rev >> root; i < 1u << sub_bits; i += 1u << (len - root))
      t[sub + i] = e;
  }
  return true;
}

/* RFC 1951, 3.2.6: the fixed codes' lengths. */
static void load_fixed(struct lz_deflate_dec *st) {
  uint8_t *lens = st->lens;
  memset(lens, 8, 144);
  memset(lens + 144, 9, 112);
  memset(lens + 256, 7, 24);
  memset(lens + 280, 8, 8);
  memset(lens + 288, 5, 32);
  unsigned bits = LITLEN_BITS, dist_bits = DIST_BITS;
  build(st->litlen, LZ_INFLATE_LITLEN_ENOUGH, &bits, lens, 288, LITLEN);
  build(st->dist, LZ_INFLATE_DIST_ENOUGH, &dist_bits, lens + 288, 32, DIST);
  st->litlen_bits = (uint8_t)bits;
  st->fixed = true;
}

/* Bits */

static inline uint64_t low(uint64_t x, unsigned n) {
  return x & ((UINT64_C(1) << n) - 1);
}

static inline uint64_t load64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

/* The byte holding bit [bit], counted from the call's first input byte:
   negative for a bit that arrived in an earlier call. */
static inline int64_t byte_of(int64_t bit) {
  return bit >= 0 ? bit / 8 : -((-bit + 7) / 8);
}

/* [peek t root bits &e &v] decodes the symbol at the bottom of [bits]
   through table [t]: [e] is its entry, [v] its value with the extra bits
   added (for an entry that is not E_X), and the result is the number of
   bits it takes. Bits past those the buffer holds read as zero; the
   result tells the caller whether the decode used them. */
static inline unsigned peek(const uint32_t *t, unsigned root, uint64_t bits,
                            uint32_t *e, uint32_t *v) {
  uint32_t x = t[low(bits, root)];
  if ((x & E_X) && KIND(x) == K_SUB)
    x = t[VALUE(x) + low(bits >> TAKES(x), CODE_LEN(x))];
  *e = x;
  if (!(x & E_X))
    *v = VALUE(x) + (uint32_t)(low(bits, TAKES(x)) >> CODE_LEN(x));
  return TAKES(x);
}

/* Matches */

/* A match's copy loop runs a few times at most, or byte by byte where the
   match overlaps itself: vectorised, it would be preceded by a test of
   its overlap that costs more than the loop. */
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#define NOVECTOR _Pragma("GCC novector")
#else
#define NOVECTOR
#endif

/* [far out out0 hist dist n] copies a match of [n] bytes at distance
   [dist] that starts before [out0], where the member's output in this
   segment begins, in the window's earlier segment, which ends at [hist].
   The match goes on from [out0] where it passes the segment's end. */
static uint8_t *far(uint8_t *out, const uint8_t *out0, const uint8_t *hist,
                    size_t dist, size_t n) {
  size_t back = dist - (size_t)(out - out0);
  size_t k = back < n ? back : n;
  memcpy(out, hist - back, k);
  out += k;
  const uint8_t *s = out0;
  NOVECTOR
  for (n -= k; n > 0; n--) *out++ = *s++;
  return out;
}

/* [copy out out0 hist dist n] copies a match exactly, byte by byte where
   it overlaps itself. */
static uint8_t *copy(uint8_t *out, const uint8_t *out0, const uint8_t *hist,
                     size_t dist, size_t n) {
  if (dist > (size_t)(out - out0)) return far(out, out0, hist, dist, n);
  const uint8_t *s = out - dist;
  NOVECTOR
  for (; n > 0; n--) *out++ = *s++;
  return out;
}

static inline void store8(uint8_t *d, uint64_t v) { memcpy(d, &v, 8); }

/* Through a value, so that the source is read whole before the store. */
static inline void copy8(uint8_t *d, const uint8_t *s) {
  uint64_t v;
  memcpy(&v, s, 8);
  memcpy(d, &v, 8);
}

/* [wide d s n dist] copies a match of [n] bytes from [s] to [d] in
   stores of 8 bytes that may write up to 47 bytes past [d + n]. [s] is
   [dist] bytes before [d], or in another buffer when [dist] is 0. From a
   distance of 8, every store's source is complete before it. Below, a
   store from [dist] bytes back is right in its first [dist] bytes, and
   the next store starts there. */
static inline uint8_t *wide(uint8_t *d, const uint8_t *s, size_t n,
                            size_t dist) {
  uint8_t *end = d + n;
  if (likely(dist == 0 || dist >= 8)) {
    copy8(d, s);
    copy8(d + 8, s + 8);
    copy8(d + 16, s + 16);
    copy8(d + 24, s + 24);
    copy8(d + 32, s + 32);
    copy8(d + 40, s + 40);
    NOVECTOR
    for (d += 48, s += 48; d < end; d += 16, s += 16) {
      copy8(d, s);
      copy8(d + 8, s + 8);
    }
    return end;
  }
  if (dist == 1) {
    uint64_t p = UINT64_C(0x0101010101010101) * *s;
    NOVECTOR
    do {
      store8(d, p);
      store8(d + 8, p);
      d += 16;
    } while (d < end);
    return end;
  }
  NOVECTOR
  do {
    copy8(d, d - dist);
    d += dist;
  } while (d < end);
  return end;
}

/* The fast loop */

/* An iteration fills the bit buffer at most twice, the first at most
   three times, each fill loading 8 bytes from at most 7 bytes past the
   last: FAST_IN keeps its loads inside the input. It writes at most three
   literals and a fourth, or three and a match of at most 258 bytes whose
   wide stores run under 48 bytes past its end: FAST_OUT keeps them inside
   the destination. */
#define FAST_IN 32
#define FAST_OUT (3 + 258 + 48)

/* How the fast loop ended. On an error the cursor is at the first bit of
   the code that is wrong. */
enum fast_exit {
  FAST_SLOW,
  FAST_END,
  FAST_BAD_LITLEN,
  FAST_BAD_DIST,
  FAST_TOO_FAR
};

struct cursor {
  const uint8_t *in;
  uint8_t *out;
  uint64_t bitbuf;
  uint32_t bitsleft;
};

/* [fast st c out0 hist in_end out_end] decodes symbols while the input
   and the destination have the room above. It returns FAST_SLOW at that
   limit, FAST_END after an end-of-block, and otherwise the error it
   found.

   The loop is compiled twice on x86-64, once for any CPU and once with
   BMI2, whose shifts by a register and bit extraction take one
   instruction each; [fast] takes the second on a CPU that has BMI2. */

#define FAST fast_any
#define FAST_TARGET
#define EXTRACT(x, n) low((x), (n) & 63)
#include "inflate_fast.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_BMI2_PATH 1
#include <immintrin.h>
#define FAST fast_bmi2
#define FAST_TARGET __attribute__((target("bmi2")))
#define EXTRACT(x, n) _bzhi_u64((x), (unsigned)(n))
#include "inflate_fast.h"
#endif

static enum fast_exit fast(const struct lz_deflate_dec *st, struct cursor *c,
                           uint8_t *out0, const uint8_t *hist,
                           const uint8_t *in_end, uint8_t *out_end) {
#ifdef HAVE_BMI2_PATH
  if (__builtin_cpu_supports("bmi2"))
    return fast_bmi2(st, c, out0, hist, in_end, out_end);
#endif
  return fast_any(st, c, out0, hist, in_end, out_end);
}

/* The decoder */

enum mode {
  M_START,
  M_ZLIB,
  M_GZIP_ID,
  M_GZIP_FLAGS,
  M_GZIP_REST,
  M_GZIP_XLEN,
  M_GZIP_EXTRA,
  M_GZIP_NAME,
  M_GZIP_COMMENT,
  M_GZIP_HCRC,
  M_BLOCK,
  M_STORED,
  M_COPY,
  M_TABLE,
  M_PRECODE,
  M_LENS,
  M_HUFF,
  M_MATCH,
  M_CHECK,
  M_ISIZE
};

/* gzip's ID1 and ID2, and its FLG bits (RFC 1952, 2.3.1). */
#define ID1 0x1f
#define ID2 0x8b
#define FHCRC 0x02
#define FEXTRA 0x04
#define FNAME 0x08
#define FCOMMENT 0x10
#define FRESERVED 0xe0

void lz_deflate_dec_init(struct lz_deflate_dec *st, enum lz_wrap wrap,
                         bool stream) {
  st->wrap = wrap;
  st->stream = stream;
  st->fixed = false;
  st->seg = 0;
  st->ext_end = 0;
  st->ext = 0;
  lz_deflate_dec_reset(st);
}

void lz_deflate_dec_reset(struct lz_deflate_dec *st) {
  st->mode = M_START;
  st->bitbuf = 0;
  st->bitsleft = 0;
}

/* The buffer is the stubs'. */
void lz_deflate_dec_free(struct lz_deflate_dec *st) { (void)st; }

/* [wrap st io] moves a streaming decoder's empty run to the buffer's
   start once a run and the slack no longer fit after it. The output
   behind becomes the window's earlier segment: the member's bytes in it,
   at most the window, followed by those it had before. */
static void wrap(struct lz_deflate_dec *st, struct lz_io *io) {
  if (io->produced != io->run) return;
  if (io->produced + LZ_INFLATE_RUN + LZ_INFLATE_SLACK <= io->dst_cap) return;
  size_t have = io->produced - st->seg + st->ext;
  st->ext = have < LZ_INFLATE_WINDOW ? (uint32_t)have : LZ_INFLATE_WINDOW;
  st->ext_end = io->produced;
  st->seg = 0;
  io->produced = 0;
  io->run = 0;
}

/* [incomplete space used] is whether lengths leaving [space] units of
   codespace unused form a code zlib refuses: one that is neither complete,
   empty, nor a single code of length 1. */
static bool incomplete(int32_t space, uint32_t used) {
  return space > 0 && used > 0 && !(used == 1 && space == 1 << (MAX_CODE - 1));
}

/* [set_len st i v] records the code length [v] of a dynamic block's
   symbol [i], counted over the literal/length then the distance symbols,
   and is the message of the error it proves, or NULL: the lengths so far
   over-subscribe their code, symbol 256 has none, or the last length of a
   code leaves it incomplete. */
static const char *set_len(struct lz_deflate_dec *st, uint32_t i, uint32_t v) {
  int c = i >= st->nlen;
  st->lens[i] = (uint8_t)v;
  if (v) {
    st->space[c] -= 1 << (MAX_CODE - v);
    st->used[c]++;
  }
  const char *set = c ? "invalid distances set" : "invalid literal/lengths set";
  if (st->space[c] < 0) return set;
  if (i == 256 && v == 0) return "invalid code -- missing end-of-block";
  if ((i == st->nlen - 1 || i == st->nlen + st->ndist - 1) &&
      incomplete(st->space[c], st->used[c]))
    return set;
  return NULL;
}

/* [hcrc st bits n] adds the [n] low bytes of [bits] to the gzip header's
   CRC. */
static void hcrc(struct lz_deflate_dec *st, uint64_t bits, unsigned n) {
  uint8_t b[8];
  for (unsigned i = 0; i < n; i++) b[i] = (uint8_t)(bits >> (8 * i));
  st->hcrc = lz_crc32(st->hcrc, b, n);
}

/* [account st from to] adds output to the member's length and check. */
static void account(struct lz_deflate_dec *st, const uint8_t *from,
                    const uint8_t *to) {
  size_t n = (size_t)(to - from);
  st->total += n;
  if (st->wrap == LZ_GZIP) st->check = lz_crc32(st->check, from, n);
  if (st->wrap == LZ_ZLIB) st->check = lz_adler32(st->check, from, n);
}

static inline uint32_t bswap32(uint32_t x) {
  return x >> 24 | (x >> 8 & 0xff00) | (x << 8 & 0xff0000) | x << 24;
}

enum lz_status lz_deflate_decode(struct lz_deflate_dec *st, struct lz_io *io,
                                 bool final, struct lz_error *err) {
  size_t end = io->dst_cap;
  if (st->stream) {
    if (io->dst_cap < LZ_INFLATE_BUFFER) {
      io->need = LZ_INFLATE_BUFFER;
      return LZ_NEED_BUFFER;
    }
    wrap(st, io);
    end = io->run + LZ_INFLATE_RUN;
  }

  /* [out0] is where the member's output in this segment begins, [hist]
     the end of the window's earlier segment. */
  const uint8_t *src = io->src, *in = src + io->consumed;
  const uint8_t *in0 = in, *in_end = src + io->src_len;
  uint8_t *out = io->dst + io->produced, *out_end = io->dst + end;
  uint8_t *out0 = io->dst + st->seg;
  const uint8_t *hist = io->dst + st->ext_end;
  const uint8_t *checked = out;
  uint64_t bitbuf = st->bitbuf;
  uint32_t bitsleft = st->bitsleft;
  enum mode mode = (enum mode)st->mode;
  enum lz_status status;
  const char *msg;
  int64_t at;

  /* The input taken and the output written when work was last reported
     (lz.h). */
  const uint8_t *in_told = in;
  const uint8_t *out_told = out;

#define PULL()                                                               \
  do {                                                                       \
    if (in == in_end) goto starved;                                          \
    bitbuf |= (uint64_t)*in++ << bitsleft;                                   \
    bitsleft += 8;                                                           \
  } while (0)
#define NEED(n)                                                              \
  do {                                                                       \
    while (bitsleft < (n)) PULL();                                           \
  } while (0)
#define BITS(n) ((uint32_t)low(bitbuf, (n)))
#define DROP(n)                                                              \
  do {                                                                       \
    bitbuf >>= (n);                                                          \
    bitsleft -= (n);                                                         \
  } while (0)
/* The position of the buffer's next bit, and the byte that holds it. */
#define BITPOS() ((int64_t)(in - src) * 8 - (int64_t)bitsleft)
#define HERE() byte_of(BITPOS())
#define FAIL(where, text)                                                    \
  do {                                                                       \
    at = (where);                                                            \
    msg = (text);                                                            \
    goto malformed;                                                          \
  } while (0)
/* Reports the input taken and output written since the last report. */
#define TELL()                                                               \
  do {                                                                       \
    lz_did(io->work, (size_t)(in - in_told) + (size_t)(out - out_told));     \
    in_told = in;                                                            \
    out_told = out;                                                          \
  } while (0)
/* The destination is full and the code at [where] asks for more: the end
   of a step's run, or past the whole-buffer decode's destination. */
#define FULL(where)                                                          \
  do {                                                                       \
    if (st->stream) goto full;                                               \
    at = (where);                                                            \
    goto too_long;                                                           \
  } while (0)

  for (;;) {
    switch (mode) {
    case M_START:
      st->last = false;
      st->total = 0;
      st->seg = (size_t)(out - io->dst);
      st->ext = 0;
      out0 = out;
      st->hcrc = 0;
      st->check = st->wrap == LZ_ZLIB ? 1 : 0;
      mode = st->wrap == LZ_RAW    ? M_BLOCK
             : st->wrap == LZ_ZLIB ? M_ZLIB
                                   : M_GZIP_ID;
      break;

    /* RFC 1950, 2.2: CMF, FLG, checked in zlib's order. */
    case M_ZLIB: {
      NEED(16);
      uint32_t cmf = BITS(8), flg = BITS(16) >> 8;
      at = HERE();
      if ((cmf << 8 | flg) % 31) FAIL(at, "incorrect header check");
      if ((cmf & 15) != 8) FAIL(at, "unknown compression method");
      if ((cmf >> 4) > 7) FAIL(at, "invalid window size");
      if (flg & 0x20)
        return lz_fail(err, LZ_UNSUPPORTED, at + 1, "preset dictionary");
      DROP(16);
      mode = M_BLOCK;
      break;
    }

    /* RFC 1952, 2.3: ID1 ID2, CM FLG, MTIME XFL OS, then the optional
       fields FLG names. The first four bytes are checked one by one as
       they arrive, so that a byte that begins no member is refused
       whatever follows it. */
    case M_GZIP_ID:
      NEED(8);
      if (BITS(8) != ID1) FAIL(HERE(), "incorrect header check");
      NEED(16);
      if (BITS(16) >> 8 != ID2) FAIL(HERE(), "incorrect header check");
      hcrc(st, bitbuf, 2);
      DROP(16);
      mode = M_GZIP_FLAGS;
      break;

    case M_GZIP_FLAGS:
      NEED(8);
      if (BITS(8) != 8) FAIL(HERE(), "unknown compression method");
      NEED(16);
      st->flags = BITS(16) >> 8;
      if (st->flags & FRESERVED) FAIL(HERE() + 1, "unknown header flags set");
      hcrc(st, bitbuf, 2);
      DROP(16);
      mode = M_GZIP_REST;
      break;

    case M_GZIP_REST:
      NEED(48);
      hcrc(st, bitbuf, 6);
      DROP(48);
      mode = M_GZIP_XLEN;
      break;

    case M_GZIP_XLEN:
      st->count = 0;
      if (st->flags & FEXTRA) {
        NEED(16);
        st->count = BITS(16);
        hcrc(st, bitbuf, 2);
        DROP(16);
      }
      mode = M_GZIP_EXTRA;
      break;

    /* The header's variable fields are read from the input itself: the
       fixed fields before them leave the bit buffer empty. */
    case M_GZIP_EXTRA: {
      size_t n = (size_t)(in_end - in);
      if (n > st->count) n = st->count;
      st->hcrc = lz_crc32(st->hcrc, in, n);
      in += n;
      st->count -= (uint32_t)n;
      if (st->count) goto starved;
      mode = M_GZIP_NAME;
      break;
    }

    /* A field of any length, read in spans of LZ_WORK_SPAN with their
       work reported between them. */
    case M_GZIP_NAME:
    case M_GZIP_COMMENT:
      if (st->flags & (mode == M_GZIP_NAME ? FNAME : FCOMMENT)) {
        size_t n = (size_t)(in_end - in);
        if (n > LZ_WORK_SPAN) n = LZ_WORK_SPAN;
        const uint8_t *z = memchr(in, 0, n);
        const uint8_t *end = z ? z + 1 : in + n;
        st->hcrc = lz_crc32(st->hcrc, in, (size_t)(end - in));
        in = end;
        if (!z && in == in_end) goto starved;
        if (!z) {
          TELL();
          break;
        }
      }
      mode = mode == M_GZIP_NAME ? M_GZIP_COMMENT : M_GZIP_HCRC;
      break;

    case M_GZIP_HCRC:
      if (st->flags & FHCRC) {
        NEED(16);
        if (BITS(16) != (st->hcrc & 0xffff))
          FAIL(HERE(), "header crc mismatch");
        DROP(16);
      }
      mode = M_BLOCK;
      break;

    /* RFC 1951, 3.2.3: BFINAL, BTYPE. The check takes in each block's
       output as the block ends, while it is still in cache. */
    case M_BLOCK: {
      account(st, checked, out);
      checked = out;
      if (st->last) {
        mode = M_CHECK;
        break;
      }
      TELL();
      NEED(3);
      uint32_t type = BITS(3) >> 1;
      if (type == 3) FAIL(byte_of(BITPOS() + 1), "invalid block type");
      st->last = BITS(1);
      DROP(3);
      if (type == 0) {
        mode = M_STORED;
      } else if (type == 1) {
        if (!st->fixed) load_fixed(st);
        mode = M_HUFF;
      } else {
        mode = M_TABLE;
      }
      break;
    }

    /* RFC 1951, 3.2.4: LEN and NLEN from the next byte boundary. */
    case M_STORED: {
      DROP(bitsleft & 7);
      NEED(32);
      uint32_t len = BITS(16), nlen = BITS(32) >> 16;
      if (len != (~nlen & 0xffff))
        FAIL(HERE() + 2, "invalid stored block lengths");
      DROP(32);
      st->count = len;
      mode = M_COPY;
      break;
    }

    /* The bytes the bit buffer holds, then the input's. A byte the
       destination has no room for is the code that asks for output. */
    case M_COPY: {
      while (st->count > 0 && bitsleft >= 8) {
        if (out == out_end) FULL(HERE());
        *out++ = (uint8_t)bitbuf;
        DROP(8);
        st->count--;
      }
      size_t n = st->count;
      if (n > (size_t)(in_end - in)) n = (size_t)(in_end - in);
      if (n > (size_t)(out_end - out)) n = (size_t)(out_end - out);
      memcpy(out, in, n);
      in += n;
      out += n;
      st->count -= (uint32_t)n;
      if (st->count > 0) {
        if (in == in_end) goto starved;
        FULL(HERE());
      }
      mode = M_BLOCK;
      break;
    }

    /* RFC 1951, 3.2.7: HLIT, HDIST, HCLEN, the code length code, then the
       code lengths it encodes. */
    case M_TABLE: {
      NEED(14);
      int64_t hlit = BITPOS();
      st->nlen = 257 + BITS(5);
      st->ndist = 1 + (BITS(10) >> 5);
      st->ncode = 4 + (BITS(14) >> 10);
      if (st->nlen > 286)
        FAIL(byte_of(hlit), "too many length or distance symbols");
      if (st->ndist > 30)
        FAIL(byte_of(hlit + 5), "too many length or distance symbols");
      DROP(14);
      st->index = 0;
      st->space[0] = 1 << PRECODE_BITS;
      mode = M_PRECODE;
      break;
    }

    /* A length that over-subscribes the code length code proves it wrong;
       the last length, one that leaves it incomplete. */
    case M_PRECODE:
      while (st->index < st->ncode) {
        NEED(3);
        uint32_t l = BITS(3);
        if (l) st->space[0] -= 1 << (PRECODE_BITS - l);
        if (st->space[0] < 0) FAIL(HERE(), "invalid code lengths set");
        st->lens[precode_order[st->index++]] = (uint8_t)l;
        DROP(3);
      }
      for (unsigned i = st->ncode; i < 19; i++) st->lens[precode_order[i]] = 0;
      unsigned pre_bits = PRECODE_BITS;
      if (!build(st->precode, 1u << PRECODE_BITS, &pre_bits, st->lens, 19,
                 PRECODE))
        FAIL(byte_of(BITPOS() - 3), "invalid code lengths set");
      st->index = 0;
      st->space[0] = st->space[1] = 1 << MAX_CODE;
      st->used[0] = st->used[1] = 0;
      mode = M_LENS;
      break;

    /* Each item, a code length symbol and its extra bits, writes [count]
       lengths. A set of lengths fails at the item that proves it wrong.
       A repeat past the last length is zlib's first check of an item. */
    case M_LENS: {
      uint32_t total = st->nlen + st->ndist;
      while (st->index < total) {
        uint32_t e = st->precode[low(bitbuf, PRECODE_BITS)];
        uint32_t sym = VALUE(e), n = TAKES(e);
        uint32_t extra = sym < 16 ? 0 : sym == 16 ? 2 : sym == 17 ? 3 : 7;
        if (n + extra > bitsleft) {
          PULL();
          continue;
        }
        at = HERE();
        uint32_t x = (uint32_t)low(bitbuf >> n, extra);
        uint32_t val = sym < 16 ? sym : 0;
        uint32_t count = sym < 16 ? 1 : (sym == 18 ? 11 : 3) + x;
        if (sym == 16) {
          if (st->index == 0) FAIL(at, "invalid bit length repeat");
          val = st->lens[st->index - 1];
        }
        if (count > total - st->index) FAIL(at, "invalid bit length repeat");
        DROP(n + extra);
        for (uint32_t j = 0; j < count; j++) {
          const char *why = set_len(st, st->index++, val);
          if (why) FAIL(at, why);
        }
      }
      /* set_len has refused every set that build refuses. */
      st->fixed = false;
      unsigned bits = LITLEN_BITS, dist_bits = DIST_BITS;
      if (!build(st->litlen, LZ_INFLATE_LITLEN_ENOUGH, &bits, st->lens,
                 st->nlen, LITLEN))
        FAIL(HERE(), "invalid literal/lengths set");
      st->litlen_bits = (uint8_t)bits;
      if (!build(st->dist, LZ_INFLATE_DIST_ENOUGH, &dist_bits,
                 st->lens + st->nlen, st->ndist, DIST))
        FAIL(HERE(), "invalid distances set");
      mode = M_HUFF;
      break;
    }

    case M_HUFF: {
      /* The fast loop, in spans of LZ_WORK_SPAN of output with their work
         reported between them, until the input or the destination has
         too little room left. */
      enum fast_exit why = FAST_SLOW;
      while (in_end - in >= FAST_IN && out_end - out >= FAST_OUT) {
        uint8_t *span = (size_t)(out_end - out) > LZ_WORK_SPAN + FAST_OUT
                            ? out + LZ_WORK_SPAN + FAST_OUT
                            : out_end;
        struct cursor c = {in, out, bitbuf, bitsleft};
        why = fast(st, &c, out0, hist, in_end, span);
        in = c.in;
        out = c.out;
        bitbuf = c.bitbuf;
        bitsleft = c.bitsleft;
        if (why != FAST_SLOW || span == out_end) break;
        TELL();
      }
      if (why == FAST_END) {
        mode = M_BLOCK;
        break;
      }
      if (why == FAST_BAD_LITLEN) FAIL(HERE(), "invalid literal/length code");
      if (why == FAST_BAD_DIST) FAIL(HERE(), "invalid distance code");
      if (why == FAST_TOO_FAR) FAIL(HERE(), "invalid distance too far back");

      /* One symbol at a time: a literal, an end of block, or a length and
         its distance, taken whole or not at all. A distance is checked
         before the room for its match, so that a step, which only stops
         at a full run, and the whole-buffer decode refuse it alike. */
      for (;;) {
        uint32_t e, d, v = 0, dist = 0;
        unsigned n = peek(st->litlen, st->litlen_bits, bitbuf, &e, &v);
        if (n > bitsleft) {
          PULL();
          continue;
        }
        if (e & E_LIT) {
          if (out == out_end) FULL(HERE());
          *out++ = (uint8_t)v;
          DROP(n);
          continue;
        }
        if (e & E_X) {
          if (KIND(e) == K_BAD) FAIL(HERE(), "invalid literal/length code");
          DROP(n);
          mode = M_BLOCK;
          break;
        }
        unsigned m = peek(st->dist, DIST_BITS, bitbuf >> n, &d, &dist);
        if (n + m > bitsleft) {
          PULL();
          continue;
        }
        if (d & E_X) FAIL(byte_of(BITPOS() + n), "invalid distance code");
        if (dist > (size_t)(out - out0) + st->ext)
          FAIL(byte_of(BITPOS() + n), "invalid distance too far back");
        size_t room = (size_t)(out_end - out);
        if (v > room && (room == 0 || !st->stream)) FULL(HERE());
        DROP(n + m);
        uint32_t k = v < room ? v : (uint32_t)room;
        out = copy(out, out0, hist, dist, k);
        if (k < v) {
          st->match_len = v - k;
          st->match_dist = dist;
          mode = M_MATCH;
          goto full;
        }
      }
      break;
    }

    /* The rest of a match that a step's run cut. */
    case M_MATCH: {
      size_t room = (size_t)(out_end - out);
      uint32_t k = st->match_len < room ? st->match_len : (uint32_t)room;
      out = copy(out, out0, hist, st->match_dist, k);
      st->match_len -= k;
      if (st->match_len > 0) goto full;
      mode = M_HUFF;
      break;
    }

    /* RFC 1950's ADLER32, big-endian; RFC 1952's CRC32 and ISIZE,
       little-endian; each from the byte boundary after the last block. */
    case M_CHECK:
      DROP(bitsleft & 7);
      if (st->wrap == LZ_RAW) goto done;
      account(st, checked, out);
      checked = out;
      NEED(32);
      if (st->wrap == LZ_ZLIB) {
        if (bswap32(BITS(32)) != st->check)
          FAIL(HERE(), "incorrect data check");
        DROP(32);
        goto done;
      }
      if (BITS(32) != st->check) FAIL(HERE(), "incorrect data check");
      DROP(32);
      mode = M_ISIZE;
      break;

    case M_ISIZE:
      NEED(32);
      if (BITS(32) != (uint32_t)st->total)
        FAIL(HERE(), "incorrect length check");
      DROP(32);
      goto done;
    }
  }

malformed:
  return lz_fail(err, LZ_MALFORMED, at, msg);

too_long:
  return lz_fail(err, LZ_TOO_LONG, at, NULL);

starved:
  if (final) return lz_fail(err, LZ_TRUNCATED, (int64_t)(in_end - src), NULL);
  status = LZ_NEED_INPUT;
  goto leave;

full:
  status = LZ_NEED_OUTPUT;
  goto leave;

done:
  mode = M_START;
  status = LZ_DONE;

leave:
  /* Hand back the whole bytes this call pulled past the bits it used. */
  if (status != LZ_NEED_INPUT) {
    size_t k = bitsleft >> 3;
    if (k > (size_t)(in - in0)) k = (size_t)(in - in0);
    in -= k;
    bitsleft -= 8 * (uint32_t)k;
    bitbuf = low(bitbuf, bitsleft);
  }
  account(st, checked, out);
  st->mode = (uint8_t)mode;
  st->bitbuf = bitbuf;
  st->bitsleft = bitsleft;
  io->consumed = (size_t)(in - src);
  io->produced = (size_t)(out - io->dst);
  return status;

#undef PULL
#undef NEED
#undef BITS
#undef DROP
#undef BITPOS
#undef HERE
#undef FAIL
#undef FULL
#undef TELL
}
