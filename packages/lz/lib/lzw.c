/* TIFF's LZW (TIFF 6.0, section 13): codes of 9 to 12 bits, most
   significant bit first, each naming a string of bytes. Codes 0 to 255
   are the bytes themselves, 256 is ClearCode, 257 EndOfInformation, and
   each code read after the first one since a ClearCode adds an entry to
   the table, numbered from 258: the string of the code before it and the
   first byte of its own.

   That string is a run of the output: the code before wrote its string,
   and this code's string begins right after it. So entry [e] is [len[e]]
   bytes of [dst] at [off[e]], and a code above 257 is a copy from earlier
   output, as a match is in deflate. A code equal to the entry about to be
   added (the section's "else" branch: the string before it and that
   string's first byte) is the copy of the string before it, one byte
   longer, whose last byte is its first.

   The decoder widens its codes one entry before the encoder, which
   switches "after adding table entry 511": the decoder's table lags the
   encoder's by one code, so it switches "as soon as string #510 is
   stored", that is, a code read when the next entry is [next] has the
   width of [next + 1], at most 12 bits.

   An error is at the byte that holds the first bit of the code at
   fault, or at the input's end when it ends inside a code. */

#include <string.h>

#include "lzw.h"

/* Codes (section 13) */

#define CLEAR 256
#define EOI 257
#define FIRST 258

/* "our table can have at most 4096 entries" */
#define ENTRIES 4096

/* "we will arbitrarily limit ourselves to 12-bit codes" */
#define WIDEST 12

/* The width of the code read when the next entry is [next]. */
static inline unsigned width(unsigned next) {
  return next < 511 ? 9 : next < 1023 ? 10 : next < 2047 ? 11 : WIDEST;
}

static inline uint64_t be64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

/* [peek src len bit w code] sets [code] to the [w] bits of [src] from bit
   [bit], the first bit the most significant of byte 0, and is false when
   [src] ends before them. */
static inline bool peek(const uint8_t *src, size_t len, size_t bit,
                        unsigned w, uint32_t *code) {
  size_t byte = bit >> 3;
  uint64_t v;
  if (len >= 8 && byte <= len - 8) v = be64(src + byte);
  else {
    if (bit + w > (uint64_t)len * 8) return false;
    v = 0;
    for (size_t i = 0; i < 8 && byte + i < len; i++)
      v |= (uint64_t)src[byte + i] << (56 - 8 * i);
  }
  *code = (uint32_t)((v << (bit & 7)) >> (64 - w));
  return true;
}

static inline void copy16(uint8_t *dst, const uint8_t *src) {
  uint8_t t[16];
  memcpy(t, src, 16);
  memcpy(dst, t, 16);
}

static enum lz_status malformed(struct lz_error *err, size_t at,
                                const char *msg) {
  return lz_fail(err, LZ_MALFORMED, (int64_t)at, msg);
}

enum lz_status lz_lzw_decode(struct lz_io *io, struct lz_error *err) {
  const uint8_t *src = io->src;
  size_t len = io->src_len, cap = io->dst_cap;
  uint8_t *dst = io->dst;

  /* ClearCode written least significant bit first sets byte 1's low bit
     after a zero byte, where the most significant bit first sets byte 0's
     high bit. */
  if (len >= 2 && src[0] == 0 && (src[1] & 1))
    return lz_fail(err, LZ_UNSUPPORTED, 0,
                   "codes least significant bit first");

  size_t off[ENTRIES];
  uint16_t lens[ENTRIES];

  /* [next] is the entry the next code adds, if a string precedes it:
     [prev] bytes at [prev_off], none right after a ClearCode. */
  unsigned next = FIRST;
  size_t prev = 0, prev_off = 0;
  size_t bit = 0, op = 0;

  /* The input and output when work was last reported (lz.h). */
  size_t in_told = 0, out_told = 0;

  bool started = false;
  for (;;) {
    size_t at = bit >> 3;
    /* Work is input taken and output written, so that a run of ClearCodes,
       which write nothing, reports too. */
    if (at - in_told + (op - out_told) >= LZ_WORK_SPAN) {
      lz_did(io->work, at - in_told + (op - out_told));
      in_told = at;
      out_told = op;
    }
    uint32_t code;
    if (!peek(src, len, bit, width(next), &code))
      return lz_fail(err, LZ_TRUNCATED, (int64_t)len, NULL);
    bit += width(next);

    if (!started) {
      if (code != CLEAR)
        return malformed(err, at, "the first code is not ClearCode");
      started = true;
    }
    if (code == CLEAR) {
      next = FIRST;
      prev = 0;
      continue;
    }
    if (code == EOI) break;

    /* The string, written at [op]: [n] bytes. */
    size_t n;
    if (code < CLEAR) {
      n = 1;
      if (op == cap) return lz_fail(err, LZ_TOO_LONG, (int64_t)at, NULL);
      dst[op] = (uint8_t)code;
    } else if (code < next) {
      /* The entry ends before [op], so source and destination share no
         byte; a 16-byte copy reads and writes inside [dst]. */
      n = lens[code];
      if (n > cap - op) return lz_fail(err, LZ_TOO_LONG, (int64_t)at, NULL);
      if (n <= 16 && cap - op >= 16) copy16(dst + op, dst + off[code]);
      else memcpy(dst + op, dst + off[code], n);
    } else if (code == next && prev > 0) {
      n = prev + 1;
      if (n > cap - op) return lz_fail(err, LZ_TOO_LONG, (int64_t)at, NULL);
      memcpy(dst + op, dst + prev_off, prev);
      dst[op + prev] = dst[prev_off];
    } else
      return malformed(err, at, "invalid code");

    if (prev > 0) {
      if (next == ENTRIES) return malformed(err, at, "the table is full");
      off[next] = prev_off;
      lens[next] = (uint16_t)(prev + 1);
      next++;
    }
    prev_off = op;
    prev = n;
    op += n;
  }
  io->consumed = (bit + 7) >> 3;
  io->produced = op;
  return LZ_DONE;
}
