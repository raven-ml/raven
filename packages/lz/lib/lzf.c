/* LZF, as liblzf's decoder (lzf_d.c, the BSD licence's text) reads it.
   The format has no written specification: a block is a series of
   elements, each beginning with a control byte [c], and ends with its
   input. An element is

     c < 32      a literal run: the next c + 1 bytes;
     c >= 32     a match of (c >> 5) + 2 bytes, (c >> 5) + 9 + the next
                 byte when c >> 5 is 7, copied from
                 ((c & 31) << 8 | b) + 1 bytes back, b the byte after.

   So a literal run holds 1 to 32 bytes, a match 3 to 264 bytes from at
   most 8 KiB back. A match may overlap its own output, as deflate's may.

   Every error is at the element's control byte, except a truncation, at
   the input's end. A literal run whose output passes [dst] is
   LZ_TOO_LONG before its input is checked, as liblzf checks; a match is
   read whole first, then checked against the output written, then
   against [dst]. */

#include <string.h>

#include "lzf.h"

/* Control bytes below this begin a literal run. */
#define LITERAL_LIMIT 32

/* The length field of a match whose length continues in the next byte. */
#define LONG_MATCH 7

/* A match copies its length field plus this many bytes. */
#define MATCH_MIN 2

enum lz_status lz_lzf_decode(struct lz_io *io, struct lz_error *err) {
  const uint8_t *src = io->src;
  size_t len = io->src_len, cap = io->dst_cap;
  uint8_t *dst = io->dst;

  if (len == 0) return lz_fail(err, LZ_TRUNCATED, 0, NULL);

  size_t ip = 0, op = 0;

  /* The input and output when work was last reported (lz.h). */
  size_t in_told = 0, out_told = 0;

  while (ip < len) {
    if (ip - in_told + (op - out_told) >= LZ_WORK_SPAN) {
      lz_did(io->work, ip - in_told + (op - out_told));
      in_told = ip;
      out_told = op;
    }
    size_t at = ip;
    unsigned c = src[ip++];

    if (c < LITERAL_LIMIT) {
      size_t n = (size_t)c + 1;
      if (n > cap - op) return lz_fail(err, LZ_TOO_LONG, (int64_t)at, NULL);
      if (n > len - ip) return lz_fail(err, LZ_TRUNCATED, (int64_t)len, NULL);
      memcpy(dst + op, src + ip, n);
      ip += n;
      op += n;
      continue;
    }

    size_t n = c >> 5;
    if (n == LONG_MATCH) {
      if (ip == len) return lz_fail(err, LZ_TRUNCATED, (int64_t)len, NULL);
      n += src[ip++];
    }
    if (ip == len) return lz_fail(err, LZ_TRUNCATED, (int64_t)len, NULL);
    size_t distance = ((size_t)(c & 31) << 8 | src[ip++]) + 1;
    n += MATCH_MIN;
    if (distance > op)
      return lz_fail(err, LZ_MALFORMED, (int64_t)at,
                     "invalid offset too far back");
    if (n > cap - op) return lz_fail(err, LZ_TOO_LONG, (int64_t)at, NULL);

    /* A match shorter than its distance reads only bytes written before
       it; a longer one repeats them, byte by byte. */
    uint8_t *from = dst + op - distance;
    if (distance >= n) memcpy(dst + op, from, n);
    else
      for (size_t k = 0; k < n; k++) dst[op + k] = from[k];
    op += n;
  }
  io->consumed = len;
  io->produced = op;
  return LZ_DONE;
}
