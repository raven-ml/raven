/* Deflate (RFC 1951) and its zlib (RFC 1950) and gzip (RFC 1952)
   wrappers: the decoder, parameterised by the wrapper. The encoder is
   deflate_enc.h. The entries follow lz.h's contract. */

#ifndef LZ_DEFLATE_H
#define LZ_DEFLATE_H

#include "lz.h"

enum lz_wrap { LZ_RAW, LZ_ZLIB, LZ_GZIP };

/* The decoder's tables: a root indexed by the next 11 (literal/length) or
   8 (distance) bits, followed by its subtables for longer codes. The sizes
   are the most entries any valid code needs, as zlib's examples/enough.c
   counts them for 288 and 32 symbols of at most 15 bits. */
#define LZ_INFLATE_LITLEN_ENOUGH 2342
#define LZ_INFLATE_DIST_ENOUGH 402

/* The history a match may reach back into (RFC 1951, section 2). */
#define LZ_INFLATE_WINDOW 32768

/* A streaming decoder's buffer (lz.h): the window, the room the fast
   loop's wide stores write past the output (3 literals, the longest
   match, 258, and 48 bytes past it), a run, and the bytes a wide copy
   reads past the window's earlier segment. A step wraps once fewer than
   a run and the slack remain, which leaves at least the window and the
   slack behind it. A step costs the matches that reach back past the
   wrap, which halve as the run doubles: 256 KiB costs 5 % on text against
   the whole-buffer loop, 1 % on Silesia, and the buffer fits a 512 KiB
   L2. */
#define LZ_INFLATE_RUN (256 * 1024)
#define LZ_INFLATE_SLACK 320
#define LZ_INFLATE_BUFFER                                                     \
  (LZ_INFLATE_WINDOW + LZ_INFLATE_SLACK + LZ_INFLATE_RUN + LZ_INFLATE_SLACK)

/* A decoder state. A whole-buffer decode keeps it on the stack (12 KiB),
   where it writes only the fields the parse reaches. deflate_decode.c
   describes the fields. */
struct lz_deflate_dec {
  enum lz_wrap wrap;
  bool stream;
  bool last;  /* the block being read ends the member */
  bool fixed; /* the tables hold the fixed codes */
  uint8_t mode;
  uint8_t litlen_bits; /* the literal/length table's root */
  uint32_t bitsleft;
  uint64_t bitbuf;
  uint32_t flags; /* gzip's FLG */
  uint32_t count; /* bytes left of a stored block or gzip's extra field */
  uint32_t nlen, ndist, ncode, index; /* a dynamic block's header */
  int32_t space[2];       /* codespace left in its lengths read so far */
  uint32_t used[2];       /* their codes */
  uint32_t match_len, match_dist;     /* a match the destination cut */
  uint32_t check, hcrc;               /* the data's and gzip header's */
  uint64_t total;                     /* the member's output so far */
  size_t seg;     /* where the member's output in this segment begins */
  size_t ext_end; /* the end of the window's earlier segment */
  uint32_t ext;   /* the member's bytes before it, at most the window */
  uint8_t lens[288 + 32];
  uint32_t precode[128];
  uint32_t litlen[LZ_INFLATE_LITLEN_ENOUGH];
  uint32_t dist[LZ_INFLATE_DIST_ENOUGH];
};

void lz_deflate_dec_init(struct lz_deflate_dec *st, enum lz_wrap wrap,
                         bool stream);
void lz_deflate_dec_reset(struct lz_deflate_dec *st);
void lz_deflate_dec_free(struct lz_deflate_dec *st);
enum lz_status lz_deflate_decode(struct lz_deflate_dec *st, struct lz_io *io,
                                 bool final, struct lz_error *err);

#endif
