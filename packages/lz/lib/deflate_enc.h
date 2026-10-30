/* The deflate encoder, for raw deflate, zlib and gzip (deflate.h's
   [enum lz_wrap]), levels 1 to 9. The entries follow lz.h's contract.

   A streaming step gives [lz_deflate_encode] at most
   LZ_DEFLATE_ENC_QUOTA bytes of input and LZ_DEFLATE_ENC_RUN bytes of
   output: the encoder takes all of that input, and the output it owes for
   it, the blocks that end in the step, the header and under [last] the
   final block and the trailer, always fits the run. deflate_encode.c
   derives the run from its block lengths and checks it at compile time.

   After LZ_DONE the next call begins a new stream: the encoder resets
   itself, so [lz_deflate_enc_reset] is needed only to drop a stream
   midway. */

#ifndef LZ_DEFLATE_ENC_H
#define LZ_DEFLATE_ENC_H

#include "deflate.h"

#define LZ_DEFLATE_ENC_QUOTA (64 * 1024)
#define LZ_DEFLATE_ENC_RUN (384 * 1024)

/* The encoder's tables, window and parse, allocated with [lz_alloc] at
   the first [lz_deflate_encode]; deflate_encode.c defines it. */
struct lz_deflate_tables;

struct lz_deflate_enc {
  enum lz_wrap wrap;
  int level;
  bool stream;
  bool generic;
  struct lz_deflate_tables *t;
};

/* [lz_deflate_enc_mem level] is what a stream encoder at [level]
   allocates, for the collector's accounting. */
size_t lz_deflate_enc_mem(int level);
void lz_deflate_enc_init(struct lz_deflate_enc *st, enum lz_wrap wrap,
                         int level, bool stream);
void lz_deflate_enc_reset(struct lz_deflate_enc *st);
void lz_deflate_enc_free(struct lz_deflate_enc *st);
enum lz_status lz_deflate_encode(struct lz_deflate_enc *st, struct lz_io *io,
                                 bool last, struct lz_error *err);
size_t lz_deflate_bound(enum lz_wrap wrap, size_t n);

/* The builds of the matchfinders' rebase this host can run, for the tests
   that hold them equal: the generic loop first, then SSE2 where the
   compiler targets it. An encoder takes the last at its init; both write
   the same bytes. [lz_deflate_enc_paths names] sets [*names] to their
   names and is their number; [lz_deflate_enc_path st i] makes [st]
   rebase with path [i], before its first encode. */
size_t lz_deflate_enc_paths(const char *const **names);
void lz_deflate_enc_path(struct lz_deflate_enc *st, size_t i);

#endif
