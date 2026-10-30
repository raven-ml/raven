/* The Zstandard encoder, levels 1 to 12. The entries follow lz.h's
   contract.

   A streaming step gives [lz_zstd_encode] at most LZ_ZSTD_ENC_QUOTA bytes
   of input and LZ_ZSTD_ENC_RUN bytes of output: the encoder takes all of
   that input, and the output it owes for it, held-back bytes and the
   frame's end included under [last], always fits the run. It holds back
   up to a block of input, 128 KiB, until the next byte arrives or [last]:
   a step compresses the blocks that leave at most that much pending, so
   one step's output covers at most two blocks' worth of input, in blocks
   of at least 8 KiB but the last, each at most 3 bytes longer than its
   input, with the frame's header and checksum.

   The state itself is small, for the whole-buffer encode keeps it on the
   stub's stack: the tables, the block buffers and, streaming, the window
   are allocated at the first [lz_zstd_encode]. */

#ifndef LZ_ZSTD_ENC_H
#define LZ_ZSTD_ENC_H

#include "lz.h"

#define LZ_ZSTD_ENC_QUOTA (128 * 1024)
#define LZ_ZSTD_ENC_RUN (2 * LZ_ZSTD_ENC_QUOTA + 256)

struct ze_work;

struct lz_zstd_enc {
  int level;
  bool stream;
  bool bmi2;
  uint32_t limit;
  struct ze_work *w;
};

/* [lz_zstd_enc_mem level] is the most a stream encoder at [level]
   allocates over its life, for the collector's accounting. */
size_t lz_zstd_enc_mem(int level);
void lz_zstd_enc_init(struct lz_zstd_enc *st, int level, bool stream);
void lz_zstd_enc_reset(struct lz_zstd_enc *st);
void lz_zstd_enc_free(struct lz_zstd_enc *st);
enum lz_status lz_zstd_encode(struct lz_zstd_enc *st, struct lz_io *io,
                              bool last, struct lz_error *err);
size_t lz_zstd_bound(size_t n);

/* The builds of the entropy stage's bit writers this host can run, for
   the tests that hold them equal: the generic build first, then the BMI2
   build where the CPU has BMI1, BMI2 and LZCNT. An encoder takes the last
   at its init; both write the same bytes. [lz_zstd_enc_paths names] sets
   [*names] to their names and is their number; [lz_zstd_enc_path st i]
   makes [st] write with path [i], before its first encode. */
size_t lz_zstd_enc_paths(const char *const **names);
void lz_zstd_enc_path(struct lz_zstd_enc *st, size_t i);

/* The match finders' indices are 32 bits: once a block would end past an
   index limit, 3 GiB into the frame, they are rebased, which changes no
   sequence. [lz_zstd_enc_limit st limit] lowers the limit for [st], before
   its first encode, for the tests that hold the bytes equal across limits:
   a block rebases once it starts past the window and would end past
   [limit], so a limit of 0 rebases every block after the window's first
   fill. */
void lz_zstd_enc_limit(struct lz_zstd_enc *st, uint32_t limit);

#endif
