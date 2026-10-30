/* Zstandard (RFC 8878): frames and skippable frames. The entries follow
   lz.h's contract. */

#ifndef LZ_ZSTD_H
#define LZ_ZSTD_H

#include "checksum.h"
#include "lz.h"

/* The decoder's limits (RFC 8878 section 3.1.1.2.4: a block holds at most
   128 KiB, compressed or not) and the bytes a wide copy may write past
   the end of what it copies. */
#define LZ_ZSTD_BLOCK_MAX (128 * 1024)
#define LZ_ZSTD_SLACK 32

/* A streaming decoder's run (lz.h): the blocks a step writes, of at most
   this many bytes. A block decodes whole, and a compressed block's output
   is known only once decoded, so a step places one only while the run
   has room for the largest. With room for two, a step ends only once its
   run holds at least a block's maximum, however small its blocks. The
   buffer holds the window, a run and twice the slack. */
#define LZ_ZSTD_RUN (256 * 1024)
_Static_assert(LZ_ZSTD_RUN >= 2 * LZ_ZSTD_BLOCK_MAX, "a run holds two blocks");

/* A cell of an FSE decoding table (RFC 8878 section 4.1): the state's
   symbol, as the baseline and extra bits of its code, and how to reach the
   next state. */
struct lz_zstd_cell {
  uint16_t next;
  uint8_t bits;
  uint8_t extra;
  uint32_t base;
};

/* The decoder of one stream of frames. The whole-buffer decode keeps it on
   the stub's stack, which bounds its size (about 24 KiB), and allocates
   nothing: the destination is the window and holds each block's literals
   at its end. A streaming decoder writes into the buffer the stubs lend
   it (lz.h), asking for a larger one when a frame header fixes a larger
   window, and allocates [in], where a block that arrives in pieces is
   gathered, and [lits], a block's literals; reset keeps both. [path]
   selects the hot loops' build (zstd_dec_paths.h). */
struct lz_zstd_dec {
  uint64_t window_max;
  bool stream;
  uint8_t path;

  /* Where the decoder is in the stream, in input bytes since the last
     reset, and the bytes of the current header gathered in [unit]. */
  int stage;
  uint64_t total;
  uint64_t call;
  size_t have;
  uint8_t unit[24];

  /* The frame: its start, Window_Size, declared content size and where
     its field begins, block limit, the output so far and the running
     content checksum. */
  uint64_t frame_at;
  uint64_t window;
  uint64_t content_size;
  uint64_t fcs_at;
  bool sized;
  bool checked;
  uint32_t block_max;
  uint64_t produced;
  struct lz_xxh64 hash;

  /* The block being read, or the skippable frame's bytes left, and
     whether the block has its place in the output. */
  uint64_t block_at;
  uint32_t block_size;
  uint8_t block_type;
  bool placed;
  bool last;
  uint64_t skip;

  /* What a compressed block inherits from the previous ones of its frame:
     the repeat offsets, the Huffman table and the three FSE tables. */
  uint64_t rep[3];
  uint8_t huf;
  bool seq_tables;
  bool defaults;
  uint8_t ll_log, of_log, ml_log;
  const struct lz_zstd_cell *ll, *of, *ml;
  struct lz_zstd_cell ll_t[512], of_t[256], ml_t[512];
  struct lz_zstd_cell ll_d[64], of_d[32], ml_d[64];
  uint16_t huf1[2048];
  uint32_t huf2[2048];

  /* Where the frame's output in this segment of the buffer begins, set
     at its first block ([fresh] until then); the end of the window's
     earlier segment and the frame's bytes before it; and, streaming only,
     the gathered block and the literals. */
  bool fresh;
  size_t seg;
  size_t ext_end;
  size_t ext;
  uint8_t *in;
  uint8_t *lits;
};

/* [window_max] bounds the window a frame may declare; a frame above it is
   LZ_UNSUPPORTED at its header. */
void lz_zstd_dec_init(struct lz_zstd_dec *st, uint64_t window_max,
                      bool stream);
void lz_zstd_dec_reset(struct lz_zstd_dec *st);
void lz_zstd_dec_free(struct lz_zstd_dec *st);
enum lz_status lz_zstd_decode(struct lz_zstd_dec *st, struct lz_io *io,
                              bool final, struct lz_error *err);

#endif
