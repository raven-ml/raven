/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#ifndef COMPRESS_ZSTD_H
#define COMPRESS_ZSTD_H

#include <stddef.h>
#include <stdint.h>

enum {
  COMPRESS_ZSTD_OK,
  COMPRESS_ZSTD_TRUNCATED,
  COMPRESS_ZSTD_LITERALS,
  COMPRESS_ZSTD_HUFFMAN,
  COMPRESS_ZSTD_SEQUENCES,
  COMPRESS_ZSTD_OFFSET,
  COMPRESS_ZSTD_TOO_LONG,
  COMPRESS_ZSTD_ZERO_OFFSET
};

/* A state of an FSE table: its symbol, and the bits and base of the next
   state. Sequence tables also hold the symbol's value: its baseline and
   extra bits. */
typedef struct {
  uint32_t baseline;
  uint16_t base;
  uint8_t symbol;
  uint8_t bits;
  uint8_t extra;
} compress_zstd_fse_entry;

typedef struct {
  compress_zstd_fse_entry table[512];
  unsigned log;
} compress_zstd_fse;

typedef struct {
  uint8_t symbol;
  uint8_t bits;
} compress_zstd_huf;

/* The state that a frame's compressed blocks share: the tables a block may
   repeat from the one before, and the repeat offsets. */
typedef struct {
  compress_zstd_huf huf[1u << 11];
  unsigned huf_bits;
  int huf_valid;
  compress_zstd_fse ll, of, ml;
  int ll_valid, of_valid, ml_valid;
  size_t rep[3];
  uint8_t literals[128u * 1024u + 32u];
} compress_zstd;

/* Starts a frame. */
void compress_zstd_reset(compress_zstd *z);

/* Decodes the compressed block [src[pos, end)] into [dst[*out, dst_end)].
   Matches reach back to [hist]. Advances [*out]. */
int compress_zstd_block(compress_zstd *z, const uint8_t *src, size_t pos,
                        size_t end, uint8_t *dst, size_t hist, size_t *out,
                        size_t dst_end);

uint64_t compress_zstd_xxh64(const uint8_t *p, size_t len);

#endif
