/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#ifndef COMPRESS_DEFLATE_H
#define COMPRESS_DEFLATE_H

#include <stddef.h>
#include <stdint.h>

/* Fill the static tables of the library. The OCaml module initializer calls
   them once, before any other function. */
void compress_inflate_init(void);
void compress_crc32_init(void);

/* Checksums */

uint32_t compress_deflate_crc32(uint32_t crc, const uint8_t *src, size_t len);
uint32_t compress_deflate_adler32(uint32_t adler, const uint8_t *src,
                                  size_t len);

/* Inflating. [compress_inflate_run] decodes the raw deflate stream that
   [src[src_pos, src_end)] continues into [dst[dst_pos, dst_end)]. Matches may
   reach back to [dst_hist]. It returns at a symbol boundary with the
   positions advanced, having consumed only whole symbols, so that a call
   with more input or more room resumes where it stopped. */

enum {
  COMPRESS_INFLATE_END,    /* The stream ended at [src_pos]. */
  COMPRESS_INFLATE_INPUT,  /* More input is needed; [final] was false. */
  COMPRESS_INFLATE_OUTPUT, /* [dst] is full. */
  COMPRESS_INFLATE_TRUNCATED,
  COMPRESS_INFLATE_BLOCK_TYPE,
  COMPRESS_INFLATE_STORED_LENGTH,
  COMPRESS_INFLATE_CODE_COUNTS,
  COMPRESS_INFLATE_HUFFMAN,
  COMPRESS_INFLATE_REPEAT,
  COMPRESS_INFLATE_NO_END_CODE,
  COMPRESS_INFLATE_SYMBOL,
  COMPRESS_INFLATE_DISTANCE
};

#define COMPRESS_INFLATE_LIT_ENOUGH 1334
#define COMPRESS_INFLATE_DIST_ENOUGH 402

typedef struct {
  int mode;
  int last;
  uint64_t bits;
  unsigned nbits;
  size_t stored;
  size_t copy_length;
  size_t copy_distance;
  const uint32_t *lit;
  const uint32_t *dist;
  uint32_t lit_table[COMPRESS_INFLATE_LIT_ENOUGH];
  uint32_t dist_table[COMPRESS_INFLATE_DIST_ENOUGH];
} compress_inflate;

typedef struct {
  const uint8_t *src;
  size_t src_pos, src_end;
  int final; /* [src_end] is the end of the input. */
  uint8_t *dst;
  size_t dst_hist, dst_pos, dst_end;
} compress_inflate_io;

void compress_inflate_reset(compress_inflate *s);
int compress_inflate_run(compress_inflate *s, compress_inflate_io *io);

/* Deflating. Input is buffered with [compress_deflate_input]; each
   [compress_deflate_encode] encodes at most one block into [out], which holds
   [COMPRESS_DEFLATE_OUT_MAX] bytes, and returns the number of bytes written.
   A block is encoded once the input reaches past it or [eod] is set. The
   bytes depend only on the input and the level. */

#define COMPRESS_DEFLATE_OUT_MAX (65536 + 16)

typedef struct compress_deflate compress_deflate;

compress_deflate *compress_deflate_create(int level);
void compress_deflate_free(compress_deflate *e);
size_t compress_deflate_input(compress_deflate *e, const uint8_t *src,
                              size_t len);
size_t compress_deflate_encode(compress_deflate *e, uint8_t *out, int eod);

#endif
