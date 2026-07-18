/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*/

#ifndef NX_IO_CODEC_H
#define NX_IO_CODEC_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
  NX_IO_OK = 0,
  NX_IO_STOPPED,
  NX_IO_TRUNCATED,
  NX_IO_INVALID_BLOCK,
  NX_IO_INVALID_HUFFMAN,
  NX_IO_INVALID_SYMBOL,
  NX_IO_INVALID_DISTANCE,
  NX_IO_OUTPUT_SIZE,
  NX_IO_OVERFLOW,
  NX_IO_NOMEM,
  NX_IO_SYSTEM
} nx_io_status;

typedef struct {
  nx_io_status status;
  size_t input_offset;
  size_t output_size;
} nx_io_result;

typedef nx_io_status (*nx_io_consume_fn)(void *context, uint8_t byte);

uint32_t nx_io_crc32(const uint8_t *src, size_t len);
uint32_t nx_io_adler32(const uint8_t *src, size_t len);

nx_io_result nx_io_inflate_raw(const uint8_t *src, size_t src_len, size_t skip,
                               uint8_t *dst, size_t dst_len, int stop_at_limit,
                               int fd, uint32_t *crc);

nx_io_result nx_io_inflate_raw_sink(const uint8_t *src, size_t src_len,
                                    size_t output_size,
                                    nx_io_consume_fn consume, void *context,
                                    uint32_t *crc);

nx_io_result nx_io_deflate_raw(const uint8_t *prefix, size_t prefix_len,
                               const uint8_t *src, size_t src_len, int fd,
                               uint8_t **buffer, uint32_t *crc);

const char *nx_io_status_message(nx_io_status status);

#endif
