/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Static PNG decoding and encoding following the W3C PNG specification. The
  decoder validates the complete chunk stream before allocating codec state,
  inflates scanlines through a row consumer, and writes pixels directly into
  the Nx-owned destination Bigarray.
  --------------------------------------------------------------------------*/

#include "nx_io_codec.h"

#ifndef NX_IO_CODEC_NO_OCAML
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>
#endif

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PNG_IDAT_CHUNK (1u << 20)

typedef enum {
  PNG_OK = 0,
  PNG_TRUNCATED,
  PNG_SIGNATURE,
  PNG_CRC,
  PNG_ORDER,
  PNG_IHDR,
  PNG_UNSUPPORTED,
  PNG_PALETTE,
  PNG_ZLIB,
  PNG_FILTER,
  PNG_SIZE,
  PNG_NOMEM,
  PNG_SYSTEM
} png_status;

typedef struct {
  size_t width;
  size_t height;
  unsigned depth;
  unsigned color;
  unsigned channels;
  unsigned interlace;
  uint8_t palette[256 * 3];
  unsigned palette_size;
  size_t idat_size;
} png_info;

static const uint8_t png_signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};

#ifndef NX_IO_CODEC_NO_OCAML
static const char *png_message(png_status status) {
  switch (status) {
  case PNG_OK:
    return "ok";
  case PNG_TRUNCATED:
    return "truncated PNG stream";
  case PNG_SIGNATURE:
    return "invalid PNG signature";
  case PNG_CRC:
    return "PNG chunk CRC mismatch";
  case PNG_ORDER:
    return "invalid PNG chunk ordering";
  case PNG_IHDR:
    return "invalid PNG image header";
  case PNG_UNSUPPORTED:
    return "unsupported PNG feature";
  case PNG_PALETTE:
    return "invalid PNG palette";
  case PNG_ZLIB:
    return "invalid PNG zlib stream";
  case PNG_FILTER:
    return "invalid PNG scanline filter";
  case PNG_SIZE:
    return "PNG dimensions are too large";
  case PNG_NOMEM:
    return "PNG codec allocation failed";
  case PNG_SYSTEM:
    return "PNG output error";
  }
  return "unknown PNG error";
}
#endif

static uint32_t read_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static void write_be32(uint8_t p[4], uint32_t value) {
  p[0] = (uint8_t)(value >> 24);
  p[1] = (uint8_t)(value >> 16);
  p[2] = (uint8_t)(value >> 8);
  p[3] = (uint8_t)value;
}

static int type_is(const uint8_t *type, const char name[4]) {
  return memcmp(type, name, 4) == 0;
}

static int valid_depth(unsigned color, unsigned depth) {
  switch (color) {
  case 0:
    return depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16;
  case 2:
    return depth == 8 || depth == 16;
  case 3:
    return depth == 1 || depth == 2 || depth == 4 || depth == 8;
  case 4:
  case 6:
    return depth == 8 || depth == 16;
  default:
    return 0;
  }
}

static unsigned color_channels(unsigned color) {
  switch (color) {
  case 0:
  case 3:
    return 1;
  case 2:
    return 3;
  case 4:
    return 2;
  case 6:
    return 4;
  default:
    return 0;
  }
}

static png_status parse_png(const uint8_t *src, size_t len, png_info *info) {
  memset(info, 0, sizeof(*info));
  if (len < sizeof(png_signature))
    return PNG_TRUNCATED;
  if (memcmp(src, png_signature, sizeof(png_signature)) != 0)
    return PNG_SIGNATURE;

  size_t off = 8;
  int seen_header = 0;
  int seen_palette = 0;
  int seen_transparency = 0;
  int seen_data = 0;
  int data_ended = 0;
  int seen_end = 0;
  while (off < len) {
    if (len - off < 12)
      return PNG_TRUNCATED;
    uint32_t length32 = read_be32(src + off);
    size_t length = length32;
    if (length > len - off - 12)
      return PNG_TRUNCATED;
    const uint8_t *type = src + off + 4;
    const uint8_t *data = type + 4;
    uint32_t expected_crc = read_be32(data + length);
    if (nx_io_crc32(type, length + 4) != expected_crc)
      return PNG_CRC;
    if ((type[2] & 0x20u) != 0)
      return PNG_ORDER;

    if (!seen_header && !type_is(type, "IHDR"))
      return PNG_ORDER;
    if (type_is(type, "IHDR")) {
      if (seen_header || off != 8 || length != 13)
        return PNG_ORDER;
      uint32_t width = read_be32(data);
      uint32_t height = read_be32(data + 4);
      unsigned depth = data[8];
      unsigned color = data[9];
      if (width == 0 || height == 0 || !valid_depth(color, depth) ||
          data[10] != 0 || data[11] != 0 || data[12] > 1)
        return PNG_IHDR;
      info->width = width;
      info->height = height;
      info->depth = depth;
      info->color = color;
      info->channels = color_channels(color);
      info->interlace = data[12];
      seen_header = 1;
    } else if (type_is(type, "PLTE")) {
      if (!seen_header || seen_palette || seen_data || info->color == 0 ||
          info->color == 4 || length == 0 || length % 3 != 0 || length > 768)
        return PNG_PALETTE;
      info->palette_size = (unsigned)(length / 3);
      if (info->color == 3 && info->palette_size > (1u << info->depth))
        return PNG_PALETTE;
      memcpy(info->palette, data, length);
      seen_palette = 1;
    } else if (type_is(type, "tRNS")) {
      if (!seen_header || seen_transparency || seen_data)
        return PNG_ORDER;
      if ((info->color == 0 && length != 2) ||
          (info->color == 2 && length != 6) ||
          (info->color == 3 &&
           (length == 0 || !seen_palette || length > info->palette_size)) ||
          info->color == 4 || info->color == 6)
        return PNG_PALETTE;
      seen_transparency = 1;
    } else if (type_is(type, "IDAT")) {
      if (!seen_header || data_ended || (info->color == 3 && !seen_palette))
        return PNG_ORDER;
      if (info->idat_size > SIZE_MAX - length)
        return PNG_SIZE;
      info->idat_size += length;
      seen_data = 1;
    } else if (type_is(type, "IEND")) {
      if (!seen_header || !seen_data || seen_end || length != 0)
        return PNG_ORDER;
      seen_end = 1;
      off += length + 12;
      if (off != len)
        return PNG_ORDER;
      break;
    } else {
      if (seen_data)
        data_ended = 1;
      if ((type[0] & 0x20u) == 0)
        return PNG_UNSUPPORTED;
    }
    if (seen_data && !type_is(type, "IDAT"))
      data_ended = 1;
    off += length + 12;
  }
  if (!seen_end || info->idat_size < 6)
    return PNG_ORDER;
  return PNG_OK;
}

static png_status collect_idat(const uint8_t *src, size_t len,
                               const png_info *info, uint8_t **result) {
  uint8_t *compressed = malloc(info->idat_size);
  if (compressed == NULL)
    return PNG_NOMEM;
  size_t dst = 0;
  size_t off = 8;
  while (off < len) {
    size_t length = read_be32(src + off);
    const uint8_t *type = src + off + 4;
    if (type_is(type, "IDAT")) {
      memcpy(compressed + dst, type + 4, length);
      dst += length;
    }
    off += length + 12;
  }
  if (dst != info->idat_size) {
    free(compressed);
    return PNG_SIZE;
  }
  *result = compressed;
  return PNG_OK;
}

static size_t pass_extent(size_t size, unsigned start, unsigned step) {
  if (size <= start)
    return 0;
  return (size - start + step - 1) / step;
}

static png_status scanline_size(const png_info *info, size_t *total,
                                size_t *max_row) {
  static const uint8_t start_x[7] = {0, 4, 0, 2, 0, 1, 0};
  static const uint8_t start_y[7] = {0, 0, 4, 0, 2, 0, 1};
  static const uint8_t step_x[7] = {8, 8, 4, 4, 2, 2, 1};
  static const uint8_t step_y[7] = {8, 8, 8, 4, 4, 2, 2};
  unsigned passes = info->interlace ? 7 : 1;
  size_t sum = 0;
  size_t maximum = 0;
  for (unsigned pass = 0; pass < passes; pass++) {
    size_t width = info->interlace
                       ? pass_extent(info->width, start_x[pass], step_x[pass])
                       : info->width;
    size_t height = info->interlace
                        ? pass_extent(info->height, start_y[pass], step_y[pass])
                        : info->height;
    if (width == 0 || height == 0)
      continue;
    if (width > (SIZE_MAX - 7) / (info->channels * info->depth))
      return PNG_SIZE;
    size_t row = (width * info->channels * info->depth + 7) / 8;
    if (row > maximum)
      maximum = row;
    if (row == SIZE_MAX || height > (SIZE_MAX - sum) / (row + 1))
      return PNG_SIZE;
    sum += height * (row + 1);
  }
  *total = sum;
  *max_row = maximum;
  return PNG_OK;
}

static unsigned paeth(unsigned left, unsigned up, unsigned upper_left) {
  int p = (int)left + (int)up - (int)upper_left;
  unsigned pa = (unsigned)abs(p - (int)left);
  unsigned pb = (unsigned)abs(p - (int)up);
  unsigned pc = (unsigned)abs(p - (int)upper_left);
  if (pa <= pb && pa <= pc)
    return left;
  return pb <= pc ? up : upper_left;
}

static png_status unfilter(uint8_t *row, const uint8_t *previous,
                           size_t row_size, unsigned bpp, unsigned filter) {
  if (filter > 4)
    return PNG_FILTER;
  for (size_t i = 0; i < row_size; i++) {
    unsigned left = i >= bpp ? row[i - bpp] : 0;
    unsigned up = previous == NULL ? 0 : previous[i];
    unsigned upper_left = previous != NULL && i >= bpp ? previous[i - bpp] : 0;
    unsigned predictor = 0;
    switch (filter) {
    case 0:
      predictor = 0;
      break;
    case 1:
      predictor = left;
      break;
    case 2:
      predictor = up;
      break;
    case 3:
      predictor = (left + up) >> 1;
      break;
    case 4:
      predictor = paeth(left, up, upper_left);
      break;
    }
    row[i] = (uint8_t)(row[i] + predictor);
  }
  return PNG_OK;
}

static unsigned sample(const uint8_t *row, size_t index, unsigned depth) {
  if (depth == 8)
    return row[index];
  if (depth == 16)
    return ((unsigned)row[index * 2] << 8) | row[index * 2 + 1];
  unsigned per_byte = 8 / depth;
  unsigned shift = 8 - depth * (unsigned)(index % per_byte + 1);
  return (row[index / per_byte] >> shift) & ((1u << depth) - 1u);
}

static uint8_t scale_sample(unsigned value, unsigned depth) {
  if (depth == 8)
    return (uint8_t)value;
  if (depth == 16)
    return (uint8_t)(value >> 8);
  return (uint8_t)((value * 255u + ((1u << depth) - 1u) / 2u) /
                   ((1u << depth) - 1u));
}

typedef struct {
  const png_info *info;
  uint8_t *dst;
  unsigned output_channels;
  uint8_t *current;
  uint8_t *previous;
  size_t max_row;
  unsigned pass;
  size_t pass_width;
  size_t pass_height;
  size_t pass_row;
  size_t row_size;
  size_t row_pos;
  unsigned filter;
  uint32_t adler_a;
  uint32_t adler_b;
  png_status status;
  int complete;
} png_consumer;

static void pass_geometry(const png_info *info, unsigned pass, size_t *width,
                          size_t *height) {
  static const uint8_t start_x[7] = {0, 4, 0, 2, 0, 1, 0};
  static const uint8_t start_y[7] = {0, 0, 4, 0, 2, 0, 1};
  static const uint8_t step_x[7] = {8, 8, 4, 4, 2, 2, 1};
  static const uint8_t step_y[7] = {8, 8, 8, 4, 4, 2, 2};
  if (!info->interlace) {
    *width = info->width;
    *height = info->height;
  } else {
    *width = pass_extent(info->width, start_x[pass], step_x[pass]);
    *height = pass_extent(info->height, start_y[pass], step_y[pass]);
  }
}

static void consumer_next_pass(png_consumer *consumer) {
  unsigned passes = consumer->info->interlace ? 7 : 1;
  while (consumer->pass < passes) {
    pass_geometry(consumer->info, consumer->pass, &consumer->pass_width,
                  &consumer->pass_height);
    if (consumer->pass_width != 0 && consumer->pass_height != 0) {
      consumer->row_size = (consumer->pass_width * consumer->info->channels *
                                consumer->info->depth +
                            7) /
                           8;
      consumer->pass_row = 0;
      consumer->row_pos = 0;
      memset(consumer->previous, 0, consumer->max_row);
      return;
    }
    consumer->pass++;
  }
  consumer->complete = 1;
}

static png_status write_scanline(png_consumer *consumer) {
  static const uint8_t start_x[7] = {0, 4, 0, 2, 0, 1, 0};
  static const uint8_t start_y[7] = {0, 0, 4, 0, 2, 0, 1};
  static const uint8_t step_x[7] = {8, 8, 4, 4, 2, 2, 1};
  static const uint8_t step_y[7] = {8, 8, 8, 4, 4, 2, 2};
  unsigned bpp = (consumer->info->channels * consumer->info->depth + 7) / 8;
  if (bpp == 0)
    bpp = 1;
  png_status status = unfilter(
      consumer->current, consumer->pass_row == 0 ? NULL : consumer->previous,
      consumer->row_size, bpp, consumer->filter);
  if (status != PNG_OK)
    return status;

  unsigned sx = consumer->info->interlace ? start_x[consumer->pass] : 0;
  unsigned sy = consumer->info->interlace ? start_y[consumer->pass] : 0;
  unsigned dx = consumer->info->interlace ? step_x[consumer->pass] : 1;
  unsigned dy = consumer->info->interlace ? step_y[consumer->pass] : 1;
  size_t y = sy + consumer->pass_row * dy;
  for (size_t x_pass = 0; x_pass < consumer->pass_width; x_pass++) {
    size_t component = x_pass * consumer->info->channels;
    unsigned r;
    unsigned g;
    unsigned b;
    if (consumer->info->color == 0 || consumer->info->color == 4) {
      uint8_t gray = scale_sample(
          sample(consumer->current, component, consumer->info->depth),
          consumer->info->depth);
      r = g = b = gray;
    } else if (consumer->info->color == 3) {
      unsigned index = sample(consumer->current, x_pass, consumer->info->depth);
      if (index >= consumer->info->palette_size)
        return PNG_PALETTE;
      r = consumer->info->palette[index * 3];
      g = consumer->info->palette[index * 3 + 1];
      b = consumer->info->palette[index * 3 + 2];
    } else {
      r = scale_sample(
          sample(consumer->current, component, consumer->info->depth),
          consumer->info->depth);
      g = scale_sample(
          sample(consumer->current, component + 1, consumer->info->depth),
          consumer->info->depth);
      b = scale_sample(
          sample(consumer->current, component + 2, consumer->info->depth),
          consumer->info->depth);
    }
    size_t x = sx + x_pass * dx;
    size_t dst = (y * consumer->info->width + x) * consumer->output_channels;
    if (consumer->output_channels == 1) {
      consumer->dst[dst] =
          (uint8_t)((77u * r + 150u * g + 29u * b + 128u) >> 8);
    } else {
      consumer->dst[dst] = (uint8_t)r;
      consumer->dst[dst + 1] = (uint8_t)g;
      consumer->dst[dst + 2] = (uint8_t)b;
    }
  }
  uint8_t *swap = consumer->previous;
  consumer->previous = consumer->current;
  consumer->current = swap;
  consumer->pass_row++;
  consumer->row_pos = 0;
  if (consumer->pass_row == consumer->pass_height) {
    consumer->pass++;
    consumer_next_pass(consumer);
  }
  return PNG_OK;
}

static nx_io_status consume_scanline(void *context, uint8_t byte) {
  png_consumer *consumer = context;
  consumer->adler_a += byte;
  if (consumer->adler_a >= 65521u)
    consumer->adler_a -= 65521u;
  consumer->adler_b += consumer->adler_a;
  if (consumer->adler_b >= 65521u)
    consumer->adler_b -= 65521u;
  if (consumer->complete)
    return NX_IO_OUTPUT_SIZE;
  if (consumer->row_pos == 0) {
    consumer->filter = byte;
    consumer->row_pos = 1;
    return byte <= 4 ? NX_IO_OK : NX_IO_INVALID_BLOCK;
  }
  consumer->current[consumer->row_pos - 1] = byte;
  consumer->row_pos++;
  if (consumer->row_pos == consumer->row_size + 1) {
    png_status status = write_scanline(consumer);
    if (status != PNG_OK) {
      consumer->status = status;
      return NX_IO_STOPPED;
    }
  }
  return NX_IO_OK;
}

static png_status decode_png(const uint8_t *src, size_t len, uint8_t *dst,
                             size_t dst_len, int grayscale) {
  png_info info;
  png_status status = parse_png(src, len, &info);
  if (status != PNG_OK)
    return status;
  unsigned output_channels = grayscale ? 1 : 3;
  if (info.width > SIZE_MAX / info.height ||
      info.width * info.height > SIZE_MAX / output_channels ||
      info.width * info.height * output_channels != dst_len)
    return PNG_SIZE;
  size_t filtered_size;
  size_t max_row;
  status = scanline_size(&info, &filtered_size, &max_row);
  if (status != PNG_OK)
    return status;
  if (max_row > SIZE_MAX / 2)
    return PNG_SIZE;
  uint8_t *zlib = NULL;
  status = collect_idat(src, len, &info, &zlib);
  if (status != PNG_OK)
    return status;
  unsigned cmf = zlib[0];
  unsigned flg = zlib[1];
  if ((cmf & 15u) != 8 || (cmf >> 4) > 7 || ((cmf << 8) | flg) % 31 != 0 ||
      (flg & 0x20u) != 0) {
    free(zlib);
    return PNG_ZLIB;
  }
  uint32_t expected_adler = read_be32(zlib + info.idat_size - 4);
  uint8_t *rows = malloc(max_row == 0 ? 1 : max_row * 2);
  if (rows == NULL) {
    free(zlib);
    return PNG_NOMEM;
  }
  png_consumer consumer;
  memset(&consumer, 0, sizeof(consumer));
  consumer.info = &info;
  consumer.dst = dst;
  consumer.output_channels = output_channels;
  consumer.current = rows;
  consumer.previous = rows + max_row;
  consumer.max_row = max_row;
  consumer.adler_a = 1;
  consumer_next_pass(&consumer);
  nx_io_result result =
      nx_io_inflate_raw_sink(zlib + 2, info.idat_size - 6, filtered_size,
                             consume_scanline, &consumer, NULL);
  free(rows);
  free(zlib);
  if (consumer.status != PNG_OK)
    return consumer.status;
  if (result.status != NX_IO_OK)
    return result.status == NX_IO_NOMEM ? PNG_NOMEM : PNG_ZLIB;
  uint32_t actual_adler = (consumer.adler_b << 16) | consumer.adler_a;
  if (!consumer.complete || actual_adler != expected_adler)
    return PNG_ZLIB;
  return PNG_OK;
}

static uint32_t crc_table_entry(uint32_t value) {
  for (int bit = 0; bit < 8; bit++)
    value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
  return value;
}

static uint32_t crc_update(uint32_t crc, const uint8_t *data, size_t len) {
  uint32_t table[256];
  for (uint32_t i = 0; i < 256; i++)
    table[i] = crc_table_entry(i);
  for (size_t i = 0; i < len; i++)
    crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
  return crc;
}

static png_status write_all(int fd, const uint8_t *data, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t written = write(fd, data + off, len - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return PNG_SYSTEM;
    off += (size_t)written;
  }
  return PNG_OK;
}

static png_status write_chunk(int fd, const char type[4], const uint8_t *data,
                              size_t len) {
  if (len > UINT32_MAX)
    return PNG_SIZE;
  uint8_t header[8];
  write_be32(header, (uint32_t)len);
  memcpy(header + 4, type, 4);
  png_status status = write_all(fd, header, sizeof(header));
  if (status == PNG_OK)
    status = write_all(fd, data, len);
  uint32_t crc = crc_update(0xffffffffu, (const uint8_t *)type, 4);
  crc = crc_update(crc, data, len) ^ 0xffffffffu;
  uint8_t trailer[4];
  write_be32(trailer, crc);
  if (status == PNG_OK)
    status = write_all(fd, trailer, sizeof(trailer));
  return status;
}

static unsigned filter_byte(unsigned filter, unsigned raw, unsigned left,
                            unsigned up, unsigned upper_left) {
  unsigned predictor;
  switch (filter) {
  case 0:
    predictor = 0;
    break;
  case 1:
    predictor = left;
    break;
  case 2:
    predictor = up;
    break;
  case 3:
    predictor = (left + up) >> 1;
    break;
  default:
    predictor = paeth(left, up, upper_left);
    break;
  }
  return (raw - predictor) & 0xffu;
}

static png_status filter_image(const uint8_t *src, size_t width, size_t height,
                               unsigned channels, uint8_t **result,
                               size_t *result_len) {
  if (width > SIZE_MAX / channels)
    return PNG_SIZE;
  size_t stride = width * channels;
  if (stride == SIZE_MAX || height > SIZE_MAX / (stride + 1))
    return PNG_SIZE;
  size_t length = height * (stride + 1);
  uint8_t *filtered = malloc(length == 0 ? 1 : length);
  if (filtered == NULL)
    return PNG_NOMEM;
  for (size_t y = 0; y < height; y++) {
    const uint8_t *row = src + y * stride;
    const uint8_t *previous = y == 0 ? NULL : row - stride;
    unsigned best_filter = 0;
    uint64_t best_score = UINT64_MAX;
    for (unsigned filter = 0; filter <= 4; filter++) {
      uint64_t score = 0;
      for (size_t i = 0; i < stride; i++) {
        unsigned left = i >= channels ? row[i - channels] : 0;
        unsigned up = previous == NULL ? 0 : previous[i];
        unsigned upper_left =
            previous != NULL && i >= channels ? previous[i - channels] : 0;
        unsigned value = filter_byte(filter, row[i], left, up, upper_left);
        score += value < 128 ? value : 256 - value;
      }
      if (score < best_score) {
        best_score = score;
        best_filter = filter;
      }
    }
    uint8_t *out = filtered + y * (stride + 1);
    out[0] = (uint8_t)best_filter;
    for (size_t i = 0; i < stride; i++) {
      unsigned left = i >= channels ? row[i - channels] : 0;
      unsigned up = previous == NULL ? 0 : previous[i];
      unsigned upper_left =
          previous != NULL && i >= channels ? previous[i - channels] : 0;
      out[i + 1] =
          (uint8_t)filter_byte(best_filter, row[i], left, up, upper_left);
    }
  }
  *result = filtered;
  *result_len = length;
  return PNG_OK;
}

static png_status encode_png(int fd, const uint8_t *src, size_t src_len,
                             size_t width, size_t height, unsigned channels) {
  if (width == 0 || height == 0 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      width > SIZE_MAX / height || width * height > SIZE_MAX / channels ||
      width * height * channels != src_len || width > UINT32_MAX ||
      height > UINT32_MAX)
    return PNG_SIZE;
  uint8_t *filtered = NULL;
  size_t filtered_len;
  png_status status =
      filter_image(src, width, height, channels, &filtered, &filtered_len);
  if (status != PNG_OK)
    return status;
  uint8_t *raw = NULL;
  uint32_t crc;
  nx_io_result compressed =
      nx_io_deflate_raw(NULL, 0, filtered, filtered_len, -1, &raw, &crc);
  if (compressed.status != NX_IO_OK) {
    free(filtered);
    return compressed.status == NX_IO_NOMEM ? PNG_NOMEM : PNG_SIZE;
  }
  uint32_t adler = nx_io_adler32(filtered, filtered_len);
  free(filtered);
  if (compressed.output_size > SIZE_MAX - 6) {
    free(raw);
    return PNG_SIZE;
  }
  uint8_t *zlib = realloc(raw, compressed.output_size + 6);
  if (zlib == NULL) {
    free(raw);
    return PNG_NOMEM;
  }
  memmove(zlib + 2, zlib, compressed.output_size);
  zlib[0] = 0x78;
  zlib[1] = 0x01;
  write_be32(zlib + compressed.output_size + 2, adler);
  size_t zlib_len = compressed.output_size + 6;

  status = write_all(fd, png_signature, sizeof(png_signature));
  uint8_t ihdr[13];
  write_be32(ihdr, (uint32_t)width);
  write_be32(ihdr + 4, (uint32_t)height);
  ihdr[8] = 8;
  ihdr[9] = channels == 1 ? 0 : channels == 3 ? 2 : 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  if (status == PNG_OK)
    status = write_chunk(fd, "IHDR", ihdr, sizeof(ihdr));
  size_t off = 0;
  while (status == PNG_OK && off < zlib_len) {
    size_t chunk = zlib_len - off;
    if (chunk > PNG_IDAT_CHUNK)
      chunk = PNG_IDAT_CHUNK;
    status = write_chunk(fd, "IDAT", zlib + off, chunk);
    off += chunk;
  }
  if (status == PNG_OK)
    status = write_chunk(fd, "IEND", NULL, 0);
  free(zlib);
  return status;
}

#ifndef NX_IO_CODEC_NO_OCAML
static void checked_bytes(value vbuf, const uint8_t **data, size_t *len) {
  struct caml_ba_array *array = Caml_ba_array_val(vbuf);
  if ((array->flags & CAML_BA_KIND_MASK) != CAML_BA_UINT8)
    caml_invalid_argument("Nx_io PNG: expected a uint8 Bigarray");
  *data = Caml_ba_data_val(vbuf);
  *len = caml_ba_byte_size(array);
}

CAMLprim value caml_nx_io_png_probe(value vsrc) {
  CAMLparam1(vsrc);
  CAMLlocal1(vresult);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  png_info info;
  caml_release_runtime_system();
  png_status status = parse_png(src, src_len, &info);
  caml_acquire_runtime_system();
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  if (info.width > (size_t)Max_long || info.height > (size_t)Max_long)
    caml_failwith(png_message(PNG_SIZE));
  vresult = caml_alloc_tuple(2);
  Store_field(vresult, 0, Val_long(info.width));
  Store_field(vresult, 1, Val_long(info.height));
  CAMLreturn(vresult);
}

CAMLprim value caml_nx_io_png_decode(value vsrc, value vdst, value vgrayscale) {
  CAMLparam3(vsrc, vdst, vgrayscale);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  const uint8_t *dst_const;
  size_t dst_len;
  checked_bytes(vdst, &dst_const, &dst_len);
  uint8_t *dst = (uint8_t *)dst_const;
  int grayscale = Bool_val(vgrayscale);
  caml_release_runtime_system();
  png_status status = decode_png(src, src_len, dst, dst_len, grayscale);
  caml_acquire_runtime_system();
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_io_png_encode(value vfd, value vsrc, value vwidth,
                                     value vheight, value vchannels) {
  CAMLparam5(vfd, vsrc, vwidth, vheight, vchannels);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  intnat width_i = Long_val(vwidth);
  intnat height_i = Long_val(vheight);
  intnat channels_i = Long_val(vchannels);
  if (width_i <= 0 || height_i <= 0 || channels_i <= 0)
    caml_invalid_argument("Nx_io PNG: invalid image dimensions");
  int fd = Int_val(vfd);
  caml_release_runtime_system();
  png_status status = encode_png(fd, src, src_len, (size_t)width_i,
                                 (size_t)height_i, (unsigned)channels_i);
  int saved_errno = errno;
  caml_acquire_runtime_system();
  if (status == PNG_SYSTEM)
    unix_error(saved_errno == 0 ? EIO : saved_errno, "write", Nothing);
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  CAMLreturn(Val_unit);
}
#endif
