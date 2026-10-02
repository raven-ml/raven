/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Static PNG decoding and encoding following the W3C PNG specification: the
  chunk stream, scanline filters and pixel formats. The caller checks chunk
  CRC-32s, and compresses and decompresses image data with compress.deflate's
  zlib streams: the decoder unfilters the decompressed scanlines in place and writes pixels
  directly into the Nx-owned destination Bigarray, and the encoder returns the
  filtered scanlines.
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
  PNG_ORDER,
  PNG_IHDR,
  PNG_UNSUPPORTED,
  PNG_PALETTE,
  PNG_FILTER,
  PNG_SIZE,
  PNG_NOMEM
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
  case PNG_ORDER:
    return "invalid PNG chunk ordering";
  case PNG_IHDR:
    return "invalid PNG image header";
  case PNG_UNSUPPORTED:
    return "unsupported PNG feature";
  case PNG_PALETTE:
    return "invalid PNG palette";
  case PNG_FILTER:
    return "invalid PNG scanline filter";
  case PNG_SIZE:
    return "PNG dimensions are too large";
  case PNG_NOMEM:
    return "PNG codec allocation failed";
  }
  return "unknown PNG error";
}
#endif

static uint32_t read_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8) | (uint32_t)p[3];
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

/* Reads the chunk stream [src], whose chunk CRC-32s the caller checks. */
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

/* Copies the IDAT chunks of [src] into [dst], which they must fill: [src] may
   be a mapped file that changed since it was parsed. */
static png_status copy_idat(const uint8_t *src, size_t len, uint8_t *dst,
                            size_t dst_len) {
  size_t off = 8;
  size_t at = 0;
  while (len - off >= 12) {
    size_t length = read_be32(src + off);
    if (length > len - off - 12)
      return PNG_TRUNCATED;
    const uint8_t *type = src + off + 4;
    if (type_is(type, "IDAT")) {
      if (length > dst_len - at)
        return PNG_SIZE;
      memcpy(dst + at, type + 4, length);
      at += length;
    }
    off += length + 12;
  }
  return at == dst_len ? PNG_OK : PNG_SIZE;
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

/* Writes the pixels of the unfiltered scanline [line], row [row] of pass
   [pass], into [dst]. */
static png_status write_row(const png_info *info, unsigned pass, size_t row,
                            size_t pass_width, const uint8_t *line,
                            uint8_t *dst, unsigned output_channels) {
  static const uint8_t start_x[7] = {0, 4, 0, 2, 0, 1, 0};
  static const uint8_t start_y[7] = {0, 0, 4, 0, 2, 0, 1};
  static const uint8_t step_x[7] = {8, 8, 4, 4, 2, 2, 1};
  static const uint8_t step_y[7] = {8, 8, 8, 4, 4, 2, 2};
  unsigned sx = info->interlace ? start_x[pass] : 0;
  unsigned sy = info->interlace ? start_y[pass] : 0;
  unsigned dx = info->interlace ? step_x[pass] : 1;
  unsigned dy = info->interlace ? step_y[pass] : 1;
  size_t y = sy + row * dy;
  for (size_t x_pass = 0; x_pass < pass_width; x_pass++) {
    size_t component = x_pass * info->channels;
    unsigned r;
    unsigned g;
    unsigned b;
    if (info->color == 0 || info->color == 4) {
      uint8_t gray =
          scale_sample(sample(line, component, info->depth), info->depth);
      r = g = b = gray;
    } else if (info->color == 3) {
      unsigned index = sample(line, x_pass, info->depth);
      if (index >= info->palette_size)
        return PNG_PALETTE;
      r = info->palette[index * 3];
      g = info->palette[index * 3 + 1];
      b = info->palette[index * 3 + 2];
    } else {
      r = scale_sample(sample(line, component, info->depth), info->depth);
      g = scale_sample(sample(line, component + 1, info->depth), info->depth);
      b = scale_sample(sample(line, component + 2, info->depth), info->depth);
    }
    size_t x = sx + x_pass * dx;
    size_t at = (y * info->width + x) * output_channels;
    if (output_channels == 1) {
      dst[at] = (uint8_t)((77u * r + 150u * g + 29u * b + 128u) >> 8);
    } else {
      dst[at] = (uint8_t)r;
      dst[at + 1] = (uint8_t)g;
      dst[at + 2] = (uint8_t)b;
    }
  }
  return PNG_OK;
}

/* Unfilters the scanlines of [filtered] in place, pass by pass, and writes
   their pixels into [dst]. */
static png_status decode_png(const png_info *info, uint8_t *filtered,
                             size_t filtered_len, uint8_t *dst, size_t dst_len,
                             int grayscale) {
  unsigned output_channels = grayscale ? 1 : 3;
  if (info->width > SIZE_MAX / info->height ||
      info->width * info->height > SIZE_MAX / output_channels ||
      info->width * info->height * output_channels != dst_len)
    return PNG_SIZE;
  size_t total;
  size_t max_row;
  png_status status = scanline_size(info, &total, &max_row);
  if (status != PNG_OK)
    return status;
  if (total != filtered_len)
    return PNG_SIZE;
  unsigned bpp = (info->channels * info->depth + 7) / 8;
  if (bpp == 0)
    bpp = 1;
  unsigned passes = info->interlace ? 7 : 1;
  uint8_t *scanline = filtered;
  for (unsigned pass = 0; pass < passes; pass++) {
    size_t width;
    size_t height;
    pass_geometry(info, pass, &width, &height);
    if (width == 0 || height == 0)
      continue;
    size_t row_size = (width * info->channels * info->depth + 7) / 8;
    const uint8_t *previous = NULL;
    for (size_t row = 0; row < height; row++) {
      uint8_t *line = scanline + 1;
      status = unfilter(line, previous, row_size, bpp, scanline[0]);
      if (status == PNG_OK)
        status = write_row(info, pass, row, width, line, dst, output_channels);
      if (status != PNG_OK)
        return status;
      previous = line;
      scanline += row_size + 1;
    }
  }
  return PNG_OK;
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

/* The image data of [src], a probed PNG stream, and the length of its
   decompressed scanlines. */
CAMLprim value caml_nx_io_png_idat(value vsrc) {
  CAMLparam1(vsrc);
  CAMLlocal2(vidat, vresult);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  png_info info;
  size_t filtered;
  size_t max_row;
  png_status status = parse_png(src, src_len, &info);
  if (status == PNG_OK)
    status = scanline_size(&info, &filtered, &max_row);
  if (status == PNG_OK && filtered > (size_t)Max_long)
    status = PNG_SIZE;
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  vidat = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                             (intnat)info.idat_size);
  status = copy_idat(Caml_ba_data_val(vsrc), src_len, Caml_ba_data_val(vidat), info.idat_size);
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  vresult = caml_alloc_tuple(2);
  Store_field(vresult, 0, vidat);
  Store_field(vresult, 1, Val_long(filtered));
  CAMLreturn(vresult);
}

CAMLprim value caml_nx_io_png_decode(value vsrc, value vfiltered, value vdst,
                                     value vgrayscale) {
  CAMLparam4(vsrc, vfiltered, vdst, vgrayscale);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  const uint8_t *filtered;
  size_t filtered_len;
  checked_bytes(vfiltered, &filtered, &filtered_len);
  const uint8_t *dst;
  size_t dst_len;
  checked_bytes(vdst, &dst, &dst_len);
  int grayscale = Bool_val(vgrayscale);
  png_info info;
  caml_release_runtime_system();
  png_status status = parse_png(src, src_len, &info);
  if (status == PNG_OK)
    status = decode_png(&info, (uint8_t *)filtered, filtered_len,
                        (uint8_t *)dst, dst_len, grayscale);
  caml_acquire_runtime_system();
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  CAMLreturn(Val_unit);
}

/* The filtered scanlines of the 8-bit image [src]. */
CAMLprim value caml_nx_io_png_filter(value vsrc, value vwidth, value vheight,
                                     value vchannels) {
  CAMLparam4(vsrc, vwidth, vheight, vchannels);
  CAMLlocal1(vresult);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  size_t width = (size_t)Long_val(vwidth);
  size_t height = (size_t)Long_val(vheight);
  unsigned channels = (unsigned)Long_val(vchannels);
  if (width == 0 || height == 0 ||
      (channels != 1 && channels != 3 && channels != 4) ||
      width > SIZE_MAX / height || width * height > SIZE_MAX / channels ||
      width * height * channels != src_len || width > UINT32_MAX ||
      height > UINT32_MAX)
    caml_failwith(png_message(PNG_SIZE));
  uint8_t *filtered = NULL;
  size_t filtered_len;
  caml_release_runtime_system();
  png_status status =
      filter_image(src, width, height, channels, &filtered, &filtered_len);
  caml_acquire_runtime_system();
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  vresult = caml_alloc_initialized_string(filtered_len, (const char *)filtered);
  free(filtered);
  CAMLreturn(vresult);
}
#endif
