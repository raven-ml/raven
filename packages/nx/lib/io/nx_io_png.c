/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Static PNG decoding and encoding following the W3C PNG specification: the
  chunk stream, scanline filters and pixel formats. The caller checks chunk
  CRC-32s, and compresses and decompresses image data with lz's zlib
  streams. The decoder unfilters the decompressed scanlines in place and
  writes pixels directly into the Nx-owned destination Bigarray. The encoder
  filters one row at a time, which the caller compresses.
  --------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
  PNG_SHORT
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
  /* The tRNS chunk: a palette entry's alpha, or the one grey or RGB sample
     value that is transparent, its bits above the depth masked. */
  int has_transparency;
  uint8_t palette_alpha[256];
  unsigned transparent[3];
  size_t idat_size;
  /* The offset after the IEND chunk, past which bytes are ignored. */
  size_t end;
} png_info;

static const uint8_t png_signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};

/* The largest four-byte unsigned integer PNG allows, a side's bound (PNG,
   7.1). */
#define PNG_MAX_INTEGER 0x7fffffffu

/* The most bytes a byte of a zlib stream inflates to: a deflate match of 258
   bytes takes at least two bits, a code of one bit for its length and one
   for its distance (RFC 1951, 3.2.5 and 3.2.7). */
#define INFLATE_RATIO 1032

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
  case PNG_SHORT:
    return "PNG image data too short for its dimensions";
  }
  return "unknown PNG error";
}

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

/* The channels of a pixel as the file stores it. */
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
      if (width == 0 || height == 0 || width > PNG_MAX_INTEGER ||
          height > PNG_MAX_INTEGER || !valid_depth(color, depth) ||
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
      /* PNG-3 11.3.1.1: a grey or RGB value is two bytes whatever the depth,
         and a decoder masks the bits above it. Palette entries past the
         chunk's end are opaque. */
      if (info->color == 3) {
        memset(info->palette_alpha, 255, sizeof(info->palette_alpha));
        memcpy(info->palette_alpha, data, length);
      } else {
        unsigned mask = info->depth == 16 ? 0xffffu : (1u << info->depth) - 1u;
        for (unsigned i = 0; i < length / 2; i++)
          info->transparent[i] =
              (((unsigned)data[2 * i] << 8) | data[2 * i + 1]) & mask;
      }
      info->has_transparency = 1;
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
      /* Bytes after IEND are ignored, as libpng ignores them. */
      seen_end = 1;
      info->end = off + 12;
      break;
    } else if ((type[0] & 0x20u) == 0) {
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

/* Copies the IDAT chunks of [src], whose IEND chunk ends at [len], into [dst],
   which they must fill: [src] may be a mapped file that changed since it was
   parsed. */
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

/* A pass of an image: the column and row of its first pixel, and the steps
   between its pixels. */
typedef struct {
  uint8_t x;
  uint8_t y;
  uint8_t dx;
  uint8_t dy;
} png_pass;

/* The seven passes of Adam7 interlacing (PNG-3, 8.2). */
static const png_pass adam7[7] = {{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8},
                                  {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2},
                                  {0, 1, 1, 2}};

/* The one pass of an image without interlacing. */
static const png_pass whole = {0, 0, 1, 1};

static const png_pass *pass_of(const png_info *info, unsigned pass) {
  return info->interlace ? &adam7[pass] : &whole;
}

static size_t pass_extent(size_t size, unsigned start, unsigned step) {
  if (size <= start)
    return 0;
  return (size - start + step - 1) / step;
}

static void pass_geometry(const png_info *info, unsigned pass, size_t *width,
                          size_t *height) {
  const png_pass *p = pass_of(info, pass);
  *width = pass_extent(info->width, p->x, p->dx);
  *height = pass_extent(info->height, p->y, p->dy);
}

static png_status scanline_size(const png_info *info, size_t *total) {
  unsigned passes = info->interlace ? 7 : 1;
  size_t sum = 0;
  for (unsigned pass = 0; pass < passes; pass++) {
    size_t width;
    size_t height;
    pass_geometry(info, pass, &width, &height);
    if (width == 0 || height == 0)
      continue;
    if (width > (SIZE_MAX - 7) / (info->channels * info->depth))
      return PNG_SIZE;
    size_t row = (width * info->channels * info->depth + 7) / 8;
    if (row == SIZE_MAX || height > (SIZE_MAX - sum) / (row + 1))
      return PNG_SIZE;
    sum += height * (row + 1);
  }
  *total = sum;
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

/* The channels of a decoded pixel: the file's, a palette's three colours, and
   an alpha channel where a tRNS chunk gives transparency. */
static unsigned output_channels(const png_info *info) {
  unsigned colours = info->color == 3 ? 3 : info->channels;
  return colours + (info->has_transparency ? 1u : 0u);
}

/* Writes the pixels of the unfiltered scanline [line], row [row] of pass
   [pass], into [dst], [output_channels (info)] samples a pixel. A sample is
   compared with a tRNS value before it is scaled to 8 bits. */
static png_status write_row(const png_info *info, unsigned pass, size_t row,
                            size_t pass_width, const uint8_t *line,
                            uint8_t *dst) {
  const png_pass *p = pass_of(info, pass);
  unsigned out = output_channels(info);
  size_t y = p->y + row * p->dy;
  for (size_t x_pass = 0; x_pass < pass_width; x_pass++) {
    size_t x = p->x + x_pass * p->dx;
    uint8_t *pixel = dst + (y * info->width + x) * out;
    if (info->color == 3) {
      unsigned index = sample(line, x_pass, info->depth);
      if (index >= info->palette_size)
        return PNG_PALETTE;
      memcpy(pixel, info->palette + index * 3, 3);
      if (info->has_transparency)
        pixel[3] = info->palette_alpha[index];
      continue;
    }
    size_t component = x_pass * info->channels;
    int transparent = info->has_transparency;
    for (unsigned c = 0; c < info->channels; c++) {
      unsigned value = sample(line, component + c, info->depth);
      transparent = transparent && value == info->transparent[c];
      pixel[c] = scale_sample(value, info->depth);
    }
    if (info->has_transparency)
      pixel[info->channels] = transparent ? 0 : 255;
  }
  return PNG_OK;
}

/* Unfilters the scanlines of [filtered] in place, pass by pass, and writes
   their pixels into [dst]. */
static png_status decode_png(const png_info *info, uint8_t *filtered,
                             size_t filtered_len, uint8_t *dst,
                             size_t dst_len) {
  unsigned out = output_channels(info);
  if (info->width > SIZE_MAX / info->height ||
      info->width * info->height > SIZE_MAX / out ||
      info->width * info->height * out != dst_len)
    return PNG_SIZE;
  size_t total;
  png_status status = scanline_size(info, &total);
  if (status != PNG_OK)
    return status;
  if (total != filtered_len)
    return PNG_SIZE;
  unsigned bpp = (info->channels * info->depth + 7) / 8;
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
        status = write_row(info, pass, row, width, line, dst);
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

/* Writes into [out] the filter type, then the [stride] filtered bytes, of the
   8-bit scanline [row] of [channels] samples a pixel, below [previous], or
   NULL for the first row. The type is the one whose bytes, read as signed,
   have the least sum of magnitudes, the heuristic PNG-3 suggests (12.8). */
static void filter_row(const uint8_t *row, const uint8_t *previous,
                       size_t stride, unsigned channels, uint8_t *out) {
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

/* The bytes of [v], a one-dimensional uint8 bigarray, as its external's type
   states. */
static void bytes_of(value v, const uint8_t **data, size_t *len) {
  *data = Caml_ba_data_val(v);
  *len = Caml_ba_array_val(v)->dim[0];
}

CAMLprim value caml_nx_io_png_probe(value vsrc) {
  CAMLparam1(vsrc);
  CAMLlocal1(vresult);
  const uint8_t *src;
  size_t src_len;
  bytes_of(vsrc, &src, &src_len);
  png_info info;
  caml_release_runtime_system();
  png_status status = parse_png(src, src_len, &info);
  caml_acquire_runtime_system();
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  if (info.width > (size_t)Max_long || info.height > (size_t)Max_long)
    caml_failwith(png_message(PNG_SIZE));
  /* Image data that cannot inflate to the scanlines refuses the image, so
     that the buffer they inflate into holds at most INFLATE_RATIO times the
     file's image data, whatever its header states. */
  size_t filtered;
  status = scanline_size(&info, &filtered);
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  if (info.idat_size < SIZE_MAX / INFLATE_RATIO &&
      filtered > (info.idat_size + 1) * INFLATE_RATIO)
    caml_failwith(png_message(PNG_SHORT));
  vresult = caml_alloc_tuple(3);
  Store_field(vresult, 0, Val_long(info.width));
  Store_field(vresult, 1, Val_long(info.height));
  Store_field(vresult, 2, Val_long(output_channels(&info)));
  CAMLreturn(vresult);
}

/* The image data of [src], a probed PNG stream, and the length of its
   decompressed scanlines. */
CAMLprim value caml_nx_io_png_idat(value vsrc) {
  CAMLparam1(vsrc);
  CAMLlocal2(vidat, vresult);
  const uint8_t *src;
  size_t src_len;
  bytes_of(vsrc, &src, &src_len);
  png_info info;
  size_t filtered;
  png_status status = parse_png(src, src_len, &info);
  if (status == PNG_OK)
    status = scanline_size(&info, &filtered);
  if (status == PNG_OK && filtered > (size_t)Max_long)
    status = PNG_SIZE;
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  vidat = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                             (intnat)info.idat_size);
  status = copy_idat(Caml_ba_data_val(vsrc), info.end, Caml_ba_data_val(vidat),
                     info.idat_size);
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  vresult = caml_alloc_tuple(2);
  Store_field(vresult, 0, vidat);
  Store_field(vresult, 1, Val_long(filtered));
  CAMLreturn(vresult);
}

CAMLprim value caml_nx_io_png_decode(value vsrc, value vfiltered, value vdst) {
  CAMLparam3(vsrc, vfiltered, vdst);
  const uint8_t *src;
  size_t src_len;
  bytes_of(vsrc, &src, &src_len);
  const uint8_t *filtered;
  size_t filtered_len;
  bytes_of(vfiltered, &filtered, &filtered_len);
  const uint8_t *dst;
  size_t dst_len;
  bytes_of(vdst, &dst, &dst_len);
  png_info info;
  caml_release_runtime_system();
  png_status status = parse_png(src, src_len, &info);
  if (status == PNG_OK)
    status = decode_png(&info, (uint8_t *)filtered, filtered_len,
                        (uint8_t *)dst, dst_len);
  caml_acquire_runtime_system();
  if (status != PNG_OK)
    caml_failwith(png_message(status));
  CAMLreturn(Val_unit);
}

/* Filters row [vy] of the 8-bit image [vsrc], of [vchannels] samples a pixel,
   into [vrow], whose length is one more than the row's. It allocates nothing
   and keeps the runtime lock. The caller makes [vy] a row of [vsrc]. */
CAMLprim value caml_nx_io_png_filter_row(value vsrc, value vrow, value vy,
                                         value vchannels) {
  size_t stride = Caml_ba_array_val(vrow)->dim[0] - 1;
  size_t y = (size_t)Long_val(vy);
  const uint8_t *row = (const uint8_t *)Caml_ba_data_val(vsrc) + y * stride;
  filter_row(row, y == 0 ? NULL : row - stride, stride,
             (unsigned)Long_val(vchannels), Caml_ba_data_val(vrow));
  return Val_unit;
}
