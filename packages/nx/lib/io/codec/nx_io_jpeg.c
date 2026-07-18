/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  8-bit Huffman JPEG decoder for sequential and progressive DCT frames. The
  implementation follows ITU-T T.81 and keeps marker parsing, entropy decode,
  coefficient storage, inverse transform, resampling, and color conversion as
  separate checked stages.
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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define JPEG_MAX_COMPONENTS 4
#define JPEG_MAX_TABLES 4

typedef enum {
  JPEG_OK = 0,
  JPEG_TRUNCATED,
  JPEG_MARKER,
  JPEG_UNSUPPORTED,
  JPEG_FRAME,
  JPEG_TABLE,
  JPEG_SCAN,
  JPEG_HUFFMAN,
  JPEG_RESTART,
  JPEG_SIZE,
  JPEG_NOMEM,
  JPEG_SYSTEM
} jpeg_status;

typedef struct {
  uint8_t values[256];
  int32_t min_code[17];
  int32_t max_code[18];
  uint16_t value_offset[17];
  int valid;
} jpeg_huffman;

typedef struct {
  uint8_t id;
  unsigned h;
  unsigned v;
  unsigned quant;
  unsigned dc_table;
  unsigned ac_table;
  size_t blocks_x;
  size_t blocks_y;
  size_t allocated_x;
  size_t allocated_y;
  int32_t *coefficients;
  uint8_t *samples;
  size_t sample_stride;
  int dc_predictor;
} jpeg_component;

typedef struct {
  const uint8_t *src;
  size_t len;
  size_t width;
  size_t height;
  unsigned component_count;
  unsigned max_h;
  unsigned max_v;
  size_t mcu_columns;
  size_t mcu_rows;
  int progressive;
  int saw_scan;
  int jfif;
  int adobe_transform;
  unsigned restart_interval;
  uint16_t quant[JPEG_MAX_TABLES][64];
  uint8_t quant_valid[JPEG_MAX_TABLES];
  jpeg_huffman dc[JPEG_MAX_TABLES];
  jpeg_huffman ac[JPEG_MAX_TABLES];
  jpeg_component components[JPEG_MAX_COMPONENTS];
} jpeg_decoder;

typedef struct {
  const uint8_t *src;
  size_t len;
  size_t pos;
  uint64_t bits;
  unsigned nbits;
  int marker;
  size_t marker_offset;
} entropy_reader;

static const uint8_t zigzag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

static const int16_t idct_matrix[8][8] = {
    {5793, 8035, 7568, 6811, 5793, 4551, 3135, 1598},
    {5793, 6811, 3135, -1598, -5793, -8035, -7568, -4551},
    {5793, 4551, -3135, -8035, -5793, 1598, 7568, 6811},
    {5793, 1598, -7568, -4551, 5793, 6811, -3135, -8035},
    {5793, -1598, -7568, 4551, 5793, -6811, -3135, 8035},
    {5793, -4551, -3135, 8035, -5793, -1598, 7568, -6811},
    {5793, -6811, 3135, 1598, -5793, 8035, -7568, 4551},
    {5793, -8035, 7568, -6811, 5793, -4551, 3135, -1598}};

#ifndef NX_IO_CODEC_NO_OCAML
static const char *jpeg_message(jpeg_status status) {
  switch (status) {
  case JPEG_OK:
    return "ok";
  case JPEG_TRUNCATED:
    return "truncated JPEG stream";
  case JPEG_MARKER:
    return "invalid JPEG marker stream";
  case JPEG_UNSUPPORTED:
    return "unsupported JPEG feature";
  case JPEG_FRAME:
    return "invalid JPEG frame";
  case JPEG_TABLE:
    return "invalid JPEG table";
  case JPEG_SCAN:
    return "invalid JPEG scan";
  case JPEG_HUFFMAN:
    return "invalid JPEG Huffman data";
  case JPEG_RESTART:
    return "invalid JPEG restart sequence";
  case JPEG_SIZE:
    return "JPEG dimensions are too large";
  case JPEG_NOMEM:
    return "JPEG codec allocation failed";
  case JPEG_SYSTEM:
    return "JPEG output error";
  }
  return "unknown JPEG error";
}
#endif

static uint16_t be16(const uint8_t *p) {
  return (uint16_t)(((unsigned)p[0] << 8) | p[1]);
}

static int checked_product(size_t a, size_t b, size_t *result) {
  if (b != 0 && a > SIZE_MAX / b)
    return 0;
  *result = a * b;
  return 1;
}

static jpeg_status next_marker(const uint8_t *src, size_t len, size_t *off,
                               unsigned *marker) {
  if (*off >= len || src[*off] != 0xff)
    return JPEG_MARKER;
  while (*off < len && src[*off] == 0xff)
    (*off)++;
  if (*off >= len || src[*off] == 0x00)
    return JPEG_MARKER;
  *marker = src[(*off)++];
  return JPEG_OK;
}

static jpeg_status segment(const uint8_t *src, size_t len, size_t *off,
                           const uint8_t **data, size_t *data_len) {
  if (*off > len || len - *off < 2)
    return JPEG_TRUNCATED;
  unsigned size = be16(src + *off);
  if (size < 2 || size > len - *off)
    return JPEG_TRUNCATED;
  *data = src + *off + 2;
  *data_len = size - 2;
  *off += size;
  return JPEG_OK;
}

static jpeg_status build_huffman(jpeg_huffman *table, const uint8_t counts[16],
                                 const uint8_t *values, size_t value_count) {
  memset(table, 0, sizeof(*table));
  unsigned count = 0;
  int32_t code = 0;
  for (unsigned length = 1; length <= 16; length++) {
    unsigned amount = counts[length - 1];
    if (count + amount > value_count)
      return JPEG_TABLE;
    table->value_offset[length] = (uint16_t)count;
    if (amount == 0) {
      table->min_code[length] = -1;
      table->max_code[length] = -1;
    } else {
      table->min_code[length] = code;
      code += (int32_t)amount - 1;
      table->max_code[length] = code;
      code++;
      count += amount;
    }
    if (code > (1 << length))
      return JPEG_TABLE;
    code <<= 1;
  }
  if (count != value_count || count == 0)
    return JPEG_TABLE;
  memcpy(table->values, values, count);
  table->max_code[17] = INT32_MAX;
  table->valid = 1;
  return JPEG_OK;
}

static jpeg_status parse_dqt(jpeg_decoder *decoder, const uint8_t *data,
                             size_t len) {
  size_t off = 0;
  while (off < len) {
    unsigned info = data[off++];
    unsigned precision = info >> 4;
    unsigned table = info & 15u;
    if (table >= JPEG_MAX_TABLES || precision > 1)
      return JPEG_TABLE;
    size_t bytes = precision ? 128 : 64;
    if (bytes > len - off)
      return JPEG_TRUNCATED;
    for (unsigned i = 0; i < 64; i++) {
      unsigned value = precision ? be16(data + off + i * 2) : data[off + i];
      if (value == 0)
        return JPEG_TABLE;
      decoder->quant[table][zigzag[i]] = (uint16_t)value;
    }
    decoder->quant_valid[table] = 1;
    off += bytes;
  }
  return off == len ? JPEG_OK : JPEG_TRUNCATED;
}

static jpeg_status parse_dht(jpeg_decoder *decoder, const uint8_t *data,
                             size_t len) {
  size_t off = 0;
  while (off < len) {
    if (len - off < 17)
      return JPEG_TRUNCATED;
    unsigned info = data[off++];
    unsigned class_ = info >> 4;
    unsigned index = info & 15u;
    if (class_ > 1 || index >= JPEG_MAX_TABLES)
      return JPEG_TABLE;
    const uint8_t *counts = data + off;
    off += 16;
    size_t count = 0;
    for (unsigned i = 0; i < 16; i++)
      count += counts[i];
    if (count > 256 || count > len - off)
      return JPEG_TRUNCATED;
    jpeg_huffman *table =
        class_ == 0 ? &decoder->dc[index] : &decoder->ac[index];
    jpeg_status status = build_huffman(table, counts, data + off, count);
    if (status != JPEG_OK)
      return status;
    off += count;
  }
  return JPEG_OK;
}

static void free_decoder(jpeg_decoder *decoder) {
  for (unsigned i = 0; i < JPEG_MAX_COMPONENTS; i++) {
    free(decoder->components[i].coefficients);
    free(decoder->components[i].samples);
    decoder->components[i].coefficients = NULL;
    decoder->components[i].samples = NULL;
  }
}

static jpeg_status parse_frame(jpeg_decoder *decoder, const uint8_t *data,
                               size_t len, int progressive) {
  if (decoder->component_count != 0 || len < 6 || data[0] != 8)
    return JPEG_FRAME;
  size_t height = be16(data + 1);
  size_t width = be16(data + 3);
  unsigned count = data[5];
  if (width == 0 || height == 0 || (count != 1 && count != 3 && count != 4) ||
      len != 6 + count * 3)
    return JPEG_FRAME;
  decoder->width = width;
  decoder->height = height;
  decoder->component_count = count;
  decoder->progressive = progressive;
  for (unsigned i = 0; i < count; i++) {
    jpeg_component *component = &decoder->components[i];
    component->id = data[6 + i * 3];
    component->h = data[7 + i * 3] >> 4;
    component->v = data[7 + i * 3] & 15u;
    component->quant = data[8 + i * 3];
    if (component->h == 0 || component->v == 0 || component->h > 4 ||
        component->v > 4 || component->quant >= JPEG_MAX_TABLES)
      return JPEG_FRAME;
    for (unsigned previous = 0; previous < i; previous++)
      if (decoder->components[previous].id == component->id)
        return JPEG_FRAME;
    if (component->h > decoder->max_h)
      decoder->max_h = component->h;
    if (component->v > decoder->max_v)
      decoder->max_v = component->v;
  }
  size_t mcu_width = decoder->max_h * 8;
  size_t mcu_height = decoder->max_v * 8;
  decoder->mcu_columns = (width + mcu_width - 1) / mcu_width;
  decoder->mcu_rows = (height + mcu_height - 1) / mcu_height;
  for (unsigned i = 0; i < count; i++) {
    jpeg_component *component = &decoder->components[i];
    component->blocks_x =
        (width * component->h + decoder->max_h * 8 - 1) / (decoder->max_h * 8);
    component->blocks_y =
        (height * component->v + decoder->max_v * 8 - 1) / (decoder->max_v * 8);
    component->allocated_x = decoder->mcu_columns * component->h;
    component->allocated_y = decoder->mcu_rows * component->v;
    size_t blocks;
    size_t coefficients;
    if (!checked_product(component->allocated_x, component->allocated_y,
                         &blocks) ||
        !checked_product(blocks, 64, &coefficients) || coefficients == 0 ||
        coefficients > SIZE_MAX / sizeof(int32_t))
      return JPEG_SIZE;
    component->coefficients = calloc(coefficients, sizeof(int32_t));
    if (component->coefficients == NULL)
      return JPEG_NOMEM;
  }
  return JPEG_OK;
}

static jpeg_component *find_component(jpeg_decoder *decoder, unsigned id) {
  for (unsigned i = 0; i < decoder->component_count; i++)
    if (decoder->components[i].id == id)
      return &decoder->components[i];
  return NULL;
}

static jpeg_status entropy_byte(entropy_reader *reader, uint8_t *byte) {
  if (reader->pos >= reader->len)
    return JPEG_TRUNCATED;
  size_t marker_offset = reader->pos;
  unsigned value = reader->src[reader->pos++];
  if (value != 0xff) {
    *byte = (uint8_t)value;
    return JPEG_OK;
  }
  while (reader->pos < reader->len && reader->src[reader->pos] == 0xff)
    reader->pos++;
  if (reader->pos >= reader->len)
    return JPEG_TRUNCATED;
  value = reader->src[reader->pos++];
  if (value == 0x00) {
    *byte = 0xff;
    return JPEG_OK;
  }
  reader->marker = (int)value;
  reader->marker_offset = marker_offset;
  return JPEG_SCAN;
}

static jpeg_status entropy_bits(entropy_reader *reader, unsigned count,
                                unsigned *value) {
  if (count > 16)
    return JPEG_SCAN;
  while (reader->nbits < count) {
    uint8_t byte;
    jpeg_status status = entropy_byte(reader, &byte);
    if (status != JPEG_OK)
      return status;
    reader->bits = (reader->bits << 8) | byte;
    reader->nbits += 8;
  }
  reader->nbits -= count;
  *value =
      count == 0
          ? 0
          : (unsigned)((reader->bits >> reader->nbits) & ((1u << count) - 1u));
  return JPEG_OK;
}

static jpeg_status huffman_symbol(entropy_reader *reader,
                                  const jpeg_huffman *table, unsigned *symbol) {
  if (!table->valid)
    return JPEG_TABLE;
  int32_t code = 0;
  for (unsigned length = 1; length <= 16; length++) {
    unsigned bit;
    jpeg_status status = entropy_bits(reader, 1, &bit);
    if (status != JPEG_OK)
      return status;
    code = (code << 1) | (int32_t)bit;
    if (table->max_code[length] >= 0 && code <= table->max_code[length]) {
      int32_t index =
          table->value_offset[length] + code - table->min_code[length];
      if (index < 0 || index >= 256)
        return JPEG_HUFFMAN;
      *symbol = table->values[index];
      return JPEG_OK;
    }
  }
  return JPEG_HUFFMAN;
}

static jpeg_status receive_extend(entropy_reader *reader, unsigned count,
                                  int *value) {
  if (count > 16)
    return JPEG_HUFFMAN;
  unsigned bits;
  jpeg_status status = entropy_bits(reader, count, &bits);
  if (status != JPEG_OK)
    return status;
  if (count != 0 && bits < (1u << (count - 1)))
    *value = (int)bits - (int)((1u << count) - 1u);
  else
    *value = (int)bits;
  return JPEG_OK;
}

static jpeg_status decode_sequential(entropy_reader *reader,
                                     jpeg_component *component,
                                     const jpeg_decoder *decoder,
                                     int32_t block[64]) {
  unsigned category;
  jpeg_status status =
      huffman_symbol(reader, &decoder->dc[component->dc_table], &category);
  if (status != JPEG_OK || category > 11)
    return status == JPEG_OK ? JPEG_HUFFMAN : status;
  int difference;
  status = receive_extend(reader, category, &difference);
  if (status != JPEG_OK)
    return status;
  component->dc_predictor += difference;
  block[0] = component->dc_predictor;
  unsigned k = 1;
  while (k < 64) {
    unsigned rs;
    status = huffman_symbol(reader, &decoder->ac[component->ac_table], &rs);
    if (status != JPEG_OK)
      return status;
    unsigned run = rs >> 4;
    unsigned size = rs & 15u;
    if (size == 0) {
      if (run == 0)
        break;
      if (run != 15)
        return JPEG_HUFFMAN;
      k += 16;
      continue;
    }
    if (size > 10 || run > 63 - k)
      return JPEG_HUFFMAN;
    k += run;
    int coefficient;
    status = receive_extend(reader, size, &coefficient);
    if (status != JPEG_OK)
      return status;
    block[zigzag[k++]] = coefficient;
  }
  return k <= 64 ? JPEG_OK : JPEG_HUFFMAN;
}

static void refine_nonzero(int32_t *coefficient, unsigned bit, int delta) {
  if (bit != 0 && ((unsigned)abs(*coefficient) & (unsigned)delta) == 0)
    *coefficient += *coefficient >= 0 ? delta : -delta;
}

static jpeg_status
decode_progressive(entropy_reader *reader, jpeg_component *component,
                   const jpeg_decoder *decoder, int32_t block[64],
                   unsigned spectral_start, unsigned spectral_end,
                   unsigned high, unsigned low, unsigned *eob_run) {
  int delta = 1 << low;
  if (spectral_start == 0) {
    if (high == 0) {
      unsigned category;
      jpeg_status status =
          huffman_symbol(reader, &decoder->dc[component->dc_table], &category);
      if (status != JPEG_OK || category > 11)
        return status == JPEG_OK ? JPEG_HUFFMAN : status;
      int difference;
      status = receive_extend(reader, category, &difference);
      if (status != JPEG_OK)
        return status;
      component->dc_predictor += difference;
      block[0] = component->dc_predictor * delta;
      return JPEG_OK;
    }
    unsigned bit;
    jpeg_status status = entropy_bits(reader, 1, &bit);
    if (status == JPEG_OK)
      refine_nonzero(&block[0], bit, delta);
    return status;
  }

  if (high == 0) {
    if (*eob_run != 0) {
      (*eob_run)--;
      return JPEG_OK;
    }
    unsigned k = spectral_start;
    while (k <= spectral_end) {
      unsigned rs;
      jpeg_status status =
          huffman_symbol(reader, &decoder->ac[component->ac_table], &rs);
      if (status != JPEG_OK)
        return status;
      unsigned run = rs >> 4;
      unsigned size = rs & 15u;
      if (size == 0) {
        if (run == 15) {
          k += 16;
          continue;
        }
        unsigned extra = 0;
        status = entropy_bits(reader, run, &extra);
        if (status != JPEG_OK)
          return status;
        *eob_run = (1u << run) + extra - 1u;
        break;
      }
      if (size > 10 || run > spectral_end - k)
        return JPEG_HUFFMAN;
      k += run;
      int value;
      status = receive_extend(reader, size, &value);
      if (status != JPEG_OK)
        return status;
      block[zigzag[k++]] = value * delta;
    }
    return k <= spectral_end + 1 ? JPEG_OK : JPEG_HUFFMAN;
  }

  unsigned k = spectral_start;
  if (*eob_run == 0) {
    while (k <= spectral_end) {
      unsigned rs;
      jpeg_status status =
          huffman_symbol(reader, &decoder->ac[component->ac_table], &rs);
      if (status != JPEG_OK)
        return status;
      unsigned run = rs >> 4;
      unsigned size = rs & 15u;
      int new_coefficient = 0;
      if (size == 0) {
        if (run < 15) {
          unsigned extra = 0;
          status = entropy_bits(reader, run, &extra);
          if (status != JPEG_OK)
            return status;
          *eob_run = (1u << run) + extra;
          break;
        }
        run = 16;
      } else {
        if (size != 1)
          return JPEG_HUFFMAN;
        int sign;
        status = receive_extend(reader, 1, &sign);
        if (status != JPEG_OK)
          return status;
        new_coefficient = sign * delta;
      }
      while (k <= spectral_end) {
        int32_t *coefficient = &block[zigzag[k]];
        if (*coefficient != 0) {
          unsigned bit;
          status = entropy_bits(reader, 1, &bit);
          if (status != JPEG_OK)
            return status;
          refine_nonzero(coefficient, bit, delta);
        } else if (run == 0) {
          break;
        } else {
          run--;
        }
        k++;
      }
      if (new_coefficient != 0) {
        if (k > spectral_end)
          return JPEG_HUFFMAN;
        block[zigzag[k++]] = new_coefficient;
      } else if (run != 0) {
        return JPEG_HUFFMAN;
      }
    }
  }
  if (*eob_run != 0) {
    while (k <= spectral_end) {
      int32_t *coefficient = &block[zigzag[k++]];
      if (*coefficient != 0) {
        unsigned bit;
        jpeg_status status = entropy_bits(reader, 1, &bit);
        if (status != JPEG_OK)
          return status;
        refine_nonzero(coefficient, bit, delta);
      }
    }
    (*eob_run)--;
  }
  return JPEG_OK;
}

static jpeg_status restart(entropy_reader *reader, unsigned expected) {
  reader->bits = 0;
  reader->nbits = 0;
  if (reader->marker == 0) {
    size_t marker_offset = reader->pos;
    if (reader->pos >= reader->len || reader->src[reader->pos++] != 0xff)
      return JPEG_RESTART;
    while (reader->pos < reader->len && reader->src[reader->pos] == 0xff)
      reader->pos++;
    if (reader->pos >= reader->len)
      return JPEG_TRUNCATED;
    reader->marker = reader->src[reader->pos++];
    reader->marker_offset = marker_offset;
  }
  if (reader->marker != (int)(0xd0u + expected))
    return JPEG_RESTART;
  reader->marker = 0;
  return JPEG_OK;
}

static jpeg_status finish_scan(entropy_reader *reader, size_t *next) {
  if (reader->marker != 0) {
    *next = reader->marker_offset;
    return JPEG_OK;
  }
  size_t off = reader->pos;
  if (off >= reader->len || reader->src[off] != 0xff)
    return JPEG_SCAN;
  *next = off;
  return JPEG_OK;
}

static jpeg_status decode_scan(jpeg_decoder *decoder, const uint8_t *header,
                               size_t header_len, size_t entropy_off,
                               size_t *next) {
  if (header_len < 4)
    return JPEG_SCAN;
  unsigned scan_count = header[0];
  if (scan_count == 0 || scan_count > decoder->component_count ||
      header_len != 1 + scan_count * 2 + 3)
    return JPEG_SCAN;
  jpeg_component *scan_components[JPEG_MAX_COMPONENTS];
  for (unsigned i = 0; i < scan_count; i++) {
    jpeg_component *component = find_component(decoder, header[1 + i * 2]);
    unsigned tables = header[2 + i * 2];
    if (component == NULL || (tables >> 4) >= JPEG_MAX_TABLES ||
        (tables & 15u) >= JPEG_MAX_TABLES)
      return JPEG_SCAN;
    for (unsigned previous = 0; previous < i; previous++)
      if (scan_components[previous] == component)
        return JPEG_SCAN;
    component->dc_table = tables >> 4;
    component->ac_table = tables & 15u;
    scan_components[i] = component;
  }
  unsigned spectral_start = header[1 + scan_count * 2];
  unsigned spectral_end = header[2 + scan_count * 2];
  unsigned approximation = header[3 + scan_count * 2];
  unsigned high = approximation >> 4;
  unsigned low = approximation & 15u;
  if ((!decoder->progressive &&
       (spectral_start != 0 || spectral_end != 63 || high != 0 || low != 0)) ||
      (decoder->progressive &&
       (spectral_start > spectral_end || spectral_end > 63 || high > 13 ||
        low > 13 || (spectral_start == 0 && spectral_end != 0) ||
        (spectral_start != 0 && scan_count != 1) ||
        (high != 0 && high != low + 1))))
    return JPEG_SCAN;

  entropy_reader reader = {decoder->src, decoder->len, entropy_off, 0, 0, 0, 0};
  for (unsigned i = 0; i < decoder->component_count; i++)
    decoder->components[i].dc_predictor = 0;
  unsigned eob_run = 0;
  size_t mcu_columns =
      scan_count == 1 ? scan_components[0]->blocks_x : decoder->mcu_columns;
  size_t mcu_rows =
      scan_count == 1 ? scan_components[0]->blocks_y : decoder->mcu_rows;
  size_t mcu_index = 0;
  unsigned expected_restart = 0;
  for (size_t mcu_y = 0; mcu_y < mcu_rows; mcu_y++) {
    for (size_t mcu_x = 0; mcu_x < mcu_columns; mcu_x++, mcu_index++) {
      if (decoder->restart_interval != 0 && mcu_index != 0 &&
          mcu_index % decoder->restart_interval == 0) {
        jpeg_status status = restart(&reader, expected_restart);
        if (status != JPEG_OK)
          return status;
        expected_restart = (expected_restart + 1) & 7u;
        for (unsigned i = 0; i < decoder->component_count; i++)
          decoder->components[i].dc_predictor = 0;
        eob_run = 0;
      }
      for (unsigned scan = 0; scan < scan_count; scan++) {
        jpeg_component *component = scan_components[scan];
        unsigned horizontal = scan_count == 1 ? 1 : component->h;
        unsigned vertical = scan_count == 1 ? 1 : component->v;
        for (unsigned by = 0; by < vertical; by++) {
          for (unsigned bx = 0; bx < horizontal; bx++) {
            size_t block_x =
                scan_count == 1 ? mcu_x : mcu_x * component->h + bx;
            size_t block_y =
                scan_count == 1 ? mcu_y : mcu_y * component->v + by;
            if (block_x >= component->allocated_x ||
                block_y >= component->allocated_y)
              return JPEG_SIZE;
            int32_t *block = component->coefficients +
                             (block_y * component->allocated_x + block_x) * 64;
            jpeg_status status =
                decoder->progressive
                    ? decode_progressive(&reader, component, decoder, block,
                                         spectral_start, spectral_end, high,
                                         low, &eob_run)
                    : decode_sequential(&reader, component, decoder, block);
            if (status != JPEG_OK)
              return status;
          }
        }
      }
    }
  }
  decoder->saw_scan = 1;
  return finish_scan(&reader, next);
}

static int clamp_sample(int64_t value) {
  if (value < 0)
    return 0;
  if (value > 255)
    return 255;
  return (int)value;
}

static void inverse_block(const int32_t *coefficients, const uint16_t *quant,
                          uint8_t *dst, size_t stride) {
  int64_t horizontal[8][8];
  for (unsigned v = 0; v < 8; v++) {
    for (unsigned x = 0; x < 8; x++) {
      int64_t sum = 0;
      for (unsigned u = 0; u < 8; u++)
        sum += (int64_t)coefficients[v * 8 + u] * quant[v * 8 + u] *
               idct_matrix[x][u];
      horizontal[v][x] = sum;
    }
  }
  const int64_t rounding = INT64_C(1) << 27;
  for (unsigned y = 0; y < 8; y++) {
    for (unsigned x = 0; x < 8; x++) {
      int64_t sum = 0;
      for (unsigned v = 0; v < 8; v++)
        sum += horizontal[v][x] * idct_matrix[y][v];
      int64_t value =
          sum >= 0 ? (sum + rounding) >> 28 : -(((-sum) + rounding) >> 28);
      dst[y * stride + x] = (uint8_t)clamp_sample(value + 128);
    }
  }
}

static jpeg_status reconstruct(jpeg_decoder *decoder) {
  for (unsigned index = 0; index < decoder->component_count; index++) {
    jpeg_component *component = &decoder->components[index];
    if (!decoder->quant_valid[component->quant])
      return JPEG_TABLE;
    if (component->blocks_x > SIZE_MAX / 8 ||
        component->blocks_y > SIZE_MAX / 8)
      return JPEG_SIZE;
    component->sample_stride = component->blocks_x * 8;
    size_t sample_rows = component->blocks_y * 8;
    size_t sample_count;
    if (!checked_product(component->sample_stride, sample_rows, &sample_count))
      return JPEG_SIZE;
    component->samples = malloc(sample_count == 0 ? 1 : sample_count);
    if (component->samples == NULL)
      return JPEG_NOMEM;
    for (size_t block_y = 0; block_y < component->blocks_y; block_y++) {
      for (size_t block_x = 0; block_x < component->blocks_x; block_x++) {
        const int32_t *block =
            component->coefficients +
            (block_y * component->allocated_x + block_x) * 64;
        inverse_block(block, decoder->quant[component->quant],
                      component->samples +
                          block_y * 8 * component->sample_stride + block_x * 8,
                      component->sample_stride);
      }
    }
  }
  return JPEG_OK;
}

static unsigned resample(const jpeg_decoder *decoder,
                         const jpeg_component *component, size_t x, size_t y) {
  int64_t x_num = (int64_t)(2 * x + 1) * component->h - decoder->max_h;
  int64_t y_num = (int64_t)(2 * y + 1) * component->v - decoder->max_v;
  int64_t x_den = 2 * decoder->max_h;
  int64_t y_den = 2 * decoder->max_v;
  int64_t x0 = x_num >= 0 ? x_num / x_den : -(((-x_num) + x_den - 1) / x_den);
  int64_t y0 = y_num >= 0 ? y_num / y_den : -(((-y_num) + y_den - 1) / y_den);
  int64_t x_rem = x_num - x0 * x_den;
  int64_t y_rem = y_num - y0 * y_den;
  int64_t max_x = (int64_t)component->sample_stride - 1;
  int64_t max_y = (int64_t)component->blocks_y * 8 - 1;
  int64_t x1 = x0 + 1;
  int64_t y1 = y0 + 1;
  if (x0 < 0) {
    x0 = x1 = 0;
    x_rem = 0;
  } else if (x1 > max_x) {
    x0 = x1 = max_x;
    x_rem = 0;
  }
  if (y0 < 0) {
    y0 = y1 = 0;
    y_rem = 0;
  } else if (y1 > max_y) {
    y0 = y1 = max_y;
    y_rem = 0;
  }
  unsigned p00 = component->samples[y0 * component->sample_stride + x0];
  unsigned p10 = component->samples[y0 * component->sample_stride + x1];
  unsigned p01 = component->samples[y1 * component->sample_stride + x0];
  unsigned p11 = component->samples[y1 * component->sample_stride + x1];
  int64_t top = p00 * (x_den - x_rem) + p10 * x_rem;
  int64_t bottom = p01 * (x_den - x_rem) + p11 * x_rem;
  return (
      unsigned)((top * (y_den - y_rem) + bottom * y_rem + (x_den * y_den) / 2) /
                (x_den * y_den));
}

static int color_clamp(int value) {
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

static void ycbcr(unsigned y, unsigned cb, unsigned cr, unsigned *r,
                  unsigned *g, unsigned *b) {
  int cb_shift = (int)cb - 128;
  int cr_shift = (int)cr - 128;
  *r = (unsigned)color_clamp((int)y + ((91881 * cr_shift + 32768) >> 16));
  *g = (unsigned)color_clamp(
      (int)y - ((22554 * cb_shift + 46802 * cr_shift + 32768) >> 16));
  *b = (unsigned)color_clamp((int)y + ((116130 * cb_shift + 32768) >> 16));
}

static unsigned multiply_255(unsigned a, unsigned b) {
  unsigned value = a * b + 128;
  return (value + (value >> 8)) >> 8;
}

static jpeg_status write_pixels(const jpeg_decoder *decoder, uint8_t *dst,
                                size_t dst_len, int grayscale) {
  unsigned output_channels = grayscale ? 1 : 3;
  if (decoder->width > SIZE_MAX / decoder->height ||
      decoder->width * decoder->height > SIZE_MAX / output_channels ||
      decoder->width * decoder->height * output_channels != dst_len)
    return JPEG_SIZE;
  for (size_t y = 0; y < decoder->height; y++) {
    for (size_t x = 0; x < decoder->width; x++) {
      unsigned r;
      unsigned g;
      unsigned b;
      if (decoder->component_count == 1) {
        r = g = b = resample(decoder, &decoder->components[0], x, y);
      } else if (decoder->component_count == 3) {
        unsigned first = resample(decoder, &decoder->components[0], x, y);
        unsigned second = resample(decoder, &decoder->components[1], x, y);
        unsigned third = resample(decoder, &decoder->components[2], x, y);
        if (decoder->adobe_transform == 0) {
          r = first;
          g = second;
          b = third;
        } else {
          ycbcr(first, second, third, &r, &g, &b);
        }
      } else {
        unsigned first = resample(decoder, &decoder->components[0], x, y);
        unsigned second = resample(decoder, &decoder->components[1], x, y);
        unsigned third = resample(decoder, &decoder->components[2], x, y);
        unsigned key = resample(decoder, &decoder->components[3], x, y);
        if (decoder->adobe_transform == 2) {
          unsigned rr;
          unsigned gg;
          unsigned bb;
          ycbcr(first, second, third, &rr, &gg, &bb);
          r = multiply_255(255 - rr, key);
          g = multiply_255(255 - gg, key);
          b = multiply_255(255 - bb, key);
        } else if (decoder->adobe_transform == 0) {
          r = multiply_255(first, key);
          g = multiply_255(second, key);
          b = multiply_255(third, key);
        } else {
          r = multiply_255(255 - first, 255 - key);
          g = multiply_255(255 - second, 255 - key);
          b = multiply_255(255 - third, 255 - key);
        }
      }
      size_t out = (y * decoder->width + x) * output_channels;
      if (grayscale) {
        dst[out] = (uint8_t)((77u * r + 150u * g + 29u * b + 128u) >> 8);
      } else {
        dst[out] = (uint8_t)r;
        dst[out + 1] = (uint8_t)g;
        dst[out + 2] = (uint8_t)b;
      }
    }
  }
  return JPEG_OK;
}

static jpeg_status decode_jpeg(const uint8_t *src, size_t len, uint8_t *dst,
                               size_t dst_len, int grayscale) {
  if (len < 4 || src[0] != 0xff || src[1] != 0xd8)
    return JPEG_MARKER;
  jpeg_decoder decoder;
  memset(&decoder, 0, sizeof(decoder));
  decoder.src = src;
  decoder.len = len;
  decoder.adobe_transform = -1;
  jpeg_status status = JPEG_OK;
  size_t off = 2;
  int saw_end = 0;
  while (off < len && status == JPEG_OK) {
    unsigned marker;
    status = next_marker(src, len, &off, &marker);
    if (status != JPEG_OK)
      break;
    if (marker == 0xd9) {
      saw_end = 1;
      break;
    }
    if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7) ||
        marker == 0x01) {
      status = JPEG_MARKER;
      break;
    }
    const uint8_t *data;
    size_t data_len;
    status = segment(src, len, &off, &data, &data_len);
    if (status != JPEG_OK)
      break;
    if (marker == 0xc0 || marker == 0xc2) {
      status = parse_frame(&decoder, data, data_len, marker == 0xc2);
    } else if (marker == 0xdb) {
      status = parse_dqt(&decoder, data, data_len);
    } else if (marker == 0xc4) {
      status = parse_dht(&decoder, data, data_len);
    } else if (marker == 0xdd) {
      if (data_len != 2)
        status = JPEG_TABLE;
      else
        decoder.restart_interval = be16(data);
    } else if (marker == 0xda) {
      if (decoder.component_count == 0)
        status = JPEG_SCAN;
      else
        status = decode_scan(&decoder, data, data_len, off, &off);
    } else if (marker == 0xe0 && data_len >= 5 &&
               memcmp(data, "JFIF\0", 5) == 0) {
      decoder.jfif = 1;
    } else if (marker == 0xee && data_len >= 12 &&
               memcmp(data, "Adobe", 5) == 0) {
      decoder.adobe_transform = data[11];
    } else if ((marker >= 0xc1 && marker <= 0xcf && marker != 0xc4 &&
                marker != 0xc8 && marker != 0xcc) ||
               marker == 0xcc) {
      status = JPEG_UNSUPPORTED;
    }
  }
  if (status == JPEG_OK && (!saw_end || !decoder.saw_scan))
    status = JPEG_TRUNCATED;
  if (status == JPEG_OK && off != len)
    status = JPEG_MARKER;
  if (status == JPEG_OK && decoder.component_count == 3 &&
      decoder.adobe_transform < 0) {
    int rgb_ids = decoder.components[0].id == 'R' &&
                  decoder.components[1].id == 'G' &&
                  decoder.components[2].id == 'B';
    decoder.adobe_transform = rgb_ids && !decoder.jfif ? 0 : 1;
  }
  if (status == JPEG_OK)
    status = reconstruct(&decoder);
  if (status == JPEG_OK)
    status = write_pixels(&decoder, dst, dst_len, grayscale);
  free_decoder(&decoder);
  return status;
}

static jpeg_status probe_jpeg(const uint8_t *src, size_t len, size_t *width,
                              size_t *height) {
  if (len < 4 || src[0] != 0xff || src[1] != 0xd8)
    return JPEG_MARKER;
  size_t off = 2;
  while (off < len) {
    unsigned marker;
    jpeg_status status = next_marker(src, len, &off, &marker);
    if (status != JPEG_OK)
      return status;
    if (marker == 0xd9 || marker == 0xda)
      return JPEG_FRAME;
    const uint8_t *data;
    size_t data_len;
    status = segment(src, len, &off, &data, &data_len);
    if (status != JPEG_OK)
      return status;
    if (marker == 0xc0 || marker == 0xc2) {
      if (data_len < 6 || data[0] != 8 || be16(data + 1) == 0 ||
          be16(data + 3) == 0)
        return JPEG_FRAME;
      *height = be16(data + 1);
      *width = be16(data + 3);
      return JPEG_OK;
    }
    if ((marker >= 0xc1 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 &&
         marker != 0xcc))
      return JPEG_UNSUPPORTED;
  }
  return JPEG_TRUNCATED;
}

typedef struct {
  uint16_t code[256];
  uint8_t size[256];
} encoder_huffman;

typedef struct {
  int fd;
  uint8_t staging[16384];
  size_t staged;
  uint64_t bits;
  unsigned nbits;
} jpeg_writer;

static const uint8_t quant_luminance[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

static const uint8_t quant_chrominance[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

static const uint8_t dc_luminance_counts[16] = {0, 1, 5, 1, 1, 1, 1, 1,
                                                1, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t dc_chrominance_counts[16] = {0, 3, 1, 1, 1, 1, 1, 1,
                                                  1, 1, 1, 0, 0, 0, 0, 0};
static const uint8_t dc_values[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const uint8_t ac_luminance_counts[16] = {0, 2, 1, 3, 3, 2, 4, 3,
                                                5, 5, 4, 4, 0, 0, 1, 0x7d};
static const uint8_t ac_chrominance_counts[16] = {0, 2, 1, 2, 4, 4, 3, 4,
                                                  7, 5, 4, 4, 0, 1, 2, 0x77};

static const uint8_t ac_luminance_values[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
    0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
    0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

static const uint8_t ac_chrominance_values[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41,
    0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1,
    0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74,
    0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
    0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

static jpeg_status writer_flush(jpeg_writer *writer) {
  size_t off = 0;
  while (off < writer->staged) {
    ssize_t written =
        write(writer->fd, writer->staging + off, writer->staged - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      if (written == 0)
        errno = EIO;
      return JPEG_SYSTEM;
    }
    off += (size_t)written;
  }
  writer->staged = 0;
  return JPEG_OK;
}

static jpeg_status writer_byte(jpeg_writer *writer, uint8_t byte) {
  writer->staging[writer->staged++] = byte;
  return writer->staged == sizeof(writer->staging) ? writer_flush(writer)
                                                   : JPEG_OK;
}

static jpeg_status writer_data(jpeg_writer *writer, const uint8_t *data,
                               size_t len) {
  for (size_t i = 0; i < len; i++) {
    jpeg_status status = writer_byte(writer, data[i]);
    if (status != JPEG_OK)
      return status;
  }
  return JPEG_OK;
}

static jpeg_status writer_marker(jpeg_writer *writer, unsigned marker,
                                 const uint8_t *data, size_t len) {
  if (len > 65533)
    return JPEG_SIZE;
  uint8_t header[4] = {0xff, (uint8_t)marker, (uint8_t)((len + 2) >> 8),
                       (uint8_t)(len + 2)};
  jpeg_status status = writer_data(writer, header, sizeof(header));
  return status == JPEG_OK ? writer_data(writer, data, len) : status;
}

static jpeg_status writer_bits(jpeg_writer *writer, unsigned bits,
                               unsigned count) {
  if (count == 0)
    return JPEG_OK;
  writer->bits = (writer->bits << count) | (bits & ((1u << count) - 1u));
  writer->nbits += count;
  while (writer->nbits >= 8) {
    writer->nbits -= 8;
    uint8_t byte = (uint8_t)(writer->bits >> writer->nbits);
    jpeg_status status = writer_byte(writer, byte);
    if (status != JPEG_OK)
      return status;
    if (byte == 0xff) {
      status = writer_byte(writer, 0x00);
      if (status != JPEG_OK)
        return status;
    }
  }
  return JPEG_OK;
}

static jpeg_status writer_finish_bits(jpeg_writer *writer) {
  if (writer->nbits != 0) {
    unsigned padding = 8 - writer->nbits;
    return writer_bits(writer, (1u << padding) - 1u, padding);
  }
  return JPEG_OK;
}

static void build_encoder_huffman(encoder_huffman *table,
                                  const uint8_t counts[16],
                                  const uint8_t *values) {
  memset(table, 0, sizeof(*table));
  unsigned code = 0;
  unsigned index = 0;
  for (unsigned length = 1; length <= 16; length++) {
    for (unsigned i = 0; i < counts[length - 1]; i++) {
      table->code[values[index]] = (uint16_t)code;
      table->size[values[index]] = (uint8_t)length;
      code++;
      index++;
    }
    code <<= 1;
  }
}

static unsigned coefficient_bits(int value, unsigned *bits) {
  unsigned magnitude = value < 0 ? (unsigned)(-value) : (unsigned)value;
  unsigned count = 0;
  for (unsigned copy = magnitude; copy != 0; copy >>= 1)
    count++;
  *bits = value < 0 ? (unsigned)(value + (int)((1u << count) - 1u)) : magnitude;
  return count;
}

static void forward_block(const int16_t samples[64], const uint8_t quant[64],
                          int16_t output[64]) {
  int64_t horizontal[8][8];
  for (unsigned y = 0; y < 8; y++) {
    for (unsigned u = 0; u < 8; u++) {
      int64_t sum = 0;
      for (unsigned x = 0; x < 8; x++)
        sum += (int64_t)samples[y * 8 + x] * idct_matrix[x][u];
      horizontal[y][u] = sum;
    }
  }
  for (unsigned v = 0; v < 8; v++) {
    for (unsigned u = 0; u < 8; u++) {
      int64_t sum = 0;
      for (unsigned y = 0; y < 8; y++)
        sum += horizontal[y][u] * idct_matrix[y][v];
      int64_t transformed = sum >= 0 ? (sum + (INT64_C(1) << 27)) >> 28
                                     : -(((-sum) + (INT64_C(1) << 27)) >> 28);
      int divisor = quant[v * 8 + u];
      int64_t quantized = transformed >= 0
                              ? (transformed + divisor / 2) / divisor
                              : -(((-transformed) + divisor / 2) / divisor);
      if (quantized < INT16_MIN)
        quantized = INT16_MIN;
      if (quantized > INT16_MAX)
        quantized = INT16_MAX;
      output[v * 8 + u] = (int16_t)quantized;
    }
  }
}

static jpeg_status encode_block(jpeg_writer *writer, const encoder_huffman *dc,
                                const encoder_huffman *ac,
                                const int16_t coefficients[64],
                                int *previous_dc) {
  int difference = coefficients[0] - *previous_dc;
  *previous_dc = coefficients[0];
  unsigned bits;
  unsigned count = coefficient_bits(difference, &bits);
  jpeg_status status = writer_bits(writer, dc->code[count], dc->size[count]);
  if (status == JPEG_OK)
    status = writer_bits(writer, bits, count);
  unsigned zeroes = 0;
  for (unsigned k = 1; status == JPEG_OK && k < 64; k++) {
    int value = coefficients[zigzag[k]];
    if (value == 0) {
      zeroes++;
      continue;
    }
    while (zeroes >= 16) {
      status = writer_bits(writer, ac->code[0xf0], ac->size[0xf0]);
      zeroes -= 16;
      if (status != JPEG_OK)
        return status;
    }
    count = coefficient_bits(value, &bits);
    if (count > 10)
      return JPEG_SIZE;
    unsigned symbol = (zeroes << 4) | count;
    status = writer_bits(writer, ac->code[symbol], ac->size[symbol]);
    if (status == JPEG_OK)
      status = writer_bits(writer, bits, count);
    zeroes = 0;
  }
  if (status == JPEG_OK && zeroes != 0)
    status = writer_bits(writer, ac->code[0], ac->size[0]);
  return status;
}

static void rgb_to_ycbcr(unsigned r, unsigned g, unsigned b, int *y, int *cb,
                         int *cr) {
  *y = (19595 * (int)r + 38470 * (int)g + 7471 * (int)b + 32768) >> 16;
  *cb = (-11059 * (int)r - 21709 * (int)g + 32768 * (int)b + 8421376) >> 16;
  *cr = (32768 * (int)r - 27439 * (int)g - 5329 * (int)b + 8421376) >> 16;
}

static void image_pixel(const uint8_t *src, size_t width, size_t height,
                        unsigned channels, size_t x, size_t y, int *luma,
                        int *cb, int *cr) {
  if (x >= width)
    x = width - 1;
  if (y >= height)
    y = height - 1;
  const uint8_t *pixel = src + (y * width + x) * channels;
  if (channels == 1) {
    *luma = pixel[0];
    *cb = *cr = 128;
  } else {
    rgb_to_ycbcr(pixel[0], pixel[1], pixel[2], luma, cb, cr);
  }
}

static void luma_block(const uint8_t *src, size_t width, size_t height,
                       unsigned channels, size_t start_x, size_t start_y,
                       int16_t block[64]) {
  for (size_t y = 0; y < 8; y++) {
    for (size_t x = 0; x < 8; x++) {
      int luma;
      int cb;
      int cr;
      image_pixel(src, width, height, channels, start_x + x, start_y + y, &luma,
                  &cb, &cr);
      block[y * 8 + x] = (int16_t)(luma - 128);
    }
  }
}

static void chroma_block(const uint8_t *src, size_t width, size_t height,
                         size_t start_x, size_t start_y, int choose_cr,
                         int16_t block[64]) {
  for (size_t y = 0; y < 8; y++) {
    for (size_t x = 0; x < 8; x++) {
      int sum = 0;
      for (size_t dy = 0; dy < 2; dy++) {
        for (size_t dx = 0; dx < 2; dx++) {
          int luma;
          int cb;
          int cr;
          image_pixel(src, width, height, 3, start_x + x * 2 + dx,
                      start_y + y * 2 + dy, &luma, &cb, &cr);
          sum += choose_cr ? cr : cb;
        }
      }
      block[y * 8 + x] = (int16_t)((sum + 2) / 4 - 128);
    }
  }
}

static size_t append_huffman(uint8_t *dst, unsigned class_, unsigned index,
                             const uint8_t counts[16], const uint8_t *values) {
  size_t count = 0;
  for (unsigned i = 0; i < 16; i++)
    count += counts[i];
  dst[0] = (uint8_t)((class_ << 4) | index);
  memcpy(dst + 1, counts, 16);
  memcpy(dst + 17, values, count);
  return count + 17;
}

static jpeg_status encode_jpeg(int fd, const uint8_t *src, size_t src_len,
                               size_t width, size_t height, unsigned channels) {
  if (width == 0 || height == 0 || width > 65535 || height > 65535 ||
      (channels != 1 && channels != 3) || width > SIZE_MAX / height ||
      width * height > SIZE_MAX / channels ||
      width * height * channels != src_len)
    return JPEG_SIZE;
  uint8_t luma_quant[64];
  uint8_t chroma_quant[64];
  for (unsigned i = 0; i < 64; i++) {
    unsigned luma = (quant_luminance[i] * 20u + 50u) / 100u;
    unsigned chroma = (quant_chrominance[i] * 20u + 50u) / 100u;
    luma_quant[i] = (uint8_t)(luma == 0 ? 1 : luma);
    chroma_quant[i] = (uint8_t)(chroma == 0 ? 1 : chroma);
  }
  encoder_huffman dc_luma;
  encoder_huffman ac_luma;
  encoder_huffman dc_chroma;
  encoder_huffman ac_chroma;
  build_encoder_huffman(&dc_luma, dc_luminance_counts, dc_values);
  build_encoder_huffman(&ac_luma, ac_luminance_counts, ac_luminance_values);
  build_encoder_huffman(&dc_chroma, dc_chrominance_counts, dc_values);
  build_encoder_huffman(&ac_chroma, ac_chrominance_counts,
                        ac_chrominance_values);

  jpeg_writer writer;
  memset(&writer, 0, sizeof(writer));
  writer.fd = fd;
  const uint8_t soi[2] = {0xff, 0xd8};
  jpeg_status status = writer_data(&writer, soi, sizeof(soi));
  const uint8_t app0[14] = {'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0};
  if (status == JPEG_OK)
    status = writer_marker(&writer, 0xe0, app0, sizeof(app0));

  uint8_t dqt[130];
  size_t dqt_len = 65;
  dqt[0] = 0;
  for (unsigned i = 0; i < 64; i++)
    dqt[i + 1] = luma_quant[zigzag[i]];
  if (channels == 3) {
    dqt[65] = 1;
    for (unsigned i = 0; i < 64; i++)
      dqt[66 + i] = chroma_quant[zigzag[i]];
    dqt_len = 130;
  }
  if (status == JPEG_OK)
    status = writer_marker(&writer, 0xdb, dqt, dqt_len);

  uint8_t sof[15];
  size_t sof_len = channels == 1 ? 9 : 15;
  sof[0] = 8;
  sof[1] = (uint8_t)(height >> 8);
  sof[2] = (uint8_t)height;
  sof[3] = (uint8_t)(width >> 8);
  sof[4] = (uint8_t)width;
  sof[5] = channels == 1 ? 1 : 3;
  sof[6] = 1;
  sof[7] = channels == 1 ? 0x11 : 0x22;
  sof[8] = 0;
  if (channels == 3) {
    sof[9] = 2;
    sof[10] = 0x11;
    sof[11] = 1;
    sof[12] = 3;
    sof[13] = 0x11;
    sof[14] = 1;
  }
  if (status == JPEG_OK)
    status = writer_marker(&writer, 0xc0, sof, sof_len);

  uint8_t dht[416];
  size_t dht_len = 0;
  dht_len +=
      append_huffman(dht + dht_len, 0, 0, dc_luminance_counts, dc_values);
  dht_len += append_huffman(dht + dht_len, 1, 0, ac_luminance_counts,
                            ac_luminance_values);
  if (channels == 3) {
    dht_len +=
        append_huffman(dht + dht_len, 0, 1, dc_chrominance_counts, dc_values);
    dht_len += append_huffman(dht + dht_len, 1, 1, ac_chrominance_counts,
                              ac_chrominance_values);
  }
  if (status == JPEG_OK)
    status = writer_marker(&writer, 0xc4, dht, dht_len);

  uint8_t sos[10];
  size_t sos_len = channels == 1 ? 6 : 10;
  sos[0] = channels == 1 ? 1 : 3;
  sos[1] = 1;
  sos[2] = 0x00;
  if (channels == 3) {
    sos[3] = 2;
    sos[4] = 0x11;
    sos[5] = 3;
    sos[6] = 0x11;
    sos[7] = 0;
    sos[8] = 63;
    sos[9] = 0;
  } else {
    sos[3] = 0;
    sos[4] = 63;
    sos[5] = 0;
  }
  if (status == JPEG_OK)
    status = writer_marker(&writer, 0xda, sos, sos_len);

  int previous_y = 0;
  int previous_cb = 0;
  int previous_cr = 0;
  size_t mcu_width = channels == 1 ? 8 : 16;
  size_t mcu_height = channels == 1 ? 8 : 16;
  int16_t samples[64];
  int16_t coefficients[64];
  for (size_t mcu_y = 0; status == JPEG_OK && mcu_y < height;
       mcu_y += mcu_height) {
    for (size_t mcu_x = 0; status == JPEG_OK && mcu_x < width;
         mcu_x += mcu_width) {
      unsigned y_blocks = channels == 1 ? 1 : 4;
      for (unsigned block = 0; block < y_blocks; block++) {
        size_t block_x = mcu_x + (block & 1u) * 8;
        size_t block_y = mcu_y + (block >> 1) * 8;
        luma_block(src, width, height, channels, block_x, block_y, samples);
        forward_block(samples, luma_quant, coefficients);
        status = encode_block(&writer, &dc_luma, &ac_luma, coefficients,
                              &previous_y);
      }
      if (channels == 3 && status == JPEG_OK) {
        chroma_block(src, width, height, mcu_x, mcu_y, 0, samples);
        forward_block(samples, chroma_quant, coefficients);
        status = encode_block(&writer, &dc_chroma, &ac_chroma, coefficients,
                              &previous_cb);
      }
      if (channels == 3 && status == JPEG_OK) {
        chroma_block(src, width, height, mcu_x, mcu_y, 1, samples);
        forward_block(samples, chroma_quant, coefficients);
        status = encode_block(&writer, &dc_chroma, &ac_chroma, coefficients,
                              &previous_cr);
      }
    }
  }
  if (status == JPEG_OK)
    status = writer_finish_bits(&writer);
  const uint8_t eoi[2] = {0xff, 0xd9};
  if (status == JPEG_OK)
    status = writer_data(&writer, eoi, sizeof(eoi));
  if (status == JPEG_OK)
    status = writer_flush(&writer);
  return status;
}

#ifndef NX_IO_CODEC_NO_OCAML
static void checked_bytes(value vbuf, const uint8_t **data, size_t *len) {
  struct caml_ba_array *array = Caml_ba_array_val(vbuf);
  if ((array->flags & CAML_BA_KIND_MASK) != CAML_BA_UINT8)
    caml_invalid_argument("Nx_io JPEG: expected a uint8 Bigarray");
  *data = Caml_ba_data_val(vbuf);
  *len = caml_ba_byte_size(array);
}

CAMLprim value caml_nx_io_jpeg_probe(value vsrc) {
  CAMLparam1(vsrc);
  CAMLlocal1(vresult);
  const uint8_t *src;
  size_t len;
  checked_bytes(vsrc, &src, &len);
  size_t width;
  size_t height;
  caml_release_runtime_system();
  jpeg_status status = probe_jpeg(src, len, &width, &height);
  caml_acquire_runtime_system();
  if (status != JPEG_OK)
    caml_failwith(jpeg_message(status));
  if (width > (size_t)Max_long || height > (size_t)Max_long)
    caml_failwith(jpeg_message(JPEG_SIZE));
  vresult = caml_alloc_tuple(2);
  Store_field(vresult, 0, Val_long(width));
  Store_field(vresult, 1, Val_long(height));
  CAMLreturn(vresult);
}

CAMLprim value caml_nx_io_jpeg_decode(value vsrc, value vdst,
                                      value vgrayscale) {
  CAMLparam3(vsrc, vdst, vgrayscale);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  const uint8_t *dst_const;
  size_t dst_len;
  checked_bytes(vdst, &dst_const, &dst_len);
  int grayscale = Bool_val(vgrayscale);
  caml_release_runtime_system();
  jpeg_status status =
      decode_jpeg(src, src_len, (uint8_t *)dst_const, dst_len, grayscale);
  caml_acquire_runtime_system();
  if (status != JPEG_OK)
    caml_failwith(jpeg_message(status));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_io_jpeg_encode(value vfd, value vsrc, value vwidth,
                                      value vheight, value vchannels) {
  CAMLparam5(vfd, vsrc, vwidth, vheight, vchannels);
  const uint8_t *src;
  size_t src_len;
  checked_bytes(vsrc, &src, &src_len);
  intnat width_i = Long_val(vwidth);
  intnat height_i = Long_val(vheight);
  intnat channels_i = Long_val(vchannels);
  if (width_i <= 0 || height_i <= 0 || channels_i <= 0)
    caml_invalid_argument("Nx_io JPEG: invalid image dimensions");
  int fd = Int_val(vfd);
  caml_release_runtime_system();
  jpeg_status status = encode_jpeg(fd, src, src_len, (size_t)width_i,
                                   (size_t)height_i, (unsigned)channels_i);
  int saved_errno = errno;
  caml_acquire_runtime_system();
  if (status == JPEG_SYSTEM)
    unix_error(saved_errno == 0 ? EIO : saved_errno, "write", Nothing);
  if (status != JPEG_OK)
    caml_failwith(jpeg_message(status));
  CAMLreturn(Val_unit);
}
#endif
