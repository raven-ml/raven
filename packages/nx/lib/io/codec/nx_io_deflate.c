/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Raw DEFLATE decoder implementing RFC 1951 sections 3.1 and 3.2. The decoder
  keeps an explicit 32 KiB history ring, so callers may discard an initial
  output prefix without weakening distance validation.
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

#define NX_IO_WINDOW 32768u
#define NX_IO_MAX_BITS 15u
#define NX_IO_MAX_LITLEN 288u
#define NX_IO_MAX_DIST 32u
#define NX_IO_MAX_CODES (NX_IO_MAX_LITLEN + NX_IO_MAX_DIST)
#define NX_IO_WRITE_BUFFER 32768u
#define NX_IO_FAST_BITS 9u
#define NX_IO_FAST_SIZE (1u << NX_IO_FAST_BITS)

typedef struct {
  const uint8_t *src;
  size_t len;
  size_t pos;
  uint64_t bits;
  unsigned nbits;
} bit_reader;

typedef struct {
  uint16_t count[NX_IO_MAX_BITS + 1];
  uint16_t symbol[NX_IO_MAX_CODES];
  uint16_t fast[NX_IO_FAST_SIZE];
  int empty;
} huffman;

typedef struct {
  uint8_t window[NX_IO_WINDOW];
  size_t produced;
  size_t skip;
  uint8_t *dst;
  size_t dst_len;
  int stop_at_limit;
  int unbounded;
  int fd;
  nx_io_consume_fn consume;
  void *consume_context;
  uint8_t staging[NX_IO_WRITE_BUFFER];
  size_t staged;
  uint32_t crc;
  uint32_t crc_table[8][256];
  int calculate_crc;
  int batch_crc;
} output;

static uint32_t read_le32(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) |
         ((uint32_t)src[3] << 24);
}

static uint32_t crc_update(const uint32_t table[8][256], uint32_t crc,
                           const uint8_t *src, size_t len) {
  while (len >= 8) {
    uint32_t first = crc ^ read_le32(src);
    uint32_t second = read_le32(src + 4);
    crc = table[7][first & 0xffu] ^ table[6][(first >> 8) & 0xffu] ^
          table[5][(first >> 16) & 0xffu] ^ table[4][first >> 24] ^
          table[3][second & 0xffu] ^ table[2][(second >> 8) & 0xffu] ^
          table[1][(second >> 16) & 0xffu] ^ table[0][second >> 24];
    src += 8;
    len -= 8;
  }
  while (len-- != 0)
    crc = table[0][(crc ^ *src++) & 0xffu] ^ (crc >> 8);
  return crc;
}

static nx_io_status output_flush(output *out) {
  if (out->fd < 0)
    return NX_IO_OK;
  if (out->calculate_crc && out->batch_crc)
    out->crc = crc_update(out->crc_table, out->crc, out->staging, out->staged);
  size_t off = 0;
  while (off < out->staged) {
    ssize_t written = write(out->fd, out->staging + off, out->staged - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      if (written == 0)
        errno = EIO;
      return NX_IO_SYSTEM;
    }
    off += (size_t)written;
  }
  out->staged = 0;
  return NX_IO_OK;
}

static nx_io_status bits_get(bit_reader *br, unsigned count, uint32_t *value) {
  if (count > 24)
    return NX_IO_INVALID_BLOCK;
  while (br->nbits < count) {
    if (br->pos == br->len)
      return NX_IO_TRUNCATED;
    br->bits |= (uint64_t)br->src[br->pos++] << br->nbits;
    br->nbits += 8;
  }
  *value = (uint32_t)(br->bits & ((UINT64_C(1) << count) - 1));
  br->bits >>= count;
  br->nbits -= count;
  return NX_IO_OK;
}

static nx_io_status bit_get(bit_reader *br, uint32_t *value) {
  return bits_get(br, 1, value);
}

static void bits_align_byte(bit_reader *br) {
  unsigned discard = br->nbits & 7u;
  br->bits >>= discard;
  br->nbits -= discard;
}

static unsigned reverse_bits(unsigned value, unsigned count) {
  unsigned reversed = 0;
  for (unsigned i = 0; i < count; i++) {
    reversed = (reversed << 1) | (value & 1u);
    value >>= 1;
  }
  return reversed;
}

static nx_io_status huffman_build(huffman *h, const uint8_t *lengths,
                                  unsigned n, int allow_empty) {
  unsigned offsets[NX_IO_MAX_BITS + 1];
  memset(h, 0, sizeof(*h));
  for (unsigned symbol = 0; symbol < n; symbol++) {
    unsigned len = lengths[symbol];
    if (len > NX_IO_MAX_BITS)
      return NX_IO_INVALID_HUFFMAN;
    h->count[len]++;
  }
  if (h->count[0] == n) {
    h->empty = 1;
    return allow_empty ? NX_IO_OK : NX_IO_INVALID_HUFFMAN;
  }

  int left = 1;
  for (unsigned len = 1; len <= NX_IO_MAX_BITS; len++) {
    left <<= 1;
    left -= h->count[len];
    if (left < 0)
      return NX_IO_INVALID_HUFFMAN;
  }

  offsets[1] = 0;
  for (unsigned len = 1; len < NX_IO_MAX_BITS; len++)
    offsets[len + 1] = offsets[len] + h->count[len];
  for (unsigned symbol = 0; symbol < n; symbol++) {
    unsigned len = lengths[symbol];
    if (len != 0)
      h->symbol[offsets[len]++] = (uint16_t)symbol;
  }
  unsigned next_code[NX_IO_MAX_BITS + 1] = {0};
  unsigned code = 0;
  for (unsigned len = 1; len <= NX_IO_MAX_BITS; len++) {
    unsigned previous_count = len == 1 ? 0 : h->count[len - 1];
    code = (code + previous_count) << 1;
    next_code[len] = code;
  }
  for (unsigned symbol = 0; symbol < n; symbol++) {
    unsigned len = lengths[symbol];
    if (len == 0)
      continue;
    unsigned canonical = next_code[len]++;
    if (len <= NX_IO_FAST_BITS) {
      unsigned reversed = reverse_bits(canonical, len);
      uint16_t entry = (uint16_t)((len << 9) | symbol);
      for (unsigned index = reversed; index < NX_IO_FAST_SIZE;
           index += 1u << len)
        h->fast[index] = entry;
    }
  }
  return NX_IO_OK;
}

static nx_io_status huffman_decode(bit_reader *br, const huffman *h,
                                   uint16_t *symbol) {
  if (h->empty)
    return NX_IO_INVALID_HUFFMAN;
  while (br->nbits < NX_IO_FAST_BITS && br->pos < br->len) {
    br->bits |= (uint64_t)br->src[br->pos++] << br->nbits;
    br->nbits += 8;
  }
  if (br->nbits >= NX_IO_FAST_BITS) {
    uint16_t entry = h->fast[br->bits & (NX_IO_FAST_SIZE - 1u)];
    if (entry != 0) {
      unsigned len = entry >> 9;
      *symbol = entry & 0x1ffu;
      br->bits >>= len;
      br->nbits -= len;
      return NX_IO_OK;
    }
  }
  unsigned code = 0;
  unsigned first = 0;
  unsigned index = 0;
  for (unsigned len = 1; len <= NX_IO_MAX_BITS; len++) {
    uint32_t bit;
    nx_io_status status = bit_get(br, &bit);
    if (status != NX_IO_OK)
      return status;
    code |= bit;
    unsigned count = h->count[len];
    if (code < first + count) {
      *symbol = h->symbol[index + (code - first)];
      return NX_IO_OK;
    }
    index += count;
    first = (first + count) << 1;
    code <<= 1;
  }
  return NX_IO_INVALID_HUFFMAN;
}

static void crc_table_init(uint32_t table[8][256]) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int bit = 0; bit < 8; bit++)
      c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    table[0][i] = c;
  }
  for (unsigned slice = 1; slice < 8; slice++)
    for (unsigned i = 0; i < 256; i++) {
      uint32_t previous = table[slice - 1][i];
      table[slice][i] = table[0][previous & 0xffu] ^ (previous >> 8);
    }
}

static nx_io_status output_byte(output *out, uint8_t byte) {
  size_t limit;
  if (!out->unbounded) {
    if (out->skip > SIZE_MAX - out->dst_len)
      return NX_IO_OVERFLOW;
    limit = out->skip + out->dst_len;
    if (out->produced == limit) {
      if (out->stop_at_limit)
        return NX_IO_STOPPED;
      return NX_IO_OUTPUT_SIZE;
    }
  }
  if (out->consume == NULL && out->fd < 0 && out->dst != NULL &&
      out->produced >= out->skip) {
    size_t index = out->produced - out->skip;
    if (out->stop_at_limit || index < NX_IO_WINDOW)
      out->window[out->produced & (NX_IO_WINDOW - 1u)] = byte;
    out->dst[index] = byte;
    out->produced++;
    return NX_IO_OK;
  }
  out->window[out->produced & (NX_IO_WINDOW - 1u)] = byte;
  if (out->calculate_crc && (!out->batch_crc || out->produced < out->skip))
    out->crc = out->crc_table[0][(out->crc ^ byte) & 0xffu] ^ (out->crc >> 8);
  if (out->produced >= out->skip) {
    if (out->consume != NULL) {
      nx_io_status status = out->consume(out->consume_context, byte);
      if (status != NX_IO_OK)
        return status;
    } else if (out->fd >= 0) {
      out->staging[out->staged++] = byte;
      if (out->staged == sizeof(out->staging)) {
        nx_io_status status = output_flush(out);
        if (status != NX_IO_OK)
          return status;
      }
    } else if (out->dst != NULL) {
      out->dst[out->produced - out->skip] = byte;
    } else {
      return NX_IO_OUTPUT_SIZE;
    }
  }
  out->produced++;
  return NX_IO_OK;
}

static void window_write(output *out, const uint8_t *src, size_t len) {
  size_t ring = out->produced & (NX_IO_WINDOW - 1u);
  size_t first = len;
  if (first > NX_IO_WINDOW - ring)
    first = NX_IO_WINDOW - ring;
  memcpy(out->window + ring, src, first);
  if (first < len)
    memcpy(out->window, src + first, len - first);
}

static nx_io_status output_span(output *out, const uint8_t *src, size_t len) {
  if (out->consume != NULL || out->stop_at_limit)
    goto bytewise;
  if (!out->unbounded) {
    size_t limit = out->skip + out->dst_len;
    if (out->produced > limit || len > limit - out->produced)
      return NX_IO_OUTPUT_SIZE;
  }
  while (len != 0 && out->produced < out->skip) {
    nx_io_status status = output_byte(out, *src++);
    if (status != NX_IO_OK)
      return status;
    len--;
  }
  while (len != 0) {
    if (out->fd >= 0) {
      size_t chunk = sizeof(out->staging) - out->staged;
      if (chunk > len)
        chunk = len;
      window_write(out, src, chunk);
      memcpy(out->staging + out->staged, src, chunk);
      out->staged += chunk;
      out->produced += chunk;
      src += chunk;
      len -= chunk;
      if (out->staged == sizeof(out->staging)) {
        nx_io_status status = output_flush(out);
        if (status != NX_IO_OK)
          return status;
      }
    } else if (out->dst != NULL) {
      size_t chunk = len > NX_IO_WINDOW ? NX_IO_WINDOW : len;
      window_write(out, src, chunk);
      memcpy(out->dst + out->produced - out->skip, src, chunk);
      out->produced += chunk;
      src += chunk;
      len -= chunk;
    } else {
      return NX_IO_OUTPUT_SIZE;
    }
  }
  return NX_IO_OK;

bytewise:
  for (size_t i = 0; i < len; i++) {
    nx_io_status status = output_byte(out, src[i]);
    if (status != NX_IO_OK)
      return status;
  }
  return NX_IO_OK;
}

static nx_io_status output_match(output *out, unsigned distance,
                                 unsigned length) {
  if (distance == 0 || distance > NX_IO_WINDOW || distance > out->produced)
    return NX_IO_INVALID_DISTANCE;
  if (!out->stop_at_limit && out->consume == NULL && out->fd < 0 &&
      out->dst != NULL && out->produced >= out->skip &&
      distance <= out->produced - out->skip) {
    if (!out->unbounded) {
      size_t limit = out->skip + out->dst_len;
      if (out->produced > limit || length > limit - out->produced)
        return NX_IO_OUTPUT_SIZE;
    }
    size_t offset = out->produced - out->skip;
    size_t copied = 0;
    while (copied < length) {
      size_t chunk = length - copied;
      if (chunk > distance)
        chunk = distance;
      memcpy(out->dst + offset + copied, out->dst + offset + copied - distance,
             chunk);
      copied += chunk;
    }
    window_write(out, out->dst + offset, length);
    out->produced += length;
    return NX_IO_OK;
  }
  for (unsigned i = 0; i < length; i++) {
    uint8_t byte =
        out->window[(out->produced - distance) & (NX_IO_WINDOW - 1u)];
    nx_io_status status = output_byte(out, byte);
    if (status != NX_IO_OK)
      return status;
  }
  return NX_IO_OK;
}

static void fixed_trees(huffman *litlen, huffman *dist) {
  uint8_t lengths[NX_IO_MAX_LITLEN];
  uint8_t distances[NX_IO_MAX_DIST];
  for (unsigned i = 0; i <= 143; i++)
    lengths[i] = 8;
  for (unsigned i = 144; i <= 255; i++)
    lengths[i] = 9;
  for (unsigned i = 256; i <= 279; i++)
    lengths[i] = 7;
  for (unsigned i = 280; i < NX_IO_MAX_LITLEN; i++)
    lengths[i] = 8;
  memset(distances, 5, sizeof(distances));
  (void)huffman_build(litlen, lengths, NX_IO_MAX_LITLEN, 0);
  (void)huffman_build(dist, distances, NX_IO_MAX_DIST, 0);
}

static nx_io_status dynamic_trees(bit_reader *br, huffman *litlen,
                                  huffman *dist) {
  static const uint8_t order[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                    11, 4,  12, 3, 13, 2, 14, 1, 15};
  uint32_t value;
  nx_io_status status = bits_get(br, 5, &value);
  if (status != NX_IO_OK)
    return status;
  unsigned nlit = value + 257;
  status = bits_get(br, 5, &value);
  if (status != NX_IO_OK)
    return status;
  unsigned ndist = value + 1;
  status = bits_get(br, 4, &value);
  if (status != NX_IO_OK)
    return status;
  unsigned ncode = value + 4;
  if (nlit > 286 || ndist > NX_IO_MAX_DIST)
    return NX_IO_INVALID_BLOCK;

  uint8_t code_lengths[19] = {0};
  for (unsigned i = 0; i < ncode; i++) {
    status = bits_get(br, 3, &value);
    if (status != NX_IO_OK)
      return status;
    code_lengths[order[i]] = (uint8_t)value;
  }
  huffman code_tree;
  status = huffman_build(&code_tree, code_lengths, 19, 0);
  if (status != NX_IO_OK)
    return status;

  uint8_t lengths[NX_IO_MAX_CODES] = {0};
  unsigned count = nlit + ndist;
  unsigned index = 0;
  while (index < count) {
    uint16_t symbol;
    status = huffman_decode(br, &code_tree, &symbol);
    if (status != NX_IO_OK)
      return status;
    if (symbol <= 15) {
      lengths[index++] = (uint8_t)symbol;
    } else if (symbol == 16) {
      if (index == 0)
        return NX_IO_INVALID_HUFFMAN;
      status = bits_get(br, 2, &value);
      if (status != NX_IO_OK)
        return status;
      unsigned repeat = value + 3;
      if (repeat > count - index)
        return NX_IO_INVALID_HUFFMAN;
      uint8_t previous = lengths[index - 1];
      while (repeat-- != 0)
        lengths[index++] = previous;
    } else if (symbol == 17 || symbol == 18) {
      unsigned extra = symbol == 17 ? 3 : 7;
      unsigned base = symbol == 17 ? 3 : 11;
      status = bits_get(br, extra, &value);
      if (status != NX_IO_OK)
        return status;
      unsigned repeat = value + base;
      if (repeat > count - index)
        return NX_IO_INVALID_HUFFMAN;
      while (repeat-- != 0)
        lengths[index++] = 0;
    } else {
      return NX_IO_INVALID_SYMBOL;
    }
  }
  if (lengths[256] == 0)
    return NX_IO_INVALID_HUFFMAN;
  status = huffman_build(litlen, lengths, nlit, 0);
  if (status != NX_IO_OK)
    return status;
  return huffman_build(dist, lengths + nlit, ndist, 1);
}

static nx_io_status compressed_block(bit_reader *br, output *out,
                                     const huffman *litlen,
                                     const huffman *dist) {
  static const uint16_t length_base[29] = {
      3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
      31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const uint8_t length_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                           1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                           4, 4, 4, 4, 5, 5, 5, 5, 0};
  static const uint16_t dist_base[30] = {
      1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
      33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
      1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
  static const uint8_t dist_extra[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                         4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                         9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

  for (;;) {
    uint16_t symbol;
    nx_io_status status = huffman_decode(br, litlen, &symbol);
    if (status != NX_IO_OK)
      return status;
    if (symbol < 256) {
      status = output_byte(out, (uint8_t)symbol);
      if (status != NX_IO_OK)
        return status;
      continue;
    }
    if (symbol == 256)
      return NX_IO_OK;
    if (symbol < 257 || symbol > 285)
      return NX_IO_INVALID_SYMBOL;

    unsigned length_index = symbol - 257;
    uint32_t extra_value = 0;
    status = bits_get(br, length_extra[length_index], &extra_value);
    if (status != NX_IO_OK)
      return status;
    unsigned length = length_base[length_index] + extra_value;

    uint16_t distance_symbol;
    status = huffman_decode(br, dist, &distance_symbol);
    if (status != NX_IO_OK)
      return status;
    if (distance_symbol >= 30)
      return NX_IO_INVALID_SYMBOL;
    extra_value = 0;
    status = bits_get(br, dist_extra[distance_symbol], &extra_value);
    if (status != NX_IO_OK)
      return status;
    unsigned distance = dist_base[distance_symbol] + extra_value;
    status = output_match(out, distance, length);
    if (status != NX_IO_OK)
      return status;
  }
}

static nx_io_status stored_block(bit_reader *br, output *out) {
  bits_align_byte(br);
  uint32_t len;
  uint32_t nlen;
  nx_io_status status = bits_get(br, 16, &len);
  if (status != NX_IO_OK)
    return status;
  status = bits_get(br, 16, &nlen);
  if (status != NX_IO_OK)
    return status;
  if ((len ^ 0xffffu) != nlen)
    return NX_IO_INVALID_BLOCK;
  if (br->nbits == 0) {
    if (len > br->len - br->pos)
      return NX_IO_TRUNCATED;
    status = output_span(out, br->src + br->pos, len);
    if (status == NX_IO_OK)
      br->pos += len;
    return status;
  }
  for (uint32_t i = 0; i < len; i++) {
    uint32_t byte;
    status = bits_get(br, 8, &byte);
    if (status != NX_IO_OK)
      return status;
    status = output_byte(out, (uint8_t)byte);
    if (status != NX_IO_OK)
      return status;
  }
  return NX_IO_OK;
}

static nx_io_result inflate_raw_impl(const uint8_t *src, size_t src_len,
                                     size_t skip, uint8_t *dst, size_t dst_len,
                                     int stop_at_limit, int fd,
                                     nx_io_consume_fn consume,
                                     void *consume_context, int unbounded,
                                     int allow_trailing, uint32_t *crc) {
  nx_io_result result = {NX_IO_OK, 0, 0};
  if (skip > SIZE_MAX - dst_len) {
    result.status = NX_IO_OVERFLOW;
    return result;
  }
  bit_reader br = {src, src_len, 0, 0, 0};
  output out;
  memset(&out, 0, sizeof(out));
  out.skip = skip;
  out.dst = dst;
  out.dst_len = dst_len;
  out.stop_at_limit = stop_at_limit;
  out.unbounded = unbounded;
  out.fd = fd;
  out.consume = consume;
  out.consume_context = consume_context;
  out.crc = 0xffffffffu;
  out.calculate_crc = crc != NULL;
  out.batch_crc =
      out.calculate_crc && consume == NULL && (dst != NULL || fd >= 0);
  if (out.calculate_crc)
    crc_table_init(out.crc_table);

  int final = 0;
  while (!final) {
    uint32_t value;
    nx_io_status status = bits_get(&br, 1, &value);
    if (status != NX_IO_OK) {
      result.status = status;
      break;
    }
    final = (int)value;
    status = bits_get(&br, 2, &value);
    if (status != NX_IO_OK) {
      result.status = status;
      break;
    }
    if (value == 0) {
      status = stored_block(&br, &out);
    } else if (value == 1 || value == 2) {
      huffman litlen;
      huffman dist;
      if (value == 1) {
        fixed_trees(&litlen, &dist);
        status = NX_IO_OK;
      } else {
        status = dynamic_trees(&br, &litlen, &dist);
      }
      if (status == NX_IO_OK)
        status = compressed_block(&br, &out, &litlen, &dist);
    } else {
      status = NX_IO_INVALID_BLOCK;
    }
    if (status == NX_IO_STOPPED && stop_at_limit) {
      result.status = NX_IO_STOPPED;
      break;
    }
    if (status != NX_IO_OK) {
      result.status = status;
      break;
    }
  }

  result.input_offset = br.pos;
  result.output_size = out.produced;
  if (result.status == NX_IO_OK) {
    if (unbounded) {
      if (!allow_trailing && br.pos != src_len)
        result.status = NX_IO_INVALID_BLOCK;
    } else if (!stop_at_limit) {
      size_t expected = skip + dst_len;
      if (out.produced != expected)
        result.status = NX_IO_OUTPUT_SIZE;
      else if (br.pos != src_len)
        result.status = NX_IO_INVALID_BLOCK;
    } else if (br.pos != src_len) {
      result.status = NX_IO_INVALID_BLOCK;
    }
  }
  if (result.status == NX_IO_OK)
    result.status = output_flush(&out);
  if (result.status == NX_IO_OK && out.batch_crc && out.fd < 0 &&
      out.dst != NULL)
    out.crc = crc_update(out.crc_table, out.crc, out.dst, out.dst_len);
  if (crc != NULL)
    *crc = out.crc ^ 0xffffffffu;
  return result;
}

nx_io_result nx_io_inflate_raw(const uint8_t *src, size_t src_len, size_t skip,
                               uint8_t *dst, size_t dst_len, int stop_at_limit,
                               int fd, uint32_t *crc) {
  return inflate_raw_impl(src, src_len, skip, dst, dst_len, stop_at_limit, fd,
                          NULL, NULL, 0, 0, crc);
}

nx_io_result nx_io_inflate_raw_sink(const uint8_t *src, size_t src_len,
                                    size_t output_size,
                                    nx_io_consume_fn consume, void *context,
                                    uint32_t *crc) {
  if (consume == NULL) {
    nx_io_result result = {NX_IO_OUTPUT_SIZE, 0, 0};
    return result;
  }
  return inflate_raw_impl(src, src_len, 0, NULL, output_size, 0, -1, consume,
                          context, 0, 0, crc);
}

#ifndef NX_IO_CODEC_NO_OCAML
static nx_io_result inflate_raw_member(const uint8_t *src, size_t src_len,
                                       int fd, uint32_t *crc) {
  return inflate_raw_impl(src, src_len, 0, NULL, 0, 0, fd, NULL, NULL, 1, 1,
                          crc);
}
#endif

const char *nx_io_status_message(nx_io_status status) {
  switch (status) {
  case NX_IO_OK:
    return "ok";
  case NX_IO_STOPPED:
    return "output prefix complete";
  case NX_IO_TRUNCATED:
    return "truncated DEFLATE stream";
  case NX_IO_INVALID_BLOCK:
    return "invalid DEFLATE block";
  case NX_IO_INVALID_HUFFMAN:
    return "invalid DEFLATE Huffman tree";
  case NX_IO_INVALID_SYMBOL:
    return "invalid DEFLATE symbol";
  case NX_IO_INVALID_DISTANCE:
    return "invalid DEFLATE distance";
  case NX_IO_OUTPUT_SIZE:
    return "unexpected DEFLATE output size";
  case NX_IO_OVERFLOW:
    return "codec size overflow";
  case NX_IO_NOMEM:
    return "codec allocation failed";
  case NX_IO_SYSTEM:
    return "codec system error";
  }
  return "unknown codec error";
}

#ifndef NX_IO_CODEC_NO_OCAML
static void checked_input(value vsrc, value voff, value vlen,
                          const uint8_t **src, size_t *len) {
  intnat off_i = Long_val(voff);
  intnat len_i = Long_val(vlen);
  if (off_i < 0 || len_i < 0)
    caml_invalid_argument("Nx_io.inflate: negative input span");
  size_t off = (size_t)off_i;
  size_t n = (size_t)len_i;
  size_t total = caml_ba_byte_size(Caml_ba_array_val(vsrc));
  if (off > total || n > total - off)
    caml_invalid_argument("Nx_io.inflate: input span out of bounds");
  *src = (const uint8_t *)Caml_ba_data_val(vsrc) + off;
  *len = n;
}

CAMLprim value caml_nx_io_inflate_raw_prefix(value vsrc, value voff, value vlen,
                                             value vmax) {
  CAMLparam4(vsrc, voff, vlen, vmax);
  CAMLlocal1(vdst);
  const uint8_t *src;
  size_t src_len;
  checked_input(vsrc, voff, vlen, &src, &src_len);
  intnat max_i = Long_val(vmax);
  if (max_i < 0)
    caml_invalid_argument("Nx_io.inflate: negative output limit");
  size_t max = (size_t)max_i;
  uint8_t *buffer = malloc(max == 0 ? 1 : max);
  if (buffer == NULL)
    caml_raise_out_of_memory();

  caml_release_runtime_system();
  nx_io_result result =
      nx_io_inflate_raw(src, src_len, 0, buffer, max, 1, -1, NULL);
  caml_acquire_runtime_system();
  if (result.status != NX_IO_OK && result.status != NX_IO_STOPPED) {
    const char *message = nx_io_status_message(result.status);
    free(buffer);
    caml_failwith(message);
  }
  vdst = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                            result.output_size);
  if (result.output_size != 0)
    memcpy(Caml_ba_data_val(vdst), buffer, result.output_size);
  free(buffer);
  CAMLreturn(vdst);
}

CAMLprim value caml_nx_io_inflate_raw_into(value vsrc, value vsrc_off,
                                           value vsrc_len, value vskip,
                                           value vdst, value vdst_off,
                                           value vdst_len) {
  CAMLparam5(vsrc, vsrc_off, vsrc_len, vskip, vdst);
  CAMLxparam2(vdst_off, vdst_len);
  const uint8_t *src;
  size_t src_len;
  checked_input(vsrc, vsrc_off, vsrc_len, &src, &src_len);
  intnat skip_i = Long_val(vskip);
  intnat dst_off_i = Long_val(vdst_off);
  intnat dst_len_i = Long_val(vdst_len);
  if (skip_i < 0 || dst_off_i < 0 || dst_len_i < 0)
    caml_invalid_argument("Nx_io.inflate: negative output span");
  size_t skip = (size_t)skip_i;
  size_t dst_off = (size_t)dst_off_i;
  size_t dst_len = (size_t)dst_len_i;
  size_t dst_total = caml_ba_byte_size(Caml_ba_array_val(vdst));
  if (dst_off > dst_total || dst_len > dst_total - dst_off)
    caml_invalid_argument("Nx_io.inflate: output span out of bounds");
  uint8_t *dst = (uint8_t *)Caml_ba_data_val(vdst) + dst_off;

  uint32_t crc;
  caml_release_runtime_system();
  nx_io_result result =
      nx_io_inflate_raw(src, src_len, skip, dst, dst_len, 0, -1, &crc);
  caml_acquire_runtime_system();
  if (result.status != NX_IO_OK)
    caml_failwith(nx_io_status_message(result.status));
  CAMLreturn(caml_copy_int32((int32_t)crc));
}

CAMLprim value caml_nx_io_inflate_raw_into_bytecode(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_inflate_raw_into(argv[0], argv[1], argv[2], argv[3],
                                     argv[4], argv[5], argv[6]);
}

CAMLprim value caml_nx_io_inflate_raw_to_fd(value vfd, value vsrc,
                                            value vsrc_off, value vsrc_len,
                                            value voutput_size) {
  CAMLparam5(vfd, vsrc, vsrc_off, vsrc_len, voutput_size);
  const uint8_t *src;
  size_t src_len;
  checked_input(vsrc, vsrc_off, vsrc_len, &src, &src_len);
  intnat output_i = Long_val(voutput_size);
  if (output_i < 0)
    caml_invalid_argument("Nx_io.inflate: negative output size");
  size_t output_size = (size_t)output_i;
  int fd = Int_val(vfd);
  uint32_t crc;
  caml_release_runtime_system();
  nx_io_result result =
      nx_io_inflate_raw(src, src_len, 0, NULL, output_size, 0, fd, &crc);
  int saved_errno = errno;
  caml_acquire_runtime_system();
  if (result.status == NX_IO_SYSTEM)
    unix_error(saved_errno == 0 ? EIO : saved_errno, "write", Nothing);
  if (result.status != NX_IO_OK)
    caml_failwith(nx_io_status_message(result.status));
  CAMLreturn(caml_copy_int32((int32_t)crc));
}

CAMLprim value caml_nx_io_inflate_raw_member_to_fd(value vfd, value vsrc,
                                                   value vsrc_off,
                                                   value vsrc_len) {
  CAMLparam4(vfd, vsrc, vsrc_off, vsrc_len);
  CAMLlocal2(vresult, vcrc);
  const uint8_t *src;
  size_t src_len;
  checked_input(vsrc, vsrc_off, vsrc_len, &src, &src_len);
  int fd = Int_val(vfd);
  uint32_t crc;
  caml_release_runtime_system();
  nx_io_result result = inflate_raw_member(src, src_len, fd, &crc);
  int saved_errno = errno;
  caml_acquire_runtime_system();
  if (result.status == NX_IO_SYSTEM)
    unix_error(saved_errno == 0 ? EIO : saved_errno, "write", Nothing);
  if (result.status != NX_IO_OK)
    caml_failwith(nx_io_status_message(result.status));
  if (result.input_offset > (size_t)Max_long ||
      result.output_size > (size_t)Max_long)
    caml_failwith("Nx_io.inflate: member exceeds OCaml integer range");
  vcrc = caml_copy_int32((int32_t)crc);
  vresult = caml_alloc_tuple(3);
  Store_field(vresult, 0, Val_long(result.input_offset));
  Store_field(vresult, 1, Val_long(result.output_size));
  Store_field(vresult, 2, vcrc);
  CAMLreturn(vresult);
}
#endif
