/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  A bounded-memory RFC 1951 encoder. Input is divided into 64 KiB blocks and
  each block independently chooses fixed Huffman coding or storage based on
  its exact encoded bit cost. A hash-chain match finder retains the full
  32 KiB DEFLATE history across block boundaries.
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

#define NX_IO_BLOCK 65535u
#define NX_IO_WINDOW 32768u
#define NX_IO_HASH_BITS 16u
#define NX_IO_HASH_SIZE (1u << NX_IO_HASH_BITS)
#define NX_IO_CHAIN_LIMIT 4u
#define NX_IO_SINK_BUFFER 32768u
#define NX_IO_LITLEN_CODES 286u
#define NX_IO_DIST_CODES 30u
#define NX_IO_CODELEN_CODES 19u
#define NX_IO_MAX_CODELEN_TOKENS 640u

typedef struct {
  const uint8_t *prefix;
  size_t prefix_len;
  const uint8_t *src;
  size_t src_len;
  size_t total;
} input;

typedef struct {
  uint16_t value;
  uint16_t distance;
} token;

typedef struct {
  int fd;
  int failed;
  uint8_t staging[NX_IO_SINK_BUFFER];
  size_t staged;
  uint8_t *buffer;
  size_t length;
  size_t capacity;
} sink;

typedef struct {
  sink *out;
  uint64_t bits;
  unsigned nbits;
} bit_writer;

typedef struct {
  uint32_t frequency;
  int parent;
  unsigned symbol;
} huffman_node;

typedef struct {
  uint8_t symbol;
  uint8_t extra_bits;
  uint16_t extra;
} codelen_token;

typedef struct {
  uint8_t litlen_length[NX_IO_LITLEN_CODES];
  uint8_t dist_length[NX_IO_DIST_CODES];
  uint16_t litlen_code[NX_IO_LITLEN_CODES];
  uint16_t dist_code[NX_IO_DIST_CODES];
  uint8_t codelen_length[NX_IO_CODELEN_CODES];
  uint16_t codelen_code[NX_IO_CODELEN_CODES];
  codelen_token codelen_tokens[NX_IO_MAX_CODELEN_TOKENS];
  size_t codelen_count;
  unsigned nlit;
  unsigned ndist;
  unsigned ncode;
  size_t bit_cost;
} dynamic_plan;

static uint8_t input_get(const input *in, size_t pos) {
  if (pos < in->prefix_len)
    return in->prefix[pos];
  return in->src[pos - in->prefix_len];
}

static nx_io_status sink_flush(sink *s) {
  if (s->failed)
    return NX_IO_SYSTEM;
  if (s->fd < 0)
    return NX_IO_OK;
  size_t off = 0;
  while (off < s->staged) {
    ssize_t written = write(s->fd, s->staging + off, s->staged - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      if (written == 0)
        errno = EIO;
      s->failed = 1;
      return NX_IO_SYSTEM;
    }
    off += (size_t)written;
  }
  if (s->length > SIZE_MAX - s->staged)
    return NX_IO_OVERFLOW;
  s->length += s->staged;
  s->staged = 0;
  return NX_IO_OK;
}

static nx_io_status sink_byte(sink *s, uint8_t byte) {
  if (s->fd >= 0) {
    s->staging[s->staged++] = byte;
    if (s->staged == sizeof(s->staging))
      return sink_flush(s);
    return NX_IO_OK;
  }
  if (s->length == s->capacity) {
    size_t capacity = s->capacity == 0 ? 1024 : s->capacity * 2;
    if (capacity < s->capacity)
      return NX_IO_OVERFLOW;
    uint8_t *buffer = realloc(s->buffer, capacity);
    if (buffer == NULL)
      return NX_IO_NOMEM;
    s->buffer = buffer;
    s->capacity = capacity;
  }
  s->buffer[s->length++] = byte;
  return NX_IO_OK;
}

static nx_io_status bits_put(bit_writer *bw, uint32_t bits, unsigned count) {
  bw->bits |= (uint64_t)bits << bw->nbits;
  bw->nbits += count;
  while (bw->nbits >= 8) {
    nx_io_status status = sink_byte(bw->out, (uint8_t)bw->bits);
    if (status != NX_IO_OK)
      return status;
    bw->bits >>= 8;
    bw->nbits -= 8;
  }
  return NX_IO_OK;
}

static nx_io_status bits_align(bit_writer *bw) {
  if (bw->nbits == 0)
    return NX_IO_OK;
  nx_io_status status = sink_byte(bw->out, (uint8_t)bw->bits);
  bw->bits = 0;
  bw->nbits = 0;
  return status;
}

static uint32_t reverse_bits(uint32_t value, unsigned count) {
  uint32_t reversed = 0;
  for (unsigned i = 0; i < count; i++) {
    reversed = (reversed << 1) | (value & 1u);
    value >>= 1;
  }
  return reversed;
}

static void fixed_code(unsigned symbol, uint32_t *code, unsigned *bits) {
  if (symbol <= 143) {
    *code = reverse_bits(0x30u + symbol, 8);
    *bits = 8;
  } else if (symbol <= 255) {
    *code = reverse_bits(0x190u + symbol - 144u, 9);
    *bits = 9;
  } else if (symbol <= 279) {
    *code = reverse_bits(symbol - 256u, 7);
    *bits = 7;
  } else {
    *code = reverse_bits(0xc0u + symbol - 280u, 8);
    *bits = 8;
  }
}

static unsigned fixed_code_bits(unsigned symbol) {
  if (symbol <= 143)
    return 8;
  if (symbol <= 255)
    return 9;
  if (symbol <= 279)
    return 7;
  return 8;
}

static unsigned hash3(const input *in, size_t pos) {
  uint32_t value = (uint32_t)input_get(in, pos) * 0x1e35a7bdu;
  value ^= (uint32_t)input_get(in, pos + 1) * 0x9e3779b1u;
  value ^= (uint32_t)input_get(in, pos + 2) * 0x85ebca77u;
  return value >> (32u - NX_IO_HASH_BITS);
}

static void insert_position(const input *in, size_t pos, size_t *head,
                            size_t *previous) {
  if (pos + 2 >= in->total)
    return;
  unsigned hash = hash3(in, pos);
  previous[pos & (NX_IO_WINDOW - 1u)] = head[hash];
  head[hash] = pos;
}

static unsigned match_length(const input *in, size_t left, size_t right,
                             unsigned limit) {
  unsigned length = 0;
  while (length < limit &&
         input_get(in, left + length) == input_get(in, right + length))
    length++;
  return length;
}

static unsigned find_match(const input *in, size_t pos, size_t block_end,
                           const size_t *head, const size_t *previous,
                           unsigned *best_distance) {
  if (pos + 2 >= block_end)
    return 0;
  size_t candidate = head[hash3(in, pos)];
  unsigned best = 2;
  unsigned limit = (unsigned)(block_end - pos);
  if (limit > 258)
    limit = 258;
  for (unsigned chain = 0; candidate != SIZE_MAX && chain < NX_IO_CHAIN_LIMIT;
       chain++) {
    if (candidate >= pos || pos - candidate > NX_IO_WINDOW)
      break;
    if (input_get(in, candidate + best) == input_get(in, pos + best)) {
      unsigned length = match_length(in, candidate, pos, limit);
      if (length > best) {
        best = length;
        *best_distance = (unsigned)(pos - candidate);
        if (best == limit)
          break;
      }
    }
    size_t next = previous[candidate & (NX_IO_WINDOW - 1u)];
    if (next >= candidate)
      break;
    candidate = next;
  }
  return best >= 3 ? best : 0;
}

static void length_symbol(unsigned length, unsigned *symbol, unsigned *extra,
                          unsigned *extra_bits) {
  static const uint16_t base[29] = {3,  4,  5,  6,   7,   8,   9,   10,  11, 13,
                                    15, 17, 19, 23,  27,  31,  35,  43,  51, 59,
                                    67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const uint8_t bits[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
  unsigned index = 0;
  while (index < 28 && length >= base[index + 1])
    index++;
  *symbol = index + 257;
  *extra = length - base[index];
  *extra_bits = bits[index];
}

static void distance_symbol(unsigned distance, unsigned *symbol,
                            unsigned *extra, unsigned *extra_bits) {
  static const uint16_t base[30] = {
      1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
      33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
      1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
  static const uint8_t bits[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                   4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                   9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  unsigned index = 0;
  while (index < 29 && distance >= base[index + 1])
    index++;
  *symbol = index;
  *extra = distance - base[index];
  *extra_bits = bits[index];
}

static int heap_less(const huffman_node *nodes, int left, int right) {
  if (nodes[left].frequency != nodes[right].frequency)
    return nodes[left].frequency < nodes[right].frequency;
  return nodes[left].symbol < nodes[right].symbol;
}

static void heap_push(int *heap, unsigned *count, int node,
                      const huffman_node *nodes) {
  unsigned child = (*count)++;
  while (child != 0) {
    unsigned parent = (child - 1) / 2;
    if (!heap_less(nodes, node, heap[parent]))
      break;
    heap[child] = heap[parent];
    child = parent;
  }
  heap[child] = node;
}

static int heap_pop(int *heap, unsigned *count, const huffman_node *nodes) {
  int result = heap[0];
  int tail = heap[--(*count)];
  unsigned parent = 0;
  while (parent * 2 + 1 < *count) {
    unsigned child = parent * 2 + 1;
    if (child + 1 < *count && heap_less(nodes, heap[child + 1], heap[child]))
      child++;
    if (!heap_less(nodes, heap[child], tail))
      break;
    heap[parent] = heap[child];
    parent = child;
  }
  if (*count != 0)
    heap[parent] = tail;
  return result;
}

static void sort_symbols_by_frequency(unsigned *symbols, unsigned count,
                                      const uint32_t *frequency) {
  for (unsigned i = 1; i < count; i++) {
    unsigned symbol = symbols[i];
    unsigned position = i;
    while (position != 0) {
      unsigned previous = symbols[position - 1];
      if (frequency[previous] < frequency[symbol] ||
          (frequency[previous] == frequency[symbol] && previous > symbol))
        break;
      symbols[position] = previous;
      position--;
    }
    symbols[position] = symbol;
  }
}

static int build_lengths(const uint32_t *frequency, unsigned symbols,
                         unsigned max_bits, uint8_t *length) {
  huffman_node nodes[2 * NX_IO_LITLEN_CODES];
  int heap[2 * NX_IO_LITLEN_CODES];
  unsigned used_symbols[NX_IO_LITLEN_CODES];
  if (symbols > NX_IO_LITLEN_CODES || max_bits == 0 || max_bits > 15)
    return 0;
  unsigned heap_count = 0;
  unsigned used = 0;
  memset(length, 0, symbols);
  for (unsigned symbol = 0; symbol < symbols; symbol++) {
    if (frequency[symbol] == 0)
      continue;
    nodes[used] = (huffman_node){frequency[symbol], -1, symbol};
    used_symbols[used] = symbol;
    heap_push(heap, &heap_count, (int)used, nodes);
    used++;
  }
  if (used == 0)
    return 0;
  if (used == 1) {
    length[used_symbols[0]] = 1;
    return 1;
  }

  unsigned node_count = used;
  while (heap_count > 1) {
    int left = heap_pop(heap, &heap_count, nodes);
    int right = heap_pop(heap, &heap_count, nodes);
    uint32_t sum = nodes[left].frequency + nodes[right].frequency;
    if (sum < nodes[left].frequency)
      return 0;
    unsigned tie = nodes[left].symbol < nodes[right].symbol
                       ? nodes[left].symbol
                       : nodes[right].symbol;
    nodes[node_count] = (huffman_node){sum, -1, tie};
    nodes[left].parent = (int)node_count;
    nodes[right].parent = (int)node_count;
    heap_push(heap, &heap_count, (int)node_count, nodes);
    node_count++;
  }

  unsigned counts[16] = {0};
  unsigned overflow = 0;
  for (unsigned leaf = 0; leaf < used; leaf++) {
    unsigned depth = 0;
    for (int node = (int)leaf; nodes[node].parent >= 0;
         node = nodes[node].parent)
      depth++;
    if (depth > max_bits) {
      depth = max_bits;
      overflow++;
    }
    counts[depth]++;
  }
  while (overflow != 0) {
    unsigned bits = max_bits - 1;
    while (bits != 0 && counts[bits] == 0)
      bits--;
    if (bits == 0 || counts[max_bits] == 0 || overflow < 2)
      return 0;
    counts[bits]--;
    counts[bits + 1] += 2;
    counts[max_bits]--;
    overflow -= 2;
  }

  sort_symbols_by_frequency(used_symbols, used, frequency);
  unsigned index = 0;
  for (unsigned bits = max_bits; bits != 0; bits--)
    for (unsigned count = 0; count < counts[bits]; count++)
      length[used_symbols[index++]] = (uint8_t)bits;
  return index == used;
}

static int build_codes(const uint8_t *length, unsigned symbols,
                       unsigned max_bits, uint16_t *codes) {
  unsigned counts[16] = {0};
  unsigned next[16] = {0};
  for (unsigned symbol = 0; symbol < symbols; symbol++) {
    if (length[symbol] > max_bits)
      return 0;
    counts[length[symbol]]++;
  }
  counts[0] = 0;
  unsigned code = 0;
  for (unsigned bits = 1; bits <= max_bits; bits++) {
    code = (code + counts[bits - 1]) << 1;
    next[bits] = code;
    if (code + counts[bits] > (1u << bits))
      return 0;
  }
  for (unsigned symbol = 0; symbol < symbols; symbol++) {
    unsigned bits = length[symbol];
    codes[symbol] = bits == 0 ? 0 : (uint16_t)reverse_bits(next[bits]++, bits);
  }
  return 1;
}

static int add_codelen_token(dynamic_plan *plan, uint32_t *frequency,
                             unsigned symbol, unsigned extra,
                             unsigned extra_bits) {
  if (plan->codelen_count == NX_IO_MAX_CODELEN_TOKENS)
    return 0;
  plan->codelen_tokens[plan->codelen_count++] =
      (codelen_token){(uint8_t)symbol, (uint8_t)extra_bits, (uint16_t)extra};
  frequency[symbol]++;
  return 1;
}

static int encode_codelengths(dynamic_plan *plan, const uint8_t *lengths,
                              unsigned count, uint32_t *frequency) {
  unsigned index = 0;
  while (index < count) {
    unsigned value = lengths[index];
    unsigned run = 1;
    while (index + run < count && lengths[index + run] == value)
      run++;
    index += run;
    if (value == 0) {
      while (run >= 11) {
        unsigned repeat = run > 138 ? 138 : run;
        if (!add_codelen_token(plan, frequency, 18, repeat - 11, 7))
          return 0;
        run -= repeat;
      }
      if (run >= 3) {
        unsigned repeat = run > 10 ? 10 : run;
        if (!add_codelen_token(plan, frequency, 17, repeat - 3, 3))
          return 0;
        run -= repeat;
      }
      while (run-- != 0)
        if (!add_codelen_token(plan, frequency, 0, 0, 0))
          return 0;
    } else {
      if (!add_codelen_token(plan, frequency, value, 0, 0))
        return 0;
      run--;
      while (run >= 3) {
        unsigned repeat = run > 6 ? 6 : run;
        if (!add_codelen_token(plan, frequency, 16, repeat - 3, 2))
          return 0;
        run -= repeat;
      }
      while (run-- != 0)
        if (!add_codelen_token(plan, frequency, value, 0, 0))
          return 0;
    }
  }
  return 1;
}

static int make_dynamic_plan(const token *tokens, size_t count,
                             dynamic_plan *plan) {
  static const uint8_t order[NX_IO_CODELEN_CODES] = {
      16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
  uint32_t litlen_frequency[NX_IO_LITLEN_CODES] = {0};
  uint32_t dist_frequency[NX_IO_DIST_CODES] = {0};
  memset(plan, 0, sizeof(*plan));
  litlen_frequency[256] = 1;
  for (size_t i = 0; i < count; i++) {
    if (tokens[i].distance == 0) {
      litlen_frequency[tokens[i].value]++;
    } else {
      unsigned symbol;
      unsigned extra;
      unsigned extra_bits;
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      litlen_frequency[symbol]++;
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      dist_frequency[symbol]++;
    }
  }
  int has_distances = 0;
  for (unsigned symbol = 0; symbol < NX_IO_DIST_CODES; symbol++)
    has_distances |= dist_frequency[symbol] != 0;
  if (!has_distances)
    dist_frequency[0] = 1;
  if (!build_lengths(litlen_frequency, NX_IO_LITLEN_CODES, 15,
                     plan->litlen_length) ||
      !build_lengths(dist_frequency, NX_IO_DIST_CODES, 15, plan->dist_length) ||
      !build_codes(plan->litlen_length, NX_IO_LITLEN_CODES, 15,
                   plan->litlen_code) ||
      !build_codes(plan->dist_length, NX_IO_DIST_CODES, 15, plan->dist_code))
    return 0;

  plan->nlit = NX_IO_LITLEN_CODES;
  while (plan->nlit > 257 && plan->litlen_length[plan->nlit - 1] == 0)
    plan->nlit--;
  plan->ndist = NX_IO_DIST_CODES;
  while (plan->ndist > 1 && plan->dist_length[plan->ndist - 1] == 0)
    plan->ndist--;
  uint8_t lengths[NX_IO_LITLEN_CODES + NX_IO_DIST_CODES];
  memcpy(lengths, plan->litlen_length, plan->nlit);
  memcpy(lengths + plan->nlit, plan->dist_length, plan->ndist);
  uint32_t codelen_frequency[NX_IO_CODELEN_CODES] = {0};
  if (!encode_codelengths(plan, lengths, plan->nlit + plan->ndist,
                          codelen_frequency) ||
      !build_lengths(codelen_frequency, NX_IO_CODELEN_CODES, 7,
                     plan->codelen_length) ||
      !build_codes(plan->codelen_length, NX_IO_CODELEN_CODES, 7,
                   plan->codelen_code))
    return 0;
  plan->ncode = NX_IO_CODELEN_CODES;
  while (plan->ncode > 4 && plan->codelen_length[order[plan->ncode - 1]] == 0)
    plan->ncode--;

  size_t cost = 3 + 5 + 5 + 4 + plan->ncode * 3;
  for (size_t i = 0; i < plan->codelen_count; i++) {
    codelen_token item = plan->codelen_tokens[i];
    cost += plan->codelen_length[item.symbol] + item.extra_bits;
  }
  cost += plan->litlen_length[256];
  for (size_t i = 0; i < count; i++) {
    if (tokens[i].distance == 0) {
      cost += plan->litlen_length[tokens[i].value];
    } else {
      unsigned symbol;
      unsigned extra;
      unsigned extra_bits;
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      cost += plan->litlen_length[symbol] + extra_bits;
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      cost += plan->dist_length[symbol] + extra_bits;
    }
  }
  plan->bit_cost = cost;
  return 1;
}

static size_t token_bit_cost(const token *tokens, size_t count) {
  size_t cost = 3 + fixed_code_bits(256);
  for (size_t i = 0; i < count; i++) {
    if (tokens[i].distance == 0) {
      cost += fixed_code_bits(tokens[i].value);
    } else {
      unsigned symbol;
      unsigned extra;
      unsigned extra_bits;
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      (void)extra;
      cost += fixed_code_bits(symbol) + extra_bits;
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      cost += 5 + extra_bits;
    }
  }
  return cost;
}

static nx_io_status emit_fixed_symbol(bit_writer *bw, unsigned symbol) {
  uint32_t code;
  unsigned bits;
  fixed_code(symbol, &code, &bits);
  return bits_put(bw, code, bits);
}

static nx_io_status emit_fixed_block(bit_writer *bw, const token *tokens,
                                     size_t count, int final) {
  nx_io_status status = bits_put(bw, (uint32_t) final | 2u, 3);
  for (size_t i = 0; status == NX_IO_OK && i < count; i++) {
    if (tokens[i].distance == 0) {
      status = emit_fixed_symbol(bw, tokens[i].value);
    } else {
      unsigned symbol;
      unsigned extra;
      unsigned extra_bits;
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      status = emit_fixed_symbol(bw, symbol);
      if (status == NX_IO_OK)
        status = bits_put(bw, extra, extra_bits);
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      if (status == NX_IO_OK)
        status = bits_put(bw, reverse_bits(symbol, 5), 5);
      if (status == NX_IO_OK)
        status = bits_put(bw, extra, extra_bits);
    }
  }
  if (status == NX_IO_OK)
    status = emit_fixed_symbol(bw, 256);
  return status;
}

static nx_io_status emit_dynamic_symbol(bit_writer *bw, unsigned symbol,
                                        const uint16_t *codes,
                                        const uint8_t *lengths) {
  if (lengths[symbol] == 0)
    return NX_IO_INVALID_HUFFMAN;
  return bits_put(bw, codes[symbol], lengths[symbol]);
}

static nx_io_status emit_dynamic_block(bit_writer *bw, const token *tokens,
                                       size_t count, int final,
                                       const dynamic_plan *plan) {
  static const uint8_t order[NX_IO_CODELEN_CODES] = {
      16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
  nx_io_status status = bits_put(bw, (uint32_t) final | 4u, 3);
  if (status == NX_IO_OK)
    status = bits_put(bw, plan->nlit - 257, 5);
  if (status == NX_IO_OK)
    status = bits_put(bw, plan->ndist - 1, 5);
  if (status == NX_IO_OK)
    status = bits_put(bw, plan->ncode - 4, 4);
  for (unsigned i = 0; status == NX_IO_OK && i < plan->ncode; i++)
    status = bits_put(bw, plan->codelen_length[order[i]], 3);
  for (size_t i = 0; status == NX_IO_OK && i < plan->codelen_count; i++) {
    codelen_token item = plan->codelen_tokens[i];
    status = emit_dynamic_symbol(bw, item.symbol, plan->codelen_code,
                                 plan->codelen_length);
    if (status == NX_IO_OK)
      status = bits_put(bw, item.extra, item.extra_bits);
  }
  for (size_t i = 0; status == NX_IO_OK && i < count; i++) {
    if (tokens[i].distance == 0) {
      status = emit_dynamic_symbol(bw, tokens[i].value, plan->litlen_code,
                                   plan->litlen_length);
    } else {
      unsigned symbol;
      unsigned extra;
      unsigned extra_bits;
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      status = emit_dynamic_symbol(bw, symbol, plan->litlen_code,
                                   plan->litlen_length);
      if (status == NX_IO_OK)
        status = bits_put(bw, extra, extra_bits);
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      if (status == NX_IO_OK)
        status =
            emit_dynamic_symbol(bw, symbol, plan->dist_code, plan->dist_length);
      if (status == NX_IO_OK)
        status = bits_put(bw, extra, extra_bits);
    }
  }
  if (status == NX_IO_OK)
    status =
        emit_dynamic_symbol(bw, 256, plan->litlen_code, plan->litlen_length);
  return status;
}

static nx_io_status emit_stored_block(bit_writer *bw, const input *in,
                                      size_t start, size_t length, int final) {
  nx_io_status status = bits_put(bw, (uint32_t) final, 3);
  if (status == NX_IO_OK)
    status = bits_align(bw);
  if (status != NX_IO_OK)
    return status;
  uint16_t len = (uint16_t)length;
  uint16_t complement = (uint16_t)~len;
  status = sink_byte(bw->out, (uint8_t)len);
  if (status == NX_IO_OK)
    status = sink_byte(bw->out, (uint8_t)(len >> 8));
  if (status == NX_IO_OK)
    status = sink_byte(bw->out, (uint8_t)complement);
  if (status == NX_IO_OK)
    status = sink_byte(bw->out, (uint8_t)(complement >> 8));
  for (size_t i = 0; status == NX_IO_OK && i < length; i++)
    status = sink_byte(bw->out, input_get(in, start + i));
  return status;
}

static void crc_table_init(uint32_t table[256]) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int bit = 0; bit < 8; bit++)
      c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    table[i] = c;
  }
}

static uint32_t input_crc32(const input *in) {
  uint32_t table[256];
  uint32_t crc = 0xffffffffu;
  crc_table_init(table);
  for (size_t i = 0; i < in->total; i++)
    crc = table[(crc ^ input_get(in, i)) & 0xffu] ^ (crc >> 8);
  return crc ^ 0xffffffffu;
}

nx_io_result nx_io_deflate_raw(const uint8_t *prefix, size_t prefix_len,
                               const uint8_t *src, size_t src_len, int fd,
                               uint8_t **buffer, uint32_t *crc) {
  nx_io_result result = {NX_IO_OK, 0, 0};
  if (prefix_len > SIZE_MAX - src_len) {
    result.status = NX_IO_OVERFLOW;
    return result;
  }
  input in = {prefix, prefix_len, src, src_len, prefix_len + src_len};
  sink out;
  memset(&out, 0, sizeof(out));
  out.fd = fd;
  bit_writer bw = {&out, 0, 0};

  size_t *head = malloc(NX_IO_HASH_SIZE * sizeof(*head));
  size_t *previous = malloc(NX_IO_WINDOW * sizeof(*previous));
  token *tokens = malloc(NX_IO_BLOCK * sizeof(*tokens));
  if (head == NULL || previous == NULL || tokens == NULL) {
    result.status = NX_IO_NOMEM;
    goto done;
  }
  for (size_t i = 0; i < NX_IO_HASH_SIZE; i++)
    head[i] = SIZE_MAX;
  for (size_t i = 0; i < NX_IO_WINDOW; i++)
    previous[i] = SIZE_MAX;

  if (in.total == 0) {
    result.status = emit_fixed_block(&bw, tokens, 0, 1);
  } else {
    size_t block_start = 0;
    while (block_start < in.total && result.status == NX_IO_OK) {
      size_t block_end = block_start + NX_IO_BLOCK;
      if (block_end < block_start || block_end > in.total)
        block_end = in.total;
      size_t count = 0;
      size_t pos = block_start;
      while (pos < block_end) {
        unsigned distance = 0;
        unsigned length =
            find_match(&in, pos, block_end, head, previous, &distance);
        insert_position(&in, pos, head, previous);
        if (length == 0) {
          tokens[count++] = (token){input_get(&in, pos), 0};
          pos++;
        } else {
          tokens[count++] = (token){(uint16_t)length, (uint16_t)distance};
          for (unsigned i = 1; i < length; i++)
            insert_position(&in, pos + i, head, previous);
          pos += length;
        }
      }
      size_t fixed_bits = token_bit_cost(tokens, count);
      dynamic_plan dynamic;
      int has_dynamic = make_dynamic_plan(tokens, count, &dynamic);
      unsigned after_header = (bw.nbits + 3u) & 7u;
      size_t stored_bits = 3u + ((8u - after_header) & 7u) + 32u;
      if (block_end - block_start > (SIZE_MAX - stored_bits) / 8u) {
        result.status = NX_IO_OVERFLOW;
        break;
      }
      stored_bits += (block_end - block_start) * 8u;
      int final = block_end == in.total;
      if (has_dynamic && dynamic.bit_cost < fixed_bits &&
          dynamic.bit_cost < stored_bits)
        result.status = emit_dynamic_block(&bw, tokens, count, final, &dynamic);
      else if (fixed_bits < stored_bits)
        result.status = emit_fixed_block(&bw, tokens, count, final);
      else
        result.status = emit_stored_block(&bw, &in, block_start,
                                          block_end - block_start, final);
      block_start = block_end;
    }
  }
  if (result.status == NX_IO_OK)
    result.status = bits_align(&bw);
  if (result.status == NX_IO_OK)
    result.status = sink_flush(&out);
  if (result.status == NX_IO_OK) {
    result.input_offset = in.total;
    result.output_size = out.length;
    *crc = input_crc32(&in);
    if (buffer != NULL) {
      *buffer = out.buffer;
      out.buffer = NULL;
    }
  }

done:
  free(head);
  free(previous);
  free(tokens);
  free(out.buffer);
  return result;
}

#ifndef NX_IO_CODEC_NO_OCAML
static void checked_input(value vsrc, value voff, value vlen,
                          const uint8_t **src, size_t *len) {
  intnat off_i = Long_val(voff);
  intnat len_i = Long_val(vlen);
  if (off_i < 0 || len_i < 0)
    caml_invalid_argument("Nx_io.deflate: negative input span");
  size_t off = (size_t)off_i;
  size_t n = (size_t)len_i;
  size_t total = caml_ba_byte_size(Caml_ba_array_val(vsrc));
  if (off > total || n > total - off)
    caml_invalid_argument("Nx_io.deflate: input span out of bounds");
  *src = (const uint8_t *)Caml_ba_data_val(vsrc) + off;
  *len = n;
}

CAMLprim value caml_nx_io_deflate_raw(value vprefix, value vsrc, value voff,
                                      value vlen) {
  CAMLparam4(vprefix, vsrc, voff, vlen);
  CAMLlocal1(vdst);
  const uint8_t *src;
  size_t src_len;
  checked_input(vsrc, voff, vlen, &src, &src_len);
  size_t prefix_len = caml_string_length(vprefix);
  if (prefix_len > SIZE_MAX - src_len)
    caml_invalid_argument("Nx_io.deflate: input is too large");
  uint8_t *prefix = prefix_len == 0 ? NULL : malloc(prefix_len);
  if (prefix_len != 0 && prefix == NULL)
    caml_raise_out_of_memory();
  if (prefix_len != 0)
    memcpy(prefix, String_val(vprefix), prefix_len);
  uint8_t *buffer = NULL;
  uint32_t crc;
  caml_release_runtime_system();
  nx_io_result result =
      nx_io_deflate_raw(prefix, prefix_len, src, src_len, -1, &buffer, &crc);
  caml_acquire_runtime_system();
  free(prefix);
  if (result.status != NX_IO_OK) {
    free(buffer);
    caml_failwith(nx_io_status_message(result.status));
  }
  if (result.output_size > (size_t)Max_long) {
    free(buffer);
    caml_failwith("Nx_io.deflate: output exceeds OCaml integer range");
  }
  vdst = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                            result.output_size);
  if (result.output_size != 0)
    memcpy(Caml_ba_data_val(vdst), buffer, result.output_size);
  free(buffer);
  CAMLreturn(vdst);
}

CAMLprim value caml_nx_io_deflate_raw_to_fd(value vfd, value vprefix,
                                            value vsrc, value voff,
                                            value vlen) {
  CAMLparam5(vfd, vprefix, vsrc, voff, vlen);
  CAMLlocal2(vresult, vcrc);
  const uint8_t *src;
  size_t src_len;
  checked_input(vsrc, voff, vlen, &src, &src_len);
  size_t prefix_len = caml_string_length(vprefix);
  if (prefix_len > SIZE_MAX - src_len)
    caml_invalid_argument("Nx_io.deflate: input is too large");
  uint8_t *prefix = prefix_len == 0 ? NULL : malloc(prefix_len);
  if (prefix_len != 0 && prefix == NULL)
    caml_raise_out_of_memory();
  if (prefix_len != 0)
    memcpy(prefix, String_val(vprefix), prefix_len);
  int fd = Int_val(vfd);
  uint32_t crc;
  caml_release_runtime_system();
  nx_io_result result =
      nx_io_deflate_raw(prefix, prefix_len, src, src_len, fd, NULL, &crc);
  int saved_errno = errno;
  caml_acquire_runtime_system();
  free(prefix);
  if (result.status == NX_IO_SYSTEM)
    unix_error(saved_errno == 0 ? EIO : saved_errno, "write", Nothing);
  if (result.status != NX_IO_OK)
    caml_failwith(nx_io_status_message(result.status));
  if (result.input_offset > (size_t)Max_long ||
      result.output_size > (size_t)Max_long)
    caml_failwith("Nx_io.deflate: result exceeds OCaml integer range");
  vcrc = caml_copy_int32((int32_t)crc);
  vresult = caml_alloc_tuple(3);
  Store_field(vresult, 0, vcrc);
  Store_field(vresult, 1, Val_long(result.input_offset));
  Store_field(vresult, 2, Val_long(result.output_size));
  CAMLreturn(vresult);
}
#endif
