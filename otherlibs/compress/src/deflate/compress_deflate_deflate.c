/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  A streaming RFC 1951 encoder. Input is cut into blocks of 65535 bytes at
  fixed offsets, so the output does not depend on how it is fed. Each block
  is stored, or coded with fixed or dynamic Huffman codes, whichever takes
  fewest bits. Matches are found on hash chains over the 32 KiB history,
  followed at most a level-dependent number of links, and never cross a block
  end.
  ---------------------------------------------------------------------------*/

#include "compress_deflate.h"

#include <stdlib.h>
#include <string.h>

#define BLOCK 65535u
#define WINDOW 32768u
#define BUFFER (WINDOW + 2u * 65536u)
#define HASH_BITS 16u
#define HASH_SIZE (1u << HASH_BITS)
#define NONE SIZE_MAX
#define TOO_FAR 4096u
#define LITLEN_CODES 286u
#define DIST_CODES 30u
#define CODELEN_CODES 19u
#define MAX_CODELEN_TOKENS (LITLEN_CODES + DIST_CODES)

typedef struct {
  uint16_t value; /* A literal byte, or a match length when [distance > 0]. */
  uint16_t distance;
} token;

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
  uint8_t litlen_length[LITLEN_CODES];
  uint8_t dist_length[DIST_CODES];
  uint16_t litlen_code[LITLEN_CODES];
  uint16_t dist_code[DIST_CODES];
  uint8_t codelen_length[CODELEN_CODES];
  uint16_t codelen_code[CODELEN_CODES];
  codelen_token codelen_tokens[MAX_CODELEN_TOKENS];
  size_t codelen_count;
  unsigned nlit;
  unsigned ndist;
  unsigned ncode;
  size_t bit_cost;
} dynamic_plan;

/* Positions are absolute input offsets: [buf[0]] is the byte at [base]. */
struct compress_deflate {
  unsigned chain; /* Links followed per match search; 0 stores blocks. */
  uint8_t *buf;
  size_t base;
  size_t end;   /* The end of the input given so far. */
  size_t block; /* The start of the next block to encode. */
  int finished;
  size_t *head;
  size_t *previous;
  token *tokens;
  uint64_t bits; /* Bits not yet written as a whole byte. */
  unsigned nbits;
  uint8_t *out;
  size_t out_len;
};

static const uint8_t codelen_order[CODELEN_CODES] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

static inline uint8_t at(const compress_deflate *e, size_t pos) {
  return e->buf[pos - e->base];
}

/* Bit output */

static void bits_put(compress_deflate *e, uint32_t value, unsigned count) {
  e->bits |= (uint64_t)value << e->nbits;
  e->nbits += count;
  while (e->nbits >= 8) {
    e->out[e->out_len++] = (uint8_t)e->bits;
    e->bits >>= 8;
    e->nbits -= 8;
  }
}

static void bits_align(compress_deflate *e) {
  if (e->nbits != 0)
    e->out[e->out_len++] = (uint8_t)e->bits;
  e->bits = 0;
  e->nbits = 0;
}

static uint32_t reverse_bits(uint32_t value, unsigned count) {
  uint32_t reversed = 0;
  for (unsigned i = 0; i < count; i++) {
    reversed = (reversed << 1) | (value & 1u);
    value >>= 1;
  }
  return reversed;
}

/* Symbols */

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

/* Matching */

static unsigned hash3(const compress_deflate *e, size_t pos) {
  uint32_t value = (uint32_t)at(e, pos) * 0x1e35a7bdu;
  value ^= (uint32_t)at(e, pos + 1) * 0x9e3779b1u;
  value ^= (uint32_t)at(e, pos + 2) * 0x85ebca77u;
  return value >> (32u - HASH_BITS);
}

static void insert_position(compress_deflate *e, size_t pos) {
  if (pos + 2 >= e->end)
    return;
  unsigned hash = hash3(e, pos);
  e->previous[pos & (WINDOW - 1u)] = e->head[hash];
  e->head[hash] = pos;
}

static unsigned find_match(const compress_deflate *e, size_t pos,
                           size_t block_end, unsigned *best_distance) {
  if (pos + 2 >= block_end)
    return 0;
  size_t candidate = e->head[hash3(e, pos)];
  unsigned best = 2;
  unsigned limit = (unsigned)(block_end - pos);
  if (limit > 258)
    limit = 258;
  const uint8_t *here = e->buf + (pos - e->base);
  for (unsigned chain = 0; candidate != NONE && chain < e->chain; chain++) {
    if (candidate >= pos || pos - candidate > WINDOW)
      break;
    const uint8_t *there = e->buf + (candidate - e->base);
    if (there[best] == here[best]) {
      unsigned length = 0;
      while (length < limit && there[length] == here[length])
        length++;
      if (length > best) {
        best = length;
        *best_distance = (unsigned)(pos - candidate);
        if (best == limit)
          break;
      }
    }
    size_t next = e->previous[candidate & (WINDOW - 1u)];
    if (next >= candidate)
      break;
    candidate = next;
  }
  /* Long searches find 3-byte matches far back, whose distances cost more
     bits than the literals they replace. */
  if (best == 3 && *best_distance > TOO_FAR && e->chain > 4)
    return 0;
  return best >= 3 ? best : 0;
}

/* Huffman codes */

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

/* Code lengths of at most [max_bits] for [frequency]. A single used symbol
   gets a second, unused one, so that every code is complete as decoders
   require. */
static int build_lengths(const uint32_t *frequency, unsigned symbols,
                         unsigned max_bits, uint8_t *length) {
  huffman_node nodes[2 * LITLEN_CODES];
  int heap[2 * LITLEN_CODES];
  unsigned used_symbols[LITLEN_CODES];
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
    length[used_symbols[0] == 0 ? 1 : 0] = 1;
    return 1;
  }

  unsigned node_count = used;
  while (heap_count > 1) {
    int left = heap_pop(heap, &heap_count, nodes);
    int right = heap_pop(heap, &heap_count, nodes);
    uint32_t sum = nodes[left].frequency + nodes[right].frequency;
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
  for (unsigned leaf = 0; leaf < used; leaf++) {
    unsigned depth = 0;
    for (int node = (int)leaf; nodes[node].parent >= 0;
         node = nodes[node].parent)
      depth++;
    counts[depth > max_bits ? max_bits : depth]++;
  }
  /* Clamping deep leaves to [max_bits] over-subscribes the code by [excess]
     units of 2^-max_bits, fewer units than there are leaves at [max_bits].
     Each step moves the deepest leaf above [max_bits] one level down, beside a
     leaf taken from [max_bits], and lowers the excess by one unit. */
  unsigned long excess = 0;
  for (unsigned bits = 1; bits <= max_bits; bits++)
    excess += (unsigned long)counts[bits] << (max_bits - bits);
  excess -= 1ul << max_bits;
  while (excess != 0) {
    unsigned bits = max_bits - 1;
    while (counts[bits] == 0)
      bits--;
    counts[bits]--;
    counts[bits + 1] += 2;
    counts[max_bits]--;
    excess--;
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
  for (unsigned symbol = 0; symbol < symbols; symbol++)
    counts[length[symbol]]++;
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

static void add_codelen_token(dynamic_plan *plan, uint32_t *frequency,
                              unsigned symbol, unsigned extra,
                              unsigned extra_bits) {
  plan->codelen_tokens[plan->codelen_count++] =
      (codelen_token){(uint8_t)symbol, (uint8_t)extra_bits, (uint16_t)extra};
  frequency[symbol]++;
}

/* Run-length codes [lengths]: at most one token per length. */
static void encode_codelengths(dynamic_plan *plan, const uint8_t *lengths,
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
        add_codelen_token(plan, frequency, 18, repeat - 11, 7);
        run -= repeat;
      }
      if (run >= 3) {
        unsigned repeat = run > 10 ? 10 : run;
        add_codelen_token(plan, frequency, 17, repeat - 3, 3);
        run -= repeat;
      }
      while (run-- != 0)
        add_codelen_token(plan, frequency, 0, 0, 0);
    } else {
      add_codelen_token(plan, frequency, value, 0, 0);
      run--;
      while (run >= 3) {
        unsigned repeat = run > 6 ? 6 : run;
        add_codelen_token(plan, frequency, 16, repeat - 3, 2);
        run -= repeat;
      }
      while (run-- != 0)
        add_codelen_token(plan, frequency, value, 0, 0);
    }
  }
}

static int make_dynamic_plan(const token *tokens, size_t count,
                             dynamic_plan *plan) {
  uint32_t litlen_frequency[LITLEN_CODES] = {0};
  uint32_t dist_frequency[DIST_CODES] = {0};
  memset(plan, 0, sizeof(*plan));
  litlen_frequency[256] = 1;
  for (size_t i = 0; i < count; i++) {
    unsigned symbol, extra, extra_bits;
    if (tokens[i].distance == 0) {
      litlen_frequency[tokens[i].value]++;
    } else {
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      litlen_frequency[symbol]++;
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      dist_frequency[symbol]++;
    }
  }
  int has_distances = 0;
  for (unsigned symbol = 0; symbol < DIST_CODES; symbol++)
    has_distances |= dist_frequency[symbol] != 0;
  if (!has_distances)
    dist_frequency[0] = 1;
  if (!build_lengths(litlen_frequency, LITLEN_CODES, 15, plan->litlen_length) ||
      !build_lengths(dist_frequency, DIST_CODES, 15, plan->dist_length) ||
      !build_codes(plan->litlen_length, LITLEN_CODES, 15, plan->litlen_code) ||
      !build_codes(plan->dist_length, DIST_CODES, 15, plan->dist_code))
    return 0;

  plan->nlit = LITLEN_CODES;
  while (plan->nlit > 257 && plan->litlen_length[plan->nlit - 1] == 0)
    plan->nlit--;
  plan->ndist = DIST_CODES;
  while (plan->ndist > 1 && plan->dist_length[plan->ndist - 1] == 0)
    plan->ndist--;
  uint8_t lengths[LITLEN_CODES + DIST_CODES];
  memcpy(lengths, plan->litlen_length, plan->nlit);
  memcpy(lengths + plan->nlit, plan->dist_length, plan->ndist);
  uint32_t codelen_frequency[CODELEN_CODES] = {0};
  encode_codelengths(plan, lengths, plan->nlit + plan->ndist,
                     codelen_frequency);
  if (!build_lengths(codelen_frequency, CODELEN_CODES, 7,
                     plan->codelen_length) ||
      !build_codes(plan->codelen_length, CODELEN_CODES, 7, plan->codelen_code))
    return 0;
  plan->ncode = CODELEN_CODES;
  while (plan->ncode > 4 &&
         plan->codelen_length[codelen_order[plan->ncode - 1]] == 0)
    plan->ncode--;

  size_t cost = 3 + 5 + 5 + 4 + plan->ncode * 3;
  for (size_t i = 0; i < plan->codelen_count; i++) {
    codelen_token item = plan->codelen_tokens[i];
    cost += plan->codelen_length[item.symbol] + item.extra_bits;
  }
  cost += plan->litlen_length[256];
  for (size_t i = 0; i < count; i++) {
    unsigned symbol, extra, extra_bits;
    if (tokens[i].distance == 0) {
      cost += plan->litlen_length[tokens[i].value];
    } else {
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      cost += plan->litlen_length[symbol] + extra_bits;
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      cost += plan->dist_length[symbol] + extra_bits;
    }
  }
  plan->bit_cost = cost;
  return 1;
}

static size_t fixed_bit_cost(const token *tokens, size_t count) {
  size_t cost = 3 + fixed_code_bits(256);
  for (size_t i = 0; i < count; i++) {
    unsigned symbol, extra, extra_bits;
    if (tokens[i].distance == 0) {
      cost += fixed_code_bits(tokens[i].value);
    } else {
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      cost += fixed_code_bits(symbol) + extra_bits;
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      cost += 5 + extra_bits;
    }
  }
  return cost;
}

/* Blocks */

static void emit_fixed_symbol(compress_deflate *e, unsigned symbol) {
  uint32_t code;
  unsigned bits;
  fixed_code(symbol, &code, &bits);
  bits_put(e, code, bits);
}

static void emit_fixed_block(compress_deflate *e, const token *tokens,
                             size_t count, int final) {
  bits_put(e, (uint32_t)final | 2u, 3);
  for (size_t i = 0; i < count; i++) {
    unsigned symbol, extra, extra_bits;
    if (tokens[i].distance == 0) {
      emit_fixed_symbol(e, tokens[i].value);
    } else {
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      emit_fixed_symbol(e, symbol);
      bits_put(e, extra, extra_bits);
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      bits_put(e, reverse_bits(symbol, 5), 5);
      bits_put(e, extra, extra_bits);
    }
  }
  emit_fixed_symbol(e, 256);
}

static void emit_dynamic_block(compress_deflate *e, const token *tokens,
                               size_t count, int final,
                               const dynamic_plan *plan) {
  bits_put(e, (uint32_t)final | 4u, 3);
  bits_put(e, plan->nlit - 257, 5);
  bits_put(e, plan->ndist - 1, 5);
  bits_put(e, plan->ncode - 4, 4);
  for (unsigned i = 0; i < plan->ncode; i++)
    bits_put(e, plan->codelen_length[codelen_order[i]], 3);
  for (size_t i = 0; i < plan->codelen_count; i++) {
    codelen_token item = plan->codelen_tokens[i];
    bits_put(e, plan->codelen_code[item.symbol],
             plan->codelen_length[item.symbol]);
    bits_put(e, item.extra, item.extra_bits);
  }
  for (size_t i = 0; i < count; i++) {
    unsigned symbol, extra, extra_bits;
    if (tokens[i].distance == 0) {
      bits_put(e, plan->litlen_code[tokens[i].value],
               plan->litlen_length[tokens[i].value]);
    } else {
      length_symbol(tokens[i].value, &symbol, &extra, &extra_bits);
      bits_put(e, plan->litlen_code[symbol], plan->litlen_length[symbol]);
      bits_put(e, extra, extra_bits);
      distance_symbol(tokens[i].distance, &symbol, &extra, &extra_bits);
      bits_put(e, plan->dist_code[symbol], plan->dist_length[symbol]);
      bits_put(e, extra, extra_bits);
    }
  }
  bits_put(e, plan->litlen_code[256], plan->litlen_length[256]);
}

static void emit_stored_block(compress_deflate *e, size_t start, size_t length,
                              int final) {
  bits_put(e, (uint32_t)final, 3);
  bits_align(e);
  uint8_t *p = e->out + e->out_len;
  p[0] = (uint8_t)length;
  p[1] = (uint8_t)(length >> 8);
  p[2] = (uint8_t)~length;
  p[3] = (uint8_t)(~length >> 8);
  memcpy(p + 4, e->buf + (start - e->base), length);
  e->out_len += 4 + length;
}

static void encode_block(compress_deflate *e, size_t start, size_t end,
                         int final) {
  if (e->chain == 0) {
    emit_stored_block(e, start, end - start, final);
    return;
  }
  size_t count = 0;
  size_t pos = start;
  while (pos < end) {
    unsigned distance = 0;
    unsigned length = find_match(e, pos, end, &distance);
    insert_position(e, pos);
    if (length == 0) {
      e->tokens[count++] = (token){at(e, pos), 0};
      pos++;
    } else {
      e->tokens[count++] = (token){(uint16_t)length, (uint16_t)distance};
      for (unsigned i = 1; i < length; i++)
        insert_position(e, pos + i);
      pos += length;
    }
  }
  size_t fixed_bits = fixed_bit_cost(e->tokens, count);
  dynamic_plan dynamic;
  int has_dynamic = make_dynamic_plan(e->tokens, count, &dynamic);
  unsigned after_header = (e->nbits + 3u) & 7u;
  size_t stored_bits = 3u + ((8u - after_header) & 7u) + 32u + (end - start) * 8u;
  if (has_dynamic && dynamic.bit_cost < fixed_bits &&
      dynamic.bit_cost < stored_bits)
    emit_dynamic_block(e, e->tokens, count, final, &dynamic);
  else if (fixed_bits < stored_bits)
    emit_fixed_block(e, e->tokens, count, final);
  else
    emit_stored_block(e, start, end - start, final);
}

/* Encoders */

compress_deflate *compress_deflate_create(int level) {
  static const unsigned chains[10] = {0, 1, 2, 2, 3, 3, 4, 16, 64, 256};
  compress_deflate *e = calloc(1, sizeof(*e));
  if (e == NULL)
    return NULL;
  e->chain = chains[level];
  e->buf = malloc(BUFFER);
  if (e->buf == NULL)
    goto fail;
  if (e->chain != 0) {
    e->head = malloc(HASH_SIZE * sizeof(*e->head));
    e->previous = malloc(WINDOW * sizeof(*e->previous));
    e->tokens = malloc(BLOCK * sizeof(*e->tokens));
    if (e->head == NULL || e->previous == NULL || e->tokens == NULL)
      goto fail;
    for (size_t i = 0; i < HASH_SIZE; i++)
      e->head[i] = NONE;
    for (size_t i = 0; i < WINDOW; i++)
      e->previous[i] = NONE;
  }
  return e;
fail:
  compress_deflate_free(e);
  return NULL;
}

void compress_deflate_free(compress_deflate *e) {
  if (e == NULL)
    return;
  free(e->buf);
  free(e->head);
  free(e->previous);
  free(e->tokens);
  free(e);
}

size_t compress_deflate_input(compress_deflate *e, const uint8_t *src,
                              size_t len) {
  size_t keep = e->block > WINDOW ? e->block - WINDOW : 0;
  if (keep > e->base && e->end - e->base + len > BUFFER) {
    memmove(e->buf, e->buf + (keep - e->base), e->end - keep);
    e->base = keep;
  }
  size_t room = BUFFER - (e->end - e->base);
  if (len > room)
    len = room;
  memcpy(e->buf + (e->end - e->base), src, len);
  e->end += len;
  return len;
}

/* A block is encoded once two bytes past it are known, or at [eod]: hashing
   the last positions of a block reads them, as the encoder of a whole input
   would. */
size_t compress_deflate_encode(compress_deflate *e, uint8_t *out, int eod) {
  if (e->finished)
    return 0;
  size_t available = e->end - e->block;
  if (available < BLOCK + 2 && !eod)
    return 0;
  e->out = out;
  e->out_len = 0;
  size_t block_end = e->block + (available < BLOCK ? available : BLOCK);
  int final = eod && block_end == e->end;
  if (available == 0 && e->chain != 0)
    emit_fixed_block(e, NULL, 0, 1);
  else
    encode_block(e, e->block, block_end, final);
  e->block = block_end;
  if (final) {
    bits_align(e);
    e->finished = 1;
  }
  return e->out_len;
}
