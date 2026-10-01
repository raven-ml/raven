/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Snappy blocks, after the format description:
  https://github.com/google/snappy/blob/main/format_description.txt
  The decoder checks every element against both ends before it copies it.
  The encoder follows the reference: fragments of 64 KiB, a hash table of
  4-byte sequences sized by the fragment, and a search that skips faster the
  longer it finds nothing.
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdint.h>
#include <string.h>

enum {
  OK,
  TRUNCATED,
  LENGTH_VARINT,
  LENGTH_MISMATCH,
  LITERAL_PAST_INPUT,
  DATA_TOO_LONG,
  OFFSET_ZERO,
  OFFSET_BEYOND,
  DATA_TOO_SHORT
};

static inline void copy8(uint8_t *d, const uint8_t *s) {
  uint64_t v;
  memcpy(&v, s, 8);
  memcpy(d, &v, 8);
}

/* Copies [n] bytes from [offset] back. Wide copies may write up to 15 bytes
   past the match, which [room] holds and are not yet produced. */
static inline void copy_match(uint8_t *d, size_t offset, size_t n,
                              size_t room) {
  const uint8_t *s = d - offset;
  if (n + 16 > room) {
    for (size_t k = 0; k < n; k++)
      d[k] = s[k];
    return;
  }
  if (offset < 8) {
    /* The data repeats every [offset] bytes, hence every [step >= 8]. */
    size_t step = offset;
    while (step < 8)
      step += offset;
    for (size_t k = 0; k < step; k++)
      d[k] = s[k];
    if (n <= step)
      return;
    s = d;
    d += step;
    n -= step;
  }
  for (size_t k = 0; k < n; k += 8)
    copy8(d + k, s + k);
}

/* Decodes [src[0, n)] into [dst[0, m)]. [*at] is the offset in [src] of the
   element being decoded when it returns. */
static int decode(const uint8_t *src, size_t n, uint8_t *dst, size_t m,
                  size_t *at) {
  size_t ip = 0, op = 0;
  uint64_t length = 0;
  for (unsigned shift = 0;; shift += 7) {
    *at = ip;
    if (ip == n)
      return TRUNCATED;
    if (shift == 35)
      return LENGTH_VARINT;
    uint8_t b = src[ip++];
    length |= (uint64_t)(b & 0x7f) << shift;
    if ((b & 0x80) == 0)
      break;
  }
  if (length > 0xffffffffu)
    return LENGTH_VARINT;
  if (length != m)
    return LENGTH_MISMATCH;
  while (ip < n) {
    *at = ip;
    uint8_t tag = src[ip++];
    size_t len;
    if ((tag & 3) == 0) {
      len = tag >> 2;
      if (len >= 60) {
        size_t k = len - 59;
        if (n - ip < k)
          return TRUNCATED;
        len = 0;
        for (size_t i = k; i-- > 0;)
          len = (len << 8) | src[ip + i];
        ip += k;
      }
      len += 1;
      if (len > n - ip)
        return LITERAL_PAST_INPUT;
      if (len > m - op)
        return DATA_TOO_LONG;
      if (len <= 16 && n - ip >= 16 && m - op >= 16) {
        copy8(dst + op, src + ip);
        copy8(dst + op + 8, src + ip + 8);
      } else {
        memcpy(dst + op, src + ip, len);
      }
      ip += len;
      op += len;
      continue;
    }
    size_t offset;
    if ((tag & 3) == 1) {
      if (n - ip < 1)
        return TRUNCATED;
      len = 4 + ((tag >> 2) & 7);
      offset = ((size_t)(tag >> 5) << 8) | src[ip];
      ip += 1;
    } else if ((tag & 3) == 2) {
      if (n - ip < 2)
        return TRUNCATED;
      len = 1 + (tag >> 2);
      offset = src[ip] | ((size_t)src[ip + 1] << 8);
      ip += 2;
    } else {
      if (n - ip < 4)
        return TRUNCATED;
      len = 1 + (tag >> 2);
      offset = src[ip] | ((size_t)src[ip + 1] << 8) |
               ((size_t)src[ip + 2] << 16) | ((size_t)src[ip + 3] << 24);
      ip += 4;
    }
    if (offset == 0)
      return OFFSET_ZERO;
    if (offset > op)
      return OFFSET_BEYOND;
    if (len > m - op)
      return DATA_TOO_LONG;
    copy_match(dst + op, offset, len, m - op);
    op += len;
  }
  *at = n;
  return op == m ? OK : DATA_TOO_SHORT;
}

/* Encoding */

#define FRAGMENT 65536u
#define MAX_TABLE 16384u
#define INPUT_MARGIN 15u

static inline uint32_t load32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap32(v);
#endif
  return v;
}

static inline unsigned hash(uint32_t v, unsigned shift) {
  return (v * 0x1e35a7bdu) >> shift;
}

static uint8_t *emit_literal(uint8_t *op, const uint8_t *src, size_t len) {
  size_t n = len - 1;
  if (n < 60) {
    *op++ = (uint8_t)(n << 2);
  } else {
    unsigned bytes = n < 0x100 ? 1 : n < 0x10000 ? 2 : n < 0x1000000 ? 3 : 4;
    *op++ = (uint8_t)((59 + bytes) << 2);
    for (unsigned i = 0; i < bytes; i++)
      *op++ = (uint8_t)(n >> (8 * i));
  }
  memcpy(op, src, len);
  return op + len;
}

static uint8_t *emit_copy_at_most_64(uint8_t *op, size_t offset, size_t len) {
  if (len < 12 && offset < 2048) {
    *op++ = (uint8_t)(1 | ((len - 4) << 2) | ((offset >> 8) << 5));
    *op++ = (uint8_t)offset;
  } else {
    *op++ = (uint8_t)(2 | ((len - 1) << 2));
    *op++ = (uint8_t)offset;
    *op++ = (uint8_t)(offset >> 8);
  }
  return op;
}

static uint8_t *emit_copy(uint8_t *op, size_t offset, size_t len) {
  while (len >= 68) {
    op = emit_copy_at_most_64(op, offset, 64);
    len -= 64;
  }
  if (len > 64) {
    op = emit_copy_at_most_64(op, offset, 60);
    len -= 60;
  }
  return emit_copy_at_most_64(op, offset, len);
}

/* The length of the common prefix of [a] and [b], with [a < b], that ends
   before [end]. */
static size_t match_length(const uint8_t *a, const uint8_t *b,
                           const uint8_t *end) {
  const uint8_t *start = b;
  while (end - b >= 8) {
    uint64_t x, y;
    memcpy(&x, a, 8);
    memcpy(&y, b, 8);
    if (x != y)
      break;
    a += 8;
    b += 8;
  }
  while (b < end && *a == *b) {
    a++;
    b++;
  }
  return (size_t)(b - start);
}

static uint8_t *compress_fragment(const uint8_t *src, size_t len, uint8_t *op,
                                  uint16_t *table) {
  unsigned size = 256, shift = 24;
  while (size < MAX_TABLE && size < len) {
    size <<= 1;
    shift--;
  }
  memset(table, 0, size * sizeof(*table));
  size_t next_emit = 0;
  if (len >= INPUT_MARGIN) {
    size_t limit = len - INPUT_MARGIN;
    size_t ip = 1;
    unsigned next_hash = hash(load32(src + 1), shift);
    for (;;) {
      unsigned skip = 32;
      size_t next_ip = ip;
      size_t candidate;
      do {
        ip = next_ip;
        unsigned h = next_hash;
        next_ip = ip + (skip++ >> 5);
        if (next_ip > limit)
          goto remainder;
        next_hash = hash(load32(src + next_ip), shift);
        candidate = table[h];
        table[h] = (uint16_t)ip;
      } while (load32(src + ip) != load32(src + candidate));
      op = emit_literal(op, src + next_emit, ip - next_emit);
      do {
        size_t start = ip;
        size_t matched =
            4 + match_length(src + candidate + 4, src + ip + 4, src + len);
        ip += matched;
        op = emit_copy(op, start - candidate, matched);
        next_emit = ip;
        if (ip >= limit)
          goto remainder;
        table[hash(load32(src + ip - 1), shift)] = (uint16_t)(ip - 1);
        unsigned h = hash(load32(src + ip), shift);
        candidate = table[h];
        table[h] = (uint16_t)ip;
      } while (load32(src + ip) == load32(src + candidate));
      next_hash = hash(load32(src + ip + 1), shift);
      ip++;
    }
  }
remainder:
  if (next_emit < len)
    op = emit_literal(op, src + next_emit, len - next_emit);
  return op;
}

static size_t encode(const uint8_t *src, size_t n, uint8_t *dst) {
  uint16_t table[MAX_TABLE];
  uint8_t *op = dst;
  for (size_t v = n;; v >>= 7) {
    if (v < 0x80) {
      *op++ = (uint8_t)v;
      break;
    }
    *op++ = (uint8_t)(v | 0x80);
  }
  for (size_t at = 0; at < n; at += FRAGMENT) {
    size_t len = n - at < FRAGMENT ? n - at : FRAGMENT;
    op = compress_fragment(src + at, len, op, table);
  }
  return (size_t)(op - dst);
}

CAMLprim value caml_compress_snappy_overlap(value a, value b) {
  uintptr_t pa = (uintptr_t)Caml_ba_data_val(a);
  uintptr_t pb = (uintptr_t)Caml_ba_data_val(b);
  size_t la = Caml_ba_array_val(a)->dim[0], lb = Caml_ba_array_val(b)->dim[0];
  return Val_bool(la != 0 && lb != 0 && pa < pb + lb && pb < pa + la);
}

/* Returns the status; [at.(0)] receives the offset where decoding stopped. */
CAMLprim value caml_compress_snappy_decompress(value vsrc, value vdst,
                                               value vat) {
  CAMLparam3(vsrc, vdst, vat);
  const uint8_t *src = Caml_ba_data_val(vsrc);
  size_t n = Caml_ba_array_val(vsrc)->dim[0];
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t m = Caml_ba_array_val(vdst)->dim[0];
  size_t at;
  int status;
  if (n + m > 65536) {
    caml_release_runtime_system();
    status = decode(src, n, dst, m, &at);
    caml_acquire_runtime_system();
  } else {
    status = decode(src, n, dst, m, &at);
  }
  Field(vat, 0) = Val_long(at);
  CAMLreturn(Val_int(status));
}

/* [dst] holds at least [max_compressed_length] bytes. */
CAMLprim value caml_compress_snappy_compress(value vsrc, value vdst) {
  CAMLparam2(vsrc, vdst);
  const uint8_t *src = Caml_ba_data_val(vsrc);
  size_t n = Caml_ba_array_val(vsrc)->dim[0];
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t written;
  if (n > 65536) {
    caml_release_runtime_system();
    written = encode(src, n, dst);
    caml_acquire_runtime_system();
  } else {
    written = encode(src, n, dst);
  }
  CAMLreturn(Val_long(written));
}
