/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_packed.h — the sub-byte family: storage whose elements are narrower
   than a byte, `bits` wide (4 for int4 and uint4, 1 for bit).

   Storage. Element i of a buffer is bits [i * bits, (i + 1) * bits) of the
   buffer, counted from bit 0 of byte 0, the least significant bit of a byte
   first: bit i mod 8 of byte i / 8 for bit, the low nibble first for int4. A
   view's offset counts elements, so a view may start at any bit. The bits of
   a buffer past its last element belong to no element.

   Reading. A source is read by its elements in C order, up to 64 bits at a
   time (nx_c_packed_read): a unit-stride run with two word loads and a funnel
   shift, any other stride element by element. Every load stays inside the
   bytes the view reaches: a last partial word is loaded byte by byte.

   Writing. A destination is written by the 64-bit words of its storage,
   counted from the buffer's first byte (nx_c_packed_write). One worker owns
   each word and computes it from the destination elements it covers, so no
   two workers write one byte. A word the destination covers in part is read,
   merged under a mask and written back by its owner, and the store touches
   only bytes that hold destination elements. A destination whose elements
   are its buffer's last ones writes the bits past them as 0, which belong to
   no value (nx_backend_intf.mli): the destinations nx.cpu writes are fresh
   buffers or windows of them, never another value's storage. A
   destination that is not one run of its storage is written by one worker,
   one row at a time. */

#ifndef NX_C_PACKED_H
#define NX_C_PACKED_H

#include "nx_c_engine.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Words are little-endian: byte k of a word holds bits [8k, 8k + 8). */
static inline uint64_t nx_c_le64(uint64_t w) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return __builtin_bswap64(w);
#else
  return w;
#endif
}

static inline uint64_t nx_c_ld64(const uint8_t *p) {
  uint64_t w;
  memcpy(&w, p, 8);
  return nx_c_le64(w);
}

static inline void nx_c_st64(uint8_t *p, uint64_t w) {
  w = nx_c_le64(w);
  memcpy(p, &w, 8);
}

/* The low n bits set, for n in [0, 64]. */
static inline uint64_t nx_c_low_bits(int n) {
  return n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1);
}

/* Bits [bit, bit + n) of the storage at base, n in [1, 64], in the low n bits
   of the result. Reads only bytes bit / 8 to (bit + n - 1) / 8. */
static inline uint64_t nx_c_bits_load(const uint8_t *base, int64_t bit,
                                      int n) {
  const uint8_t *p = base + (bit >> 3);
  int s = (int)(bit & 7);
  int bytes = (s + n + 7) >> 3;
  uint64_t w;
  if (bytes >= 8) {
    w = nx_c_ld64(p);
    if (s) {
      w >>= s;
      if (bytes == 9) w |= (uint64_t)p[8] << (64 - s);
    }
  } else {
    w = 0;
    for (int i = 0; i < bytes; i++) w |= (uint64_t)p[i] << (8 * i);
    w >>= s;
  }
  return w & nx_c_low_bits(n);
}

/* Writes the low n bits of v at bits [bit, bit + n), keeping the other bits of
   the bytes it touches, bit / 8 to (bit + n - 1) / 8. The bits lie in one
   64-bit word of the storage, counted from base: (bit & 63) + n <= 64. */
static inline void nx_c_bits_store(uint8_t *base, int64_t bit, int n,
                                   uint64_t v) {
  uint8_t *p = base + (bit >> 3);
  int s = (int)(bit & 7);
  int bytes = (s + n + 7) >> 3;
  uint64_t mask = nx_c_low_bits(n) << s;
  uint64_t val = (v << s) & mask;
  if (bytes == 8) {
    nx_c_st64(p, (nx_c_ld64(p) & ~mask) | val);
    return;
  }
  for (int i = 0; i < bytes; i++) {
    uint8_t m = (uint8_t)(mask >> (8 * i));
    p[i] = (uint8_t)((p[i] & ~m) | (uint8_t)(val >> (8 * i)));
  }
}

/* Element i of packed storage, and its replacement, for the movers that touch
   one element at a time. */
static inline uint8_t nx_c_packed_get(const void *base, int64_t i, int bits) {
  int64_t bit = i * bits;
  uint8_t b = ((const uint8_t *)base)[bit >> 3];
  return (uint8_t)((b >> (bit & 7)) & ((1u << bits) - 1));
}

static inline void nx_c_packed_set(void *base, int64_t i, int bits,
                                   uint8_t v) {
  int64_t bit = i * bits;
  uint8_t *b = (uint8_t *)base + (bit >> 3);
  uint8_t m = (uint8_t)(((1u << bits) - 1) << (bit & 7));
  *b = (uint8_t)((*b & ~m) | ((v << (bit & 7)) & m));
}

/* k copies of the bits-wide element v, k * bits <= 64. */
static inline uint64_t nx_c_packed_splat(uint64_t v, int bits, int k) {
  uint64_t w = 0;
  if (v) {
    uint64_t one = bits == 1 ? ~(uint64_t)0 : 0x1111111111111111u * v;
    w = one;
  }
  return w & nx_c_low_bits(k * bits);
}

/* A packed operand read by its elements in C order: its view with size-1 axes
   dropped and neighbours that are one run merged, so a C-contiguous view is
   one axis of stride 1. */
typedef struct {
  const uint8_t *base;
  int bits;
  int ndim; /* >= 1 */
  int64_t offset;
  int64_t shape[NX_C_MAX_NDIM];
  int64_t strides[NX_C_MAX_NDIM];
} nx_c_packed_src;

void nx_c_packed_src_init(nx_c_packed_src *s, const nx_c_ndarray *a, int bits);

/* Whether s is one run of stride 1, read from bit (s->offset + e) * bits. */
static inline bool nx_c_packed_dense(const nx_c_packed_src *s) {
  return s->ndim == 1 && s->strides[0] == 1;
}

/* Elements [e, e + k) of s in C order, k * bits <= 64, element e + j in bits
   [j * bits, (j + 1) * bits) of the result and the bits above them 0. */
uint64_t nx_c_packed_read_any(const nx_c_packed_src *s, int64_t e, int k);

static inline uint64_t nx_c_packed_read(const nx_c_packed_src *s, int64_t e,
                                        int k) {
  if (nx_c_packed_dense(s))
    return nx_c_bits_load(s->base, (s->offset + e) * s->bits, k * s->bits);
  return nx_c_packed_read_any(s, e, k);
}

/* What a destination's elements hold: [word ctx e k] is the bits of its
   elements [e, e + k) in C order, as nx_c_packed_read returns them, and
   [words ctx e n dst], when not NULL, writes the 8 * n bytes of n whole
   words of them from element e. Both are called by several workers at
   once. */
typedef uint64_t nx_c_packed_fill(const void *ctx, int64_t e, int k);
typedef void nx_c_packed_fill_words(const void *ctx, int64_t e, int64_t n,
                                    uint8_t *dst);

typedef struct {
  nx_c_packed_fill *word;
  nx_c_packed_fill_words *words;
  const void *ctx;
} nx_c_packed_filler;

/* Writes every element of out, a packed array of `bits`-wide elements, with
   f. `bytes` is the traffic, for the parallel policy. */
nx_c_status nx_c_packed_write(const nx_c_ndarray *out, int bits,
                              const nx_c_packed_filler *f, int64_t bytes);

/* The movers. Each writes out, of in's shape (copy) or indices' (gather), as
   nx_c_packed_write does. */
nx_c_status nx_c_packed_copy(const nx_c_ndarray *out, const nx_c_ndarray *in,
                             nx_c_dtype dt);
nx_c_status nx_c_packed_gather(const nx_c_ndarray *out,
                               const nx_c_ndarray *data,
                               const nx_c_ndarray *indices, int axis,
                               nx_c_dtype dt);

/* The logical operations of bit operands of one shape, word by word. */
typedef enum {
  NX_C_BIT_AND,
  NX_C_BIT_OR,
  NX_C_BIT_XOR,
} nx_c_bit_op;

nx_c_status nx_c_bit_logic(nx_c_bit_op op, const nx_c_ndarray *out,
                           const nx_c_ndarray *a, const nx_c_ndarray *b);

/* Packing and unpacking, n <= 64: the n booleans at `bytes` as the low n
   bits of a word, element j in bit j and every non-zero byte a 1; and the low
   n bits of `bits` as n bytes of 0 and 1. NEON on arm64, and multiplies that
   clang does not vectorise elsewhere. */
static inline uint64_t nx_c_bit_pack(const uint8_t *bytes, int n) {
  uint64_t w = 0;
  int i = 0;
#if defined(__ARM_NEON)
  if (n == 64) {
    /* Each lane keeps its weight, 1 << (lane mod 8), where it is non-zero,
       and three rounds of pairwise sums add each 8 lanes into one byte. */
    static const uint8_t weights[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                        1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t m = vld1q_u8(weights);
    uint8x16_t x0 = vld1q_u8(bytes), x1 = vld1q_u8(bytes + 16);
    uint8x16_t x2 = vld1q_u8(bytes + 32), x3 = vld1q_u8(bytes + 48);
    x0 = vandq_u8(vtstq_u8(x0, x0), m);
    x1 = vandq_u8(vtstq_u8(x1, x1), m);
    x2 = vandq_u8(vtstq_u8(x2, x2), m);
    x3 = vandq_u8(vtstq_u8(x3, x3), m);
    uint8x16_t p = vpaddq_u8(vpaddq_u8(x0, x1), vpaddq_u8(x2, x3));
    p = vpaddq_u8(p, p);
    return vgetq_lane_u64(vreinterpretq_u64_u8(p), 0);
  }
#endif
  /* Eight booleans to one byte: each non-zero byte becomes 1 by carrying it
     into its top bit, and the product moves byte j's low bit to bit 56 + j. */
  for (; i + 8 <= n; i += 8) {
    uint64_t x = nx_c_ld64(bytes + i);
    x = (((x & 0x7f7f7f7f7f7f7f7fu) + 0x7f7f7f7f7f7f7f7fu) | x) >> 7;
    x &= 0x0101010101010101u;
    w |= ((x * 0x0102040810204080u) >> 56) << i;
  }
  for (; i < n; i++) w |= (uint64_t)(bytes[i] != 0) << i;
  return w;
}

static inline void nx_c_bit_unpack(uint8_t *bytes, uint64_t bits, int n) {
  int i = 0;
#if defined(__ARM_NEON)
  if (n == 64) {
    /* Lane j of each group of 16 holds byte j / 8 of the word, tested against
       bit j mod 8. */
    static const uint8_t masks[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                      1, 2, 4, 8, 16, 32, 64, 128};
    static const uint8_t lanes[4][16] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
        {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7}};
    uint8x16_t m = vld1q_u8(masks), one = vdupq_n_u8(1);
    uint8x8_t b = vcreate_u8(bits);
    uint8x16_t src = vcombine_u8(b, b);
    for (int g = 0; g < 4; g++) {
      uint8x16_t x = vqtbl1q_u8(src, vld1q_u8(lanes[g]));
      vst1q_u8(bytes + 16 * g, vandq_u8(vtstq_u8(x, m), one));
    }
    return;
  }
#endif
  /* One byte to eight: spread it to every byte, keep byte j's bit j, and turn
     each non-zero byte into 1 by carrying it into the byte's top bit. */
  for (; i + 8 <= n; i += 8) {
    uint64_t t = ((bits >> i) & 0xff) * 0x0101010101010101u;
    t &= 0x8040201008040201u;
    t = ((t + 0x7f7f7f7f7f7f7f7fu) >> 7) & 0x0101010101010101u;
    nx_c_st64(bytes + i, t);
  }
  for (; i < n; i++) bytes[i] = (uint8_t)((bits >> i) & 1);
}

/* Nibbles to bytes and back, n <= 16: the low 4 bits of the n bytes at
   `bytes` as the low 4n bits of a word, element j in bits [4j, 4j + 4); and
   the low 4n bits of `nibs` as n bytes, sign-extended when `sign`. */
static inline uint64_t nx_c_nib_pack(const uint8_t *bytes, int n) {
  uint64_t w = 0;
  int i = 0;
#if defined(__ARM_NEON)
  if (n == 16) {
    /* Even elements are the low nibbles, odd ones the high. */
    uint8x8x2_t x = vld2_u8(bytes);
    uint8x8_t lo = vand_u8(x.val[0], vdup_n_u8(0x0f));
    uint8x8_t r = vorr_u8(lo, vshl_n_u8(x.val[1], 4));
    return vget_lane_u64(vreinterpret_u64_u8(r), 0);
  }
#endif
  /* Eight bytes to eight nibbles: each round halves the gaps between them. */
  for (; i + 8 <= n; i += 8) {
    uint64_t x = nx_c_ld64(bytes + i) & 0x0f0f0f0f0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00ffu;
    x = (x | (x >> 8)) & 0x0000ffff0000ffffu;
    x = (x | (x >> 16)) & 0xffffffffu;
    w |= x << (4 * i);
  }
  for (; i < n; i++) w |= (uint64_t)(bytes[i] & 0x0f) << (4 * i);
  return w;
}

static inline void nx_c_nib_unpack(uint8_t *bytes, uint64_t nibs, int n,
                                   bool sign) {
  int i = 0;
#if defined(__ARM_NEON)
  if (n == 16) {
    uint8x8_t b = vcreate_u8(nibs);
    uint8x8x2_t z = vzip_u8(vand_u8(b, vdup_n_u8(0x0f)), vshr_n_u8(b, 4));
    uint8x16_t x = vcombine_u8(z.val[0], z.val[1]);
    if (sign)
      x = vreinterpretq_u8_s8(
          vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(x), 4), 4));
    vst1q_u8(bytes, x);
    return;
  }
#endif
  /* Eight nibbles to eight bytes: each round doubles the gaps between them,
     and a set bit 3 fills the byte's high nibble. */
  for (; i + 8 <= n; i += 8) {
    uint64_t t = (nibs >> (4 * i)) & 0xffffffffu;
    t = (t | (t << 16)) & 0x0000ffff0000ffffu;
    t = (t | (t << 8)) & 0x00ff00ff00ff00ffu;
    t = (t | (t << 4)) & 0x0f0f0f0f0f0f0f0fu;
    if (sign) t |= ((t >> 3) & 0x0101010101010101u) * 0xf0;
    nx_c_st64(bytes + i, t);
  }
  for (; i < n; i++) {
    uint8_t v = (uint8_t)((nibs >> (4 * i)) & 0x0f);
    bytes[i] = sign && (v & 8) ? (uint8_t)(v | 0xf0) : v;
  }
}

/* A sub-byte dtype's elements as bytes: bit as bool, int4 as int8 and uint4
   as uint8. Its values are the byte dtype's, and a byte becomes an element
   by its low bits, a bool's by whether it is non-zero. */
static inline nx_c_dtype nx_c_packed_via(nx_c_dtype dt) {
  switch (dt) {
  case NX_C_DTYPE_bit:
    return NX_C_DTYPE_bool_;
  case NX_C_DTYPE_i4:
    return NX_C_DTYPE_i8;
  default:
    return NX_C_DTYPE_u8;
  }
}

/* The n elements of `bits`-wide dtype dt in v as its bytes, and back. */
static inline void nx_c_packed_widen(uint8_t *bytes, uint64_t v, int n,
                                     nx_c_dtype dt) {
  if (dt == NX_C_DTYPE_bit)
    nx_c_bit_unpack(bytes, v, n);
  else
    nx_c_nib_unpack(bytes, v, n, dt == NX_C_DTYPE_i4);
}

static inline uint64_t nx_c_packed_narrow(const uint8_t *bytes, int n,
                                          nx_c_dtype dt) {
  return dt == NX_C_DTYPE_bit ? nx_c_bit_pack(bytes, n)
                              : nx_c_nib_pack(bytes, n);
}

/* Runs of whole words. The n elements of dt from element `first` of the
   storage at base, as bytes; and n words of storage packed from the bytes
   of 64 / bits elements each. The choice of width stays out of the loops. */
static inline void nx_c_packed_widen_run(uint8_t *bytes, const uint8_t *base,
                                         int64_t first, int64_t n,
                                         nx_c_dtype dt) {
  int bits = nx_c_packed_bits(dt), per = 64 / bits;
  int64_t j = 0;
  if (dt == NX_C_DTYPE_bit)
    for (; j + per <= n; j += per)
      nx_c_bit_unpack(bytes + j, nx_c_bits_load(base, first + j, 64), 64);
  else if (dt == NX_C_DTYPE_i4)
    for (; j + per <= n; j += per)
      nx_c_nib_unpack(bytes + j, nx_c_bits_load(base, (first + j) * 4, 64),
                      16, true);
  else
    for (; j + per <= n; j += per)
      nx_c_nib_unpack(bytes + j, nx_c_bits_load(base, (first + j) * 4, 64),
                      16, false);
  if (j < n)
    nx_c_packed_widen(bytes + j,
                      nx_c_bits_load(base, (first + j) * bits,
                                     (int)(n - j) * bits),
                      (int)(n - j), dt);
}

static inline void nx_c_packed_narrow_words(uint8_t *dst,
                                            const uint8_t *bytes, int64_t n,
                                            nx_c_dtype dt) {
  if (dt == NX_C_DTYPE_bit)
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_bit_pack(bytes + 64 * j, 64));
  else
    for (int64_t j = 0; j < n; j++)
      nx_c_st64(dst + 8 * j, nx_c_nib_pack(bytes + 16 * j, 16));
}

#endif /* NX_C_PACKED_H */
