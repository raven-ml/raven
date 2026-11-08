/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* xxHash64, as the xxHash specification (xxhash_spec.md in the xxHash
   repository) defines it for an input of arbitrary length and a 64-bit seed.
   Its input is little-endian bytes, so a byte row's words swap on a
   big-endian host, and a word row, already its value, does not. */

#include <stdint.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

#define P1 0x9E3779B185EBCA87ULL
#define P2 0xC2B2AE3D27D4EB4FULL
#define P3 0x165667B19E3779F9ULL
#define P4 0x85EBCA77C2B2AE63ULL
#define P5 0x27D4EB2F165667C5ULL

static inline uint64_t rot(uint64_t x, int r)
{
  return (x << r) | (x >> (64 - r));
}

static inline uint64_t mix(uint64_t acc, uint64_t w)
{
  return rot(acc + w * P2, 31) * P1;
}

static inline uint64_t merge(uint64_t h, uint64_t v)
{
  return (h ^ mix(0, v)) * P1 + P4;
}

static inline uint64_t avalanche(uint64_t h)
{
  h = (h ^ (h >> 33)) * P2;
  h = (h ^ (h >> 29)) * P3;
  return h ^ (h >> 32);
}

/* [stripes4 v] folds four accumulators into one, as the specification's
   step 3 and step 4 do for an input of at least 32 bytes. */
static inline uint64_t stripes4(uint64_t v[4])
{
  uint64_t h = rot(v[0], 1) + rot(v[1], 7) + rot(v[2], 12) + rot(v[3], 18);
  for (int i = 0; i < 4; i++) h = merge(h, v[i]);
  return h;
}

static inline void seed4(uint64_t v[4], uint64_t seed)
{
  v[0] = seed + P1 + P2;
  v[1] = seed + P2;
  v[2] = seed;
  v[3] = seed - P1;
}

static inline uint64_t read_le64(const uint8_t *p)
{
  uint64_t w;
  __builtin_memcpy(&w, p, 8);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  w = __builtin_bswap64(w);
#endif
  return w;
}

static inline uint32_t read_le32(const uint8_t *p)
{
  uint32_t w;
  __builtin_memcpy(&w, p, 4);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  w = __builtin_bswap32(w);
#endif
  return w;
}

/* [hash_bytes(p, n, seed)] is the xxHash64 of the [n] bytes at [p]. */
static inline uint64_t hash_bytes(const uint8_t *p, size_t n, uint64_t seed)
{
  const uint8_t *end = p + n;
  uint64_t h;
  if (n >= 32) {
    uint64_t v[4];
    seed4(v, seed);
    do {
      for (int i = 0; i < 4; i++) v[i] = mix(v[i], read_le64(p + 8 * i));
      p += 32;
    } while (end - p >= 32);
    h = stripes4(v);
  } else {
    h = seed + P5;
  }
  h += n;
  for (; end - p >= 8; p += 8) h = rot(h ^ mix(0, read_le64(p)), 27) * P1 + P4;
  if (end - p >= 4) {
    h = rot(h ^ (read_le32(p) * P1), 23) * P2 + P3;
    p += 4;
  }
  for (; p < end; p++) h = rot(h ^ (*p * P5), 11) * P1;
  return avalanche(h);
}

/* [hash_words(w, n, seed)] is the xxHash64 of the [n] words at [w], each
   read as its value. */
static inline uint64_t hash_words(const uint64_t *w, size_t n, uint64_t seed)
{
  const uint64_t *end = w + n;
  uint64_t h;
  if (n >= 4) {
    uint64_t v[4];
    seed4(v, seed);
    do {
      for (int i = 0; i < 4; i++) v[i] = mix(v[i], w[i]);
      w += 4;
    } while (end - w >= 4);
    h = stripes4(v);
  } else {
    h = seed + P5;
  }
  h += 8 * n;
  for (; w < end; w++) h = rot(h ^ mix(0, *w), 27) * P1 + P4;
  return avalanche(h);
}

#define Words_ba(v) ((uint64_t *)Caml_ba_data_val(v))

/* [talon_hash_words(x, w, n, seed, out)] writes [out[i]], the hash of the
   [w] words of row [i] of the [n] rows of [x], its rows consecutive. A row
   may have no words. */
value talon_hash_words(value x, intnat w, intnat n, intnat seed, value out)
{
  const uint64_t *words = Words_ba(x);
  uint64_t *h = Words_ba(out);
  uint64_t s = (uint64_t)(int64_t)seed;
  for (intnat i = 0; i < n; i++) h[i] = hash_words(words + i * w, w, s);
  return Val_unit;
}

value talon_hash_words_byte(value x, value w, value n, value seed, value out)
{
  return talon_hash_words(x, Long_val(w), Long_val(n), Long_val(seed), out);
}

/* [talon_hash_bytes(b, o, n, seed, out)] writes [out[r]], the hash of row [r]
   of the [n] rows of [b], cut by [o]: the bytes [o[r], o[r + 1]). */
value talon_hash_bytes(value b, value o, intnat n, intnat seed, value out)
{
  const uint8_t *v = (const uint8_t *)Caml_ba_data_val(b);
  const int64_t *off = (const int64_t *)Caml_ba_data_val(o);
  uint64_t *h = Words_ba(out);
  uint64_t s = (uint64_t)(int64_t)seed;
  for (intnat r = 0; r < n; r++)
    h[r] = hash_bytes(v + off[r], off[r + 1] - off[r], s);
  return Val_unit;
}

value talon_hash_bytes_byte(value b, value o, value n, value seed, value out)
{
  return talon_hash_bytes(b, o, Long_val(n), Long_val(seed), out);
}
