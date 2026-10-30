/* FSE table descriptions and decoding tables (RFC 8878 section 4.1). */

#include "zstd_dec.h"

/* [peek s n bit] is the 32 bits of [s] from bit [bit], read forward and
   little-endian, with zeros past the [n] bytes. */
static uint32_t peek(const uint8_t *s, size_t n, size_t bit) {
  size_t i = bit >> 3;
  uint64_t v = 0;
  if (i + 8 <= n) {
    v = lz_le64(s + i);
  } else {
    for (size_t k = 0; k < 8 && i + k < n; k++) v |= (uint64_t)s[i + k] << (8 * k);
  }
  return (uint32_t)(v >> (bit & 7));
}

static const char past_msg[] = "FSE table description past the block";
static const char big_log[] = "Accuracy_Log above the maximum";
static const char too_many_msg[] = "FSE table description past the alphabet";

/* Section 4.1.1. Each probability is read in the fewest bits that can hold
   any value up to the points left plus one, the low values one bit
   shorter (Table 20); a value v is the probability v - 1, where -1 means
   "less than 1" and counts as one point. A zero is followed by 2-bit
   counts of further zeros, the count 3 announcing another. The
   description ends when the points reach 1 << log, never more, since no
   value exceeds the points left. A field that ends past the [n] bytes
   makes the description run past them, whatever its value. */
size_t lz_fse_ncount(int16_t norm[256], unsigned *max, unsigned *log,
                     unsigned log_max, const uint8_t *s, size_t n,
                     struct lz_zstd_fail *f) {
  if (n == 0) {
    lz_zstd_fail(f, LZ_MALFORMED, 0, past_msg);
    return 0;
  }
  unsigned l = (s[0] & 15) + 5;
  if (l > log_max) {
    lz_zstd_fail(f, LZ_MALFORMED, 0, big_log);
    return 0;
  }

  /* [left] is the points left plus one; values of [bits] bits hold it,
     [threshold] being 1 << (bits - 1). */
  size_t limit = 8 * n;
  int left = (1 << l) + 1;
  int threshold = 1 << l;
  unsigned bits = l + 1;
  size_t bit = 4;
  unsigned sym = 0;
  while (left > 1) {
    if (bit >= limit) goto past;
    if (sym > *max) goto too_many;
    uint32_t v = peek(s, n, bit);
    int low = 2 * threshold - 1 - left;
    int value = (int)(v & (uint32_t)(threshold - 1));
    if (value < low) {
      bit += bits - 1;
    } else {
      value = (int)(v & (uint32_t)(2 * threshold - 1));
      if (value >= threshold) value -= low;
      bit += bits;
    }
    if (bit > limit) goto past;
    int p = value - 1;
    left -= p < 0 ? 1 : p;
    norm[sym++] = (int16_t)p;
    while (p == 0) {
      if (bit + 2 > limit) goto past;
      unsigned zeros = peek(s, n, bit) & 3;
      if (sym + zeros > *max + 1) goto too_many;
      bit += 2;
      for (unsigned k = 0; k < zeros; k++) norm[sym++] = 0;
      if (zeros != 3) break;
    }
    while (left < threshold) {
      threshold >>= 1;
      bits--;
    }
  }
  *max = sym - 1;
  *log = l;
  return (bit + 7) >> 3;

past:
  lz_zstd_fail(f, LZ_MALFORMED, 0, past_msg);
  return 0;
too_many:
  lz_zstd_fail(f, LZ_MALFORMED, (int64_t)(bit >> 3), too_many_msg);
  return 0;
}

/* Section 4.1.1. Symbols of probability "less than 1" take one cell each
   from the end of the table. The others are spread in their order, each
   over as many cells as its probability, every cell [step] after the
   previous one modulo the table, skipping those of the end. A symbol's
   cells, in increasing order, then read the bits that complete the next
   state: the first ones one bit more than the rest, so that its cells
   cover the table's states once (Table 21). */
void lz_fse_build(struct lz_zstd_cell *t, const int16_t *norm, unsigned max,
                  unsigned log, const uint32_t *base, const uint8_t *extra) {
  uint32_t size = 1u << log;
  uint32_t mask = size - 1;
  uint32_t step = (size >> 1) + (size >> 3) + 3;
  uint32_t high = size - 1;
  uint16_t next[256];
  uint8_t sym[512];

  for (unsigned s = 0; s <= max; s++) {
    if (norm[s] == -1) {
      sym[high--] = (uint8_t)s;
      next[s] = 1;
    } else {
      next[s] = (uint16_t)norm[s];
    }
  }

  uint32_t pos = 0;
  for (unsigned s = 0; s <= max; s++) {
    for (int i = 0; i < norm[s]; i++) {
      sym[pos] = (uint8_t)s;
      do pos = (pos + step) & mask;
      while (pos > high);
    }
  }

  for (uint32_t u = 0; u < size; u++) {
    unsigned s = sym[u];
    uint32_t x = next[s]++;
    unsigned nb = log - lz_high32(x);
    t[u].next = (uint16_t)((x << nb) - size);
    t[u].bits = (uint8_t)nb;
    t[u].extra = extra ? extra[s] : 0;
    t[u].base = base ? base[s] : s;
  }
}
