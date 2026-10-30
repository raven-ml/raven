/* CRC-32 by folding with carry-less multiplication, on 16-byte vectors:
   the part that checksum_pclmul.c and checksum_vpclmul.c share, compiled
   into each with that unit's instruction set.

   The message is a polynomial over GF(2); its CRC is M(x) x^32 mod G(x),
   with G gzip's polynomial (RFC 1952 section 8). A 16-byte vector A, of
   the polynomial A(x), followed D bits later by the message B(x), can be
   replaced by a vector F with F(x) = A(x) x^D + B(x) mod G(x) laid over B:
   the CRC of the message does not change. With A = A1 x^64 + A0 split in
   its 64-bit halves, F = A1 (x^(D+64) mod G) + A0 (x^D mod G) + B, two
   carry-less products of a half by a 32-bit constant, each at most 95
   bits: F fits the 128 bits of B. Folding many vectors at once, each
   over the vector as many places ahead, keeps that many products in
   flight.

   The bits are reflected, as the CRC reads them: bit 0 of byte 0 is the
   highest power of x. PCLMULQDQ multiplies the integers, which for
   reflected operands is the reflected product shifted down one place.
   With a 64-bit half and a 32-bit constant held in the low 32 bits of
   its lane, the product lands as A1 c x^33: the constant for a half that
   must move D' places is x^(D'-33) mod G. The constants are computed
   once, by the probe in checksum.c.

   At the end, the vector left is stored and read as a 16-byte message
   by the table path: its CRC from a zero register is the CRC of
   everything folded into it.

   Reference: Gopal et al., "Fast CRC Computation for Generic Polynomials
   Using PCLMULQDQ Instruction" (Intel, 2009), for folding. */

#ifndef LZ_CHECKSUM_CLMUL_H
#define LZ_CHECKSUM_CLMUL_H

#include <immintrin.h>
#include <string.h>

#include "checksum_paths.h"

static inline __m128i clmul_k(const struct lz_crc32_fold *k) {
  return _mm_set_epi64x((long long)k->lo, (long long)k->hi);
}

static inline __m128i clmul_load(const uint8_t *p) {
  return _mm_loadu_si128((const __m128i *)p);
}

/* [a] folded over [b], [k] the constants of the distance between them. */
static inline __m128i clmul_fold(__m128i a, __m128i b, __m128i k) {
  __m128i hi = _mm_clmulepi64_si128(a, k, 0x00);
  __m128i lo = _mm_clmulepi64_si128(a, k, 0x11);
  return _mm_xor_si128(b, _mm_xor_si128(hi, lo));
}

/* [clmul_tail x p n k128], 0 < n < 16, is a vector whose 16 bytes, read
   as a message from a zero register, have the CRC of the 16 bytes of [x]
   followed by the [n] bytes at [p]. [k128] is
   clmul_k(&lz_crc32_folds.d128). It reads [p + n - 16, p + n): the
   caller has consumed 16 bytes before [p].

   The 16 + n bytes split as H, the first n bytes of [x], then L, the 16
   that follow: the rest of [x] and the n bytes. H with 16 - n zero bytes
   before it is a vector 128 bits ahead of L, folded over it with [k128].
   One index vector places both parts of [x]: lane i takes byte i + n - 16
   for H, and with the index's top bit flipped, byte i + n for L. PSHUFB
   reads an index's low 4 bits and gives zero where its top bit is set. */
static inline __m128i clmul_tail(__m128i x, const uint8_t *p, size_t n,
                                 __m128i k128) {
  const __m128i lane =
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m128i at = _mm_add_epi8(lane, _mm_set1_epi8((char)(n - 16)));

  /* Lanes below 16 - n index below 0: zero in H, [x]'s bytes n to 15 in
     L. The lanes from 16 - n take H's bytes, and in L the input's n
     bytes, which end the 16 bytes read. */
  const __m128i low = _mm_cmplt_epi8(at, _mm_setzero_si128());
  __m128i h = _mm_shuffle_epi8(x, at);
  __m128i rest = _mm_shuffle_epi8(x, _mm_xor_si128(at, _mm_set1_epi8(-128)));
  __m128i l = _mm_or_si128(rest, _mm_andnot_si128(low, clmul_load(p + n - 16)));
  return clmul_fold(h, l, k128);
}

/* The CRC register after the 16 bytes of [x], from zero. */
static inline uint32_t clmul_finish(__m128i x) {
  uint8_t b[16];
  _mm_storeu_si128((__m128i *)b, x);
  return lz_crc32_slice8(0, b, sizeof b);
}

/* The register [r] advanced over [n] >= 16 bytes, on 16-byte vectors:
   eight folded at once while 128 bytes remain, then one. */
static inline uint32_t clmul_crc128(uint32_t r, const uint8_t *p, size_t n) {
  const struct lz_crc32_folds *f = &lz_crc32_folds;
  const __m128i k128 = clmul_k(&f->d128);
  __m128i x = _mm_xor_si128(clmul_load(p), _mm_cvtsi32_si128((int)r));
  p += 16, n -= 16;

  if (n >= 112) {
    const __m128i k1024 = clmul_k(&f->d1024);
    __m128i v0 = x, v1 = clmul_load(p), v2 = clmul_load(p + 16),
            v3 = clmul_load(p + 32), v4 = clmul_load(p + 48),
            v5 = clmul_load(p + 64), v6 = clmul_load(p + 80),
            v7 = clmul_load(p + 96);
    p += 112, n -= 112;
    for (; n >= 128; p += 128, n -= 128) {
      v0 = clmul_fold(v0, clmul_load(p), k1024);
      v1 = clmul_fold(v1, clmul_load(p + 16), k1024);
      v2 = clmul_fold(v2, clmul_load(p + 32), k1024);
      v3 = clmul_fold(v3, clmul_load(p + 48), k1024);
      v4 = clmul_fold(v4, clmul_load(p + 64), k1024);
      v5 = clmul_fold(v5, clmul_load(p + 80), k1024);
      v6 = clmul_fold(v6, clmul_load(p + 96), k1024);
      v7 = clmul_fold(v7, clmul_load(p + 112), k1024);
    }
    const __m128i k512 = clmul_k(&f->d512), k256 = clmul_k(&f->d256);
    v0 = clmul_fold(v0, v4, k512);
    v1 = clmul_fold(v1, v5, k512);
    v2 = clmul_fold(v2, v6, k512);
    v3 = clmul_fold(v3, v7, k512);
    v0 = clmul_fold(v0, v2, k256);
    v1 = clmul_fold(v1, v3, k256);
    x = clmul_fold(v0, v1, k128);
  }

  for (; n >= 16; p += 16, n -= 16) x = clmul_fold(x, clmul_load(p), k128);
  if (n > 0) x = clmul_tail(x, p, n, k128);
  return clmul_finish(x);
}

#endif
