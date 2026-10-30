/* Adler-32 on 32-byte vectors: the part that checksum_avx2.c and
   checksum_vnni.c share. The including unit defines
   ADLER_DOT(acc, data, weights): [acc] plus, in each 32-bit lane, the sum
   of the 4 products of the unsigned bytes of [data] with the signed bytes
   of [weights] at that lane.

   Over a run of n bytes b_0 .. b_(n-1), from sums s1 and s2, RFC 1950
   section 2.2's sums become s1 + sum b_i and s2 + n s1 + sum (n - i) b_i.
   With the run cut in K blocks of 64 bytes, byte i of block k weighs
   64 (K - 1 - k) + (63 - i) + 1: the first term is 64 times the sum, over
   the blocks, of the byte sums of the blocks before; the second a dot
   product with fixed weights 63 .. 0; the third the byte sum. The vectors
   hold the three in lanes, added up and reduced once per run of at most
   ADLER_RUN bytes. The loop takes two blocks a turn, with a dot product
   accumulator per vector, so that no accumulator waits on the one before.

   Reference: libdeflate's lib/x86/adler32_template.h, for the deferred
   modulo and the byte sums by PSADBW. */

#ifndef LZ_CHECKSUM_ADLER_H
#define LZ_CHECKSUM_ADLER_H

#include <immintrin.h>

#include "checksum_paths.h"

#define ADLER_MOD 65521u

/* The most bytes, a multiple of the block, after which the sums are
   reduced: RFC 1950 section 8.2's 5552 bytes keep s2 within 32 bits. Each
   vector lane stays far below it. */
#define ADLER_BLOCK 64
#define ADLER_RUN (5552 / ADLER_BLOCK * ADLER_BLOCK)

/* From this many bytes, aligning the vectors pays for its scalar head: on
   fudge, at 16 bytes past a line, the head costs about 6 ns and the split
   loads about 10 % of the run. */
#define ADLER_ALIGN_MIN 16384

static inline uint32_t adler_hsum(__m256i v) {
  __m128i x = _mm_add_epi32(_mm256_castsi256_si128(v),
                            _mm256_extracti128_si256(v, 1));
  x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4e));
  x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1));
  return (uint32_t)_mm_cvtsi128_si32(x);
}

static inline uint32_t adler_vec(uint32_t adler, const uint8_t *p, size_t n) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i wa = _mm256_setr_epi8(
      63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
      45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32);
  const __m256i wb = _mm256_setr_epi8(
      31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
      13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  /* A 32-byte load across a 64-byte line costs two: a long input starts
     its vectors at the next 32-byte boundary, the bytes before it summed
     one at a time. */
  if (n >= ADLER_ALIGN_MIN) {
    size_t head = -(uintptr_t)p & 31;
    adler = lz_adler32_generic(adler, p, head);
    p += head, n -= head;
  }
  uint32_t s1 = adler & 0xffff, s2 = adler >> 16;

  while (n >= ADLER_BLOCK) {
    size_t run = n < ADLER_RUN ? n & ~(size_t)(ADLER_BLOCK - 1) : ADLER_RUN;
    n -= run;
    s2 += s1 * (uint32_t)run;

    /* sums: the byte sums (PSADBW leaves the sum of each 8 bytes in a
       64-bit lane); before: the byte sums of the blocks before each block;
       dot: the weighted sums. */
    __m256i sums = zero, before = zero;
    __m256i dot0 = zero, dot1 = zero, dot2 = zero, dot3 = zero;
    const uint8_t *e = p + run;
    for (; e - p >= 2 * ADLER_BLOCK; p += 2 * ADLER_BLOCK) {
      __m256i a = _mm256_loadu_si256((const __m256i *)p);
      __m256i b = _mm256_loadu_si256((const __m256i *)(p + 32));
      __m256i c = _mm256_loadu_si256((const __m256i *)(p + 64));
      __m256i d = _mm256_loadu_si256((const __m256i *)(p + 96));
      __m256i first =
          _mm256_add_epi32(_mm256_sad_epu8(a, zero), _mm256_sad_epu8(b, zero));
      __m256i second =
          _mm256_add_epi32(_mm256_sad_epu8(c, zero), _mm256_sad_epu8(d, zero));
      /* Before the first block: sums; before the second: sums + first. */
      before = _mm256_add_epi32(
          before, _mm256_add_epi32(_mm256_slli_epi32(sums, 1), first));
      sums = _mm256_add_epi32(sums, _mm256_add_epi32(first, second));
      dot0 = ADLER_DOT(dot0, a, wa);
      dot1 = ADLER_DOT(dot1, b, wb);
      dot2 = ADLER_DOT(dot2, c, wa);
      dot3 = ADLER_DOT(dot3, d, wb);
    }
    if (p < e) {
      __m256i a = _mm256_loadu_si256((const __m256i *)p);
      __m256i b = _mm256_loadu_si256((const __m256i *)(p + 32));
      before = _mm256_add_epi32(before, sums);
      sums = _mm256_add_epi32(
          sums, _mm256_add_epi32(_mm256_sad_epu8(a, zero),
                                 _mm256_sad_epu8(b, zero)));
      dot0 = ADLER_DOT(dot0, a, wa);
      dot1 = ADLER_DOT(dot1, b, wb);
      p += ADLER_BLOCK;
    }
    __m256i dot = _mm256_add_epi32(_mm256_add_epi32(dot0, dot1),
                                   _mm256_add_epi32(dot2, dot3));
    uint32_t total = adler_hsum(sums);
    s1 += total;
    s2 += ADLER_BLOCK * adler_hsum(before) + total +
          adler_hsum(dot);
    s1 %= ADLER_MOD;
    s2 %= ADLER_MOD;
  }
  return lz_adler32_generic(s2 << 16 | s1, p, n);
}

#endif
