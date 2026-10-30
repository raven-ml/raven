/* Adler-32 on 32-byte vectors with AVX2 (checksum_adler.h), for x86-64
   hosts with AVX2. PMADDUBSW adds pairs of products into 16 bits, with
   saturation: weights below 64 keep the sum of two products under 2^15.
   Empty elsewhere, and where the C compiler lacks the extension. */

#include "checksum_paths.h"

#if defined(__x86_64__) && defined(__AVX2__)

#include <immintrin.h>

static inline __m256i adler_dot(__m256i acc, __m256i data, __m256i weights) {
  __m256i pairs = _mm256_maddubs_epi16(data, weights);
  return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

#define ADLER_DOT adler_dot
#include "checksum_adler.h"

static uint32_t adler32_avx2(uint32_t adler, const uint8_t *p, size_t n) {
  return adler_vec(adler, p, n);
}

lz_adler32_fn *const lz_adler32_avx2 = adler32_avx2;

#else
lz_adler32_fn *const lz_adler32_avx2 = NULL;
#endif
