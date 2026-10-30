/* CRC-32 on 32-byte vectors with VPCLMULQDQ, for x86-64 hosts with
   VPCLMULQDQ and AVX2: the folding of checksum_clmul.h, two 16-byte lanes
   per instruction, each lane folded over the same distance. Empty
   elsewhere, and where the C compiler lacks the extensions. */

#include "checksum_paths.h"

#if defined(__x86_64__) && defined(__VPCLMULQDQ__) && defined(__PCLMUL__) && \
    defined(__AVX2__)

#include "checksum_clmul.h"

/* The wide path starts from eight 32-byte vectors; shorter inputs take the
   16-byte path. */
#define WIDE_MIN 256

/* From this many bytes, aligning the vectors pays for the table path's
   head: on fudge, at 16 bytes past a line, the head costs about 6 ns and
   the split loads about 10 % of the run. */
#define ALIGN_MIN 16384

static inline __m256i wide_k(const struct lz_crc32_fold *k) {
  return _mm256_set_epi64x((long long)k->lo, (long long)k->hi,
                           (long long)k->lo, (long long)k->hi);
}

static inline __m256i wide_load(const uint8_t *p) {
  return _mm256_loadu_si256((const __m256i *)p);
}

static inline __m256i wide_fold(__m256i a, __m256i b, __m256i k) {
  __m256i hi = _mm256_clmulepi64_epi128(a, k, 0x00);
  __m256i lo = _mm256_clmulepi64_epi128(a, k, 0x11);
  return _mm256_xor_si256(b, _mm256_xor_si256(hi, lo));
}

static uint32_t crc32_vpclmul(uint32_t crc, const uint8_t *p, size_t n) {
  if (n < 16) return ~lz_crc32_slice8(~crc, p, n);
  if (n < WIDE_MIN) return ~clmul_crc128(~crc, p, n);

  /* A 32-byte load across a 64-byte line costs two: a long input starts
     its vectors at the next 32-byte boundary, the bytes before it read by
     the table path. */
  uint32_t reg = ~crc;
  if (n >= ALIGN_MIN) {
    size_t head = -(uintptr_t)p & 31;
    reg = lz_crc32_slice8(reg, p, head);
    p += head, n -= head;
  }

  const struct lz_crc32_folds *f = &lz_crc32_folds;
  const __m256i k2048 = wide_k(&f->d2048);
  __m256i r = _mm256_inserti128_si256(_mm256_setzero_si256(),
                                       _mm_cvtsi32_si128((int)reg), 0);
  __m256i v0 = _mm256_xor_si256(wide_load(p), r), v1 = wide_load(p + 32),
          v2 = wide_load(p + 64), v3 = wide_load(p + 96),
          v4 = wide_load(p + 128), v5 = wide_load(p + 160),
          v6 = wide_load(p + 192), v7 = wide_load(p + 224);
  p += 256, n -= 256;
  for (; n >= 256; p += 256, n -= 256) {
    v0 = wide_fold(v0, wide_load(p), k2048);
    v1 = wide_fold(v1, wide_load(p + 32), k2048);
    v2 = wide_fold(v2, wide_load(p + 64), k2048);
    v3 = wide_fold(v3, wide_load(p + 96), k2048);
    v4 = wide_fold(v4, wide_load(p + 128), k2048);
    v5 = wide_fold(v5, wide_load(p + 160), k2048);
    v6 = wide_fold(v6, wide_load(p + 192), k2048);
    v7 = wide_fold(v7, wide_load(p + 224), k2048);
  }

  const __m256i k1024 = wide_k(&f->d1024), k512 = wide_k(&f->d512),
                k256 = wide_k(&f->d256);
  v0 = wide_fold(v0, v4, k1024);
  v1 = wide_fold(v1, v5, k1024);
  v2 = wide_fold(v2, v6, k1024);
  v3 = wide_fold(v3, v7, k1024);
  v0 = wide_fold(v0, v2, k512);
  v1 = wide_fold(v1, v3, k512);
  v0 = wide_fold(v0, v1, k256);
  for (; n >= 32; p += 32, n -= 32) v0 = wide_fold(v0, wide_load(p), k256);

  const __m128i k128 = clmul_k(&f->d128);
  __m128i x = clmul_fold(_mm256_castsi256_si128(v0),
                         _mm256_extracti128_si256(v0, 1), k128);
  if (n >= 16) {
    x = clmul_fold(x, clmul_load(p), k128);
    p += 16, n -= 16;
  }
  if (n > 0) x = clmul_tail(x, p, n, k128);
  return ~clmul_finish(x);
}

lz_crc32_fn *const lz_crc32_vpclmul = crc32_vpclmul;

#else
lz_crc32_fn *const lz_crc32_vpclmul = NULL;
#endif
