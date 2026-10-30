/* CRC-32 for arm64 hosts: with the CRC32 instructions alone, and folded
   with PMULL (the folding of checksum_clmul.h, whose arithmetic PMULL
   shares with PCLMULQDQ) where the host has both. Empty elsewhere, and
   where the C compiler lacks the extensions.

   CRC32X and its narrower forms advance the CRC-32 register, held
   inverted, by 8, 4, 2 or 1 bytes (Arm ARM, CRC32X): the tail of a fold
   and the vector it leaves are read with them. */

#include "checksum_paths.h"

#if defined(__aarch64__) && !defined(__ARM_BIG_ENDIAN) && \
    defined(__ARM_FEATURE_CRC32)

#include <arm_acle.h>
#include <arm_neon.h>

static inline uint64_t load64(const uint8_t *p) {
  uint64_t v;
  __builtin_memcpy(&v, p, sizeof v);
  return v;
}

static inline uint32_t load32(const uint8_t *p) {
  uint32_t v;
  __builtin_memcpy(&v, p, sizeof v);
  return v;
}

static inline uint16_t load16(const uint8_t *p) {
  uint16_t v;
  __builtin_memcpy(&v, p, sizeof v);
  return v;
}

/* The register [r] advanced over [n] bytes, 8 at a time. */
static inline uint32_t crc_words(uint32_t r, const uint8_t *p, size_t n) {
  for (; n >= 32; p += 32, n -= 32) {
    r = __crc32d(r, load64(p));
    r = __crc32d(r, load64(p + 8));
    r = __crc32d(r, load64(p + 16));
    r = __crc32d(r, load64(p + 24));
  }
  for (; n >= 8; p += 8, n -= 8) r = __crc32d(r, load64(p));
  if (n & 4) r = __crc32w(r, load32(p)), p += 4;
  if (n & 2) r = __crc32h(r, load16(p)), p += 2;
  if (n & 1) r = __crc32b(r, *p);
  return r;
}

static uint32_t crc32_arm(uint32_t crc, const uint8_t *p, size_t n) {
  return ~crc_words(~crc, p, n);
}

lz_crc32_fn *const lz_crc32_arm = crc32_arm;

/* PMULL */

#if defined(__ARM_FEATURE_AES)

static inline uint8x16_t vk(const struct lz_crc32_fold *k) {
  uint64_t both[2] = {k->hi, k->lo};
  return vreinterpretq_u8_u64(vld1q_u64(both));
}

/* [a] folded over [b], [k] the constants of the distance between them:
   each 64-bit lane of [a] times the constant in the same lane of [k]. */
static inline uint8x16_t fold(uint8x16_t a, uint8x16_t b, uint8x16_t k) {
  poly64x2_t a64 = vreinterpretq_p64_u8(a), k64 = vreinterpretq_p64_u8(k);
  poly128_t lo = vmull_p64(vgetq_lane_p64(a64, 0), vgetq_lane_p64(k64, 0));
  poly128_t hi = vmull_high_p64(a64, k64);
  return veorq_u8(b, veorq_u8(vreinterpretq_u8_p128(lo),
                              vreinterpretq_u8_p128(hi)));
}

/* Below FOLD_MIN bytes, crc32_pmull takes the CRC32 chain alone: one
   CRC32X per 8 bytes, each waiting on the last, costs less than setting
   up twelve vectors, joining them and reading the last with CRC32X. */
#define FOLD_MIN 256

/* Twelve vectors folded at once, each over the one 192 bytes ahead: a
   fold is two PMULLs and two EORs, and twelve independent chains keep
   enough of them in flight to cover a chain's latency on cores with two
   to four PMULL units. The twelve join over 768, 384, then 256 and 128
   bits; the rest folds 16 bytes at a time; CRC32X reads the vector left
   and the last bytes. */
static uint32_t crc32_pmull(uint32_t crc, const uint8_t *p, size_t n) {
  uint32_t r = ~crc;
  if (n < FOLD_MIN) return ~crc_words(r, p, n);

  /* The register XORed into the first 4 bytes, then folded from zero. */
  const struct lz_crc32_folds *f = &lz_crc32_folds;
  const uint8x16_t seed =
      vreinterpretq_u8_u32(vsetq_lane_u32(r, vdupq_n_u32(0), 0));
  uint8x16_t v0 = veorq_u8(vld1q_u8(p), seed), v1 = vld1q_u8(p + 16),
             v2 = vld1q_u8(p + 32), v3 = vld1q_u8(p + 48),
             v4 = vld1q_u8(p + 64), v5 = vld1q_u8(p + 80),
             v6 = vld1q_u8(p + 96), v7 = vld1q_u8(p + 112),
             v8 = vld1q_u8(p + 128), v9 = vld1q_u8(p + 144),
             v10 = vld1q_u8(p + 160), v11 = vld1q_u8(p + 176);
  p += 192, n -= 192;

  const uint8x16_t k1536 = vk(&f->d1536);
  for (; n >= 192; p += 192, n -= 192) {
    v0 = fold(v0, vld1q_u8(p), k1536);
    v1 = fold(v1, vld1q_u8(p + 16), k1536);
    v2 = fold(v2, vld1q_u8(p + 32), k1536);
    v3 = fold(v3, vld1q_u8(p + 48), k1536);
    v4 = fold(v4, vld1q_u8(p + 64), k1536);
    v5 = fold(v5, vld1q_u8(p + 80), k1536);
    v6 = fold(v6, vld1q_u8(p + 96), k1536);
    v7 = fold(v7, vld1q_u8(p + 112), k1536);
    v8 = fold(v8, vld1q_u8(p + 128), k1536);
    v9 = fold(v9, vld1q_u8(p + 144), k1536);
    v10 = fold(v10, vld1q_u8(p + 160), k1536);
    v11 = fold(v11, vld1q_u8(p + 176), k1536);
  }

  /* A fold lands in its second vector's place: after the first join the
     six cover the last 96 bytes of the stride, after the second the last
     48, and the last two folds land on its last 16. */
  const uint8x16_t k768 = vk(&f->d768), k384 = vk(&f->d384),
                   k256 = vk(&f->d256), k128 = vk(&f->d128);
  v6 = fold(v0, v6, k768);
  v7 = fold(v1, v7, k768);
  v8 = fold(v2, v8, k768);
  v9 = fold(v3, v9, k768);
  v10 = fold(v4, v10, k768);
  v11 = fold(v5, v11, k768);
  v9 = fold(v6, v9, k384);
  v10 = fold(v7, v10, k384);
  v11 = fold(v8, v11, k384);
  uint8x16_t x = fold(v10, fold(v9, v11, k256), k128);
  for (; n >= 16; p += 16, n -= 16) x = fold(x, vld1q_u8(p), k128);

  /* The vector run from a zero register, then the bytes after it. */
  uint64x2_t x64 = vreinterpretq_u64_u8(x);
  r = __crc32d(__crc32d(0, vgetq_lane_u64(x64, 0)), vgetq_lane_u64(x64, 1));
  return ~crc_words(r, p, n);
}

lz_crc32_fn *const lz_crc32_pmull = crc32_pmull;
#else
lz_crc32_fn *const lz_crc32_pmull = NULL;
#endif

#else
lz_crc32_fn *const lz_crc32_arm = NULL;
lz_crc32_fn *const lz_crc32_pmull = NULL;
#endif
