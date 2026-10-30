/* Checksums: CRC-32 (RFC 1952 section 8), Adler-32 (RFC 1950 section 2.2),
   XXH32 and XXH64 (the xxHash specification). Each one-shot function
   continues a running value: [lz_crc32(lz_crc32(0, a), b)] is the CRC of
   a followed by b, and likewise for Adler-32 from 1. The xxHash states
   are incremental, for the decoders and encoders that hash as they go.
   Every function takes the fastest path the host has, chosen once; each
   path returns the same value. */

#ifndef LZ_CHECKSUM_H
#define LZ_CHECKSUM_H

#include <stddef.h>
#include <stdint.h>

uint32_t lz_crc32(uint32_t crc, const uint8_t *p, size_t n);
uint32_t lz_adler32(uint32_t adler, const uint8_t *p, size_t n);

struct lz_xxh32 {
  uint32_t acc[4];
  uint32_t seed;
  uint64_t total;
  uint8_t buf[16];
  uint32_t buffered;
};

void lz_xxh32_init(struct lz_xxh32 *h, uint32_t seed);
void lz_xxh32_update(struct lz_xxh32 *h, const uint8_t *p, size_t n);
uint32_t lz_xxh32_digest(const struct lz_xxh32 *h);
uint32_t lz_xxh32(uint32_t seed, const uint8_t *p, size_t n);

struct lz_xxh64 {
  uint64_t acc[4];
  uint64_t seed;
  uint64_t total;
  uint8_t buf[32];
  uint32_t buffered;
};

void lz_xxh64_init(struct lz_xxh64 *h, uint64_t seed);
void lz_xxh64_update(struct lz_xxh64 *h, const uint8_t *p, size_t n);
uint64_t lz_xxh64_digest(const struct lz_xxh64 *h);
uint64_t lz_xxh64(uint64_t seed, const uint8_t *p, size_t n);

#endif
