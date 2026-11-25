/* Loads and stores of 16-, 32- and 64-bit values in a fixed byte order,
   at any address. Every multi-byte field, bit buffer and word a kernel
   hashes or compares goes through them, so that what a kernel reads and
   writes never depends on the host's order. On a host of the same order
   each is one load or store; on the other, that and a byte swap. */

#ifndef LZ_BYTEORDER_H
#define LZ_BYTEORDER_H

#include <stdint.h>
#include <string.h>

#if !defined(__BYTE_ORDER__)
#error "lz needs the C compiler's __BYTE_ORDER__"
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define LZ_BIG_ENDIAN 0
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define LZ_BIG_ENDIAN 1
#else
#error "lz needs a little- or big-endian host"
#endif

/* Inlined always, so that each is its one load or store inside the hot
   loops of every unit, those compiled for an extension included. */
#define LZ_BYTES static inline __attribute__((always_inline))

/* Little-endian */

LZ_BYTES uint16_t lz_load_le16(const uint8_t *p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return LZ_BIG_ENDIAN ? __builtin_bswap16(v) : v;
}

LZ_BYTES uint32_t lz_load_le32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return LZ_BIG_ENDIAN ? __builtin_bswap32(v) : v;
}

LZ_BYTES uint64_t lz_load_le64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return LZ_BIG_ENDIAN ? __builtin_bswap64(v) : v;
}

LZ_BYTES void lz_store_le16(uint8_t *p, uint16_t v) {
  if (LZ_BIG_ENDIAN) v = __builtin_bswap16(v);
  memcpy(p, &v, 2);
}

LZ_BYTES void lz_store_le32(uint8_t *p, uint32_t v) {
  if (LZ_BIG_ENDIAN) v = __builtin_bswap32(v);
  memcpy(p, &v, 4);
}

LZ_BYTES void lz_store_le64(uint8_t *p, uint64_t v) {
  if (LZ_BIG_ENDIAN) v = __builtin_bswap64(v);
  memcpy(p, &v, 8);
}

/* Big-endian */

LZ_BYTES uint16_t lz_load_be16(const uint8_t *p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return LZ_BIG_ENDIAN ? v : __builtin_bswap16(v);
}

LZ_BYTES uint32_t lz_load_be32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return LZ_BIG_ENDIAN ? v : __builtin_bswap32(v);
}

LZ_BYTES uint64_t lz_load_be64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return LZ_BIG_ENDIAN ? v : __builtin_bswap64(v);
}

LZ_BYTES void lz_store_be16(uint8_t *p, uint16_t v) {
  if (!LZ_BIG_ENDIAN) v = __builtin_bswap16(v);
  memcpy(p, &v, 2);
}

LZ_BYTES void lz_store_be32(uint8_t *p, uint32_t v) {
  if (!LZ_BIG_ENDIAN) v = __builtin_bswap32(v);
  memcpy(p, &v, 4);
}

LZ_BYTES void lz_store_be64(uint8_t *p, uint64_t v) {
  if (!LZ_BIG_ENDIAN) v = __builtin_bswap64(v);
  memcpy(p, &v, 8);
}

#endif
