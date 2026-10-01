/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  CRC-32 (RFC 1952) and Adler-32 (RFC 1950). CRC-32 uses the ARMv8 CRC
  instructions where the compiler targets them, slicing by 8 elsewhere.
  ---------------------------------------------------------------------------*/

#include "compress_deflate.h"

#include <string.h>

#if defined(__ARM_FEATURE_CRC32) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#include <arm_acle.h>

void compress_crc32_init(void) {}

uint32_t compress_deflate_crc32(uint32_t crc, const uint8_t *src,
                                size_t len) {
  crc = ~crc;
  for (; len >= 8; src += 8, len -= 8) {
    uint64_t v;
    memcpy(&v, src, 8);
    crc = __crc32d(crc, v);
  }
  while (len-- != 0)
    crc = __crc32b(crc, *src++);
  return ~crc;
}
#else
static uint32_t table[8][256];

void compress_crc32_init(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int bit = 0; bit < 8; bit++)
      c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    table[0][i] = c;
  }
  for (unsigned k = 1; k < 8; k++)
    for (unsigned i = 0; i < 256; i++)
      table[k][i] = table[0][table[k - 1][i] & 0xffu] ^ (table[k - 1][i] >> 8);
}

static uint32_t load32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

uint32_t compress_deflate_crc32(uint32_t crc, const uint8_t *src,
                                size_t len) {
  crc = ~crc;
  for (; len >= 8; src += 8, len -= 8) {
    uint32_t a = crc ^ load32(src);
    uint32_t b = load32(src + 4);
    crc = table[7][a & 0xffu] ^ table[6][(a >> 8) & 0xffu] ^
          table[5][(a >> 16) & 0xffu] ^ table[4][a >> 24] ^
          table[3][b & 0xffu] ^ table[2][(b >> 8) & 0xffu] ^
          table[1][(b >> 16) & 0xffu] ^ table[0][b >> 24];
  }
  while (len-- != 0)
    crc = table[0][(crc ^ *src++) & 0xffu] ^ (crc >> 8);
  return ~crc;
}
#endif

/* 5552 is the most bytes whose sums cannot overflow 32 bits before the
   modulo (RFC 1950's NMAX). Over 16 bytes, [b] grows by 16 times [a] plus
   the bytes weighted 16 down to 1: two sums without a chain between bytes,
   which compilers vectorize. */
uint32_t compress_deflate_adler32(uint32_t adler, const uint8_t *src,
                                  size_t len) {
  uint32_t a = adler & 0xffffu;
  uint32_t b = adler >> 16;
  while (len != 0) {
    size_t n = len < 5552 ? len : 5552;
    len -= n;
    for (; n >= 16; n -= 16, src += 16) {
      uint32_t sum = 0, weighted = 0;
      for (unsigned i = 0; i < 16; i++) {
        sum += src[i];
        weighted += (16 - i) * (uint32_t)src[i];
      }
      b += 16 * a + weighted;
      a += sum;
    }
    while (n-- != 0) {
      a += *src++;
      b += a;
    }
    a %= 65521u;
    b %= 65521u;
  }
  return (b << 16) | a;
}
