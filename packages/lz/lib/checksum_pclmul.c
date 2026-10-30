/* CRC-32 on 16-byte vectors with PCLMULQDQ (checksum_clmul.h), for x86-64
   hosts with PCLMULQDQ and SSE4.1. Empty elsewhere, and where the C
   compiler lacks the extensions. */

#include "checksum_paths.h"

#if defined(__x86_64__) && defined(__PCLMUL__) && defined(__SSE4_1__)

#include "checksum_clmul.h"

static uint32_t crc32_pclmul(uint32_t crc, const uint8_t *p, size_t n) {
  if (n < 16) return ~lz_crc32_slice8(~crc, p, n);
  return ~clmul_crc128(~crc, p, n);
}

lz_crc32_fn *const lz_crc32_pclmul = crc32_pclmul;

#else
lz_crc32_fn *const lz_crc32_pclmul = NULL;
#endif
