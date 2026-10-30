/* Adler-32 on 32-byte vectors with AVX-VNNI (checksum_adler.h), for x86-64
   hosts with AVX2 and AVX-VNNI: VPDPBUSD takes the four products of a lane
   and their sum in one instruction. Empty elsewhere, and where the C
   compiler lacks the extensions. */

#include "checksum_paths.h"

#if defined(__x86_64__) && defined(__AVXVNNI__) && defined(__AVX2__)

#include <immintrin.h>

#define ADLER_DOT _mm256_dpbusd_avx_epi32
#include "checksum_adler.h"

static uint32_t adler32_vnni(uint32_t adler, const uint8_t *p, size_t n) {
  return adler_vec(adler, p, n);
}

lz_adler32_fn *const lz_adler32_vnni = adler32_vnni;

#else
lz_adler32_fn *const lz_adler32_vnni = NULL;
#endif
