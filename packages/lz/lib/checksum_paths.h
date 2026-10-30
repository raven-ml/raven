/* The paths behind checksum.h, for the units that hold a host's fast paths
   and for checksum.c, which chooses among them once. Each fast path
   returns what the generic path returns, for every length and every
   alignment. */

#ifndef LZ_CHECKSUM_PATHS_H
#define LZ_CHECKSUM_PATHS_H

#include <stddef.h>
#include <stdint.h>

typedef uint32_t lz_crc32_fn(uint32_t crc, const uint8_t *p, size_t n);
typedef uint32_t lz_adler32_fn(uint32_t adler, const uint8_t *p, size_t n);

/* [lz_crc32_slice8 r p n] advances the CRC-32 register [r], held inverted as
   RFC 1952's update_crc holds it, over [n] bytes; [lz_crc32_bytes] a byte at
   a time, for a few bytes. */
uint32_t lz_crc32_slice8(uint32_t r, const uint8_t *p, size_t n);
uint32_t lz_crc32_bytes(uint32_t r, const uint8_t *p, size_t n);

uint32_t lz_adler32_generic(uint32_t adler, const uint8_t *p, size_t n);

/* The constants that fold a 16-byte vector over the message D bits after
   its start (checksum_clmul.h): [hi] = x^(D+31) mod G for its high-order
   half, [lo] = x^(D-33) mod G for its low-order half, reflected in the
   low 32 bits. */
struct lz_crc32_fold {
  uint64_t hi, lo;
};

struct lz_crc32_folds {
  struct lz_crc32_fold d128, d256, d384, d512, d768, d1024, d1536, d2048;
};

extern struct lz_crc32_folds lz_crc32_folds;

/* The paths for an instruction set extension. Each is NULL where its unit
   compiled empty: off its architecture, or where the C compiler lacks the
   extension (lib/probe/isa.ml asks it). checksum.c chooses one only on a
   host whose CPU has its instructions.

   x86-64: PCLMULQDQ and SSE4.1; VPCLMULQDQ and AVX2; AVX2; AVX-VNNI and
   AVX2. arm64: the CRC32 instructions; those and PMULL. */
extern lz_crc32_fn *const lz_crc32_pclmul;
extern lz_crc32_fn *const lz_crc32_vpclmul;
extern lz_adler32_fn *const lz_adler32_avx2;
extern lz_adler32_fn *const lz_adler32_vnni;
extern lz_crc32_fn *const lz_crc32_arm;
extern lz_crc32_fn *const lz_crc32_pmull;

/* A path of CRC-32 or Adler-32, named for its instructions. */
struct lz_checksum_path {
  const char *name;
  uint32_t (*fn)(uint32_t, const uint8_t *, size_t);
};

/* [lz_crc32_paths(&paths)] points [paths] at the CRC-32 paths this host
   can run, the generic path first and the one lz_crc32 takes last, and
   returns their number; [lz_adler32_paths] likewise. For the tests, which
   hold every path to the same references. */
size_t lz_crc32_paths(const struct lz_checksum_path **paths);
size_t lz_adler32_paths(const struct lz_checksum_path **paths);

#endif
