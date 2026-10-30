/* What the Zstandard encoder's units share: the parameters of a level, the
   sequences a match finder emits for a block, the match finders' state,
   and the entropy tables a block may repeat from the one before it.

   The pipeline of a block (RFC 8878 section 3.1.1.3):

     bytes --match finder--> sequences + literals --entropy--> block body
              zstd_enc_match.c                     zstd_enc_entropy.c

   zstd_encode.c cuts the input into blocks, keeps the window, writes the
   frame header, the block headers and the checksum, and chooses between
   a compressed, an RLE and a raw block. */

#ifndef LZ_ZSTD_ENC_INT_H
#define LZ_ZSTD_ENC_INT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "zstd_enc.h"

#if defined(__GNUC__)
#define ZE_INLINE static inline __attribute__((always_inline))
#define ZE_NOINLINE __attribute__((noinline))
#define ZE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ZE_INLINE static inline
#define ZE_NOINLINE
#define ZE_UNLIKELY(x) (x)
#endif

/* The entropy stage's bit writers are compiled twice on x86-64: as the
   baseline allows, and for BMI1, BMI2 and LZCNT, whose shifts set no flags
   and take any register. Both write the same bits; an encoder takes the
   second where the CPU has them. */
#if defined(__x86_64__) && defined(__GNUC__)
#define ZE_BMI2 1
#define ZE_TARGET_BMI2 __attribute__((target("bmi,bmi2,lzcnt")))
#else
#define ZE_BMI2 0
#endif

/* RFC 8878 section 3.1.1.2.3: a block holds at most 128 KiB. */
#define ZE_BLOCK (128 * 1024)

/* Bytes the bit writers and wide copies may store past what they keep. */
#define ZE_SLACK 32

enum ze_strategy { ZE_FAST = 1, ZE_DFAST, ZE_GREEDY, ZE_LAZY, ZE_LAZY2 };

/* A level's parameters, named as the reference's level table names them:
   log2 of the window, of the short hash (double-fast), of the hash table,
   of the searches per position, the hashed match length, the step at
   level 1 and 2, and the strategy. */
struct ze_params {
  uint8_t wlog, clog, hlog, slog, mls, tlen, strat;
};

/* A sequence as the format codes it: literals length, match length, and
   Offset_Value (1 to 3 a repeat offset, otherwise the offset plus 3); and
   the three fields' codes, literals length, match length and offset, one
   byte each from the lowest. */
struct ze_seq {
  uint32_t ll, ml, off, code;
};

struct ze_seqs {
  struct ze_seq *seq;
  size_t n;
  uint8_t *lit;
  size_t nlit;
};

/* The match finders' state. Positions are indices from [base], kept in
   32 bits; index 0 is never a position, so a zeroed table holds no
   candidate. [low] is the lowest index the window still holds. For the
   row strategies, [next] is the first position not yet inserted,
   [cache] the hashes of the positions from [next] while [cached] holds,
   and [skipping] whether searches insert only themselves. */
struct ze_match {
  const uint8_t *base;
  uint32_t low;
  uint32_t next;
  uint32_t *t1;
  uint32_t *t2;
  uint8_t *tags;
  uint32_t cache[8];
  bool cached;
  bool skipping;
  struct ze_params p;
};

/* Entropy tables. A Huffman table codes each literal; an FSE table holds
   the encoder's transform of a normalised distribution. A mode says
   whether the next block may repeat a table. */
enum { ZE_NONE, ZE_CHECK };

struct ze_huf {
  uint16_t code[256];
  uint8_t bits[256];
  int max_sym;
  int log;
};

struct ze_fse_sym {
  int32_t find;
  uint32_t nbits;
};

struct ze_fse {
  uint16_t state[512];
  struct ze_fse_sym sym[53];
  int16_t norm[53];
  uint8_t log;
  uint8_t max_sym;
};

struct ze_entropy {
  struct ze_huf huf;
  struct ze_fse ll, of, ml;
  uint8_t huf_mode, ll_mode, of_mode, ml_mode;
};

/* The predefined distributions' tables, built once per encoder. */
struct ze_defaults {
  struct ze_fse ll, of, ml;
};

/* Little-endian loads and stores, whatever the host's order, so that the
   bytes written never depend on the machine. */
ZE_INLINE uint64_t ze_rd64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

ZE_INLINE uint32_t ze_rd32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap32(v);
#endif
  return v;
}

ZE_INLINE void ze_wr64(uint8_t *p, uint64_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  memcpy(p, &v, 8);
}

ZE_INLINE void ze_wr32(uint8_t *p, uint32_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap32(v);
#endif
  memcpy(p, &v, 4);
}

ZE_INLINE void ze_wr24(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
}

ZE_INLINE unsigned ze_highbit(uint32_t v) {
  return 31 - (unsigned)__builtin_clz(v);
}

/* The code of a literals length below 64 and of a match length minus 3
   below 128 (section 3.1.1.3.2.1.1, Tables 16 and 17); longer ones take
   their highest bit. */
static const uint8_t ze_ll_code[64] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21,
    22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24};
static const uint8_t ze_ml_code[128] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
    38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42};

/* [ze_code ll ml off] is the three fields' codes of a sequence. */
ZE_INLINE uint32_t ze_code(uint32_t ll, uint32_t ml, uint32_t off) {
  uint32_t m = ml - 3;
  uint32_t a = ll < 64 ? ze_ll_code[ll] : ze_highbit(ll) + 19;
  uint32_t b = m < 128 ? ze_ml_code[m] : ze_highbit(m) + 36;
  return a | (b << 8) | (ze_highbit(off) << 16);
}

/* zstd_enc_match.c */

/* [ze_match_reset ms] empties the tables, of the sizes [ms->p] gives: a
   row's positions are valid only under a tag, so only the tags are
   cleared. */
void ze_match_reset(struct ze_match *ms);

/* [ze_match_bytes p] is the table memory the match finder of [p] needs. */
size_t ze_match_bytes(const struct ze_params *p);

/* [ze_match_block ms ss rep src n] finds the sequences of the block of [n]
   bytes at [src], appending them, their codes and their literals to
   [ss], from the decoder's first two repeat offsets [rep], which it sets
   to theirs after the block; it uses none the window no longer holds. Its
   sequences never refer to the third repeat offset, which they leave
   unread. It reads no byte past [src + n], nor before the window. It
   returns the length of the literals after the last sequence, which it
   does not copy. */
size_t ze_match_block(struct ze_match *ms, struct ze_seqs *ss,
                      uint32_t rep[2], const uint8_t *src, size_t n);

/* zstd_enc_entropy.c */

void ze_defaults_build(struct ze_defaults *d);

/* [ze_body dst cap ss prev next d strat bmi2] writes the literals and
   sequences sections of a block whose sequences are [ss], at most [cap]
   bytes plus ZE_SLACK of scratch, and fills [next] with the tables the
   next block inherits, with the BMI2 bit writers if [bmi2]. It returns the
   length, or 0 if the body does not fit [cap]. */
size_t ze_body(uint8_t *dst, size_t cap, struct ze_seqs *ss,
               const struct ze_entropy *prev, struct ze_entropy *next,
               const struct ze_defaults *d, int strat, bool bmi2);

#endif
