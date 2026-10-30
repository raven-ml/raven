/* Deflate encoding (RFC 1951) inside the zlib (RFC 1950) and gzip
   (RFC 1952) wrappers, one kernel for both drivers of lz.h. The design is
   libdeflate's (lib/deflate_compress.c), its parsers, matchfinders, block
   splitting and constants:

     level  parser   matchfinder                        search  nice
     1      greedy   hash table, 2 entries per bucket        2    32
     2-4    greedy   hash chains on 4 bytes, 3 in front  6-16  10-30
     5-7    lazy     hash chains                        16-100 30-130
     8-9    lazy2    hash chains                       300-600   258

   A parser chooses literals and matches into a sequence store and counts
   their symbols; a block ends on those counts (its length, its matches,
   or a change in the distribution of its symbols, sampled every 512) or
   at the input's end. Ending it builds the block's Huffman codes, prices
   the block dynamic, static and stored, and writes the cheapest:

     input -> parse -> sequences, counts -> codes -> cost -> bits
               ^ matchfinder                         stored | static
                                                     | dynamic

   Determinism. The bytes are a function of the input, the level and
   where [last] fell, never of how a stream's input was cut into calls.
   Every decision that depends on where the input ends (the longest match,
   hashing near the end, a block's last check, a block's length) reads at
   most LOOKAHEAD bytes ahead of the item it decides, so a streaming call
   parses an item only once LOOKAHEAD bytes follow it, or at [last].
   Matchfinder positions are relative to a base that advances by the
   window's size at fixed positions, so a stream's window moving under
   them changes nothing. The whole-buffer call is the streaming one given
   all input at once: it parses the caller's buffer in place.

   A streaming encoder copies its input into [win] and keeps the bytes of
   the block not yet written, the last 32 KiB and the lookahead; the
   whole-buffer one reads the caller's buffer. A block is written whole in
   one call, after a check that it fits: a block that does not is kept and
   written by the next call. */

#include "deflate_enc.h"

#include <string.h>

#include "checksum.h"

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define INLINE static inline __attribute__((always_inline))
#define NOINLINE static __attribute__((noinline))
#else
#define NOINLINE static
#define likely(x) (x)
#define unlikely(x) (x)
#define INLINE static inline
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define BIG_ENDIAN_HOST 1
#else
#define BIG_ENDIAN_HOST 0
#endif

/* The format (RFC 1951) */

#define WSIZE 32768
#define MIN_MATCH 3
#define MAX_MATCH 258
#define NLITLEN 288
#define NDIST 32
#define NPRE 19
#define EOB 256
#define MAX_STORED 65535

/* The longest code of each code. Literal/length codes stop at 14 bits, one
   short of the format's 15. */
#define MAX_LITLEN_LEN 14
#define MAX_DIST_LEN 15
#define MAX_PRE_LEN 7

static const uint16_t length_base[29] = {
    3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const uint8_t length_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                         1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t dist_base[30] = {
    1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
    33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t dist_extra[30] = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,
                                       4, 4, 5, 5, 6, 6, 7,  7,  8,  8,
                                       9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/* The length slot of each match length, 3 to 258. */
static const uint8_t length_slot[MAX_MATCH + 1] = {
     0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  8,  9,  9, 10, 10, 11,
    11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16,
    16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18,
    18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22,
    22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
    25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26,
    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 28,
};

/* The order of the code length code's lengths (RFC 1951, 3.2.7), and the
   extra bits of its symbols 16, 17 and 18. */
static const uint8_t pre_order[NPRE] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                        11, 4,  12, 3, 13, 2, 14, 1, 15};
static const uint8_t pre_extra[NPRE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 2, 3, 7};

/* [dist_slot d] is the distance symbol of distance [d], 1 to 32768: two
   symbols per power of two above 4, told apart by the bit below the top. */
INLINE unsigned dist_slot(uint32_t d) {
  uint32_t x = d - 1;
  if (x < 4) return x;
  unsigned top = 31 - (unsigned)__builtin_clz(x);
  return 2 * top + ((x >> (top - 1)) & 1);
}

/* Levels: libdeflate's parser, search depth and nice length per level.
   zlib's depths and nice lengths gained 0.5 to 3.4 % of ratio on text,
   columns and Silesia for 1.2 to 5.9 times the time, and still missed
   zlib's ratio on columns at 6 and 9, which is its parser's. */

enum parser { FASTEST, GREEDY, LAZY, LAZY2 };

static const struct {
  uint8_t parser;
  uint16_t depth;
  uint16_t nice;
} levels[10] = {
    {FASTEST, 0, 0},    {FASTEST, 2, 32},  {GREEDY, 6, 10},
    {GREEDY, 12, 14},   {GREEDY, 16, 30},  {LAZY, 16, 30},
    {LAZY, 35, 65},     {LAZY, 100, 130},  {LAZY2, 300, MAX_MATCH},
    {LAZY2, 600, MAX_MATCH},
};

/* Blocks. A block other than the last is at least MIN_BLOCK bytes, which
   bounds the stored blocks [lz_deflate_bound] allows for. The parsers end
   a block once it reaches SOFT_MAX bytes (FAST_SOFT_MAX at level 1) or
   fills the sequence store, unless the input ends less than MIN_BLOCK
   bytes later, and the greedy and lazy parsers sooner when its symbols
   change. */

#define MIN_BLOCK 5000
#define SOFT_MAX 300000
#define SEQ_STORE 50000
#define FAST_SOFT_MAX 65535
#define FAST_SEQ_STORE 8192
#define CHECK_EVERY 512

/* Inputs up to [55 - 4 * level] bytes are one block of literals: a search
   cannot pay for setting up the matchfinder. */
#define PASSTHROUGH(level) ((size_t)(55 - 4 * (level)))

/* How far one parser item may move past its first byte: a lazy parser
   follows better matches one or two bytes on while 4 * length - log2
   (distance) grows by 3 or more, which is at most 345 steps, and then
   takes a match of up to 258 bytes. */
#define ITEM_MAX 1024

/* The bytes a streaming call waits for after an item before parsing it,
   so that every decision about the item reads no further than the input
   holds: the item, a block's last check MIN_BLOCK bytes after it, and a
   match from the item's last byte. */
#define LOOKAHEAD 8192

_Static_assert(LOOKAHEAD >= ITEM_MAX + MIN_BLOCK + MAX_MATCH + 8,
               "a block check reads within the lookahead");
_Static_assert(LOOKAHEAD >= 4096 + MAX_MATCH, "the literal scan fits");
_Static_assert(2 * MIN_BLOCK <= MAX_STORED, "the bound counts blocks");
_Static_assert(SEQ_STORE * MIN_MATCH >= MIN_BLOCK, "a full store is long");
_Static_assert(FAST_SEQ_STORE * 4 >= MIN_BLOCK, "a full store is long");

/* A streaming window keeps, besides the step's input, the block not yet
   written (under SOFT_MAX bytes, or the last 32 KiB if longer) and the
   unparsed lookahead. */
#define KEEP_MAX(soft) ((size_t)(soft) + LOOKAHEAD)
#define WIN_SIZE(soft) (KEEP_MAX(soft) + 2 * (size_t)LZ_DEFLATE_ENC_QUOTA)

/* What a step may write: the blocks covering what it holds unwritten at
   its start (the open block and the lookahead) and its quota, each at most
   its stored size, plus a byte of bits, the header and the trailer. */
#define STEP_DATA ((size_t)SOFT_MAX + LOOKAHEAD + LZ_DEFLATE_ENC_QUOTA)
_Static_assert(STEP_DATA + 5 * (STEP_DATA / MIN_BLOCK + 1) + 1 + 10 + 8 <=
                   LZ_DEFLATE_ENC_RUN,
               "a step's output fits the run");

/* Bytes */

INLINE uint32_t le32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
#if BIG_ENDIAN_HOST
  v = __builtin_bswap32(v);
#endif
  return v;
}

INLINE uint32_t ld32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}

INLINE uint64_t ld64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

INLINE void put64le(uint8_t *p, uint64_t v) {
#if BIG_ENDIAN_HOST
  v = __builtin_bswap64(v);
#endif
  memcpy(p, &v, 8);
}

/* [extend a b n max] is the length of the common prefix of [a] and [b],
   known to be at least [n] and taken at most to [max]. */
INLINE uint32_t first_diff(uint64_t x) {
#if BIG_ENDIAN_HOST
  return (unsigned)__builtin_clzll(x) >> 3;
#else
  return (unsigned)__builtin_ctzll(x) >> 3;
#endif
}

INLINE uint32_t extend(const uint8_t *a, const uint8_t *b, uint32_t n,
                       uint32_t max) {
  uint32_t i = n;
  uint64_t x;

  /* Most matches end within 32 bytes: four words with no bound test. */
  if (likely(max - i >= 32)) {
    if ((x = ld64(a + i) ^ ld64(b + i)) != 0) return i + first_diff(x);
    if ((x = ld64(a + i + 8) ^ ld64(b + i + 8)) != 0)
      return i + 8 + first_diff(x);
    if ((x = ld64(a + i + 16) ^ ld64(b + i + 16)) != 0)
      return i + 16 + first_diff(x);
    if ((x = ld64(a + i + 24) ^ ld64(b + i + 24)) != 0)
      return i + 24 + first_diff(x);
    i += 32;
  }

  for (; max - i >= 8; i += 8)
    if ((x = ld64(a + i) ^ ld64(b + i)) != 0) return i + first_diff(x);

  while (i < max && a[i] == b[i]) i++;
  return i;
}

/* Matchfinders

   Positions are int16 offsets from a base that moves up by WSIZE when a
   position reaches base + WSIZE: [rebase] then subtracts WSIZE from every
   entry, saturating at NIL, so an entry at or below [cutoff] (WSIZE before
   the current position) is out of the window. */

typedef int16_t mpos;
#define NIL ((mpos)-WSIZE)

#define H3_BITS 15
#define H4_BITS 16
#define HT_BITS 15

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCHW(p) __builtin_prefetch((p), 1)
#else
#define PREFETCHW(p) ((void)(p))
#endif

INLINE uint32_t hash(uint32_t seq, unsigned bits) {
  return (uint32_t)(seq * 0x1E35A7BDu) >> (32 - bits);
}

/* Each matchfinder's positions are one array of [*_ENTRIES], followed by
   the rebase it takes. */
#define HC_HEADS (((size_t)1 << H3_BITS) + ((size_t)1 << H4_BITS))
#define HC_ENTRIES (HC_HEADS + 2 * (size_t)WSIZE)
#define HT_ENTRIES ((size_t)2 << HT_BITS)

/* The chain's link for the position at offset [c] is [next[WSIZE + c]].
   Every offset in the window, above -WSIZE and below WSIZE, has its own
   entry, so a walk turns a position into its link's index with no mask,
   which would lie on the path from one link's load to the next. A rebase
   moves the upper half's links down to the lower half. */
struct hc {
  mpos h3[1 << H3_BITS]; /* the last position of each 3-byte hash */
  mpos h4[1 << H4_BITS]; /* the head of each 4-byte hash's chain */
  mpos next[2 * WSIZE];  /* the chain, by offset */
  bool generic;          /* rebase with the generic loop */
};

struct ht {
  mpos tab[1 << HT_BITS][2]; /* the last two positions of each hash */
  bool generic;
};

_Static_assert(offsetof(struct hc, generic) == HC_ENTRIES * sizeof(mpos) &&
                   offsetof(struct ht, generic) == HT_ENTRIES * sizeof(mpos),
               "the positions are one array");
_Static_assert(HC_HEADS % 32 == 0 && WSIZE % 32 == 0 && HT_ENTRIES % 32 == 0,
               "rebase's SSE2 loop takes 32 entries at a time");

static void mf_fill(mpos *t, size_t n) {
  for (size_t i = 0; i < n; i++) t[i] = NIL;
}

/* [rebase t s n generic] writes to [t] the [n] entries at [s] moved down
   by WSIZE. SSE2, the x86-64 baseline, subtracts with saturation 8
   entries an instruction; with [generic], the scalar loop gives the same
   values, which the tests hold. */
static void rebase(mpos *t, const mpos *s, size_t n, bool generic) {
#if defined(__SSE2__)
  /* Adding -32768 with signed saturation in 16-bit lanes is the scalar
     map: v - WSIZE for v >= 0, and NIL for v < 0, where it saturates. */
  if (!generic) {
    const __m128i down = _mm_set1_epi16((short)NIL);
    for (size_t i = 0; i < n; i += 32) {
      const __m128i *u = (const __m128i *)(s + i);
      __m128i *v = (__m128i *)(t + i);
      __m128i a = _mm_loadu_si128(u), b = _mm_loadu_si128(u + 1),
              c = _mm_loadu_si128(u + 2), d = _mm_loadu_si128(u + 3);
      _mm_storeu_si128(v, _mm_adds_epi16(a, down));
      _mm_storeu_si128(v + 1, _mm_adds_epi16(b, down));
      _mm_storeu_si128(v + 2, _mm_adds_epi16(c, down));
      _mm_storeu_si128(v + 3, _mm_adds_epi16(d, down));
    }
    return;
  }
#else
  (void)generic;
#endif
  /* v + (-WSIZE) saturated: v - WSIZE for v >= 0, NIL for v < 0. */
  for (size_t i = 0; i < n; i++) {
    int v = s[i];
    t[i] = (mpos)((v < 0 ? 0 : v) - WSIZE);
  }
}

/* [hc_rebase mf] rebases the heads in place and the upper links into the
   lower half. The upper half's entries are then stale: each is written
   when its position is inserted, before any walk reads it. */
static void hc_rebase(struct hc *mf) {
  rebase((mpos *)mf, (mpos *)mf, HC_HEADS, mf->generic);
  rebase(mf->next, mf->next + WSIZE, WSIZE, mf->generic);
}

/* [hc_walk_on mf base in c floor best max nice left qp] goes on with a
   walk of [hc_find] from the entry [c], for at most [left] more entries,
   with the best so far [best] at [*qp]: the same search, out of line.
   Until a chain candidate counts, an entry is rejected on its first 4
   bytes; after, on the 4 bytes ending at the best length, which rejects
   most at one load. Kept out of line, this costs the parsers' hot paths nothing
   on input where walks are short. */
NOINLINE uint32_t hc_walk_on(const struct hc *mf, const uint8_t *base,
                             const uint8_t *in, ptrdiff_t c, int32_t floor,
                             uint32_t best, uint32_t max, uint32_t nice,
                             uint32_t left, const uint8_t **qp) {
  const mpos *const links = mf->next + WSIZE;
  const uint32_t w0 = ld32(in);
  const uint8_t *q = *qp;
  uint32_t wl = 0;

  if (best >= 4) goto longer;
  for (;;) {
    if (unlikely(c <= floor)) goto done;
    const uint8_t *m = base + c;
    c = links[c];
    if (unlikely(ld32(m) == w0)) {
      best = extend(in, m, 4, max);
      q = m;
      if (best >= nice || --left == 0) goto done;
      goto longer;
    }
    if (unlikely(--left == 0)) goto done;
  }

longer:
  wl = ld32(in + best - 3);
  for (;;) {
    if (unlikely(c <= floor)) goto done;
    const uint8_t *m = base + c;
    c = links[c];
    if (unlikely(ld32(m + best - 3) == wl) && ld32(m) == w0) {
      uint32_t len = extend(in, m, 4, max);
      if (len > best) {
        best = len;
        q = m;
        if (best >= nice) goto done;
        wl = ld32(in + best - 3);
      }
    }
    if (unlikely(--left == 0)) goto done;
  }

done:
  *qp = q;
  return best;
}

/* [hc_find mf basep in best max nice depth h dist long_] inserts [in] and
   returns the longest match there longer than [best], up to [max],
   following at most [depth] links and stopping at [nice]; it is [best] if
   none. [h] holds the hashes of [in] and leaves those of [in + 1]. A
   position with fewer than 5 bytes after it is neither hashed nor
   inserted. With [long_], for a [depth] over SHORT_WALK, a walk still
   going after SHORT_WALK entries goes on in [hc_walk_on]. */
#define SHORT_WALK 8

INLINE uint32_t hc_find(struct hc *mf, const uint8_t **basep,
                        const uint8_t *in, uint32_t best, uint32_t max,
                        uint32_t nice, uint32_t depth, uint32_t *h3p,
                        uint32_t *h4p, uint32_t *dist, const bool long_) {
  *dist = 0;
  if (max < 5) return best;

  /* Insert [in], first moving the base when its offset would reach
     WSIZE. Offsets are then in [0, WSIZE), so [cur] indexes [next]. */
  const uint8_t *base = *basep;
  ptrdiff_t cur = in - base;
  if (unlikely(cur >= WSIZE)) {
    base += WSIZE;
    *basep = base;
    cur -= WSIZE;
    hc_rebase(mf);
  }
  const ptrdiff_t floor = cur - WSIZE;
  mpos *const s3 = &mf->h3[*h3p], *const s4 = &mf->h4[*h4p];
  const ptrdiff_t c3 = *s3;
  ptrdiff_t c = *s4;
  *s3 = (mpos)cur;
  *s4 = (mpos)cur;
  mf->next[WSIZE + cur] = (mpos)c;

  const uint32_t s = le32(in + 1);
  *h3p = hash(s & 0xFFFFFF, H3_BITS);
  *h4p = hash(s, H4_BITS);
  PREFETCHW(&mf->h3[*h3p]);
  PREFETCHW(&mf->h4[*h4p]);
  if (best >= nice) return best;

  /* Every position enters both tables, so an in-window position sharing
     [in]'s first 3 bytes is at most as old as [c3]: none is when [c3] is
     out of the window, and no candidate counts. */
  if (c3 <= floor) return best;

  /* The 3-byte candidate, when [best] is below 3, stands as a match of
     3 that any chain candidate replaces: those are 4 or longer. */
  uint32_t got = best;
  const uint8_t *q = NULL;
  if (best < 3) {
    got = 3;
    if (((le32(base + c3) ^ le32(in)) & 0xFFFFFF) == 0) q = base + c3;
  }

  /* The chain, newest first. A candidate longer than [got] matches [in]
     on the 4 bytes ending at [got], which rejects most at one load; the
     first 4 bytes then confirm it. */
  const mpos *const links = mf->next + WSIZE;
  const uint32_t w0 = ld32(in);
  uint32_t wt = ld32(in + got - 3);
  uint32_t walk = long_ && depth > SHORT_WALK ? SHORT_WALK : depth;
  for (; walk > 0; walk--) {
    if (c <= floor) goto searched;
    const uint8_t *m = base + c;
    c = links[c];
    if (likely(ld32(m + got - 3) != wt) || ld32(m) != w0) continue;
    uint32_t len = extend(in, m, 4, max);
    if (len <= got) continue;
    got = len;
    q = m;
    if (got >= nice) goto searched;
    wt = ld32(in + got - 3);
  }
  if (long_ && depth > SHORT_WALK && c > floor)
    got = hc_walk_on(mf, base, in, c, (int32_t)floor, got, max, nice,
                     depth - SHORT_WALK, &q);

searched:
  if (q == NULL) return best;
  *dist = (uint32_t)(in - q);
  return got;
}

/* [hc_skip] inserts the [n] positions from [in], unless the input ends
   within [n + 5] bytes of [in]. */
INLINE void hc_skip(struct hc *mf, const uint8_t **basep, const uint8_t *in,
                    const uint8_t *end, uint32_t n, uint32_t *h3p,
                    uint32_t *h4p) {
  if (unlikely((size_t)(end - in) < (size_t)n + 5)) return;

  /* A run that reaches base + WSIZE moves the base before its first
     insertion: the entries come out as if it had moved at that position,
     those inserted before it at offsets below 0. */
  const uint8_t *base = *basep;
  ptrdiff_t cur = in - base;
  if (unlikely(cur + (ptrdiff_t)n > WSIZE)) {
    base += WSIZE;
    *basep = base;
    cur -= WSIZE;
    hc_rebase(mf);
  }

  uint32_t h3 = *h3p, h4 = *h4p;
  for (const ptrdiff_t stop = cur + n; cur < stop; cur++) {
    mf->h3[h3] = (mpos)cur;
    mf->next[WSIZE + cur] = mf->h4[h4];
    mf->h4[h4] = (mpos)cur;
    uint32_t s = le32(++in);
    h3 = hash(s & 0xFFFFFF, H3_BITS);
    h4 = hash(s, H4_BITS);
  }
  *h3p = h3;
  *h4p = h4;
}

/* [ht_find] is [hc_find] for level 1: matches of 4 bytes or more among
   the last two positions of [in]'s hash. The caller ensures 5 bytes. */
INLINE uint32_t ht_find(struct ht *mf, const uint8_t **basep,
                        const uint8_t *in, uint32_t max, uint32_t nice,
                        uint32_t *h, uint32_t *dist) {
  const uint8_t *base = *basep;
  ptrdiff_t cur = in - base;
  if (unlikely(cur >= WSIZE)) {
    base += WSIZE;
    *basep = base;
    cur -= WSIZE;
    rebase((mpos *)mf, (mpos *)mf, HT_ENTRIES, mf->generic);
  }
  const ptrdiff_t floor = cur - WSIZE;

  /* A bucket holds its newer position first. */
  mpos *const bucket = mf->tab[*h];
  const ptrdiff_t newer = bucket[0], older = bucket[1];
  bucket[1] = (mpos)newer;
  bucket[0] = (mpos)cur;
  *h = hash(le32(in + 1), HT_BITS);
  PREFETCHW(mf->tab[*h]);

  /* The older is out of the window whenever the newer is. */
  if (newer <= floor) return 0;
  const uint32_t w0 = ld32(in);
  const uint8_t *q = base + newer;
  uint32_t len = 0;
  if (ld32(q) == w0) {
    len = extend(in, q, 4, max);
    if (len >= nice) goto found;
  }
  if (older > floor) {
    const uint8_t *m = base + older;
    if (ld32(m) == w0) {
      uint32_t l = extend(in, m, 4, max);
      if (l > len) {
        len = l;
        q = m;
      }
    }
  }
  if (len == 0) return 0;
found:
  *dist = (uint32_t)(in - q);
  return len;
}

INLINE void ht_skip(struct ht *mf, const uint8_t **basep, const uint8_t *in,
                    const uint8_t *end, uint32_t n, uint32_t *h) {
  if (unlikely((size_t)(end - in) < (size_t)n + 5)) return;
  const uint8_t *base = *basep;
  ptrdiff_t cur = in - base;
  if (unlikely(cur + (ptrdiff_t)n > WSIZE)) {
    base += WSIZE;
    *basep = base;
    cur -= WSIZE;
    rebase((mpos *)mf, (mpos *)mf, HT_ENTRIES, mf->generic);
  }

  /* After a match, at least MIN_MATCH positions: those go in without the
     loop's test, which leaves the loop fewer exits to mispredict. */
  uint32_t hv = *h;
  const ptrdiff_t stop = cur + n;
  if (n >= MIN_MATCH) {
    for (int k = 0; k < MIN_MATCH; k++, cur++) {
      mpos *const bucket = mf->tab[hv];
      bucket[1] = bucket[0];
      bucket[0] = (mpos)cur;
      hv = hash(le32(++in), HT_BITS);
    }
  }
  for (; cur < stop; cur++) {
    mpos *const bucket = mf->tab[hv];
    bucket[1] = bucket[0];
    bucket[0] = (mpos)cur;
    hv = hash(le32(++in), HT_BITS);
  }
  *h = hv;
}

/* Huffman codes */

/* A code over the literal/length symbols, then the 32 distance symbols:
   each symbol's length in bits and its codeword, bit-reversed for the
   writer, which emits the lowest bit first. */
#define NSYM (NLITLEN + NDIST)
#define DIST0 NLITLEN

struct code {
  uint32_t word[NSYM];
  uint8_t len[NSYM];
};

INLINE uint32_t reverse(uint32_t c, unsigned n) {
  c = ((c & 0x5555) << 1) | ((c >> 1) & 0x5555);
  c = ((c & 0x3333) << 2) | ((c >> 2) & 0x3333);
  c = ((c & 0x0F0F) << 4) | ((c >> 4) & 0x0F0F);
  c = ((c & 0x00FF) << 8) | ((c >> 8) & 0x00FF);
  return c >> (16 - n);
}

/* [canonical len n word count] gives the [n] symbols the canonical
   codewords of their lengths (RFC 1951, 3.2.2), [count[l]] of length
   [l]. */
static void canonical(const uint8_t *len, unsigned n, uint32_t *word,
                      const unsigned *count) {
  uint32_t next[MAX_DIST_LEN + 2];
  next[0] = 0;
  next[1] = 0;
  for (unsigned l = 2; l <= MAX_DIST_LEN + 1; l++)
    next[l] = (next[l - 1] + count[l - 1]) << 1;
  for (unsigned s = 0; s < n; s++)
    word[s] = len[s] ? reverse(next[len[s]]++, len[s]) : 0;
}

/* [depths w n] replaces the [n >= 2] weights [w], in increasing order,
   by the depths of their leaves in a Huffman tree, in place (Moffat and
   Katajainen, "In-place calculation of minimum-redundancy codes", 1995):
   a pass forming internal nodes, whose weights become parent indices, a
   pass turning those into depths, and a pass assigning leaf depths. */
static void depths(uint32_t *w, int n) {
  int root = 0, leaf = 2, next;
  w[0] += w[1];
  for (next = 1; next < n - 1; next++) {
    if (leaf >= n || w[root] < w[leaf]) {
      w[next] = w[root];
      w[root++] = (uint32_t)next;
    } else {
      w[next] = w[leaf++];
    }
    if (leaf >= n || (root < next && w[root] < w[leaf])) {
      w[next] += w[root];
      w[root++] = (uint32_t)next;
    } else {
      w[next] += w[leaf++];
    }
  }
  w[n - 2] = 0;
  for (next = n - 3; next >= 0; next--) w[next] = w[w[next]] + 1;
  int avail = 1, used = 0;
  uint32_t depth = 0;
  root = n - 2;
  next = n - 1;
  while (avail > 0) {
    while (root >= 0 && w[root] == depth) {
      used++;
      root--;
    }
    while (avail > used) {
      w[next--] = depth;
      avail--;
    }
    avail = 2 * used;
    depth++;
    used = 0;
  }
}

/* [huffman freq n max len word] builds a complete code over the [n]
   symbols of [freq], no longer than [max] bits; symbols of count 0 get no
   codeword. Fewer than two symbols in use still get two codewords of one
   bit, as a decoder needs a complete code. */
static void huffman(const uint32_t *freq, unsigned n, unsigned max,
                    uint8_t *len, uint32_t *word) {
  uint32_t sym[NLITLEN], w[NLITLEN], tmp[NLITLEN];
  unsigned m = 0;
  uint32_t top = 0;
  for (unsigned s = 0; s < n; s++) {
    len[s] = 0;
    if (freq[s] == 0) continue;
    sym[m++] = s;
    if (freq[s] > top) top = freq[s];
  }

  /* Fewer than two in use: the used one and partners take length 1. The
     partners come first in the order the header sends lengths, so that
     trimming the header's trailing zeros removes as much as without them:
     for the code length code, [pre_order], whose first four entries are
     always sent; for the others, by number. */
  if (m < 2) {
    unsigned partners = 2 - m;
    for (unsigned i = 0; partners > 0; i++) {
      unsigned s = n == NPRE ? pre_order[i] : i;
      if (m == 1 && s == sym[0]) continue;
      len[s] = 1;
      partners--;
    }
    if (m == 1) len[sym[0]] = 1;
    unsigned count[MAX_DIST_LEN + 2] = {0};
    count[1] = 2;
    canonical(len, n, word, count);
    return;
  }

  /* Sort the symbols by count, stably, so that equal counts stay in
     symbol order: a least significant digit radix sort, a byte a pass. */
  for (unsigned shift = 0; shift < 32 && (top >> shift) != 0; shift += 8) {
    unsigned at[257] = {0};
    for (unsigned i = 0; i < m; i++) at[((freq[sym[i]] >> shift) & 255) + 1]++;
    for (unsigned d = 1; d <= 256; d++) at[d] += at[d - 1];
    for (unsigned i = 0; i < m; i++)
      tmp[at[(freq[sym[i]] >> shift) & 255]++] = sym[i];
    memcpy(sym, tmp, m * sizeof sym[0]);
  }
  for (unsigned i = 0; i < m; i++) w[i] = freq[sym[i]];
  depths(w, (int)m);

  /* Lengths over [max] become [max], then, while the code is
     oversubscribed, a leaf at [max] goes and a shallower leaf splits in
     two one level down, which keeps the count of leaves and lowers the
     Kraft sum by one unit of 2^-max. */
  unsigned count[MAX_DIST_LEN + 2] = {0};
  for (unsigned i = 0; i < m; i++) count[w[i] < max ? w[i] : max]++;
  uint32_t kraft = 0;
  for (unsigned l = 1; l <= max; l++) kraft += count[l] << (max - l);
  while (kraft > (1u << max)) {
    count[max]--;
    for (unsigned l = max - 1; l > 0; l--) {
      if (count[l] == 0) continue;
      count[l]--;
      count[l + 1] += 2;
      break;
    }
    kraft--;
  }

  /* The rarest symbols take the longest codes. */
  unsigned i = 0;
  for (unsigned l = max; l >= 1; l--)
    for (unsigned c = count[l]; c > 0; c--) len[sym[i++]] = (uint8_t)l;
  canonical(len, n, word, count);
}

static void fixed_code(struct code *c) {
  unsigned count[MAX_DIST_LEN + 2] = {0};
  for (unsigned s = 0; s < NLITLEN; s++) {
    c->len[s] = s < 144 ? 8 : s < 256 ? 9 : s < 280 ? 7 : 8;
    count[c->len[s]]++;
  }
  canonical(c->len, NLITLEN, c->word, count);
  unsigned dcount[MAX_DIST_LEN + 2] = {0};
  dcount[5] = NDIST;
  for (unsigned s = 0; s < NDIST; s++) c->len[DIST0 + s] = 5;
  canonical(c->len + DIST0, NDIST, c->word + DIST0, dcount);
}

/* Blocks */

/* A block's parse is a list of records, each a run of literals and the
   match after it, the last a run alone. The literals' bytes are read back
   from the input. The writer finds the last record by counting the
   block's bytes, so the last record's match fields are never read. */
#define SLOT_SHIFT 9
struct seq {
  uint32_t lits;  /* the literals before the match */
  uint16_t match; /* the match's length, 3 to 258, and its distance's
                     symbol shifted by SLOT_SHIFT */
  uint16_t dist;  /* its distance, 1 to 32768 */
};

_Static_assert((uint64_t)SOFT_MAX + MIN_BLOCK + ITEM_MAX <= UINT32_MAX,
               "a record's literal count holds a block's longest run");
_Static_assert(MAX_MATCH < 1 << SLOT_SHIFT && 29 << SLOT_SHIFT < 1 << 16,
               "a length and a distance symbol share a record's 16 bits");

/* What a block needs to be written: its symbol counts, its dynamic code
   and the fixed one, and its code length code. */
struct blk {
  uint32_t freq[NSYM];
  struct code dyn, fixed;
  uint32_t pre_freq[NPRE];
  uint32_t pre_word[NPRE];
  uint8_t pre_len[NPRE];
  uint16_t items[NSYM]; /* code length items: symbol | extra << 5 */
  unsigned nitems, nlit, ndist, npre;

  /* The written code's match fields: per length its symbol's codeword
     with the extra bits above it, and their count; per distance symbol
     its codeword less its base distance shifted above it, so that adding
     the distance so shifted gives the codeword and the extra bits, the
     codeword's length, and that with the extra bits. */
  uint32_t len_word[MAX_MATCH + 1];
  uint8_t len_bits[MAX_MATCH + 1];
  uint32_t dist_word[30];
  uint8_t dist_len[30], dist_bits[30];
};

/* [rle lens n freq items] codes the [n] code lengths [lens] in the code
   length alphabet (RFC 1951, 3.2.7): it writes the items to [items], each
   its symbol with its repeat count's extra bits above, counts the 19
   symbols among them in [freq], and is the number of items. A run of a
   length is one item per piece of the most a symbol repeats (138 zeros,
   or 6 copies after the first), runs too short for a repeat symbol
   items of their own. */
static unsigned rle(const uint8_t *lens, unsigned n, uint32_t *freq,
                    uint16_t *items) {
  unsigned k = 0;
  memset(freq, 0, NPRE * sizeof *freq);
#define ITEM(s, x)                                                           \
  do {                                                                       \
    freq[s]++;                                                               \
    items[k++] = (uint16_t)((s) | (x) << 5);                                 \
  } while (0)

  /* Each maximal run of a length. A run cut into pieces leaves at least
     3 for the last: a piece that would leave 1 or 2 is shortened by 3,
     which keeps the count of pieces at its least. */
  for (unsigned i = 0; i < n;) {
    unsigned l = lens[i], r = 1;
    while (i + r < n && lens[i + r] == l) r++;
    i += r;

    if (l == 0) {
      if (r < 3) {
        for (; r > 0; r--) ITEM(0, 0u);
        continue;
      }
      while (r > 138) {
        unsigned piece = r - 138 >= 3 ? 138 : r - 3;
        ITEM(18, piece - 11);
        r -= piece;
      }
      if (r <= 10)
        ITEM(17, r - 3);
      else
        ITEM(18, r - 11);
      continue;
    }

    /* A nonzero length is sent once, then repeated. */
    ITEM(l, 0u);
    r--;
    if (r < 3) {
      for (; r > 0; r--) ITEM(l, 0u);
      continue;
    }
    while (r > 6) {
      unsigned piece = r - 6 >= 3 ? 6 : r - 3;
      ITEM(16, piece - 3);
      r -= piece;
    }
    ITEM(16, r - 3);
  }
#undef ITEM
  return k;
}

/* [header b] builds the block's dynamic code and its code length code. */
static void header(struct blk *b) {
  huffman(b->freq, NLITLEN, MAX_LITLEN_LEN, b->dyn.len, b->dyn.word);
  huffman(b->freq + DIST0, NDIST, MAX_DIST_LEN, b->dyn.len + DIST0,
          b->dyn.word + DIST0);
  unsigned nlit = NLITLEN, ndist = NDIST;
  while (nlit > 257 && b->dyn.len[nlit - 1] == 0) nlit--;
  while (ndist > 1 && b->dyn.len[DIST0 + ndist - 1] == 0) ndist--;
  uint8_t lens[NSYM];
  memcpy(lens, b->dyn.len, nlit);
  memcpy(lens + nlit, b->dyn.len + DIST0, ndist);
  b->nlit = nlit;
  b->ndist = ndist;
  b->nitems = rle(lens, nlit + ndist, b->pre_freq, b->items);
  huffman(b->pre_freq, NPRE, MAX_PRE_LEN, b->pre_len, b->pre_word);
  unsigned npre = NPRE;
  while (npre > 4 && b->pre_len[pre_order[npre - 1]] == 0) npre--;
  b->npre = npre;
}

/* The output: the next byte, the end, and the bits not yet written, the
   stream's next bit lowest. Between blocks fewer than 8 bits wait. */
struct out {
  uint8_t *p, *end;
  uint64_t bits;
  unsigned n;
};

/* The block writer's bits, in locals [acc], [n], [p] and [end]: [acc]
   holds [n] pending bits, the stream's next lowest. [PUT] appends a
   field; the caller keeps [n] under 64. [DRAIN] stores the whole bytes,
   8 at a time while 8 bytes of room remain, else one at a time, and
   leaves fewer than 8 bits. The price checked before the first bit keeps
   every store below [end]. */
#define PUT(w, l) (acc |= (uint64_t)(w) << n, n += (l))
#define DRAIN()                                                              \
  do {                                                                       \
    if (likely(end - p >= 8)) {                                              \
      put64le(p, acc);                                                       \
      p += n >> 3;                                                           \
      acc >>= n & ~7u;                                                       \
      n &= 7;                                                                \
    } else {                                                                 \
      for (; n >= 8; n -= 8) {                                               \
        *p++ = (uint8_t)acc;                                                 \
        acc >>= 8;                                                           \
      }                                                                      \
    }                                                                        \
  } while (0)

/* Four literals follow fewer than 8 pending bits between drains. */
_Static_assert(7 + 4 * MAX_LITLEN_LEN < 64, "four literals fit the bits");

/* [stored_price n len] is the bits of [len] bytes as stored blocks of at
   most MAX_STORED bytes, the first starting [n] bits into a byte: each a
   3-bit header padded to the byte, LEN, NLEN and the bytes. Every block
   after the first starts on a byte, so its header and padding are 8. */
static uint64_t stored_price(unsigned n, size_t len) {
  uint64_t blocks = len == 0 ? 1 : (len + MAX_STORED - 1) / MAX_STORED;
  uint64_t first = ((n + 3 + 7) & ~7u) - n;
  return first + 8 * (blocks - 1) + 32 * blocks + 8 * (uint64_t)len;
}

static void write_stored(struct out *o, const uint8_t *data, size_t len,
                         bool final) {
  uint8_t *p = o->p;
  uint64_t acc = o->bits;
  unsigned n = o->n;
  do {
    size_t k = len < MAX_STORED ? len : MAX_STORED;
    len -= k;
    PUT(final && len == 0, 3);
    for (n = (n + 7) & ~7u; n > 0; n -= 8) {
      *p++ = (uint8_t)acc;
      acc >>= 8;
    }
    p[0] = (uint8_t)k;
    p[1] = (uint8_t)(k >> 8);
    p[2] = (uint8_t)~k;
    p[3] = (uint8_t)(~k >> 8);
    memcpy(p + 4, data, k);
    p += 4 + k;
    data += k;
  } while (len > 0);
  o->p = p;
  o->bits = 0;
  o->n = 0;
}

/* [match_fields b c] fills [b]'s match fields from the code [c]. */
static void match_fields(struct blk *b, const struct code *c) {
  for (unsigned l = MIN_MATCH; l <= MAX_MATCH; l++) {
    unsigned slot = length_slot[l], s = 257 + slot;
    b->len_word[l] = c->word[s] | (l - length_base[slot]) << c->len[s];
    b->len_bits[l] = (uint8_t)(c->len[s] + length_extra[slot]);
  }
  for (unsigned slot = 0; slot < 30; slot++) {
    b->dist_word[slot] = c->word[DIST0 + slot] -
                         ((uint32_t)dist_base[slot] << c->len[DIST0 + slot]);
    b->dist_len[slot] = c->len[DIST0 + slot];
    b->dist_bits[slot] = (uint8_t)(c->len[DIST0 + slot] + dist_extra[slot]);
  }
}

/* [write_coded b o data len seqs final dyn] writes the block under the
   dynamic code with its header when [dyn], else under the fixed code. */
static void write_coded(struct blk *b, struct out *o, const uint8_t *data,
                        size_t len, const struct seq *seqs, bool final,
                        bool dyn) {
  const struct code *c = dyn ? &b->dyn : &b->fixed;
  uint8_t *p = o->p;
  uint8_t *const end = o->end;
  uint64_t acc = o->bits;
  unsigned n = o->n;

  PUT(final | (dyn ? 2u : 1u) << 1, 3);
  if (dyn) {
    PUT(b->nlit - 257, 5);
    PUT(b->ndist - 1, 5);
    PUT(b->npre - 4, 4);
    DRAIN();
    for (unsigned i = 0; i < b->npre; i++) {
      PUT(b->pre_len[pre_order[i]], 3);
      DRAIN();
    }
    for (unsigned i = 0; i < b->nitems; i++) {
      unsigned item = b->items[i], s = item & 31;
      PUT(b->pre_word[s], b->pre_len[s]);
      PUT(item >> 5, pre_extra[s]);
      DRAIN();
    }
  }

  bool matches = false;
  for (unsigned s = 257; s < 257 + 29; s++) matches |= b->freq[s] != 0;
  if (matches) match_fields(b, c);

  const uint32_t *const word = c->word;
  const uint8_t *const bits = c->len;
  const uint8_t *in = data, *const data_end = data + len;
  for (const struct seq *s = seqs;; s++) {
    const uint8_t *const lits_end = in + s->lits;
    for (; lits_end - in >= 4; in += 4) {
      PUT(word[in[0]], bits[in[0]]);
      PUT(word[in[1]], bits[in[1]]);
      PUT(word[in[2]], bits[in[2]]);
      PUT(word[in[3]], bits[in[3]]);
      DRAIN();
    }
    if (in < lits_end) {
      for (; in < lits_end; in++) PUT(word[*in], bits[*in]);
      DRAIN();
    }
    if (in == data_end) break;

    unsigned l = s->match & ((1u << SLOT_SHIFT) - 1),
             slot = s->match >> SLOT_SHIFT;
    uint32_t d = s->dist;
    PUT(b->len_word[l], b->len_bits[l]);
    PUT(b->dist_word[slot] + (d << b->dist_len[slot]), b->dist_bits[slot]);
    DRAIN();
    in += l;
  }
  PUT(word[EOB], bits[EOB]);
  DRAIN();

  o->p = p;
  o->bits = acc;
  o->n = n;
}

/* [write_block b o data len seqs final] writes the block of the [len]
   bytes at [data], parsed as [seqs], as the cheapest of a dynamic, a
   fixed and stored blocks, preferring stored then fixed on a tie. It is
   false, having written nothing, when the block does not fit [o]. The
   block's counts include its end-of-block symbol. */
static bool write_block(struct blk *b, struct out *o, const uint8_t *data,
                        size_t len, const struct seq *seqs, bool final) {
  header(b);

  /* The bits of each form. The two codes share the extra bits. */
  uint64_t extra = 0;
  for (unsigned slot = 0; slot < 29; slot++)
    extra += (uint64_t)b->freq[257 + slot] * length_extra[slot];
  for (unsigned slot = 0; slot < 30; slot++)
    extra += (uint64_t)b->freq[DIST0 + slot] * dist_extra[slot];
  uint64_t fixed = 3 + extra, dyn = 3 + 5 + 5 + 4 + 3 * b->npre + extra;
  for (unsigned s = 0; s < NSYM; s++) {
    fixed += (uint64_t)b->freq[s] * b->fixed.len[s];
    dyn += (uint64_t)b->freq[s] * b->dyn.len[s];
  }
  for (unsigned s = 0; s < NPRE; s++)
    dyn += (uint64_t)b->pre_freq[s] * (b->pre_len[s] + pre_extra[s]);
  uint64_t stored = stored_price(o->n, len);

  uint64_t price = stored <= fixed && stored <= dyn ? stored
                   : fixed <= dyn                  ? fixed
                                                   : dyn;
  if ((o->n + price + 7) / 8 > (uint64_t)(o->end - o->p)) return false;
  if (price == stored)
    write_stored(o, data, len, final);
  else
    write_coded(b, o, data, len, seqs, final, price != fixed);
  return true;
}

/* The encoder */

/* The block statistics that decide where a block ends: literals by their
   top two bits and lowest bit, matches as shorter or longer than 9. */
#define NOBS 10

struct split {
  uint32_t fresh[NOBS], seen[NOBS];
  uint32_t nfresh, nseen;
};

struct lz_deflate_tables {
  struct blk b;
  enum lz_wrap wrap;
  uint8_t parser;
  uint32_t depth, nice;
  size_t pass, soft, seqmax;
  bool stream;
  bool generic; /* the matchfinder rebases with the generic loop */

  /* The stream: its phase, checksum and length. */
  bool begun;    /* the header is written */
  bool finished; /* the final block is written */
  bool done;     /* the trailer is written: the next call restarts */
  bool ready;    /* the matchfinder is set up */
  uint32_t check;
  uint64_t total;
  uint64_t bits; /* bits of the last partial byte */
  unsigned nbits;

  /* The window of a streaming encoder, [cap] bytes, [fill] used; the
     positions below index it, or the caller's buffer for [encode]. */
  uint8_t *win;
  size_t cap, fill;

  /* The parse. */
  size_t pos;    /* the next byte to parse */
  size_t base;   /* the matchfinder's base */
  size_t block;  /* the open block's first byte */
  size_t recalc; /* where the lazy parsers next revise [min_len] */
  size_t told;   /* where the parse's reported work ends (lz.h) */
  bool open;     /* a block is open */
  bool held;     /* it has ended and waits to be written */
  bool held_final;
  uint32_t min_len;
  uint32_t h[2];
  size_t nseq;
  struct split split;
  void *mf;
  struct seq *seqs;
};

/* Allocation: the state, the matchfinder, the sequence store and, for a
   stream, the window, in one block. */

#define ALIGN(n) (((n) + 63) & ~(size_t)63)

static size_t mf_size(uint8_t parser) {
  return parser == FASTEST ? sizeof(struct ht) : sizeof(struct hc);
}

/* [window_at parser seqmax] is where the window starts in the block: the
   bytes of the state, the matchfinder and a store of [seqmax] matches. */
static size_t window_at(uint8_t parser, size_t seqmax) {
  size_t at_mf = ALIGN(sizeof(struct lz_deflate_tables));
  size_t at_seq = at_mf + ALIGN(mf_size(parser));
  return at_seq + ALIGN((seqmax + 1) * sizeof(struct seq));
}

static size_t seq_store(uint8_t parser) {
  return parser == FASTEST ? FAST_SEQ_STORE : SEQ_STORE;
}

static size_t window_size(int level) {
  return WIN_SIZE(levels[level].parser == FASTEST ? FAST_SOFT_MAX : SOFT_MAX);
}

size_t lz_deflate_enc_mem(int level) {
  uint8_t parser = levels[level].parser;
  return window_at(parser, seq_store(parser)) + window_size(level);
}

/* [create st n] allocates [st]'s tables for a stream, or for an encode of
   [n] bytes. Those hold at most [n / MIN_MATCH] matches, so a store of
   one more never fills: it ends no block that a full store would not. */
static struct lz_deflate_tables *create(const struct lz_deflate_enc *st,
                                        size_t n) {
  enum lz_wrap wrap = st->wrap;
  int level = st->level;
  bool stream = st->stream;
  uint8_t parser = levels[level].parser;
  size_t soft = parser == FASTEST ? FAST_SOFT_MAX : SOFT_MAX;
  size_t seqmax = seq_store(parser);
  if (!stream && n / MIN_MATCH + 1 < seqmax) seqmax = n / MIN_MATCH + 1;
  size_t at_mf = ALIGN(sizeof(struct lz_deflate_tables));
  size_t at_seq = at_mf + ALIGN(mf_size(parser));
  size_t at_win = window_at(parser, seqmax);
  size_t cap = stream ? window_size(level) : 0;
  uint8_t *m = lz_alloc(at_win + cap);
  if (m == NULL) return NULL;
  struct lz_deflate_tables *e = (struct lz_deflate_tables *)m;
  memset(e, 0, sizeof *e);
  fixed_code(&e->b.fixed);
  e->wrap = wrap;
  e->parser = parser;
  e->depth = levels[level].depth;
  e->nice = levels[level].nice;
  e->pass = PASSTHROUGH(level);
  e->soft = soft;
  e->seqmax = seqmax;
  e->stream = stream;
  e->generic = st->generic;
  e->mf = m + at_mf;
  e->seqs = (struct seq *)(m + at_seq);
  e->win = stream ? m + at_win : NULL;
  e->cap = cap;
  e->check = wrap == LZ_ZLIB ? 1 : 0;
  return e;
}

static void restart(struct lz_deflate_tables *e) {
  e->begun = e->finished = e->done = e->ready = false;
  e->check = e->wrap == LZ_ZLIB ? 1 : 0;
  e->total = 0;
  e->bits = 0;
  e->nbits = 0;
  e->fill = 0;
  e->pos = e->base = e->block = e->recalc = e->told = 0;
  e->open = e->held = e->held_final = false;
  e->nseq = 0;
}

/* Block statistics */

INLINE void observe_literal(struct split *s, unsigned c) {
  s->fresh[2 * (c >> 6) + (c & 1)]++;
  s->nfresh++;
}

INLINE void observe_match(struct split *s, uint32_t len) {
  s->fresh[8 + (len >= 9)]++;
  s->nfresh++;
}

/* [split_now s len] compares the distribution of the last observations
   with the block's so far: the block ends when the sum of the differences
   of their probabilities, plus a term growing with the block's length,
   reaches 200/512, with a higher bar for short blocks. Otherwise the
   observations join the block's. */
static bool split_now(struct split *s, size_t len) {
  const uint64_t nf = s->nfresh, ns = s->nseen;
  if (ns > 0) {
    /* The L1 distance between the two distributions, scaled by ns * nf. */
    uint64_t delta = 0;
    for (unsigned i = 0; i < NOBS; i++) {
      uint64_t a = s->fresh[i] * ns, b = s->seen[i] * nf;
      delta += a > b ? a - b : b - a;
    }
    uint64_t bar = (200 * nf / 512) * ns;
    /* A short block pays a whole header: it ends on a clearer change. */
    if (len < 10000 && ns + nf < 8192) bar += bar * (8192 - (ns + nf)) / 8192;
    if (delta + (len / 4096) * ns >= bar) return true;
  }
  for (unsigned i = 0; i < NOBS; i++) {
    s->seen[i] += s->fresh[i];
    s->fresh[i] = 0;
  }
  s->nseen += s->nfresh;
  s->nfresh = 0;
  return false;
}

/* The shortest match the greedy and lazy parsers take, from the number of
   distinct literals in use: with few, literals are cheap and short
   matches do not pay. A shallow search finds few long matches, so it caps
   the length. */
static uint32_t min_len_of(unsigned used, uint32_t depth) {
  uint32_t m = used <= 5    ? 9
               : used <= 7  ? 8
               : used <= 9  ? 7
               : used <= 15 ? 6
               : used <= 44 ? 5
               : used <= 79 ? 4
                            : 3;
  uint32_t cap = depth < 5 ? 4 : depth < 10 ? 5 : depth < 16 ? 7 : MAX_MATCH;
  return m < cap ? m : cap;
}

/* [scan_min_len p n depth] is the minimum match length from the first
   4 KiB of the [n] bytes at [p]; 3 for a block under 512 bytes, where the
   fixed code may well win. */
static uint32_t scan_min_len(const uint8_t *p, size_t n, uint32_t depth) {
  if (n < 512) return MIN_MATCH;
  size_t k = n < 4096 ? n : 4096;
  uint8_t seen[256] = {0};
  for (size_t i = 0; i < k; i++) seen[p[i]] = 1;
  unsigned used = 0;
  for (unsigned c = 0; c < 256; c++) used += seen[c];
  return min_len_of(used, depth);
}

/* [revise_min_len f depth] counts the literals the block uses more than
   once in 1024 of its literals. */
static uint32_t revise_min_len(const uint32_t *f, uint32_t depth) {
  uint32_t total = 0;
  for (unsigned c = 0; c < 256; c++) total += f[c];
  unsigned used = 0;
  for (unsigned c = 0; c < 256; c++) used += f[c] > total >> 10;
  return min_len_of(used, depth);
}

/* The parsers

   [parse e buf end last] parses items from [e->pos] over [buf], whose
   bytes run to [end]; [last] says the input ends there. It returns
   BLOCK_END when the open block ends, at an item boundary, and NEED_MORE
   when the next item needs bytes past [end]. */

enum { NEED_MORE, BLOCK_END };

/* [begin e buf end last] opens a block at [e->pos]. */
static void begin(struct lz_deflate_tables *e, const uint8_t *buf, size_t end,
                  bool last) {
  size_t pos = e->pos;
  e->open = true;
  e->block = pos;
  e->nseq = 0;
  e->seqs[0].lits = 0;
  memset(e->b.freq, 0, sizeof e->b.freq);
  memset(&e->split, 0, sizeof e->split);
  if (e->parser == FASTEST) return;
  /* The block's longest extent, as far as the scan can tell: [last] knows
     the input's end, otherwise more than 4 KiB follows. */
  size_t n = e->soft;
  if (last && end - pos < e->soft + MIN_BLOCK) n = end - pos;
  e->min_len = scan_min_len(buf + pos, n, e->depth);
  e->recalc = pos + 10000;
}

/* [ends e pos end last nseq] says whether the open block ends after the
   item that stopped at [pos]. */
INLINE bool ends(struct lz_deflate_tables *e, size_t pos, size_t end,
                 bool last, size_t nseq) {
  size_t len = pos - e->block;
  if (last && pos == end) return true;
  /* Under [last], a block that would leave a last one under MIN_BLOCK
     runs to the end instead. */
  if (len >= e->soft && (!last || end - e->block >= e->soft + MIN_BLOCK))
    return true;
  if (nseq >= e->seqmax) return true;
  if (e->parser == FASTEST) return false;
  if (e->split.nfresh < CHECK_EVERY || len < MIN_BLOCK) return false;
  if (last && end - pos < MIN_BLOCK) return false;
  return split_now(&e->split, len);
}

/* The first parsed position's hashes. */
static void prime(struct lz_deflate_tables *e, const uint8_t *buf,
                  size_t end) {
  e->ready = true;
  if (e->parser == FASTEST) {
    struct ht *mf = e->mf;
    mf_fill((mpos *)mf, HT_ENTRIES);
    mf->generic = e->generic;
  } else {
    /* A walk reads only the links of inserted positions, and [encode] of
       under WSIZE bytes inserts none from [end] on: those links are never
       read. */
    struct hc *mf = e->mf;
    size_t links = !e->stream && end < WSIZE ? end : WSIZE;
    mf_fill((mpos *)mf, HC_HEADS);
    mf_fill(mf->next + WSIZE, links);
    mf->generic = e->generic;
  }
  e->h[0] = e->h[1] = 0;
  if (end - e->pos < 4) return;
  uint32_t s = le32(buf + e->pos);
  e->h[0] =
      e->parser == FASTEST ? hash(s, HT_BITS) : hash(s & 0xFFFFFF, H3_BITS);
  e->h[1] = hash(s, H4_BITS);
}

/* [LITERAL c] emits the byte [c] into the open record; [MATCH l d] closes
   the record with a match and opens the next. Both count their symbols,
   and with [stats] observe them for the block's end. The open record's
   literal count is the parser's local [run], where those literals start,
   so that a literal costs no count and nothing the counts' stores could
   alias: [LITS_LOAD] and [LITS_STORE] move it in and out around the item
   loop. */
#define LITS_LOAD() const uint8_t *run = in - seq->lits
#define LITS_STORE() seq->lits = (uint32_t)(in - run)
#define LITERAL(c)                                                           \
  do {                                                                       \
    unsigned c_ = (c);                                                       \
    freq[c_]++;                                                              \
    if (stats) observe_literal(&e->split, c_);                               \
  } while (0)

#define MATCH(l, d)                                                          \
  do {                                                                       \
    uint32_t l_ = (l), d_ = (d);                                             \
    unsigned s_ = dist_slot(d_);                                             \
    freq[257 + length_slot[l_]]++;                                           \
    freq[DIST0 + s_]++;                                                      \
    seq->lits = (uint32_t)(in - run);                                        \
    seq->match = (uint16_t)(l_ | s_ << SLOT_SHIFT);                          \
    seq->dist = (uint16_t)d_;                                                \
    seq++;                                                                   \
    run = in + l_;                                                           \
    if (stats) observe_match(&e->split, l_);                                 \
  } while (0)

/* [better cur_len cur_dist len dist k]: a match [len] at [dist] found
   further on is worth the literals before it when 4 * length - log2
   distance grows by more than [k]. */
INLINE bool better(uint32_t cur_len, uint32_t cur_dist, uint32_t len,
                   uint32_t dist, int k) {
  if (len < cur_len) return false;
  int lg_cur = 31 - __builtin_clz(cur_dist), lg = 31 - __builtin_clz(dist);
  return 4 * (int)(len - cur_len) + lg_cur - lg > k;
}

/* The parsers keep their position, the matchfinder's base and the ends
   as pointers into [buf], and the parse state of [e] in locals. */

#define PARSE_ENTRY(nhash)                                                    \
  const uint8_t *const end = buf + end_at;                                   \
  const uint8_t *const stop =                                                \
      last ? end : buf + (end_at >= LOOKAHEAD ? end_at - LOOKAHEAD + 1 : 0); \
  const uint8_t *in = buf + e->pos;                                          \
  const uint8_t *base = buf + e->base;                                       \
  uint32_t h3 = e->h[0], h4 = e->h[1];                                       \
  uint32_t *freq = e->b.freq;                                                \
  struct seq *seq = e->seqs + e->nseq;                                       \
  int r = NEED_MORE;                                                         \
  if (!e->open) {                                                            \
    if (in >= stop && !(last && in == end)) goto out;                        \
    begin(e, buf, end_at, last);                                             \
    seq = e->seqs;                                                           \
  }                                                                          \
  const uint8_t *const block = buf + e->block;                               \
  const uint8_t *const soft_end =                                            \
      e->soft < end_at - e->block ? block + e->soft : end;                   \
  const struct seq *const seq_end = e->seqs + e->seqmax;                     \
  const uint32_t nice0 = e->nice;                                            \
  (void)h4

/* After an item: ends the block when [ends] says so. */
#define PARSE_CHECK()                                                        \
  if (unlikely(in >= soft_end || seq >= seq_end ||                           \
               (stats && e->split.nfresh >= CHECK_EVERY)) &&                 \
      ends(e, (size_t)(in - buf), end_at, last, (size_t)(seq - e->seqs))) { \
    r = BLOCK_END;                                                           \
    break;                                                                   \
  }

#define PARSE_EXIT()                                                         \
  if (last && in == end) r = BLOCK_END;                                      \
  e->nseq = (size_t)(seq - e->seqs);                                         \
  out:                                                                       \
  e->pos = (size_t)(in - buf);                                               \
  e->base = (size_t)(base - buf);                                            \
  e->h[0] = h3;                                                              \
  e->h[1] = h4;                                                              \
  return r

/* [MAXLEN] sets [max] and [nice] for a match at [in]. */
#define MAXLEN()                                                             \
  do {                                                                       \
    size_t left_ = (size_t)(end - in);                                       \
    max = left_ < MAX_MATCH ? (uint32_t)left_ : MAX_MATCH;                   \
    nice = nice0 < max ? nice0 : max;                                        \
  } while (0)

NOINLINE int parse_fastest(struct lz_deflate_tables *e, const uint8_t *buf,
                           size_t end_at, bool last) {
  const bool stats = false;
  PARSE_ENTRY();
  struct ht *mf = e->mf;
  uint32_t h = h3;

  /* Before [fast], a whole match fits before the end and the block's
     soft end is not reached: [max] and [nice] are fixed, and an item
     needs the block's checks only when it fills the store. */
  size_t fast_at = end_at > MAX_MATCH ? end_at - MAX_MATCH : 0;
  if ((size_t)(soft_end - buf) < fast_at) fast_at = (size_t)(soft_end - buf);
  if ((size_t)(stop - buf) < fast_at) fast_at = (size_t)(stop - buf);
  const uint8_t *const fast = buf + fast_at;
  LITS_LOAD();
  while (in < stop) {
    uint32_t max, nice, d;
    if (in < fast) {
      for (;;) {
        uint32_t len = ht_find(mf, &base, in, MAX_MATCH, nice0, &h, &d);
        if (len > 0) {
          MATCH(len, d);
          ht_skip(mf, &base, in + 1, end, len - 1, &h);
          in += len;
          if (unlikely(seq >= seq_end)) break;
        } else {
          LITERAL(*in);
          in++;
        }
        if (unlikely(in >= fast)) break;
      }
      PARSE_CHECK();
      continue;
    }
    MAXLEN();
    /* Within 4 bytes of the end (only at [last]): literals. */
    if (unlikely(max < 5)) {
      for (; in < end; in++) LITERAL(*in);
      break;
    }
    uint32_t len = ht_find(mf, &base, in, max, nice, &h, &d);
    if (len > 0) {
      MATCH(len, d);
      ht_skip(mf, &base, in + 1, end, len - 1, &h);
      in += len;
    } else {
      LITERAL(*in);
      in++;
    }
    PARSE_CHECK();
  }
  h3 = h;
  LITS_STORE();
  PARSE_EXIT();
}

NOINLINE int parse_greedy(struct lz_deflate_tables *e, const uint8_t *buf,
                          size_t end_at, bool last) {
  const bool stats = true;
  PARSE_ENTRY();
  struct hc *mf = e->mf;
  const uint32_t depth = e->depth, min_len = e->min_len;
  LITS_LOAD();
  while (in < stop) {
    uint32_t max, nice, d;
    MAXLEN();
    uint32_t len =
        hc_find(mf, &base, in, min_len - 1, max, nice, depth, &h3, &h4, &d,
                false);
    /* A 3-byte match beyond 4096 costs about what its literals do. */
    if (len >= min_len && (len >= 4 || d <= 4096)) {
      MATCH(len, d);
      hc_skip(mf, &base, in + 1, end, len - 1, &h3, &h4);
      in += len;
    } else {
      LITERAL(*in);
      in++;
    }
    PARSE_CHECK();
  }
  LITS_STORE();
  PARSE_EXIT();
}

/* [lazy_item ... fast] parses one item of the lazy parsers from [*inp].
   With [fast], every find of the item has MAX_MATCH bytes before [end],
   so [max] and [nice] are fixed. */
INLINE void lazy_item(struct lz_deflate_tables *e, struct hc *mf,
                      const uint8_t **inp, const uint8_t **basep,
                      const uint8_t *end, uint32_t *h3p, uint32_t *h4p,
                      struct seq **seqp, const uint8_t **runp, uint32_t *freq,
                      uint32_t min_len, uint32_t depth, uint32_t nice0,
                      const bool lazy2, const bool long_, const bool fast) {
  const bool stats = true;
  const uint8_t *in = *inp;
  struct seq *seq = *seqp;
  const uint8_t *run = *runp;
  uint32_t max, nice, d;

#define FINDLEN()                                                            \
  do {                                                                       \
    if (fast) {                                                              \
      max = MAX_MATCH;                                                       \
      nice = nice0;                                                          \
    } else {                                                                 \
      MAXLEN();                                                              \
    }                                                                        \
  } while (0)

  FINDLEN();
  uint32_t len = hc_find(mf, basep, in, min_len - 1, max, nice, depth, h3p,
                         h4p, &d, long_);
  if (len < min_len || (len == 3 && d > 8192)) {
    LITERAL(*in);
    in++;
    goto done;
  }

  /* A match (len, d) at [in]: look one byte on, and at levels 8 and 9
     two, for a match worth the literals before it. */
  for (;;) {
    if (len >= nice) {
      MATCH(len, d);
      hc_skip(mf, basep, in + 1, end, len - 1, h3p, h4p);
      in += len;
      break;
    }

    uint32_t len1, d1;
    in++;
    FINDLEN();
    len1 = hc_find(mf, basep, in, len - 1, max, nice, depth >> 1, h3p, h4p,
                   &d1, long_);
    in--;
    if (better(len, d, len1, d1, 2)) {
      LITERAL(*in);
      in++;
      len = len1;
      d = d1;
      continue;
    }

    if (!lazy2) {
      MATCH(len, d);
      hc_skip(mf, basep, in + 2, end, len - 2, h3p, h4p);
      in += len;
      break;
    }

    uint32_t len2, d2;
    in += 2;
    FINDLEN();
    len2 = hc_find(mf, basep, in, len - 1, max, nice, depth >> 2, h3p, h4p,
                   &d2, long_);
    in -= 2;
    if (better(len, d, len2, d2, 6)) {
      LITERAL(in[0]);
      LITERAL(in[1]);
      in += 2;
      len = len2;
      d = d2;
      continue;
    }

    MATCH(len, d);
    if (len > 3) hc_skip(mf, basep, in + 3, end, len - 3, h3p, h4p);
    in += len;
    break;
  }
#undef FINDLEN

done:
  *inp = in;
  *seqp = seq;
  *runp = run;
}

INLINE int lazy(struct lz_deflate_tables *e, const uint8_t *buf,
                size_t end_at, bool last, const bool lazy2, const bool long_) {
  const bool stats = true;
  PARSE_ENTRY();
  struct hc *mf = e->mf;
  const uint32_t depth = e->depth;
  uint32_t min_len = e->min_len;
  size_t recalc = e->recalc;

  /* An item moves at most ITEM_MAX bytes on: one that starts before
     [fast] finds with MAX_MATCH bytes left. */
  const size_t fast_at =
      end_at > ITEM_MAX + MAX_MATCH ? end_at - (ITEM_MAX + MAX_MATCH) : 0;
  const uint8_t *const fast = buf + fast_at;
  LITS_LOAD();
  while (in < stop) {
    /* Revise the minimum length 10000 bytes into the block, then at about
       each doubling of its length. */
    size_t at = (size_t)(in - buf);
    if (unlikely(at >= recalc)) {
      min_len = revise_min_len(freq, depth);
      recalc += (size_t)(in - block);
    }

    if (likely(in < fast))
      lazy_item(e, mf, &in, &base, end, &h3, &h4, &seq, &run, freq, min_len,
                depth, nice0, lazy2, long_, true);
    else
      lazy_item(e, mf, &in, &base, end, &h3, &h4, &seq, &run, freq, min_len,
                depth, nice0, lazy2, long_, false);
    PARSE_CHECK();
  }
  LITS_STORE();
  e->min_len = min_len;
  e->recalc = recalc;
  PARSE_EXIT();
}

/* Levels 7 to 9 search deep: their walks go on out of line (see
   [hc_walk_on]). At levels 5 and 6 the call costs more than it saves. */
#define LONG_WALK 64

INLINE int parse_lazy(struct lz_deflate_tables *e, const uint8_t *buf,
                      size_t end_at, bool last, bool lazy2) {
  if ((lazy2 || e->depth >= LONG_WALK) && e->depth >> 2 > SHORT_WALK)
    return lazy(e, buf, end_at, last, lazy2, true);
  return lazy(e, buf, end_at, last, lazy2, false);
}

NOINLINE int parse_lazy1(struct lz_deflate_tables *e, const uint8_t *buf,
                         size_t end_at, bool last) {
  return parse_lazy(e, buf, end_at, last, false);
}

NOINLINE int parse_lazy2(struct lz_deflate_tables *e, const uint8_t *buf,
                         size_t end_at, bool last) {
  return parse_lazy(e, buf, end_at, last, true);
}

static int parse(struct lz_deflate_tables *e, const uint8_t *buf, size_t end,
                 bool last) {
  if (!e->ready) {
    if (!last && end - e->pos < LOOKAHEAD) return NEED_MORE;
    prime(e, buf, end);
  }
  switch (e->parser) {
  case FASTEST:
    return parse_fastest(e, buf, end, last);
  case GREEDY:
    return parse_greedy(e, buf, end, last);
  case LAZY:
    return parse_lazy1(e, buf, end, last);
  default:
    return parse_lazy2(e, buf, end, last);
  }
}

/* Wrappers */

static size_t header_size(enum lz_wrap wrap) {
  return wrap == LZ_GZIP ? 10 : wrap == LZ_ZLIB ? 2 : 0;
}

static size_t trailer_size(enum lz_wrap wrap) {
  return wrap == LZ_GZIP ? 8 : wrap == LZ_ZLIB ? 4 : 0;
}

/* zlib's FLEVEL (RFC 1950, 2.2) and gzip's XFL (RFC 1952, 2.3.1), as zlib
   sets them per level; gzip's MTIME is 0 and OS 255, unknown, so that the
   bytes depend on the input alone. */
static void write_header(enum lz_wrap wrap, int level, uint8_t *p) {
  if (wrap == LZ_ZLIB) {
    unsigned flevel = level < 2 ? 0 : level < 6 ? 1 : level == 6 ? 2 : 3;
    unsigned cmf = 0x78, flg = flevel << 6;
    flg += 31 - (cmf << 8 | flg) % 31;
    p[0] = (uint8_t)cmf;
    p[1] = (uint8_t)flg;
  } else if (wrap == LZ_GZIP) {
    static const uint8_t gz[10] = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255};
    memcpy(p, gz, 10);
    p[8] = level == 1 ? 4 : level == 9 ? 2 : 0;
  }
}

static void write_trailer(enum lz_wrap wrap, uint32_t check, uint64_t total,
                          uint8_t *p) {
  if (wrap == LZ_ZLIB) {
    p[0] = (uint8_t)(check >> 24);
    p[1] = (uint8_t)(check >> 16);
    p[2] = (uint8_t)(check >> 8);
    p[3] = (uint8_t)check;
  } else if (wrap == LZ_GZIP) {
    uint32_t size = (uint32_t)total;
    for (unsigned i = 0; i < 4; i++) {
      p[i] = (uint8_t)(check >> (8 * i));
      p[4 + i] = (uint8_t)(size >> (8 * i));
    }
  }
}

static uint32_t checksum(enum lz_wrap wrap, uint32_t check, const uint8_t *p,
                         size_t n) {
  if (wrap == LZ_GZIP) return lz_crc32(check, p, n);
  if (wrap == LZ_ZLIB) return lz_adler32(check, p, n);
  return check;
}

/* [finish o wrap check total] writes the last partial byte and the
   trailer, or is false when they do not fit. */
static bool finish(struct out *o, enum lz_wrap wrap, uint32_t check,
                   uint64_t total) {
  size_t need = (o->n > 0) + trailer_size(wrap);
  if ((size_t)(o->end - o->p) < need) return false;
  if (o->n > 0) *o->p++ = (uint8_t)o->bits;
  o->bits = 0;
  o->n = 0;
  write_trailer(wrap, check, total, o->p);
  o->p += trailer_size(wrap);
  return true;
}

/* [literals b data n seqs] makes the [n] bytes at [data] one sequence of
   literals and counts them. */
static void literals(struct blk *b, const uint8_t *data, size_t n,
                     struct seq *seqs) {
  memset(b->freq, 0, sizeof b->freq);
  for (size_t i = 0; i < n; i++) b->freq[data[i]]++;
  b->freq[EOB]++;
  seqs[0].lits = (uint32_t)n;
}

/* The whole-buffer encode */

/* An input of at most PASSTHROUGH bytes is one block of literals, with
   its tables on the stack: no allocation. */
static enum lz_status whole_small(struct lz_deflate_enc *st, struct lz_io *io,
                                  struct out *o, uint32_t check) {
  struct blk b;
  struct seq seq[1];
  fixed_code(&b.fixed);
  literals(&b, io->src, io->src_len, seq);
  if (!write_block(&b, o, io->src, io->src_len, seq, true) ||
      !finish(o, st->wrap, check, io->src_len))
    return LZ_NEED_OUTPUT;
  io->consumed = io->src_len;
  io->produced = (size_t)(o->p - io->dst);
  return LZ_DONE;
}

static enum lz_status whole(struct lz_deflate_enc *st, struct lz_io *io,
                            struct lz_error *err) {
  struct out o = {io->dst, io->dst + io->dst_cap, 0, 0};
  size_t n = io->src_len;
  lz_did(io->work, n);
  size_t hs = header_size(st->wrap);
  if (io->dst_cap < hs) return LZ_NEED_OUTPUT;
  write_header(st->wrap, st->level, o.p);
  o.p += hs;
  uint32_t check =
      checksum(st->wrap, st->wrap == LZ_ZLIB ? 1 : 0, io->src, n);
  if (n <= PASSTHROUGH(st->level)) return whole_small(st, io, &o, check);

  if (st->t == NULL) st->t = create(st, n);
  struct lz_deflate_tables *e = st->t;
  if (e == NULL) return lz_fail(err, LZ_NOMEM, 0, NULL);
  const uint8_t *buf = io->src;
  for (;;) {
    parse(e, buf, n, true);
    bool final = e->pos == n;
    e->b.freq[EOB]++;
    if (!write_block(&e->b, &o, buf + e->block, e->pos - e->block, e->seqs,
                     final))
      return LZ_NEED_OUTPUT;
    e->open = false;
    if (final) break;
  }
  if (!finish(&o, st->wrap, check, n)) return LZ_NEED_OUTPUT;
  io->consumed = n;
  io->produced = (size_t)(o.p - io->dst);
  return LZ_DONE;
}

/* The streaming encode */

/* [slide e] moves the bytes still needed to the window's start: the open
   block, the last WSIZE bytes before [pos], and the unparsed rest. */
static void slide(struct lz_deflate_tables *e) {
  size_t keep = e->pos > WSIZE ? e->pos - WSIZE : 0;
  if (e->open && e->block < keep) keep = e->block;
  if (keep == 0) return;
  memmove(e->win, e->win + keep, e->fill - keep);
  e->fill -= keep;
  e->pos -= keep;
  e->base -= keep;
  e->block -= keep;
  e->recalc -= keep;
  e->told = e->told > keep ? e->told - keep : 0;
}

/* [tell e w last] reports the bytes the next parse covers that no parse
   has reported: up to the input's end under [last], else up to
   LOOKAHEAD before it ([parse]). A parse that ends a block stops early,
   and the next covers the rest. */
static void tell(struct lz_deflate_tables *e, struct lz_work *w, bool last) {
  size_t stop = last                    ? e->fill
                : e->fill >= LOOKAHEAD ? e->fill - LOOKAHEAD + 1
                                        : 0;
  size_t from = e->pos > e->told ? e->pos : e->told;
  if (stop <= from) return;
  e->told = stop;
  lz_did(w, stop - from);
}

/* [take e src n] copies what fits of the [n] bytes at [src] into the
   window, and is their count. */
static size_t take(struct lz_deflate_tables *e, const uint8_t *src,
                   size_t n) {
  if (e->cap - e->fill < n) slide(e);
  size_t k = e->cap - e->fill;
  if (k > n) k = n;
  memcpy(e->win + e->fill, src, k);
  e->check = checksum(e->wrap, e->check, src, k);
  e->total += k;
  e->fill += k;
  return k;
}

static enum lz_status step(struct lz_deflate_enc *st, struct lz_io *io,
                           bool last, struct lz_error *err) {
  if (st->t == NULL) st->t = create(st, 0);
  struct lz_deflate_tables *e = st->t;
  if (e == NULL) return lz_fail(err, LZ_NOMEM, 0, NULL);
  if (e->done) restart(e);

  struct out o = {io->dst + io->produced, io->dst + io->dst_cap, e->bits,
                  e->nbits};
  enum lz_status s;
  if (!e->begun) {
    size_t hs = header_size(e->wrap);
    if ((size_t)(o.end - o.p) < hs) {
      s = LZ_NEED_OUTPUT;
      goto out;
    }
    write_header(e->wrap, st->level, o.p);
    o.p += hs;
    e->begun = true;
  }

  for (;;) {
    if (e->held) {
      size_t len = e->pos - e->block;
      if (!write_block(&e->b, &o, e->win + e->block, len, e->seqs,
                       e->held_final)) {
        s = LZ_NEED_OUTPUT;
        goto out;
      }
      e->held = false;
      e->open = false;
      e->finished = e->held_final;
    }
    if (e->finished) {
      if (!finish(&o, e->wrap, e->check, e->total)) {
        s = LZ_NEED_OUTPUT;
        goto out;
      }
      e->done = true;
      s = LZ_DONE;
      goto out;
    }
    if (io->consumed < io->src_len)
      io->consumed +=
          take(e, io->src + io->consumed, io->src_len - io->consumed);
    bool fin = last && io->consumed == io->src_len;

    if (fin && e->total <= e->pass) {
      literals(&e->b, e->win, e->fill, e->seqs);
      e->open = true;
      e->block = 0;
      e->pos = e->fill;
      e->held = e->held_final = true;
      continue;
    }
    tell(e, io->work, fin);
    if (parse(e, e->win, e->fill, fin) == BLOCK_END) {
      e->b.freq[EOB]++;
      e->held = true;
      e->held_final = fin && e->pos == e->fill;
      continue;
    }
    if (io->consumed < io->src_len) continue;
    s = LZ_NEED_INPUT;
    goto out;
  }
out:
  io->produced = (size_t)(o.p - io->dst);
  e->bits = o.bits;
  e->nbits = o.n;
  return s;
}

/* Entries */

static const char *const path_names[] = {"generic", "sse2"};

#if defined(__SSE2__)
#define PATHS 2
#else
#define PATHS 1
#endif

size_t lz_deflate_enc_paths(const char *const **names) {
  *names = path_names;
  return PATHS;
}

void lz_deflate_enc_path(struct lz_deflate_enc *st, size_t i) {
  st->generic = i == 0;
}

void lz_deflate_enc_init(struct lz_deflate_enc *st, enum lz_wrap wrap,
                         int level, bool stream) {
  st->wrap = wrap;
  st->level = level;
  st->stream = stream;
  st->t = NULL;
  lz_deflate_enc_path(st, PATHS - 1);
}

void lz_deflate_enc_reset(struct lz_deflate_enc *st) {
  if (st->t != NULL) restart(st->t);
}

void lz_deflate_enc_free(struct lz_deflate_enc *st) {
  if (st->t != NULL) lz_free(st->t);
  st->t = NULL;
}

enum lz_status lz_deflate_encode(struct lz_deflate_enc *st, struct lz_io *io,
                                 bool last, struct lz_error *err) {
  if (!st->stream) return whole(st, io, err);
  return step(st, io, last, err);
}

/* Every block is the cheapest of its forms, so the stream is no longer
   than stored blocks of the same lengths: 5 bytes each per 65535 bytes,
   alignment included, and a block other than the last is at least
   MIN_BLOCK bytes. */
size_t lz_deflate_bound(enum lz_wrap wrap, size_t n) {
  size_t blocks = n == 0 ? 1 : (n + MIN_BLOCK - 1) / MIN_BLOCK;
  return n + 5 * blocks + header_size(wrap) + trailer_size(wrap);
}
