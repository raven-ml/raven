/* The Zstandard encoder's entropy stage: a block's literals section,
   Huffman-coded or not, and its sequences section, FSE-coded (RFC 8878
   sections 3.1.1.3, 4.1 and 4.2).

   Bits go through one writer: an accumulator of up to 64 bits, flushed
   eight bytes at a time and advanced by the whole bytes it held, so that
   a flush never branches on the data; a checked writer clamps its cursor
   at the capacity and remembers the overflow.

   An FSE table is built as the decoder builds its own: the same spread of
   symbols over cells, so that each encoder state x in [L, 2L) mirrors the
   decoder's x - L. Per symbol, one subtraction and a shift give the bits
   a state emits, and an offset into the symbol's run of states gives the
   next state.

   Which mode each section takes is set by the rules of lits and
   field_table. */

#include "zstd_enc_int.h"

/* RFC 8878 section 3.1.1.3.1: literals block types. Section 3.1.1.3.2.1:
   Symbol_Compression_Modes. */
enum { LIT_RAW, LIT_RLE, LIT_HUF, LIT_REPEAT };
enum { SEQ_BASIC, SEQ_RLE, SEQ_FSE, SEQ_REPEAT };

/* Section 3.1.1.3.2.1.1: the largest code of each sequence field, and the
   accuracy logs of the tables the encoder builds. */
#define LL_MAX 35
#define ML_MAX 52
#define OF_MAX 31
#define OF_DEFAULT_MAX 28
#define LL_LOG 9
#define ML_LOG 9
#define OF_LOG 8
#define HUF_LOG 11
#define WEIGHT_LOG 6
#define LONG_NBSEQ 0x7F00

#define INF ((size_t)-1)

/* Section 3.1.1.3.2.2: the predefined distributions. */
static const int16_t ll_default[LL_MAX + 1] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2,  2,
    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
static const int16_t ml_default[ML_MAX + 1] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
static const int16_t of_default[OF_DEFAULT_MAX + 1] = {
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};

static const uint8_t ll_bits[LL_MAX + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  1,
    1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const uint8_t ml_bits[ML_MAX + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
/* -256 log2(x / 256), rounded down, for x below 256: the cost in 1/256
   bits of a symbol of probability x / 256. */
static const uint16_t kinv[256] = {
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162,
    1130, 1100, 1073, 1047, 1024, 1001, 980, 960, 941, 923, 906, 889,
    874, 859, 844, 830, 817, 804, 791, 779, 768, 756, 745, 734,
    724, 714, 704, 694, 685, 676, 667, 658, 650, 642, 633, 626,
    618, 610, 603, 595, 588, 581, 574, 567, 561, 554, 548, 542,
    535, 529, 523, 517, 512, 506, 500, 495, 489, 484, 478, 473,
    468, 463, 458, 453, 448, 443, 438, 434, 429, 424, 420, 415,
    411, 407, 402, 398, 394, 390, 386, 382, 377, 373, 370, 366,
    362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322,
    318, 315, 311, 308, 305, 302, 298, 295, 292, 289, 286, 282,
    279, 276, 273, 270, 267, 264, 261, 258, 256, 253, 250, 247,
    244, 241, 239, 236, 233, 230, 228, 225, 222, 220, 217, 215,
    212, 209, 207, 204, 202, 199, 197, 194, 192, 190, 187, 185,
    182, 180, 178, 175, 173, 171, 168, 166, 164, 162, 159, 157,
    155, 153, 151, 149, 146, 144, 142, 140, 138, 136, 134, 132,
    130, 128, 126, 123, 121, 119, 117, 115, 114, 112, 110, 108,
    106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85,
    83, 82, 80, 78, 76, 74, 73, 71, 69, 67, 66, 64,
    62, 61, 59, 57, 55, 54, 52, 50, 49, 47, 46, 44,
    42, 41, 39, 37, 36, 34, 33, 31, 30, 28, 26, 25,
    23, 22, 20, 19, 17, 16, 14, 13, 11, 10, 8, 7,
    5, 4, 2, 1,
};

/* Bit writer */

/* Bits go in least significant first and are stored forward; the reader
   takes them back from the last byte (section 4.1). [acc] holds the [n]
   bits not yet stored, above them zeros. A flush stores all eight bytes
   of [acc] at [p] and moves [p] past the whole ones, so a writer stores
   up to 8 bytes past its stream. A checked writer keeps [p] at most at
   [lim], [dst + cap], so it stores nothing at or past [lim + 8], and
   remembers in [over] that the stream outgrew [cap]. */
struct bw {
  uint64_t acc;
  unsigned n;
  bool over;
  uint8_t *p;
  uint8_t *dst;
  uint8_t *lim;
};

ZE_INLINE void bw_init(struct bw *b, uint8_t *dst, size_t cap) {
  b->acc = 0;
  b->n = 0;
  b->over = false;
  b->p = dst;
  b->dst = dst;
  b->lim = dst + cap;
}

/* [v] holds at most [n] bits. */
ZE_INLINE void bw_add(struct bw *b, uint64_t v, unsigned n) {
  b->acc |= v << b->n;
  b->n += n;
}

ZE_INLINE void bw_addm(struct bw *b, uint64_t v, unsigned n) {
  bw_add(b, v & ((1ull << n) - 1), n);
}

/* [bw_flush b check] stores the whole bytes pending. Up to 64 bits may be
   pending, so the shift past them is split in two: neither half reaches
   64. */
ZE_INLINE void bw_flush(struct bw *b, bool check) {
  unsigned k = b->n >> 3;
  ze_wr64(b->p, b->acc);
  b->p += k;
  b->acc = (b->acc >> (4 * k)) >> (4 * k);
  b->n &= 7;
  if (check && b->p > b->lim) {
    b->p = b->lim;
    b->over = true;
  }
}

/* [bw_close b] ends the stream with its marker bit: its length, or 0 if
   it did not fit. */
ZE_INLINE size_t bw_close(struct bw *b) {
  bw_add(b, 1, 1);
  size_t len = (size_t)(b->p - b->dst) + ((b->n + 7) >> 3);
  if (b->over || len > (size_t)(b->lim - b->dst)) return 0;
  ze_wr64(b->p, b->acc);
  return len;
}

/* Histograms */

/* [hist count s n] counts the bytes of [s] and returns the largest count;
   [*max] is the largest byte present (0 for an empty [s]). Four tables
   break the dependency between neighbouring bytes. */
static unsigned hist(unsigned count[256], const uint8_t *s, size_t n,
                     unsigned *max) {
  uint32_t c[4][256];
  memset(c, 0, sizeof c);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint64_t a = ze_rd64(s + i), b = ze_rd64(s + i + 8);
    for (int k = 0; k < 8; k += 2) {
      c[0][(a >> (8 * k)) & 0xff]++;
      c[1][(a >> (8 * k + 8)) & 0xff]++;
      c[2][(b >> (8 * k)) & 0xff]++;
      c[3][(b >> (8 * k + 8)) & 0xff]++;
    }
  }
  for (; i < n; i++) c[0][s[i]]++;
  unsigned largest = 0;
  *max = 0;
  for (int v = 0; v < 256; v++) {
    count[v] = c[0][v] + c[1][v] + c[2][v] + c[3][v];
    if (count[v] > largest) largest = count[v];
    if (count[v]) *max = (unsigned)v;
  }
  return largest;
}

/* FSE tables (section 4.1) */

/* [fse_log max_log n max_sym minus] is the accuracy log for [n] symbols
   up to [max_sym]: no more than the data's size supports, with [minus]
   bits taken off, nor less than the alphabet needs, within [5, 12]. */
static unsigned fse_log(unsigned max_log, size_t n, unsigned max_sym,
                        unsigned minus) {
  int fill = n > 1 ? (int)ze_highbit((uint32_t)(n - 1)) - (int)minus : 0;
  if (fill > (int)max_log) fill = (int)max_log;
  int need = (n ? (int)ze_highbit((uint32_t)n) : 0) + 1;
  int alpha = (max_sym ? (int)ze_highbit(max_sym) : 0) + 2;
  if (need > alpha) need = alpha;
  int a = fill > need ? fill : need;
  return a < 5 ? 5 : a > 12 ? 12 : (unsigned)a;
}

/* [lg16 x] is log2 x in 16.16 fixed point, for x >= 1, computed in
   integers so that it is the same on every machine: the integer part is
   the top bit, and each fractional bit comes from squaring the mantissa
   once. */
static uint32_t lg16(uint32_t x) {
  unsigned e = ze_highbit(x);
  uint64_t y = (uint64_t)x << (31 - e);
  uint32_t f = 0;
  for (int i = 15; i >= 0; i--) {
    y = (y * y) >> 31;
    if (y >> 32) {
      y >>= 1;
      f |= 1u << i;
    }
  }
  return (e << 16) | f;
}

/* [gain c k] and [loss c k] are the bits, in 16.16, that one more cell
   saves a symbol counted [c] holding [k] cells, and that one fewer costs:
   [c log2((k + 1) / k)] and [c log2(k / (k - 1))]. A symbol that cannot
   lose a cell costs the most. */
static uint64_t gain(unsigned c, unsigned k) {
  return (uint64_t)c * (lg16(k + 1) - lg16(k));
}

static uint64_t loss(unsigned c, unsigned k) {
  if (k < 2) return UINT64_MAX;
  return (uint64_t)c * (lg16(k) - lg16(k - 1));
}

/* [rebalance norm count max_sym diff] moves [norm] by [diff] cells, one
   at a time, each where it changes the coded size least: given to the
   symbol it saves the most bits, or taken from the one it costs the
   fewest, the lowest symbol on a tie. "Less than 1" symbols keep their
   cell; no symbol goes below one. It is false when no cell can be
   taken. */
static bool rebalance(int16_t *norm, const unsigned *count, unsigned max_sym,
                      int diff) {
  uint64_t up[256], down[256];
  for (unsigned s = 0; s <= max_sym; s++) {
    if (norm[s] <= 0) continue;
    up[s] = gain(count[s], (unsigned)norm[s]);
    down[s] = loss(count[s], (unsigned)norm[s]);
  }
  for (; diff != 0; diff += diff > 0 ? -1 : 1) {
    int best = -1;
    for (unsigned s = 0; s <= max_sym; s++) {
      if (norm[s] <= 0) continue;
      if (diff > 0 ? best < 0 || up[s] > up[best]
                   : down[s] != UINT64_MAX &&
                         (best < 0 || down[s] < down[best]))
        best = (int)s;
    }
    if (best < 0) return false;
    norm[best] = (int16_t)(norm[best] + (diff > 0 ? 1 : -1));
    up[best] = gain(count[best], (unsigned)norm[best]);
    down[best] = loss(count[best], (unsigned)norm[best]);
  }
  return true;
}

/* Rounding thresholds, in 2^-20: a share of k cells and a fraction rounds
   up when the fraction passes tau[k], for k below 8. */
static const uint32_t tau[8] = {0,      473195, 504333, 520860,
                                550000, 700000, 750000, 830000};

/* [fse_normalize norm log count total max_sym low] scales [count] to sum
   to 2^log. A symbol too rare for one cell gets [low]: -1, a cell at the
   table's end, or 1. Below 8 cells a value rounds up past a threshold
   above one half, which favours the rare symbols. What rounding leaves
   over goes to the largest share, unless it would take half of it; then
   [rebalance] spreads it. */
static bool fse_normalize(int16_t *norm, unsigned log, const unsigned *count,
                          size_t total, unsigned max_sym, int16_t low) {
  uint32_t L = 1u << log;
  size_t rare = total >> log;
  unsigned present = 0, big = 0;
  int sum = 0, most = 0;
  for (unsigned s = 0; s <= max_sym; s++) {
    norm[s] = 0;
    if (!count[s]) continue;
    present++;
    if (count[s] <= rare) {
      norm[s] = low;
      sum++;
      continue;
    }
    uint64_t scaled = (uint64_t)count[s] << log;
    uint64_t k = scaled / total, r = scaled - k * total;
    if (k < 8 && (r << 20) > (uint64_t)tau[k] * total) k++;
    norm[s] = (int16_t)k;
    sum += (int)k;
    if ((int)k > most) {
      most = (int)k;
      big = s;
    }
  }
  if (present > L) return false;
  int diff = (int)L - sum;
  if (most > 0 && -diff < most / 2) {
    norm[big] = (int16_t)(norm[big] + diff);
    return true;
  }
  return rebalance(norm, count, max_sym, diff);
}

/* [cells v] is the cells a normalised count takes: one for "less than 1",
   none for an absent symbol. */
static unsigned cells(int16_t v) { return v < 0 ? 1u : (unsigned)v; }

/* [fse_build ct norm max_sym log] builds the encoder's table: each
   symbol's cells, spread as the decoder spreads them (section 4.1.1), and
   per symbol the transform from a state to the bits it emits and its next
   state.

   A symbol of c cells u_0 < ... < u_(c-1) owns the states L + u_j, stored
   in that order from [first], the cells of the symbols before it. From
   state x it emits k bits, the k with x >> k in [c, 2c), and moves to
   state[first + (x >> k) - c]. With M = log - hb(c - 1) (log for c = 1),
   k is M from c << M on and M - 1 below, which is (x + nbits) >> 16 for
   nbits = (M << 16) - (c << M). */
static void fse_build(struct ze_fse *ct, const int16_t *norm, unsigned max_sym,
                      unsigned log) {
  uint32_t L = 1u << log, mask = L - 1;
  uint32_t step = (L >> 1) + (L >> 3) + 3;
  uint32_t high = L - 1;
  uint8_t sym[512];
  uint16_t at[53];

  for (unsigned s = 0; s <= max_sym; s++)
    if (norm[s] < 0) sym[high--] = (uint8_t)s;
  uint32_t u = 0;
  for (unsigned s = 0; s <= max_sym; s++)
    for (int i = 0; i < norm[s]; i++) {
      sym[u] = (uint8_t)s;
      do u = (u + step) & mask;
      while (u > high);
    }

  memset(ct->sym, 0, sizeof ct->sym);
  unsigned first = 0;
  for (unsigned s = 0; s <= max_sym; s++) {
    unsigned c = cells(norm[s]);
    at[s] = (uint16_t)first;
    if (c) {
      unsigned m = c == 1 ? log : log - ze_highbit(c - 1);
      ct->sym[s].nbits = (m << 16) - (c << m);
      ct->sym[s].find = (int32_t)first - (int32_t)c;
    }
    first += c;
  }
  for (u = 0; u < L; u++) ct->state[at[sym[u]]++] = (uint16_t)(L + u);

  memset(ct->norm, 0, sizeof ct->norm);
  memcpy(ct->norm, norm, (max_sym + 1) * sizeof *norm);
  ct->log = (uint8_t)log;
  ct->max_sym = (uint8_t)max_sym;
}

/* A table for one symbol whose states emit no bits (RLE_Mode). */
static void fse_build_rle(struct ze_fse *ct, unsigned s) {
  memset(ct->sym, 0, sizeof ct->sym);
  memset(ct->norm, 0, sizeof ct->norm);
  ct->state[0] = 0;
  ct->log = 0;
  ct->max_sym = (uint8_t)s;
}

/* [fse_describe dst cap norm max_sym log] writes the FSE_Table_Description
   (section 4.1.1): its length, or 0 if it passes [cap]. A value v + 1 is
   written in the fewest bits that hold every value up to [left], the
   points left plus one: below [low] in one bit fewer, from [low] on in
   full width, those past the half-width moved up by [low] so that the
   short and long forms never share their low bits. */
static size_t fse_describe(uint8_t *dst, size_t cap, const int16_t *norm,
                           unsigned max_sym, unsigned log) {
  uint32_t L = 1u << log;
  uint32_t sum = 0;
  for (unsigned s = 0; s <= max_sym; s++) sum += cells(norm[s]);
  if (sum != L) return 0;

  uint64_t acc = log - 5;
  unsigned n = 4;
  size_t o = 0;
  int left = (int)L + 1, half = (int)L;
  unsigned bits = log + 1;
  for (unsigned s = 0; s <= max_sym && left > 1; s++) {
    int v = norm[s] + 1;
    int low = 2 * half - 1 - left;
    if (v < low) {
      acc |= (uint64_t)v << n;
      n += bits - 1;
    } else {
      acc |= (uint64_t)(v >= half ? v + low : v) << n;
      n += bits;
    }
    left -= norm[s] < 0 ? 1 : norm[s];
    if (norm[s] == 0) {
      unsigned zeros = 0;
      while (s + 1 + zeros <= max_sym && norm[s + 1 + zeros] == 0) zeros++;
      s += zeros;
      for (; zeros >= 3; zeros -= 3) {
        acc |= (uint64_t)3 << n;
        n += 2;
        for (; n >= 8; n -= 8, acc >>= 8) {
          if (o >= cap) return 0;
          dst[o++] = (uint8_t)acc;
        }
      }
      acc |= (uint64_t)zeros << n;
      n += 2;
    }
    while (left < half) {
      half >>= 1;
      bits--;
    }
    for (; n >= 8; n -= 8, acc >>= 8) {
      if (o >= cap) return 0;
      dst[o++] = (uint8_t)acc;
    }
  }
  if (n > 0) {
    if (o >= cap) return 0;
    dst[o++] = (uint8_t)acc;
  }
  return o;
}

/* FSE encoding: a state is a value in [2^log, 2^(log+1)). */
struct fse_state {
  uint32_t v;
  const struct ze_fse *ct;
};

/* [fse_start st ct s] puts the state on the lowest cell of [s], so that
   encoding [s] first emits no bits. The lowest cell is the first of the
   symbol's run, at [find + c]; [c] and [M] come back from [nbits], since
   [c << M] lies in (0, 2^16). An RLE table gives state 0. */
ZE_INLINE void fse_start(struct fse_state *st, const struct ze_fse *ct,
                         unsigned s) {
  uint32_t nbits = ct->sym[s].nbits;
  unsigned m = (nbits + 0xFFFF) >> 16;
  int32_t c = (int32_t)(((m << 16) - nbits) >> m);
  st->v = ct->state[c + ct->sym[s].find];
  st->ct = ct;
}

/* [fse_step st s out] encodes [s]: the bits the state emits, [*out] of
   them, and the state moves on. */
ZE_INLINE uint32_t fse_step(struct fse_state *st, unsigned s, unsigned *out) {
  const struct ze_fse_sym *y = &st->ct->sym[s];
  uint32_t v = st->v;
  unsigned k = (v + y->nbits) >> 16;
  *out = k;
  st->v = st->ct->state[(int32_t)(v >> k) + y->find];
  return v & ((1u << k) - 1);
}

ZE_INLINE void fse_put(struct bw *b, struct fse_state *st, unsigned s) {
  unsigned k;
  uint32_t bits = fse_step(st, s, &k);
  bw_add(b, bits, k);
}

ZE_INLINE void fse_end(struct bw *b, struct fse_state *st) {
  bw_addm(b, st->v, st->ct->log);
}

/* Huffman tables (section 4.2) */

/* [huf_lengths bits count max_sym limit] gives each present symbol a code
   length: Huffman's, built by two queues over the symbols sorted by
   count, then, if a length passes [limit], the lengths clamped and the
   code made complete again. It returns the longest length. Two symbols at
   least are present. */
static unsigned huf_lengths(uint8_t bits[256], const unsigned count[256],
                            unsigned max_sym, unsigned limit) {
  uint32_t key[256];
  unsigned m = 0;
  for (unsigned s = 0; s <= max_sym; s++)
    if (count[s]) key[m++] = (count[s] << 8) | s;

  /* Sorted ascending by count, then by symbol: two passes of a radix
     sort on the count's 18 bits, stable. */
  {
    uint32_t tmp[256];
    unsigned per[512];
    for (unsigned pass = 0; pass < 2; pass++) {
      unsigned shift = 8 + 9 * pass;
      memset(per, 0, sizeof per);
      for (unsigned i = 0; i < m; i++) per[(key[i] >> shift) & 511]++;
      for (unsigned v = 0, at = 0; v < 512; v++) {
        unsigned c = per[v];
        per[v] = at;
        at += c;
      }
      for (unsigned i = 0; i < m; i++) tmp[per[(key[i] >> shift) & 511]++] = key[i];
      memcpy(key, tmp, m * sizeof *key);
    }
  }

  /* Two queues: the leaves in order, the internal nodes as made, which
     come out in order too. Node i < m is leaf i; node m + j is the j-th
     internal node. */
  uint32_t weight[512];
  uint16_t parent[512];
  unsigned leaf = 0, inner = m, made = m;
  for (unsigned i = 0; i < m; i++) weight[i] = key[i] >> 8;
  while (made < 2 * m - 1) {
    unsigned pick[2];
    for (int k = 0; k < 2; k++) {
      if (leaf < m && (inner >= made || weight[leaf] <= weight[inner]))
        pick[k] = leaf++;
      else
        pick[k] = inner++;
    }
    weight[made] = weight[pick[0]] + weight[pick[1]];
    parent[pick[0]] = parent[pick[1]] = (uint16_t)made;
    made++;
  }

  /* Depths, from the root down, and the number of leaves per length,
     those past [limit] clamped to it. */
  uint8_t depth[512];
  unsigned per_len[32] = {0};
  depth[made - 1] = 0;
  for (int i = (int)made - 2; i >= 0; i--)
    depth[i] = (uint8_t)(depth[parent[i]] + 1);
  for (unsigned i = 0; i < m; i++)
    per_len[depth[i] > limit ? limit : depth[i]]++;

  /* Clamping overfills the code: while it does, a leaf at the limit
     leaves, and the deepest leaf above it splits into two, one level
     down, each step taking one unit of 2^-limit. */
  uint32_t total = 0;
  for (unsigned d = 1; d <= limit; d++) total += per_len[d] << (limit - d);
  while (total > (1u << limit)) {
    per_len[limit]--;
    for (unsigned d = limit - 1; d > 0; d--)
      if (per_len[d]) {
        per_len[d]--;
        per_len[d + 1] += 2;
        break;
      }
    total--;
  }

  /* The most frequent symbols take the shortest lengths. */
  memset(bits, 0, 256);
  unsigned longest = 0;
  int i = (int)m - 1;
  for (unsigned d = 1; d <= limit; d++)
    for (unsigned k = 0; k < per_len[d]; k++) {
      bits[key[i--] & 0xff] = (uint8_t)d;
      longest = d;
    }
  return longest;
}

/* [huf_codes h] assigns the codes: by length, longest first, then by
   symbol, in ascending order (section 4.2.1.3). */
static void huf_codes(struct ze_huf *h) {
  unsigned per_len[13] = {0};
  uint16_t next[13];
  for (int s = 0; s <= h->max_sym; s++) per_len[h->bits[s]]++;
  unsigned v = 0;
  for (int d = h->log; d > 0; d--) {
    next[d] = (uint16_t)v;
    v = (v + per_len[d]) >> 1;
  }
  for (int s = 0; s <= h->max_sym; s++)
    h->code[s] = h->bits[s] ? next[h->bits[s]]++ : 0;
}

/* [huf_cost h count max_sym] is the bytes [h] codes the counted literals
   in, or INF if it lacks one of them. */
static size_t huf_cost(const struct ze_huf *h, const unsigned *count,
                       unsigned max_sym) {
  if ((int)max_sym > h->max_sym) return INF;
  size_t bits = 0;
  for (unsigned s = 0; s <= max_sym; s++) {
    if (count[s] && !h->bits[s]) return INF;
    bits += (size_t)count[s] * h->bits[s];
  }
  return bits >> 3;
}

/* [huf_weights dst cap w m count top] FSE-codes the [m] weights [w],
   [count] their histogram up to [top], the highest (section 4.2.1.2): the
   table's description, then one stream that two states share, state 1
   owning the weights at even indices and state 2 those at odd ones. The
   stream is written from the last weight back: each state's first weight
   met places it, every other is a step, and state 2 is written before
   state 1 so that the decoder reads state 1 first. The decoder then
   finds the stream's start exactly when it updates the state of the
   second-to-last weight, from its lowest cell, which reads at least one
   bit. Its length, or 0 if it passes [cap]; it stores up to 8 bytes past
   [cap]. */
static size_t huf_weights(uint8_t *dst, size_t cap, const uint8_t *w,
                          unsigned m, const unsigned *count, unsigned top) {
  unsigned log = fse_log(WEIGHT_LOG, m, top, 2);
  int16_t norm[16];
  if (!fse_normalize(norm, log, count, m, top, 1)) return 0;
  size_t d = fse_describe(dst, cap, norm, top, log);
  if (!d) return 0;

  struct ze_fse ct;
  fse_build(&ct, norm, top, log);
  struct bw b;
  bw_init(&b, dst + d, cap - d);
  struct fse_state st[2];
  fse_start(&st[(m - 1) & 1], &ct, w[m - 1]);
  fse_start(&st[(m - 2) & 1], &ct, w[m - 2]);
  for (unsigned i = m - 2; i-- > 0;) {
    fse_put(&b, &st[i & 1], w[i]);
    bw_flush(&b, true);
  }
  fse_end(&b, &st[1]);
  fse_end(&b, &st[0]);
  size_t s = bw_close(&b);
  return s ? d + s : 0;
}

/* [huf_describe dst cap h] writes the Huffman_Tree_Description: the
   weights FSE-coded when that is shorter than half their count, else four
   bits each. Its length, or 0 if neither fits. The last symbol's weight
   is implied, so symbols 0 to [max_sym - 1] are listed. */
static size_t huf_describe(uint8_t *dst, size_t cap, const struct ze_huf *h) {
  unsigned m = (unsigned)h->max_sym;
  uint8_t w[256];
  unsigned count[16] = {0}, top = 0;
  for (unsigned s = 0; s < m; s++) {
    w[s] = (uint8_t)(h->bits[s] ? h->log + 1 - h->bits[s] : 0);
    count[w[s]]++;
    if (w[s] > top) top = w[s];
  }

  bool twice = false;
  for (unsigned v = 0; v <= top; v++) twice |= count[v] >= 2;
  if (m > 1 && count[w[0]] != m && twice && cap >= 2) {
    size_t s = huf_weights(dst + 1, cap - 1, w, m, count, top);
    if (s > 1 && s < m / 2) {
      dst[0] = (uint8_t)s;
      return s + 1;
    }
  }

  size_t direct = (m + 1) / 2 + 1;
  if (m > 128 || direct > cap) return 0;
  dst[0] = (uint8_t)(127 + m);
  for (unsigned i = 0; i < m; i += 2) {
    unsigned lo = i + 1 < m ? w[i + 1] : 0;
    dst[1 + i / 2] = (uint8_t)((w[i] << 4) | lo);
  }
  return direct;
}

/* [huf_stream dst cap s n tab] codes [s] backward into one stream, each
   entry of [tab] a code above its length's 8 bits: its length or 0. */
ZE_INLINE size_t huf_loop(uint8_t *dst, size_t cap, const uint8_t *s,
                          size_t n, const uint32_t *tab, bool check) {
  struct bw b;
  bw_init(&b, dst, cap);
  size_t i = n;
  while (i & 3) {
    i--;
    bw_add(&b, tab[s[i]] >> 8, tab[s[i]] & 0xff);
  }
  bw_flush(&b, check);
  for (; i > 0; i -= 4) {
    uint32_t e1 = tab[s[i - 1]], e2 = tab[s[i - 2]], e3 = tab[s[i - 3]],
             e4 = tab[s[i - 4]];
    bw_add(&b, e1 >> 8, e1 & 0xff);
    bw_add(&b, e2 >> 8, e2 & 0xff);
    bw_add(&b, e3 >> 8, e3 & 0xff);
    bw_add(&b, e4 >> 8, e4 & 0xff);
    bw_flush(&b, check);
  }
  return bw_close(&b);
}

/* [huf_stream dst cap s n tab] codes [s] backward into one stream, each
   entry of [tab] a code above its length's 8 bits: its length or 0. A
   literal takes at most 11 bits: where that fits, the loop need not watch
   the end. */
ZE_INLINE size_t huf_stream(uint8_t *dst, size_t cap, const uint8_t *s,
                            size_t n, const uint32_t *tab) {
  if (cap >= n * 11 / 8 + 16) return huf_loop(dst, cap, s, n, tab, false);
  return huf_loop(dst, cap, s, n, tab, true);
}

ZE_INLINE size_t huf_streams(uint8_t *dst, size_t cap, const uint8_t *s,
                             size_t n, const uint32_t *tab, bool four) {
  if (!four) return huf_stream(dst, cap, s, n, tab);

  if (cap < 6 + 4) return 0;
  size_t seg = (n + 3) / 4, at = 6;
  for (int k = 0; k < 4; k++) {
    size_t len = k < 3 ? seg : n - 3 * seg;
    size_t c = huf_stream(dst + at, cap - at, s + k * seg, len, tab);
    if (!c) return 0;
    if (k < 3) {
      if (c > 0xFFFF) return 0;
      dst[2 * k] = (uint8_t)c;
      dst[2 * k + 1] = (uint8_t)(c >> 8);
    }
    at += c;
  }
  return at;
}

static size_t huf_generic(uint8_t *dst, size_t cap, const uint8_t *s,
                          size_t n, const uint32_t *tab, bool four) {
  return huf_streams(dst, cap, s, n, tab, four);
}

#if ZE_BMI2
ZE_TARGET_BMI2 static size_t huf_bmi2(uint8_t *dst, size_t cap,
                                      const uint8_t *s, size_t n,
                                      const uint32_t *tab, bool four) {
  return huf_streams(dst, cap, s, n, tab, four);
}
#endif

static size_t huf_code(uint8_t *dst, size_t cap, const uint8_t *s, size_t n,
                       const struct ze_huf *h, bool four, bool bmi2) {
  uint32_t tab[256];
  for (int v = 0; v <= h->max_sym; v++)
    tab[v] = ((uint32_t)h->code[v] << 8) | h->bits[v];
#if ZE_BMI2
  if (bmi2) return huf_bmi2(dst, cap, s, n, tab, four);
#endif
  (void)bmi2;
  return huf_generic(dst, cap, s, n, tab, four);
}

/* Literals section (section 3.1.1.3.1) */

static size_t lits_header(uint8_t *dst, unsigned type, size_t n) {
  if (n < 32) {
    dst[0] = (uint8_t)(type | (n << 3));
    return 1;
  }
  if (n < 4096) {
    dst[0] = (uint8_t)(type | (1 << 2) | (n << 4));
    dst[1] = (uint8_t)(n >> 4);
    return 2;
  }
  ze_wr24(dst, (uint32_t)(type | (3 << 2) | (n << 4)));
  return 3;
}

static size_t lits_raw(uint8_t *dst, size_t cap, const uint8_t *s, size_t n) {
  if (n + 3 > cap) return 0;
  size_t h = lits_header(dst, LIT_RAW, n);
  memcpy(dst + h, s, n);
  return h + n;
}

static size_t lits_rle(uint8_t *dst, size_t cap, uint8_t v, size_t n) {
  if (cap < 4) return 0;
  size_t h = lits_header(dst, LIT_RLE, n);
  dst[h] = v;
  return h + 1;
}

/* A block whose literals outnumber its sequences this many times over is
   suspected incompressible ([suspect] below). */
#define SUSPECT_RATIO 20

/* The least a compressed literals section saves over raw, for the
   decoder's work in building a table and decoding the streams: 1/128 of
   the literals and a few bytes. */
static size_t lit_gain(size_t n) { return (n >> 7) + 4; }

/* Below LIT_MIN literals the section is raw or RLE. From HUF_FOUR on, the
   literals are coded as four streams, which the decoder runs side by
   side; below, as one. */
#define LIT_MIN 8
#define HUF_FOUR 1024

/* A suspect section of FLAT_MIN literals or more is first sampled. */
#define FLAT_MIN 4096

/* [lg256 x] is 256 log2 x for x >= 1, from [kinv] on the top 8 bits. */
static unsigned lg256(uint32_t x) {
  unsigned e = ze_highbit(x);
  uint32_t m = e >= 7 ? x >> (e - 7) : x << (7 - e);
  return 256 * e + 256 - kinv[m];
}

/* [flat s n strat] is true when a sample of [s], one byte in 17 at the
   fast strategies and one in 5 at the others, would save by its entropy
   less than [lit_gain] asks of the whole, 1/16 bit a literal: the
   literals are taken as incompressible without a table. The steps are
   odd so that records of a power-of-two width, such as 16-bit samples
   whose high bytes vary little, are sampled at every offset. */
static bool flat(const uint8_t *s, size_t n, int strat) {
  unsigned count[256] = {0};
  size_t step = strat <= ZE_DFAST ? 17 : 5, m = 0;
  for (size_t i = 0; i < n; i += step, m++) count[s[i]]++;
  uint64_t bits = 0, lm = lg256((uint32_t)m);
  for (unsigned v = 0; v < 256; v++)
    if (count[v]) bits += (uint64_t)count[v] * (lm - lg256(count[v]));
  uint64_t whole = 2048 * (uint64_t)m;
  return bits >= whole || whole - bits < 16 * (uint64_t)m;
}

static bool all_equal(const uint8_t *s, size_t n) {
  for (size_t i = 1; i < n; i++)
    if (s[i] != s[0]) return false;
  return true;
}

/* A compressed section's header (section 3.1.1.3.1.1): its size follows
   from [n] alone, since a section is kept only below [n] bytes. One
   stream under HUF_FOUR literals, with 10-bit sizes; four above, with
   14-bit sizes, or 18-bit from 16 KiB. */
static size_t huf_hsize(size_t n) {
  return n < HUF_FOUR ? 3 : n < 16384 ? 4 : 5;
}

static void huf_header(uint8_t *dst, unsigned type, size_t n, size_t c) {
  if (n < HUF_FOUR) {
    ze_wr24(dst, (uint32_t)(type | (n << 4) | (c << 14)));
    return;
  }
  if (n < 16384) {
    ze_wr32(dst, (uint32_t)(type | (2u << 2) | (n << 4) | (c << 18)));
    return;
  }
  uint64_t v = type | (3u << 2) | ((uint64_t)n << 4) | ((uint64_t)c << 22);
  ze_wr32(dst, (uint32_t)v);
  dst[4] = (uint8_t)(v >> 32);
}

/* [lits_huf dst cap s n count max prev next bmi2] writes the literals
   Huffman-coded, with a new table in [next->huf] or [prev]'s, whichever
   the estimate prices lower, if that saves [lit_gain] over raw: its
   length, or 0 if neither does, or fits. */
static size_t lits_huf(uint8_t *dst, size_t cap, const uint8_t *s, size_t n,
                       const unsigned *count, unsigned max,
                       const struct ze_entropy *prev, struct ze_entropy *next,
                       bool bmi2) {
  size_t raw = n + (n < 32 ? 1 : n < 4096 ? 2 : 3);
  size_t room = raw - lit_gain(n);
  size_t hs = huf_hsize(n);
  bool four = n >= HUF_FOUR;
  /* The header, and with four streams the jump table; each stream's last
     byte. */
  size_t extra = hs + (four ? 6 + 4 : 1);
  if (cap <= hs) return 0;

  size_t reuse = INF;
  if (prev->huf_mode == ZE_CHECK) {
    size_t c = huf_cost(&prev->huf, count, max);
    if (c != INF) reuse = c + extra;
  }

  struct ze_huf *h = &next->huf;
  h->max_sym = (int)max;
  h->log = (int)huf_lengths(h->bits, count, max, HUF_LOG);
  size_t d = huf_describe(dst + hs, cap - hs, h);
  size_t fresh = d ? d + huf_cost(h, count, max) + extra : INF;

  if (fresh < reuse && fresh <= room) {
    huf_codes(h);
    size_t c = huf_code(dst + hs + d, cap - hs - d, s, n, h, four, bmi2);
    if (!c || hs + d + c > room) return 0;
    huf_header(dst, LIT_HUF, n, d + c);
    next->huf_mode = ZE_CHECK;
    return hs + d + c;
  }
  if (reuse > room) return 0;
  size_t c = huf_code(dst + hs, cap - hs, s, n, &prev->huf, four, bmi2);
  if (!c || hs + c > room) return 0;
  huf_header(dst, LIT_REPEAT, n, c);
  next->huf = prev->huf;
  next->huf_mode = prev->huf_mode;
  return hs + c;
}

/* contract: [lits dst cap s n prev next strat suspect bmi2] writes the
   literals section of the [n] literals at [s] (RFC 8878, 3.1.1.3.1) into
   [dst] of [cap] bytes and is its length, or 0 when it does not fit. It
   chooses among raw, RLE, compressed with a new Huffman table (built in
   [next->huf]) and compressed with [prev]'s table where [prev] holds a
   usable one (treeless), by the rule of the design (cleanroom3,
   design.md). Every outcome that does not build a new table leaves
   [prev]'s table in [next]. [suspect] is true when the block's sequences
   suggest the literals are incompressible. */
static size_t lits(uint8_t *dst, size_t cap, const uint8_t *s, size_t n,
                   const struct ze_entropy *prev, struct ze_entropy *next,
                   int strat, bool suspect, bool bmi2) {
  bool same;
  if (n < LIT_MIN || (suspect && n >= FLAT_MIN && flat(s, n, strat))) {
    same = n > 0 && all_equal(s, n);
  } else {
    unsigned count[256], max;
    same = hist(count, s, n, &max) == n;
    if (!same) {
      size_t c = lits_huf(dst, cap, s, n, count, max, prev, next, bmi2);
      if (c) return c;
    }
  }
  next->huf = prev->huf;
  next->huf_mode = prev->huf_mode;
  if (same) return lits_rle(dst, cap, s[0], n);
  return lits_raw(dst, cap, s, n);
}

/* Sequences section (section 3.1.1.3.2) */

/* [fse_bits norm log top count max] is the bits, in 1/256, that the
   counted codes up to [max] take under a table of [norm], of symbols up
   to [top] and accuracy log [log]: each code log2 of the table's size
   over its cells. It is INF when a code present has no cell. */
static size_t fse_bits(const int16_t *norm, unsigned log, unsigned top,
                       const unsigned *count, unsigned max) {
  if (max > top) return INF;
  size_t bits = 0;
  for (unsigned s = 0; s <= max; s++) {
    if (!count[s]) continue;
    unsigned c = cells(norm[s]);
    if (!c) return INF;
    bits += (size_t)count[s] * (256 * log - lg256(c));
  }
  return bits;
}

/* The bytes a new table's description takes at most: 53 values of up to
   10 bits, with the zero runs' flags. */
#define DESC_MAX 128

/* One field's table choice. */
struct field {
  const struct ze_seq *seq;
  unsigned shift;
  unsigned max;
  unsigned log;
  const struct ze_fse *dflt;
  unsigned dflt_max;
  const struct ze_fse *prev;
  uint8_t prev_mode;
  struct ze_fse *next;
  uint8_t *next_mode;
};

/* contract: [field_table dst cap f n strat counts len] picks the mode of
   one sequence field (predefined, RLE, compressed with a new table,
   repeat; RFC 8878, 3.1.1.3.2.1) from the counts of its codes, builds the
   chosen table into [*f->next], writes the mode's description at [dst]
   and sets [*len] to the bytes written: the mode, or -1 if the
   description does not fit [cap]. The rule is the design's (cleanroom3,
   design.md). */
static int field_table(uint8_t *dst, size_t cap, const struct field *f,
                       size_t n, int strat, const unsigned counts[2][64],
                       size_t *len) {
  unsigned count[64], max = 0, kinds = 0;
  for (unsigned s = 0; s <= f->max; s++) {
    count[s] = counts[0][s] + counts[1][s];
    if (!count[s]) continue;
    max = s;
    kinds++;
  }

  /* Each mode's cost in 1/256 bits; on a tie the earlier mode, which the
     decoder builds faster. */
  int mode = -1;
  size_t best = INF;
  if (kinds == 1) {
    mode = SEQ_RLE;
    best = 8 * 256;
  }
  size_t c = fse_bits(f->dflt->norm, f->dflt->log, f->dflt_max, count, max);
  if (c < best) {
    mode = SEQ_BASIC;
    best = c;
  }
  if (f->prev_mode == ZE_CHECK) {
    c = fse_bits(f->prev->norm, f->prev->log, f->prev->max_sym, count, max);
    if (c < best) {
      mode = SEQ_REPEAT;
      best = c;
    }
  }

  /* A new table at the accuracy log the sequences' number suggests, and
     at the lazy2 strategy at one bit more as well, which a few fields
     code smaller. */
  int16_t norm[64];
  uint8_t desc[DESC_MAX];
  size_t d = 0;
  unsigned log = 0;
  unsigned lo = strat >= ZE_LAZY2 ? 1 : 2, hi = 2;
  for (unsigned minus = lo; kinds > 1 && minus <= hi; minus++) {
    unsigned l = fse_log(f->log, n, max, minus);
    if (minus > lo && l == fse_log(f->log, n, max, minus - 1)) continue;
    int16_t t[64];
    uint8_t b[DESC_MAX];
    if (!fse_normalize(t, l, count, n, max, -1)) continue;
    size_t k = fse_describe(b, sizeof b, t, max, l);
    if (!k) continue;
    c = 2048 * k + fse_bits(t, l, max, count, max);
    if (c >= best) continue;
    mode = SEQ_FSE;
    best = c;
    memcpy(norm, t, (max + 1) * sizeof *t);
    memcpy(desc, b, k);
    d = k;
    log = l;
  }

  *len = 0;
  switch (mode) {
  case SEQ_RLE:
    if (cap < 1) return -1;
    dst[0] = (uint8_t)max;
    *len = 1;
    fse_build_rle(f->next, max);
    *f->next_mode = ZE_NONE;
    return mode;
  case SEQ_BASIC:
    *f->next = *f->dflt;
    *f->next_mode = ZE_CHECK;
    return mode;
  case SEQ_REPEAT:
    *f->next = *f->prev;
    *f->next_mode = ZE_CHECK;
    return mode;
  case SEQ_FSE:
    if (cap < d) return -1;
    memcpy(dst, desc, d);
    *len = d;
    fse_build(f->next, norm, max, log);
    *f->next_mode = ZE_CHECK;
    return mode;
  default:
    return -1;
  }
}

/* [seq_codes ss counts] counts the codes of each sequence's three fields,
   alternating two tables per field so that runs of one code do not wait
   on their own counts. It returns the bits the sequences' bitstream takes
   at most. */
static size_t seq_codes(const struct ze_seqs *ss,
                        unsigned counts[3][2][64]) {
  memset(counts, 0, 3 * 2 * 64 * sizeof(unsigned));
  const struct ze_seq *seq = ss->seq;
  size_t n = ss->n, extra = 0;
  for (size_t i = 0; i < n; i++) {
    uint32_t code = seq[i].code;
    unsigned a = code & 0xff, b = (code >> 8) & 0xff, c = code >> 16;
    extra += ll_bits[a] + ml_bits[b] + c;
    counts[0][i & 1][a]++;
    counts[2][i & 1][b]++;
    counts[1][i & 1][c]++;
  }
  return extra + n * (LL_LOG + ML_LOG + OF_LOG);
}

/* [seq_loop b q n e check] writes the sequences before the last, from
   [q], the one before it, down, with [e]'s tables from the states [st];
   [check] keeps the writer within its end. */
ZE_INLINE void seq_loop(struct bw *b, const struct ze_seq *q, size_t n,
                        const struct ze_entropy *e, uint32_t st[3],
                        bool check) {
  struct bw w = *b;
  uint32_t vll = st[0], vml = st[1], vof = st[2];
  for (; n > 1; n--, q--) {
    uint32_t c = q->code;
    unsigned llc = c & 0xff, mlc = (c >> 8) & 0xff, ofc = c >> 16;
    unsigned lb = ll_bits[llc], mb = ml_bits[mlc];
    /* The three states' bits and the literals length's are packed apart
       from the writer, so that one sequence's work need not wait on the
       last one's bits. */
    const struct ze_fse_sym *yo = &e->of.sym[ofc], *ym = &e->ml.sym[mlc],
                            *yl = &e->ll.sym[llc];
    unsigned o1 = (vof + yo->nbits) >> 16;
    unsigned o2 = (vml + ym->nbits) >> 16;
    unsigned o3 = (vll + yl->nbits) >> 16;
    uint64_t x = (vof & ((1u << o1) - 1)) |
                 ((uint64_t)(vml & ((1u << o2) - 1)) << o1) |
                 ((uint64_t)(vll & ((1u << o3) - 1)) << (o1 + o2)) |
                 ((uint64_t)(q->ll & ((1u << lb) - 1)) << (o1 + o2 + o3));
    vof = e->of.state[(int32_t)(vof >> o1) + yo->find];
    vml = e->ml.state[(int32_t)(vml >> o2) + ym->find];
    vll = e->ll.state[(int32_t)(vll >> o3) + yl->find];
    bw_add(&w, x, o1 + o2 + o3 + lb);
    bw_flush(&w, check);
    bw_add(&w,
           ((q->ml - 3) & ((1u << mb) - 1)) |
               ((uint64_t)(q->off & ((1ull << ofc) - 1)) << mb),
           mb + ofc);
    bw_flush(&w, check);
  }
  st[0] = vll;
  st[1] = vml;
  st[2] = vof;
  *b = w;
}

/* [seq_stream dst cap ss e bits] writes the sequences' bitstream with
   [e]'s tables: the decoder reads the states' starts, then for each
   sequence the offset, match and literals lengths' extra bits and the
   literals, match and offset states' updates, all backward; the encoder
   writes the reverse, from the last sequence. [bits] bounds its length. */
ZE_INLINE size_t seq_stream(uint8_t *dst, size_t cap,
                            const struct ze_seqs *ss,
                            const struct ze_entropy *e, size_t bits) {
  struct bw b;
  bw_init(&b, dst, cap);
  struct fse_state sll, sof, sml;
  size_t n = ss->n;
  const struct ze_seq *q = &ss->seq[n - 1];
  unsigned llc = q->code & 0xff, mlc = (q->code >> 8) & 0xff,
           ofc = q->code >> 16;

  fse_start(&sml, &e->ml, mlc);
  fse_start(&sof, &e->of, ofc);
  fse_start(&sll, &e->ll, llc);
  bw_addm(&b, q->ll, ll_bits[llc]);
  bw_addm(&b, q->ml - 3, ml_bits[mlc]);
  bw_flush(&b, true);
  bw_addm(&b, q->off, ofc);
  bw_flush(&b, true);

  uint32_t st[3] = {sll.v, sml.v, sof.v};
  if (cap >= bits / 8 + 32)
    seq_loop(&b, q - 1, n, e, st, false);
  else
    seq_loop(&b, q - 1, n, e, st, true);
  sll.v = st[0];
  sml.v = st[1];
  sof.v = st[2];
  fse_end(&b, &sml);
  fse_end(&b, &sof);
  fse_end(&b, &sll);
  return bw_close(&b);
}

static size_t seq_generic(uint8_t *dst, size_t cap, const struct ze_seqs *ss,
                          const struct ze_entropy *e, size_t bits) {
  return seq_stream(dst, cap, ss, e, bits);
}

#if ZE_BMI2
ZE_TARGET_BMI2 static size_t seq_bmi2(uint8_t *dst, size_t cap,
                                      const struct ze_seqs *ss,
                                      const struct ze_entropy *e,
                                      size_t bits) {
  return seq_stream(dst, cap, ss, e, bits);
}
#endif

static size_t sequences(uint8_t *dst, size_t cap, struct ze_seqs *ss,
                        const struct ze_entropy *prev, struct ze_entropy *next,
                        const struct ze_defaults *d, int strat, bool bmi2) {
  uint8_t *p = dst, *end = dst + cap;
  size_t n = ss->n;
  if (end - p < 4) return 0;
  if (n < 128) {
    *p++ = (uint8_t)n;
  } else if (n < LONG_NBSEQ) {
    *p++ = (uint8_t)((n >> 8) + 0x80);
    *p++ = (uint8_t)n;
  } else {
    *p++ = 0xFF;
    *p++ = (uint8_t)(n - LONG_NBSEQ);
    *p++ = (uint8_t)((n - LONG_NBSEQ) >> 8);
  }
  if (n == 0) {
    next->ll = prev->ll;
    next->of = prev->of;
    next->ml = prev->ml;
    next->ll_mode = prev->ll_mode;
    next->of_mode = prev->of_mode;
    next->ml_mode = prev->ml_mode;
    return (size_t)(p - dst);
  }

  unsigned counts[3][2][64];
  size_t bits = seq_codes(ss, counts);
  uint8_t *modes = p++;
  struct field fs[3] = {
      {ss->seq, 0, LL_MAX, LL_LOG, &d->ll, LL_MAX, &prev->ll, prev->ll_mode,
       &next->ll, &next->ll_mode},
      {ss->seq, 16, OF_MAX, OF_LOG, &d->of, OF_DEFAULT_MAX, &prev->of,
       prev->of_mode, &next->of, &next->of_mode},
      {ss->seq, 8, ML_MAX, ML_LOG, &d->ml, ML_MAX, &prev->ml, prev->ml_mode,
       &next->ml, &next->ml_mode},
  };
  int mode[3];
  for (int k = 0; k < 3; k++) {
    size_t len;
    mode[k] = field_table(p, (size_t)(end - p), &fs[k], n, strat, counts[k],
                          &len);
    if (mode[k] < 0) return 0;
    p += len;
  }
  *modes = (uint8_t)((mode[0] << 6) | (mode[1] << 4) | (mode[2] << 2));

#if ZE_BMI2
  size_t s = (bmi2 ? seq_bmi2 : seq_generic)(p, (size_t)(end - p), ss, next,
                                             bits);
#else
  (void)bmi2;
  size_t s = seq_generic(p, (size_t)(end - p), ss, next, bits);
#endif
  if (!s) return 0;
  return (size_t)(p - dst) + s;
}

/* Entry points */

void ze_defaults_build(struct ze_defaults *d) {
  fse_build(&d->ll, ll_default, LL_MAX, 6);
  fse_build(&d->ml, ml_default, ML_MAX, 6);
  fse_build(&d->of, of_default, OF_DEFAULT_MAX, 5);
}

size_t ze_body(uint8_t *dst, size_t cap, struct ze_seqs *ss,
               const struct ze_entropy *prev, struct ze_entropy *next,
               const struct ze_defaults *d, int strat, bool bmi2) {
  bool suspect = ss->n == 0 || ss->nlit / ss->n >= SUSPECT_RATIO;
  size_t l =
      lits(dst, cap, ss->lit, ss->nlit, prev, next, strat, suspect, bmi2);
  if (!l) return 0;
  size_t s = sequences(dst + l, cap - l, ss, prev, next, d, strat, bmi2);
  if (!s) return 0;
  return l + s;
}
