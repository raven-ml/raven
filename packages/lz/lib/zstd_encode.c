/* The Zstandard encoder's frames (RFC 8878 section 3.1.1): the header,
   the blocks and the XXH64 content checksum, every frame carrying it.

   Both drivers run the same block loop, so their bytes agree but for the
   header: the whole-buffer encode reads the input in place and writes
   Frame_Content_Size; a stream copies its input into a window and never
   does. A block is cut only from bytes both drivers hold alike: a stream
   holds back up to 128 KiB of input until another byte or [last] arrives,
   so a block is chosen from at least 128 KiB, or from the input's end.
   When the input's end is known before the first block, the frame is
   that size and the level's parameters shrink to it, in both drivers.

   A block is compressed (zstd_enc_match.c, zstd_enc_entropy.c), then
   written as such if that saves a minimum gain, as one repeated byte if
   it is that, or raw.
   A full block may be cut shorter where the data changes, so that each
   part gets its own entropy tables (cut). */

#include "checksum.h"
#include "zstd_enc_int.h"

#define MAGIC 0xFD2FB528u

/* RFC 8878 section 3.1.1.1.2 and 3.1.1.2: the smallest window, block
   header types. */
#define MIN_WLOG 10
#define BLOCK_RAW 0u
#define BLOCK_RLE 1u
#define BLOCK_COMPRESSED 2u

/* The reference's thresholds: blocks below this are written raw; a
   compressed block this short is checked for a single repeated byte. */
#define MIN_BLOCK 7
#define RLE_MAX 25

/* A cut falls on a multiple of a span, 8 KiB or more, from the block's
   start, so that every block but the last holds 8 KiB at least
   (lz_zstd_bound). [xlg] is tabled below XLG_TAB. */
#define CUT_SPAN 8192
#define CUT_SPANS (ZE_BLOCK / CUT_SPAN)
#define XLG_TAB 4097

/* How each class of strategies cuts: spans of [CUT_SPAN << shift] bytes,
   counts of one byte in [step], read eight neighbouring bytes at a time,
   and a cut kept when the counts say it saves [gain] bytes, plus [noise]
   half-bits a byte value present: the gain that the sampling alone
   shows between two parts of the same statistics. The fast strategies
   sample least and cut only where it pays most, for each block costs a
   few microseconds of tables; the lazy ones sample more and cut more.
   A step of 8 or more keeps the counts within 16 bits. */
struct cut_rule {
  unsigned shift, step, gain, noise;
};

static const struct cut_rule cut_rules[4] = {
    {1, 512, 2400, 2}, /* fast */
    {0, 128, 1200, 2}, /* double fast */
    {0, 8, 1200, 4},   /* greedy, lazy */
    {0, 8, 300, 4},    /* lazy2 */
};

/* Index correction: past this index, the tables are rebased so that
   positions keep within 32 bits. */
#ifndef INDEX_LIMIT
#define INDEX_LIMIT (3u << 30)
#endif

/* The reference's level table (clevels.h): window, short hash, hash and
   search logs, hashed length, step and strategy, for inputs above 128 KiB
   or of unknown size, of at most 128 KiB and of at most 16 KiB. Its
   binary-tree strategies, at levels 9 to 12 for small inputs, are lazy2
   here. */
static const struct ze_params levels[3][12] = {
    {{19, 13, 14, 1, 7, 0, ZE_FAST},
     {20, 15, 16, 1, 6, 0, ZE_FAST},
     {21, 16, 17, 1, 5, 0, ZE_DFAST},
     {21, 18, 18, 1, 5, 0, ZE_DFAST},
     {21, 18, 19, 3, 5, 2, ZE_GREEDY},
     {21, 18, 19, 3, 5, 4, ZE_LAZY},
     {21, 19, 20, 4, 5, 8, ZE_LAZY},
     {21, 19, 20, 4, 5, 16, ZE_LAZY2},
     {22, 20, 21, 4, 5, 16, ZE_LAZY2},
     {22, 21, 22, 5, 5, 16, ZE_LAZY2},
     {22, 21, 22, 6, 5, 16, ZE_LAZY2},
     {22, 22, 23, 6, 5, 32, ZE_LAZY2}},
    {{17, 12, 13, 1, 6, 0, ZE_FAST},
     {17, 13, 15, 1, 5, 0, ZE_FAST},
     {17, 15, 16, 2, 5, 0, ZE_DFAST},
     {17, 17, 17, 2, 4, 0, ZE_DFAST},
     {17, 16, 17, 3, 4, 2, ZE_GREEDY},
     {17, 16, 17, 3, 4, 4, ZE_LAZY},
     {17, 16, 17, 3, 4, 8, ZE_LAZY2},
     {17, 16, 17, 4, 4, 8, ZE_LAZY2},
     {17, 16, 17, 5, 4, 8, ZE_LAZY2},
     {17, 16, 17, 6, 4, 8, ZE_LAZY2},
     {17, 17, 17, 5, 4, 8, ZE_LAZY2},
     {17, 18, 17, 7, 4, 12, ZE_LAZY2}},
    {{14, 14, 15, 1, 5, 0, ZE_FAST},
     {14, 14, 15, 1, 4, 0, ZE_FAST},
     {14, 14, 15, 2, 4, 0, ZE_DFAST},
     {14, 14, 14, 4, 4, 2, ZE_GREEDY},
     {14, 14, 14, 3, 4, 4, ZE_LAZY},
     {14, 14, 14, 4, 4, 8, ZE_LAZY2},
     {14, 14, 14, 6, 4, 8, ZE_LAZY2},
     {14, 14, 14, 8, 4, 8, ZE_LAZY2},
     {14, 15, 14, 5, 4, 8, ZE_LAZY2},
     {14, 15, 14, 9, 4, 8, ZE_LAZY2},
     {14, 15, 14, 3, 4, 12, ZE_LAZY2},
     {14, 15, 14, 4, 4, 24, ZE_LAZY2}},
};


struct ze_work {
  /* The frame: its parameters once its first block is cut, whether it
     ended, the input taken and hashed, the bytes its blocks saved, the
     repeat offsets, the current entropy tables, how far its indices were
     rebased and the index past which they are. */
  struct ze_params p;
  bool fixed;
  bool done;
  bool bmi2;
  uint64_t taken;
  int64_t saved;
  uint32_t rep[2];
  int cur;
  uint32_t corr;
  uint32_t limit;
  struct lz_xxh64 xxh;
  struct ze_match ms;

  /* The tables, the sequence buffers and a block's scratch output. */
  uint8_t *tables;
  size_t tables_cap;
  uint8_t *arrays;
  size_t arrays_cap;
  struct ze_seqs ss;
  uint8_t *out;

  /* Streaming: the window, from frame offset [pos], with [next] the first
     byte not yet in a block and [hi] the end of the input taken. */
  uint8_t *buf;
  size_t cap;
  size_t next, hi;
  uint64_t pos;

  struct ze_entropy ent[2];
  struct ze_defaults dflt;

  /* cut's scratch: the sampled byte counts of spans [0, k) in [cum k],
     and [xlg]'s table once [xlg_ready]. */
  uint16_t cum[CUT_SPANS + 1][256];
  bool xlg_ready;
  uint32_t xlg[XLG_TAB];
};

/* Parameters */

/* [params_of p level sized n] is the level's parameters; for a frame known
   to hold [n] bytes, at most a block, those of the reference's tables for
   small inputs, shrunk to [n] as the reference shrinks them. A row needs
   two at least. */
static void params_of(struct ze_params *p, int level, bool sized, uint64_t n) {
  int row = !sized ? 0 : n <= 16 * 1024 ? 2 : 1;
  *p = levels[row][level - 1];
  if (sized) {
    unsigned s = n < 64 ? 6 : ze_highbit((uint32_t)(n - 1)) + 1;
    if (p->wlog > s) p->wlog = (uint8_t)s;
    if (p->hlog > p->wlog + 1) p->hlog = (uint8_t)(p->wlog + 1);
    if (p->clog > p->wlog) p->clog = p->wlog;
    if (p->wlog < MIN_WLOG) p->wlog = MIN_WLOG;
  }
  if (p->strat >= ZE_GREEDY) {
    unsigned rlog = p->slog < 4 ? 4 : p->slog > 6 ? 6 : p->slog;
    if (p->hlog > 24 + rlog) p->hlog = (uint8_t)(24 + rlog);
    if (p->hlog < rlog + 1) p->hlog = (uint8_t)(rlog + 1);
  }
}

/* Memory */

static size_t align64(size_t n) { return (n + 63) & ~(size_t)63; }

/* The sequence buffers for blocks of up to [bmax] bytes: sequences,
   literals and a block's scratch output. A sequence covers 3 bytes at
   least. */
static size_t seqs_max(size_t bmax) { return bmax / 3 + 1; }

static size_t arrays_bytes(size_t bmax) {
  size_t ns = seqs_max(bmax);
  return align64(ns * sizeof(struct ze_seq)) + align64(bmax + ZE_SLACK) +
         align64(bmax + 3 + 2 * ZE_SLACK);
}

static bool grow(uint8_t **p, size_t *cap, size_t need) {
  if (*cap >= need) return true;
  lz_free(*p);
  *p = lz_alloc(need + 64);
  *cap = *p ? need : 0;
  return *p != NULL;
}

size_t lz_zstd_enc_mem(int level) {
  struct ze_params p;
  params_of(&p, level, false, 0);
  size_t wsize = (size_t)1 << p.wlog;
  return sizeof(struct ze_work) + ze_match_bytes(&p) + 64 +
         arrays_bytes(ZE_BLOCK) + 64 + 2 * wsize + 2 * ZE_BLOCK;
}

/* [prepare w bmax] allocates the tables of [w->p] and the buffers for
   blocks of up to [bmax] bytes, empties the tables and starts the entropy
   and repeat state. */
static bool prepare(struct ze_work *w, size_t bmax) {
  size_t tb = ze_match_bytes(&w->p);
  size_t ns = seqs_max(bmax);
  if (!grow(&w->tables, &w->tables_cap, tb)) return false;
  if (!grow(&w->arrays, &w->arrays_cap, arrays_bytes(bmax))) return false;

  uint8_t *t = (uint8_t *)align64((size_t)w->tables);
  struct ze_match *ms = &w->ms;
  ms->p = w->p;
  ms->t1 = (uint32_t *)(void *)t;
  ms->t2 = NULL;
  ms->tags = NULL;
  size_t h = (size_t)4 << w->p.hlog;
  if (w->p.strat == ZE_DFAST) ms->t2 = (uint32_t *)(void *)(t + h);
  if (w->p.strat >= ZE_GREEDY) ms->tags = t + h;
  ze_match_reset(ms);
  ms->low = 1;
  ms->next = 1;

  uint8_t *a = (uint8_t *)align64((size_t)w->arrays);
  w->ss.seq = (struct ze_seq *)(void *)a;
  a += align64(ns * sizeof(struct ze_seq));
  w->ss.lit = a;
  a += align64(bmax + ZE_SLACK);
  w->out = a;

  /* Section 3.1.1.5: a frame starts with repeat offsets 1, 4 and 8, of
     which the finders read the first two, and no table to repeat. */
  w->rep[0] = 1;
  w->rep[1] = 4;
  w->cur = 0;
  w->ent[0].huf_mode = w->ent[0].ll_mode = w->ent[0].of_mode =
      w->ent[0].ml_mode = ZE_NONE;
  w->saved = 0;
  w->corr = 0;
  return true;
}

/* Output */

struct sink {
  uint8_t *p, *end;
  bool full;
};

static void put(struct sink *o, const void *s, size_t n) {
  if (o->full || (size_t)(o->end - o->p) < n) {
    o->full = true;
    return;
  }
  memcpy(o->p, s, n);
  o->p += n;
}

static void put_block_header(struct sink *o, bool last, unsigned type,
                             size_t size) {
  uint8_t h[3];
  ze_wr24(h, (uint32_t)last | (type << 1) | ((uint32_t)size << 3));
  put(o, h, 3);
}

/* [put_header o p sized n] writes the frame header: Single_Segment when
   a known size fits the window, the Window_Descriptor otherwise, and the
   Frame_Content_Size when the size is known (section 3.1.1.1). */
static void put_header(struct sink *o, const struct ze_params *p, bool sized,
                       uint64_t n) {
  uint8_t h[14];
  size_t k = 4;
  ze_wr32(h, MAGIC);
  uint8_t wd = (uint8_t)((p->wlog - MIN_WLOG) << 3);
  if (!sized) {
    h[k++] = 0x04;
    h[k++] = wd;
    put(o, h, k);
    return;
  }
  unsigned code = (n >= 256) + (n >= 65536 + 256) + (n >= 0xFFFFFFFFu);
  bool single = ((uint64_t)1 << p->wlog) >= n;
  h[k++] = (uint8_t)((code << 6) | ((unsigned)single << 5) | 0x04);
  if (!single) h[k++] = wd;
  switch (code) {
  case 0:
    h[k++] = (uint8_t)n;
    break;
  case 1:
    h[k] = (uint8_t)(n - 256);
    h[k + 1] = (uint8_t)((n - 256) >> 8);
    k += 2;
    break;
  case 2:
    ze_wr32(h + k, (uint32_t)n);
    k += 4;
    break;
  default:
    ze_wr64(h + k, n);
    k += 8;
    break;
  }
  put(o, h, k);
}

/* Block cuts */

/* 256 log2(m / 128) for m from 128 to 255, rounded. */
static const uint8_t lg_frac[128] = {
    0,   3,   6,   9,   11,  14,  17,  20,  22,  25,  28,  30,  33,  36,
    38,  41,  44,  46,  49,  51,  54,  56,  59,  61,  63,  66,  68,  71,
    73,  75,  78,  80,  82,  85,  87,  89,  92,  94,  96,  98,  100, 103,
    105, 107, 109, 111, 113, 116, 118, 120, 122, 124, 126, 128, 130, 132,
    134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 155, 157, 159,
    161, 163, 165, 167, 169, 170, 172, 174, 176, 178, 179, 181, 183, 185,
    186, 188, 190, 192, 193, 195, 197, 198, 200, 202, 203, 205, 207, 208,
    210, 212, 213, 215, 216, 218, 220, 221, 223, 224, 226, 228, 229, 231,
    232, 234, 235, 237, 238, 240, 241, 243, 244, 246, 247, 249, 250, 252,
    253, 255};

/* [xlg x] is 256 x log2 x, 0 for 0: the bits, in 1/256, that [x] symbols
   of a part of [x] symbols would take if each were alone. */
static uint64_t xlg(uint32_t x) {
  unsigned e = ze_highbit(x | 1);
  uint32_t m = ((x << 7) >> e) & 127;
  return (uint64_t)x * (256 * e + lg_frac[m]);
}

/* [xlg_of w x] is [xlg x], from the table when [x] is in it. */
static uint64_t xlg_of(const struct ze_work *w, uint32_t x) {
  return x < XLG_TAB ? w->xlg[x] : xlg(x);
}

/* [split w end m pre gain kinds] is the span, from 1 to [end - 1], at
   which a cut of spans [0, end) saves the most, by the counts' entropy,
   with [*gain] what it saves in 1/256 bits; 0 if no cut saves anything.
   [m] is the counts a span holds, [pre c] the cost of spans [0, c),
   [*kinds] the byte values present. */
static unsigned split(const struct ze_work *w, unsigned end, uint32_t m,
                      const int64_t *pre, int64_t *gain, unsigned *kinds) {
  const uint16_t *tot = w->cum[end];
  uint8_t sym[256];
  unsigned ns = 0;
  for (unsigned v = 0; v < 256; v++)
    if (tot[v]) sym[ns++] = (uint8_t)v;
  *kinds = ns;

  int64_t best = 0;
  unsigned at = 0;
  for (unsigned c = 1; c < end; c++) {
    const uint16_t *p = w->cum[c];
    uint64_t f = 0;
    for (unsigned i = 0; i < ns; i++)
      f += xlg_of(w, (uint32_t)(tot[sym[i]] - p[sym[i]]));
    int64_t rest = (int64_t)xlg_of(w, (end - c) * m) - (int64_t)f;
    int64_t g = pre[end] - pre[c] - rest;
    if (g > best) {
      best = g;
      at = c;
    }
  }
  *gain = best;
  return at;
}

/* contract: [cut w src left] is the next block's size, from 1 to
   [ZE_BLOCK], given the [left] bytes known at [src]: all of them when
   [left < ZE_BLOCK]. It reads at most [ZE_BLOCK] bytes at [src], keeps
   any scratch it needs in [struct ze_work] (add fields as you need), and
   depends only on those bytes, [w->saved] and [w->p] so that a stream
   and a whole-buffer encode cut alike. The rule is the design's
   (cleanroom3, design.md). */
static size_t cut(struct ze_work *w, const uint8_t *src, size_t left) {
  if (left < ZE_BLOCK) return left;
  if (w->saved < 3) return ZE_BLOCK;
  if (!w->xlg_ready) {
    for (uint32_t x = 0; x < XLG_TAB; x++) w->xlg[x] = (uint32_t)xlg(x);
    w->xlg_ready = true;
  }

  int s = w->p.strat;
  const struct cut_rule *r =
      &cut_rules[s <= ZE_FAST ? 0 : s <= ZE_DFAST ? 1 : s <= ZE_LAZY ? 2 : 3];
  size_t span = (size_t)CUT_SPAN << r->shift;
  unsigned spans = CUT_SPANS >> r->shift;
  memset(w->cum[0], 0, sizeof w->cum[0]);
  for (unsigned k = 0; k < spans; k++) {
    const uint8_t *p = src + k * span;
    uint16_t *h = w->cum[k + 1];
    memcpy(h, w->cum[k], sizeof w->cum[k]);
    for (size_t i = 0; i < span; i += 8 * r->step) {
      uint64_t x = ze_rd64(p + i);
      for (unsigned b = 0; b < 64; b += 8) h[(x >> b) & 0xff]++;
    }
  }

  /* The cost of each prefix of n counts: 256 n log2 n less each value's
     [xlg]. */
  uint32_t m = (uint32_t)(span / r->step);
  const uint16_t *tot = w->cum[spans];
  uint8_t sym[256];
  unsigned ns = 0;
  for (unsigned v = 0; v < 256; v++)
    if (tot[v]) sym[ns++] = (uint8_t)v;
  int64_t pre[CUT_SPANS + 1];
  pre[0] = 0;
  for (unsigned c = 1; c <= spans; c++) {
    const uint16_t *p = w->cum[c];
    uint64_t f = 0;
    for (unsigned i = 0; i < ns; i++) f += xlg_of(w, p[sym[i]]);
    pre[c] = (int64_t)xlg_of(w, c * m) - (int64_t)f;
  }

  /* The block ends at the first change: the best cut of the window, then
     the best of what precedes it, while a cut pays. What follows is
     weighed again in the next block's window. */
  unsigned end = spans;
  for (;;) {
    int64_t g;
    unsigned kinds;
    unsigned c = split(w, end, m, pre, &g, &kinds);
    uint64_t need = (uint64_t)r->gain * 2048 +
                    (uint64_t)r->step * r->noise * kinds * 128;
    if (!c || (uint64_t)g * r->step < need) break;
    end = c;
  }
  return end * span;
}


/* Blocks */

/* [correct w block_idx] rebases every index by the same amount once
   indices near 32 bits, keeping the window's: a candidate that falls
   below it becomes 0, which is never valid, so the sequences found do
   not change. */
static uint32_t correct(struct ze_work *w, uint32_t block_idx) {
  struct ze_match *ms = &w->ms;
  uint32_t c = block_idx - (1u << w->p.wlog) - 1;
  size_t n1 = (size_t)1 << w->p.hlog;
  /* A row's slot 0 holds no position, and an untagged slot none yet. */
  unsigned slog = w->p.slog < 4 ? 4 : w->p.slog > 6 ? 6 : w->p.slog;
  size_t row_mask = ((size_t)1 << slog) - 1;
  for (size_t i = 0; i < n1; i++) {
    if (ms->tags && (!ms->tags[i] || !(i & row_mask))) continue;
    ms->t1[i] = ms->t1[i] > c ? ms->t1[i] - c : 0;
  }
  if (ms->t2) {
    size_t n2 = (size_t)1 << w->p.clog;
    for (size_t i = 0; i < n2; i++)
      ms->t2[i] = ms->t2[i] > c ? ms->t2[i] - c : 0;
  }
  ms->next = ms->next > c ? ms->next - c : 1;
  w->corr += c;
  return c;
}

static bool all_same(const uint8_t *s, size_t n) {
  for (size_t i = 1; i < n; i++)
    if (s[i] != s[0]) return false;
  return true;
}

/* [block w src n last o] writes one block of the [n] bytes at [src]. */
static void block(struct ze_work *w, const uint8_t *src, size_t n, bool last,
                  struct sink *o) {
  struct ze_match *ms = &w->ms;
  uint32_t idx = (uint32_t)(src - ms->base);
  /* Past the default limit the block always starts past the window; a
     lowered one waits for it. */
  if (idx + n > w->limit && idx > (1u << w->p.wlog) + 1)
    ms->base += correct(w, idx);

  size_t written = 0;
  if (n >= MIN_BLOCK) {
    struct ze_seqs *ss = &w->ss;
    ss->n = 0;
    ss->nlit = 0;
    uint32_t rep[2] = {w->rep[0], w->rep[1]};
    size_t tail = ze_match_block(ms, ss, rep, src, n);
    memcpy(ss->lit + ss->nlit, src + n - tail, tail);
    ss->nlit += tail;

    bool direct = (size_t)(o->end - o->p) >= n + 3 + ZE_SLACK && !o->full;
    uint8_t *body = direct ? o->p + 3 : w->out + 3;
    struct ze_entropy *prev = &w->ent[w->cur], *next = &w->ent[w->cur ^ 1];
    size_t c =
        ze_body(body, n, ss, prev, next, &w->dflt, w->p.strat, w->bmi2);
    if (c && c < n - ((n >> 6) + 2)) {
      if (c < RLE_MAX && all_same(src, n)) {
        uint8_t h[4];
        ze_wr24(h, (uint32_t)last | (BLOCK_RLE << 1) | ((uint32_t)n << 3));
        h[3] = src[0];
        put(o, h, 4);
        written = 4;
      } else {
        uint8_t *h = body - 3;
        ze_wr24(h, (uint32_t)last | (BLOCK_COMPRESSED << 1) |
                       ((uint32_t)c << 3));
        if (direct)
          o->p += c + 3;
        else
          put(o, h, c + 3);
        written = c + 3;
        w->cur ^= 1;
        w->rep[0] = rep[0];
        w->rep[1] = rep[1];
      }
    }
  }
  if (!written) {
    put_block_header(o, last, BLOCK_RAW, n);
    put(o, src, n);
    written = n + 3;
  }
  w->saved += (int64_t)n - (int64_t)written;
}

/* Frames */

static struct ze_work *work(struct lz_zstd_enc *st) {
  if (st->w) return st->w;
  struct ze_work *w = lz_alloc(sizeof *w);
  if (!w) return NULL;
  memset(w, 0, sizeof *w);
  w->bmi2 = st->bmi2;
  w->limit = st->limit;
  ze_defaults_build(&w->dflt);
  st->w = w;
  return w;
}

static void begin(struct ze_work *w) {
  w->fixed = false;
  w->done = false;
  w->taken = 0;
  w->next = w->hi = 0;
  w->pos = 0;
  lz_xxh64_init(&w->xxh, 0);
}

/* [fix w level sized n] sets the frame's parameters and readies its
   tables. */
static bool fix(struct ze_work *w, int level, bool sized, uint64_t n) {
  params_of(&w->p, level, sized, n);
  if (!prepare(w, sized ? (size_t)n + 1 : ZE_BLOCK)) return false;
  w->fixed = true;
  return true;
}

static void put_checksum(struct sink *o, struct ze_work *w) {
  uint8_t c[4];
  ze_wr32(c, (uint32_t)lz_xxh64_digest(&w->xxh));
  put(o, c, 4);
}

static enum lz_status nomem(struct lz_error *err) {
  return lz_fail(err, LZ_NOMEM, 0, NULL);
}

/* The whole-buffer encode: the input in place, its size in the header. */
static enum lz_status whole(struct lz_zstd_enc *st, struct ze_work *w,
                            struct lz_io *io, struct lz_error *err) {
  const uint8_t *src = io->src;
  size_t n = io->src_len;
  struct sink o = {io->dst, io->dst + io->dst_cap, false};

  begin(w);
  if (!fix(w, st->level, n <= ZE_BLOCK, n)) return nomem(err);
  put_header(&o, &w->p, true, n);
  w->ms.base = src - 1;

  /* Each block is hashed as it is compressed, while it is in cache. */
  if (n == 0) put_block_header(&o, true, BLOCK_RAW, 0);
  for (size_t at = 0; at < n && !o.full;) {
    size_t b = cut(w, src + at, n - at);
    w->ms.base = src - 1 + w->corr;
    block(w, src + at, b, at + b == n, &o);
    lz_xxh64_update(&w->xxh, src + at, b);
    at += b;
  }
  put_checksum(&o, w);
  io->consumed = n;
  if (o.full) return LZ_NEED_OUTPUT;
  io->produced = (size_t)(o.p - io->dst);
  return LZ_DONE;
}

/* [hold w s n] appends [n] input bytes to the window, first moving the
   window down when they do not fit: it keeps the bytes a match may
   still reach, the window before the next block. */
static bool hold(struct ze_work *w, const uint8_t *s, size_t n) {
  size_t wsize = w->fixed ? (size_t)1 << w->p.wlog : 0;
  size_t need = 2 * wsize + 2 * ZE_BLOCK;
  if (w->cap < need) {
    uint8_t *b = lz_alloc(need);
    if (!b) return false;
    if (w->buf) memcpy(b, w->buf, w->hi);
    lz_free(w->buf);
    w->buf = b;
    w->cap = need;
  }
  if (w->hi + n > w->cap) {
    size_t keep = w->next > wsize ? w->next - wsize : 0;
    memmove(w->buf, w->buf + keep, w->hi - keep);
    w->pos += keep;
    w->next -= keep;
    w->hi -= keep;
  }
  memcpy(w->buf + w->hi, s, n);
  w->hi += n;
  return true;
}

/* A step: take the input, write the blocks it completes, and with
   [last] the rest and the frame's end. */
static enum lz_status step(struct lz_zstd_enc *st, struct ze_work *w,
                           struct lz_io *io, bool last, struct lz_error *err) {
  if (w->done) begin(w);
  size_t n = io->src_len < LZ_ZSTD_ENC_QUOTA ? io->src_len : LZ_ZSTD_ENC_QUOTA;
  bool fin = last && n == io->src_len;
  struct sink o = {io->dst, io->dst + io->dst_cap, false};

  if (!hold(w, io->src, n)) return nomem(err);
  lz_xxh64_update(&w->xxh, io->src, n);
  w->taken += n;
  io->consumed = n;

  for (;;) {
    size_t left = w->hi - w->next;
    if (!fin && left <= ZE_BLOCK) break;
    if (!w->fixed) {
      if (!fix(w, st->level, fin && w->taken <= ZE_BLOCK, w->taken))
        return nomem(err);
      if (!hold(w, io->src, 0)) return nomem(err);
      put_header(&o, &w->p, false, 0);
    }
    if (left == 0) {
      put_block_header(&o, true, BLOCK_RAW, 0);
      break;
    }
    const uint8_t *src = w->buf + w->next;
    size_t b = cut(w, src, fin ? left : ZE_BLOCK);
    w->ms.base = src - (uint32_t)(w->pos + w->next + 1 - w->corr);
    block(w, src, b, fin && b == left, &o);
    w->next += b;
    if (fin && b == left) break;
  }
  if (fin) {
    put_checksum(&o, w);
    w->done = true;
  }
  io->produced = (size_t)(o.p - io->dst);
  return fin ? LZ_DONE : LZ_NEED_INPUT;
}

/* Entries */

static const char *const path_names[] = {"generic", "bmi2"};

static size_t path_count(void) {
#if ZE_BMI2
  if (__builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
      __builtin_cpu_supports("lzcnt"))
    return 2;
#endif
  return 1;
}

size_t lz_zstd_enc_paths(const char *const **names) {
  *names = path_names;
  return path_count();
}

void lz_zstd_enc_path(struct lz_zstd_enc *st, size_t i) { st->bmi2 = i == 1; }

void lz_zstd_enc_limit(struct lz_zstd_enc *st, uint32_t limit) {
  st->limit = limit;
}

void lz_zstd_enc_init(struct lz_zstd_enc *st, int level, bool stream) {
  st->level = level;
  st->stream = stream;
  st->w = NULL;
  st->limit = INDEX_LIMIT;
  lz_zstd_enc_path(st, path_count() - 1);
}

void lz_zstd_enc_reset(struct lz_zstd_enc *st) {
  if (st->w) begin(st->w);
}

void lz_zstd_enc_free(struct lz_zstd_enc *st) {
  struct ze_work *w = st->w;
  if (!w) return;
  lz_free(w->tables);
  lz_free(w->arrays);
  lz_free(w->buf);
  lz_free(w);
  st->w = NULL;
}

enum lz_status lz_zstd_encode(struct lz_zstd_enc *st, struct lz_io *io,
                              bool last, struct lz_error *err) {
  bool fresh = st->w == NULL;
  struct ze_work *w = work(st);
  if (!w) return nomem(err);
  if (fresh) begin(w);
  if (!st->stream) return whole(st, w, io, err);
  return step(st, w, io, last, err);
}

/* Each block costs at most 3 bytes over its input, and blocks are cut at
   8 KiB or more but for the last; the header takes at most 14 bytes and
   the checksum 4. */
size_t lz_zstd_bound(size_t n) { return n + 3 * (n / 8192 + 1) + 14 + 4; }
