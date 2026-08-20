/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The fused kernels: the byte-level one — the GPT-2 byte-level walker, the
   pretoken cache probe and the short byte-level BPE merge over a range of
   text, one call per chunk of spans — and the SentencePiece one at the end
   of this file, which cuts ▁/punctuation word units and feeds them through
   the same probe and merge machinery. Everything rare is handed back to
   OCaml with a resume position: an unclassified code point, a pretoken over
   15 bytes, a byte or character without a vocabulary id, a model whose
   misses may not be merged here, and a merge whose result stands for other
   bytes than its entry's. The OCaml implementations in pre_tokenizer.ml,
   char_class.ml and bpe.ml are the reference; every function here mirrors
   one there by name, and the native-vs-bytecode dump differential holds the
   two together.

   Nothing is allocated, raised or called back — the entry is [@@noalloc] —
   so no GC point exists, every derived pointer stays valid for the whole
   call and no CAMLparam rooting is needed. Stores into the ids and marks
   buffers are immediates over immediates in int arrays, which is what
   Array.unsafe_set compiles to and needs no write barrier (the OCaml 5
   deletion barrier only darkens pointer old values); Bytes stores are raw,
   and so are the int32 stores of the ids32 entries, whose sink is a
   Bigarray outside the heap. All arithmetic that must wrap is unsigned. */

#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

#if defined(__aarch64__) && defined(__ARM_NEON) \
    && (defined(__GNUC__) || defined(__clang__))
#define BROT_MASK_NEON 1
#include <arm_neon.h>
#endif

/* ABI pins, mirrored in kernel.ml — change neither without the other. */

/* Kernel.reason constructor order. */
enum { BROT_DONE, BROT_SPANS_FULL, BROT_IDS_FULL, BROT_CLASS, BROT_ENCODE };

/* Kernel.byte_level field order. */
enum {
  BROT_BL_LEAD,
  BROT_BL_FRONT,
  BROT_BL_FRONT_MASK,
  BROT_BL_CACHE,
  BROT_BL_CACHE_MASK,
  BROT_BL_BYTE_IDS,
  BROT_BL_MERGE_KEYS,
  BROT_BL_MERGE_VALUES,
  BROT_BL_MERGE_MASK,
  BROT_BL_LEN_TABLE,
  BROT_BL_MERGE
};

/* Kernel.sp field order. */
enum {
  BROT_SP_FRONT,
  BROT_SP_FRONT_MASK,
  BROT_SP_CACHE,
  BROT_SP_CACHE_MASK,
  BROT_SP_MERGE_KEYS,
  BROT_SP_MERGE_VALUES,
  BROT_SP_MERGE_MASK,
  BROT_SP_LEN_TABLE,
  BROT_SP_ASCII_IDS,
  BROT_SP_CHAR_KEYS,
  BROT_SP_CHAR_VALUES,
  BROT_SP_CHAR_MASK,
  BROT_SP_SCAN,
  BROT_SP_PUNCT_SAFE
};

/* Kernel.cursor byte offsets. */
enum { CUR_SPANS = 0, CUR_IDS = 8, CUR_MARKS = 16, CUR_RESUME = 24,
       CUR_CP = 32 };

/* Pre_tokenizer.lead_class values: the four Char_class categories, then the
   space, the apostrophe and the two shapes of non-ASCII byte. */
enum {
  CL_OTHER = 0,
  CL_WS = 1,
  CL_LETTER = 2,
  CL_NUM = 3,
  CL_SPACE = 4,
  CL_APO = 5,
  CL_LEAD = 6,
  CL_CONT = 7
};

static inline uint64_t load64(const void *p)
{
  uint64_t w;
  memcpy(&w, p, 8);
  return w;
}

static inline void store64(void *p, uint64_t w) { memcpy(p, &w, 8); }

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int ctz64(uint64_t x)
{
  unsigned long i;
  _BitScanForward64(&i, x);
  return (int)i;
}
#else
static inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

/* The prefetch is advisory: GCC/Clang emit it, other compilers drop it. */
#if defined(__GNUC__) || defined(__clang__)
#define BROT_PREFETCH(p) __builtin_prefetch(p)
#else
#define BROT_PREFETCH(p) ((void)(p))
#endif

/* The walker — Pre_tokenizer.fill_byte_level and Char_class.at, line for
   line. A class the Unicode table does not hold yet cannot be filled here
   (Uucp stays in OCaml): the walk stops, [cp] carries the code point out and
   the caller classifies it and calls again. */

typedef struct {
  const unsigned char *s; /* the text */
  intnat stop;            /* the end of the range being walked */
  const unsigned char *lead; /* Pre_tokenizer.lead_class, 256 bytes */
  const unsigned char *uni;  /* Char_class.unicode_table, byte per cp - 128 */
  intnat uni_len;
  intnat cp; /* out: the code point needing a class */
} walker;

/* Char_class.category on ASCII, recovered from the lead table by folding the
   space and the apostrophe back into their categories. Only the category is
   ever consumed here, so the packed class carries it in place of the full
   property byte the OCaml side packs. */
static inline intnat ascii_cat(const unsigned char *lead, intnat c)
{
  intnat k = lead[c];
  if (k == CL_SPACE) return CL_WS;
  if (k == CL_APO) return CL_OTHER;
  return k;
}

#define STRAY ((CL_OTHER << 3) | 1)

/* Char_class.decode + pack: (category << 3) | byte length, or -1 with
   [w->cp] set when the code point's class is not in the table yet. As in
   OCaml: no byte at or after stop is read, a sequence cut short by stop, a
   byte that cannot lead and a bad continuation are each one stray byte of
   category other, and a decoded value above 0x10FFFF is other. */
static intnat decode_class(walker *w, intnat i)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat c = s[i], cp, len;
  if (c < 0xC2) return STRAY;
  if (c < 0xE0) {
    intnat b1;
    if (i + 1 >= stop) return STRAY;
    b1 = s[i + 1];
    if ((b1 & 0xC0) != 0x80) return STRAY;
    cp = ((c & 0x1F) << 6) | (b1 & 0x3F);
    len = 2;
  } else if (c < 0xF0) {
    intnat b1, b2;
    if (i + 2 >= stop) return STRAY;
    b1 = s[i + 1];
    b2 = s[i + 2];
    if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) return STRAY;
    cp = ((c & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    len = 3;
  } else if (c < 0xF8) {
    intnat b1, b2, b3;
    if (i + 3 >= stop) return STRAY;
    b1 = s[i + 1];
    b2 = s[i + 2];
    b3 = s[i + 3];
    if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80)
      return STRAY;
    cp = ((c & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6)
         | (b3 & 0x3F);
    len = 4;
  } else
    return STRAY;
  {
    intnat cat;
    if (cp < 128) /* an overlong form decodes below ASCII */
      cat = ascii_cat(w->lead, cp);
    else if (cp > 0x10FFFF)
      cat = CL_OTHER;
    else if (cp - 128 < w->uni_len && w->uni[cp - 128] != 0)
      cat = w->uni[cp - 128] & 3;
    else {
      w->cp = cp;
      return -1;
    }
    return (cat << 3) | len;
  }
}

/* Char_class.at. */
static inline intnat char_at(walker *w, intnat i)
{
  intnat c = w->s[i];
  if (c < 0x80) return (ascii_cat(w->lead, c) << 3) | 1;
  return decode_class(w, i);
}

/* Pre_tokenizer.letters_swar: eight bytes at a time, bit 7 flagging each
   byte that is not an ASCII letter. */
static intnat letters_swar(const unsigned char *s, intnat i, intnat stop)
{
  while (i + 8 <= stop) {
    uint64_t v = load64(s + i);
    uint64_t lowered = (v & 0x7F7F7F7F7F7F7F7FULL) | 0x2020202020202020ULL;
    uint64_t ge_a = (lowered | 0x8080808080808080ULL) - 0x6161616161616161ULL;
    uint64_t le_z = 0xFAFAFAFAFAFAFAFAULL - lowered;
    uint64_t letters = ge_a & le_z & 0x8080808080808080ULL;
    uint64_t m = (~letters | v) & 0x8080808080808080ULL;
    if (m == 0)
      i += 8;
    else
      return i + (ctz64(m) >> 3);
  }
  return i;
}

/* Pre_tokenizer.category_run, or -1 propagating a class hand-back. */
static intnat category_run(walker *w, intnat j, intnat category)
{
  const unsigned char *s = w->s;
  const unsigned char *lead = w->lead;
  intnat stop = w->stop;
  if (category == CL_LETTER) j = letters_swar(s, j, stop);
  while (j < stop) {
    intnat c = lead[s[j]];
    if (c < CL_LEAD) {
      if (c == CL_APO)
        c = CL_OTHER;
      else if (c == CL_SPACE)
        c = CL_WS;
      if (c != category) break;
      j++;
    } else if (c == CL_LEAD) {
      intnat d = decode_class(w, j);
      if (d < 0) return -1;
      if (((d >> 3) & 3) != category) break;
      j += d & 7;
    } else { /* a stray continuation byte continues an "other" run */
      if (category != CL_OTHER) break;
      j++;
    }
  }
  return j;
}

/* Pre_tokenizer.whitespace_span, or -1 propagating a class hand-back. */
static intnat whitespace_span(walker *w, intnat i)
{
  const unsigned char *s = w->s;
  const unsigned char *lead = w->lead;
  intnat stop = w->stop;
  intnat j = i, last = i;
  while (j < stop) {
    intnat c = lead[s[j]];
    if (c == CL_WS || c == CL_SPACE) {
      last = j;
      j++;
    } else if (c == CL_LEAD) {
      intnat d = decode_class(w, j);
      if (d < 0) return -1;
      if (((d >> 3) & 3) != CL_WS) break;
      last = j;
      j += d & 7;
    } else
      break;
  }
  if (j == stop) return j;
  return last > i ? last : j;
}

/* Pre_tokenizer.contraction, or -1 propagating a class hand-back. */
static intnat contraction(walker *w, intnat i)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat c1;
  if (stop - i < 2) return stop;
  c1 = s[i + 1];
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
  if (stop - i >= 3) {
    intnat c2 = s[i + 2];
    if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e')
        || (c1 == 'l' && c2 == 'l'))
      return i + 3;
  }
  return category_run(w, i + 1, CL_OTHER);
}

/* One span of Pre_tokenizer.fill_byte_level: the end of the span opening at
   [i], or -1 with [w->cp] set. */
static intnat next_span(walker *w, intnat i)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  switch (w->lead[s[i]]) {
  case CL_LETTER:
    return category_run(w, i + 1, CL_LETTER);
  case CL_SPACE: {
    intnat d, cat;
    if (i + 1 >= stop) return stop;
    d = char_at(w, i + 1);
    if (d < 0) return -1;
    cat = (d >> 3) & 3;
    if (cat == CL_WS) return whitespace_span(w, i);
    return category_run(w, i + 1 + (d & 7), cat);
  }
  case CL_APO:
    return contraction(w, i);
  case CL_NUM:
    return category_run(w, i + 1, CL_NUM);
  case CL_OTHER:
    return category_run(w, i + 1, CL_OTHER);
  case CL_WS:
    return whitespace_span(w, i);
  case CL_LEAD: {
    intnat d = decode_class(w, i);
    intnat cat;
    if (d < 0) return -1;
    cat = (d >> 3) & 3;
    if (cat == CL_WS) return whitespace_span(w, i);
    return category_run(w, i + (d & 7), cat);
  }
  default: /* CL_CONT */
    return category_run(w, i + 1, CL_OTHER);
  }
}

/* The mask scanner. Instead of walking span by span, each full 64-byte
   batch of the range is classified once into per-byte class bitmasks, the
   byte-level boundary rules are evaluated on all 64 positions at once as
   u64 shift algebra, and the walk pops one "a span starts here" bit per
   span. Bits the masks cannot decide are a bad zone that [next_span]
   re-derives — the scalar walker above stays the ground truth and runs
   the last partial batch outright, so no byte at or past [stop] is read;
   the one byte of lookahead a whitespace split needs is read only below
   [stop]. Bad zones are: non-ASCII bytes and both neighbours (their
   category comes from the Unicode table, which can also be incomplete —
   the scalar path raises the Class hand-back); a whitespace byte at the
   batch edge whose lookahead is non-ASCII; an apostrophe whose bits are
   already bad, plus the two suffix bytes its contraction could absorb (the
   contraction is the one rule that reaches two bytes ahead); and an
   apostrophe opening a span less than three bytes before the batch edge,
   whose moved boundary the next batch could not see. */

/* Class bitmasks of one 64-byte batch: bit k describes byte [scan + k]. */
typedef struct {
  uint64_t letter; /* ASCII letters */
  uint64_t digit;  /* ASCII digits */
  uint64_t space;  /* 0x20 */
  uint64_t ws;     /* 0x09..0x0D and 0x20 */
  uint64_t hi;     /* >= 0x80 */
  uint64_t apo;    /* 0x27 */
} batch_classes;

#ifdef BROT_MASK_NEON

/* Four 16-lane 0x00/0xFF compare results to one u64, bit i = lane i: each
   lane keeps its bit weight and a pairwise-add tree folds the 64 lanes
   into lane 0. The tree is pinned as asm: adjacent weighted lanes hold
   disjoint bits, so the compiler rewrites vpaddq_u8 into uzp1/uzp2/orr
   triples, nearly doubling the op count of the six calls per batch. */
static inline uint64_t movemask64(uint8x16_t a0, uint8x16_t a1, uint8x16_t a2,
                                  uint8x16_t a3)
{
  static const uint8_t weights[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                      1, 2, 4, 8, 16, 32, 64, 128};
  uint8x16_t w = vld1q_u8(weights);
  a0 = vandq_u8(a0, w);
  a1 = vandq_u8(a1, w);
  a2 = vandq_u8(a2, w);
  a3 = vandq_u8(a3, w);
  __asm__("addp %[a0].16b, %[a0].16b, %[a1].16b\n\t"
          "addp %[a2].16b, %[a2].16b, %[a3].16b\n\t"
          "addp %[a0].16b, %[a0].16b, %[a2].16b\n\t"
          "addp %[a0].16b, %[a0].16b, %[a0].16b"
          : [a0] "+&w"(a0), [a2] "+w"(a2)
          : [a1] "w"(a1), [a3] "w"(a3));
  return vgetq_lane_u64(vreinterpretq_u64_u8(a0), 0);
}

static void batch_classify(const unsigned char *p, batch_classes *c)
{
  uint8x16_t l[4], d[4], sp[4], ws[4], hi[4], ap[4];
  int k;
  for (k = 0; k < 4; k++) {
    uint8x16_t v = vld1q_u8(p + 16 * k);
    uint8x16_t lowered = vorrq_u8(v, vdupq_n_u8(0x20));
    l[k] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8('a')), vdupq_n_u8(25));
    d[k] = vcleq_u8(vsubq_u8(v, vdupq_n_u8('0')), vdupq_n_u8(9));
    sp[k] = vceqq_u8(v, vdupq_n_u8(' '));
    ws[k] =
        vorrq_u8(sp[k], vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4)));
    hi[k] = vcgeq_u8(v, vdupq_n_u8(0x80));
    ap[k] = vceqq_u8(v, vdupq_n_u8('\''));
  }
  c->letter = movemask64(l[0], l[1], l[2], l[3]);
  c->digit = movemask64(d[0], d[1], d[2], d[3]);
  c->space = movemask64(sp[0], sp[1], sp[2], sp[3]);
  c->ws = movemask64(ws[0], ws[1], ws[2], ws[3]);
  c->hi = movemask64(hi[0], hi[1], hi[2], hi[3]);
  c->apo = movemask64(ap[0], ap[1], ap[2], ap[3]);
}

#else

/* The SWAR batch classifier: the same masks, eight bytes at a time. Each
   predicate leaves 0x80 per qualifying byte; the multiply gathers the
   eight high bits into bits 0..7 (bit k = byte k, little-endian order as
   [letters_swar] assumes). Range tests run on the low seven bits with the
   high bit pre-set so no borrow crosses a byte; non-ASCII bytes are
   excluded explicitly. */

#define BROT_LO 0x0101010101010101ULL
#define BROT_HI 0x8080808080808080ULL

static inline uint64_t swar_pack(uint64_t m)
{
  return ((m >> 7) * 0x0102040810204080ULL) >> 56;
}

static inline uint64_t swar_ge(uint64_t ascii7, uint64_t c)
{
  return ((ascii7 | BROT_HI) - c * BROT_LO) & BROT_HI;
}

static inline uint64_t swar_le(uint64_t ascii7, uint64_t c)
{
  return ((c * BROT_LO | BROT_HI) - ascii7) & BROT_HI;
}

static void batch_classify(const unsigned char *p, batch_classes *c)
{
  int j;
  c->letter = c->digit = c->space = c->ws = c->hi = c->apo = 0;
  for (j = 0; j < 8; j++) {
    uint64_t v = load64(p + 8 * j);
    uint64_t hi = v & BROT_HI;
    uint64_t ascii = v & ~BROT_HI;
    uint64_t lowered = ascii | 0x2020202020202020ULL;
    uint64_t letter = swar_ge(lowered, 'a') & swar_le(lowered, 'z') & ~hi;
    uint64_t digit = swar_ge(ascii, '0') & swar_le(ascii, '9') & ~hi;
    uint64_t space = swar_ge(ascii, ' ') & swar_le(ascii, ' ') & ~hi;
    uint64_t ws = (space | (swar_ge(ascii, 9) & swar_le(ascii, 13))) & ~hi;
    uint64_t apo = swar_ge(ascii, '\'') & swar_le(ascii, '\'') & ~hi;
    int sh = 8 * j;
    c->letter |= swar_pack(letter) << sh;
    c->digit |= swar_pack(digit) << sh;
    c->space |= swar_pack(space) << sh;
    c->ws |= swar_pack(ws) << sh;
    c->hi |= swar_pack(hi) << sh;
    c->apo |= swar_pack(apo) << sh;
  }
}

#endif

/* The boundary masks of the batch at [scan]: bit k of [usable] is a
   trustworthy span start at [scan + k], bit k of [bad] sends byte
   [scan + k] to the scalar walker; the two never overlap. Requires
   [scan + 64 <= stop]. [anchor] is the position this call's walk began
   at — a span start by the resume protocol — so the batch there takes no
   carries and bit 0 is trivially a start; later batches read the byte
   before the batch, which is inside the range, for the four one-step
   carries the rules need. */
static void batch_masks(const walker *w, intnat anchor, intnat scan,
                        uint64_t *usable, uint64_t *bad_out)
{
  const unsigned char *s = w->s;
  batch_classes c;
  uint64_t pl = 0, pd = 0, psp = 0, pws = 0, po = 0;
  uint64_t other, cont_same, after_sp, nb, split_ok, pwsb, boundary, bad;
  uint64_t apo_bad, cand;
  batch_classify(s + scan, &c);
  other = ~(c.letter | c.digit | c.ws);
  bad = c.hi | (c.hi << 1) | (c.hi >> 1);
  if (scan > anchor) {
    unsigned b = s[scan - 1];
    if (b >= 0x80)
      bad |= 1; /* the carries are the straddling char's class: scalar */
    else {
      pl = (unsigned)((b | 0x20) - 'a') <= 25;
      pd = (unsigned)(b - '0') <= 9;
      psp = b == ' ';
      pws = psp | ((unsigned)(b - 9) <= 4);
      po = !(pl | pd | pws);
    }
  }
  /* A span continues over its own category, [ ?] glues a space to what
     follows, and a whitespace run opens where no whitespace precedes or
     at its last character when what follows is not whitespace — the
     [\s+(?!\S)] give-back. The apostrophe and the space sit in [other]
     and [ws] exactly as [lead_class] folds them. */
  cont_same = (c.letter & ((c.letter << 1) | pl))
              | (c.digit & ((c.digit << 1) | pd))
              | (other & ((other << 1) | po));
  after_sp = (c.space << 1) | psp;
  nb = ~c.ws & ~cont_same & ~after_sp;
  split_ok = c.ws & (~c.ws >> 1);
  if (c.ws >> 63) {
    /* Bit 63's give-back needs the next byte; at the end of the range the
       run runs out instead and keeps its last character. */
    if (scan + 64 < w->stop) {
      unsigned b = s[scan + 64];
      if (b >= 0x80)
        bad |= (uint64_t)1 << 63;
      else if (!(b == ' ' || (unsigned)(b - 9) <= 4))
        split_ok |= (uint64_t)1 << 63;
    }
  }
  pwsb = (c.ws << 1) | pws;
  boundary = nb | (c.ws & (~pwsb | split_ok));
  /* Whether an apostrophe opens a span decides its contraction, so an
     apostrophe on bad bits hides the two bytes a contraction could
     absorb. On trusted bits, one that opens a span moves the boundary
     past a matching suffix, as [contraction] does; within three bytes of
     the batch edge the moved boundary could escape the batch, so the
     tail of the batch goes to the scalar walker instead. */
  apo_bad = c.apo & bad;
  bad |= (apo_bad << 1) | (apo_bad << 2);
  cand = c.apo & boundary & ~bad;
  while (cand) {
    intnat i = ctz64(cand);
    unsigned c1, c2;
    intnat k = 0;
    cand &= cand - 1;
    if (i >= 61) {
      bad |= ~(uint64_t)0 << i;
      break;
    }
    c1 = s[scan + i + 1];
    c2 = s[scan + i + 2];
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
      k = 2;
    else if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e')
             || (c1 == 'r' && c2 == 'e'))
      k = 3;
    if (k) {
      boundary &= ~((uint64_t)1 << (i + 1));
      boundary |= (uint64_t)1 << (i + k);
    }
  }
  *usable = boundary & ~bad;
  *bad_out = bad;
}

/* The batch walk. [rem] holds the pop-ready boundary bits of the current
   segment; a bad zone parks them and aims [scalar_until] at the next
   trusted boundary (or the batch end), sending every span opening before
   it through [next_span]; the batch after the current one is classified
   eagerly so its chain retires under the pops; scalar overruns keep the
   64-byte grid, so that precompute survives them. The state lives for one
   kernel call — the masks are pure functions of the bytes, so a resumed
   call rebuilds them from its own anchor. */
typedef struct {
  intnat anchor;     /* the call's start position, carry-free by contract */
  intnat scan;       /* base of the next batch to classify */
  intnat mask_base;  /* base the rem/batch bits refer to */
  uint64_t rem;      /* pop-ready boundary bits of the current segment */
  uint64_t batch_usable;
  uint64_t batch_bad;
  intnat scalar_until; /* walk spans via next_span while below this */
  intnat pre_base;     /* base of the precomputed batch, -1 for none */
  uint64_t pre_usable;
  uint64_t pre_bad;
} mask_state;

static void mask_init(mask_state *ms, intnat pos)
{
  ms->anchor = pos;
  ms->scan = pos;
  ms->mask_base = pos;
  ms->rem = 0;
  ms->batch_usable = 0;
  ms->batch_bad = 0;
  ms->scalar_until = pos;
  ms->pre_base = -1;
  ms->pre_usable = 0;
  ms->pre_bad = 0;
}

/* Load the segment of usable bits in [from_bit, next bad run) into [rem];
   a bad run parks the rest and aims [scalar_until] past it. A bit at the
   pending span's own start opens that span rather than ending it, so it
   never pops. */
static void mask_load_segment(mask_state *ms, intnat pos, intnat from_bit)
{
  uint64_t live = ~(uint64_t)0 << from_bit;
  uint64_t seg_bad = ms->batch_bad & live;
  if (seg_bad == 0) {
    ms->rem = ms->batch_usable & live;
    ms->batch_bad = 0;
  } else {
    intnat nb = ctz64(seg_bad);
    uint64_t rest = ms->batch_usable & (~(uint64_t)0 << nb);
    ms->rem = ms->batch_usable & live & (((uint64_t)1 << nb) - 1);
    ms->scalar_until =
        rest != 0 ? ms->mask_base + ctz64(rest) : ms->mask_base + 64;
  }
  if (pos == ms->mask_base + from_bit)
    ms->rem &= ~((uint64_t)1 << from_bit);
}

/* [next_span] through the masks: the end of the span opening at [i], or
   -1 with [w->cp] set. [i] is the previous call's return (the pending
   span's start), as the entry loop maintains. */
static intnat mask_next_span(mask_state *ms, walker *w, intnat i)
{
  for (;;) {
    uint64_t usable, bad;
    if (ms->rem != 0) {
      intnat e = ms->mask_base + ctz64(ms->rem);
      ms->rem &= ms->rem - 1;
      return e;
    }
    if (i < ms->scalar_until) return next_span(w, i);
    /* Continue with the current batch's next trusted segment after a
       scalar gap; each batch is classified exactly once. */
    if (ms->batch_bad != 0 && i < ms->mask_base + 64) {
      mask_load_segment(ms, i, i - ms->mask_base);
      continue;
    }
    ms->batch_bad = 0;
    while (ms->scan + 64 <= i) ms->scan += 64;
    if (ms->scan + 64 > w->stop) {
      ms->scalar_until = w->stop; /* partial tail: scalar to the end */
      continue;
    }
    if (ms->pre_base == ms->scan) {
      usable = ms->pre_usable;
      bad = ms->pre_bad;
    } else
      batch_masks(w, ms->anchor, ms->scan, &usable, &bad);
    ms->mask_base = ms->scan;
    ms->scan += 64;
    ms->batch_usable = usable;
    ms->batch_bad = bad;
    if (ms->scan + 64 <= w->stop) {
      batch_masks(w, ms->anchor, ms->scan, &ms->pre_usable, &ms->pre_bad);
      ms->pre_base = ms->scan;
    } else
      ms->pre_base = -1;
    mask_load_segment(ms, i, i > ms->mask_base ? i - ms->mask_base : 0);
  }
}

/* The cache probe — Bpe.key0/key1/hash_of, the direct-mapped front table and
   the two-way layout of the 64-byte back set, bit for bit: every word is
   loaded and stored native-endian, so the table OCaml writes and the table
   written here are one format. A key
   word is one eight-byte read masked to the bytes it keeps; a pretoken
   ending within eight bytes of the end of the text reads the last eight
   bytes shifted down, and a text shorter than eight bytes gathers bytes one
   at a time — semantically masked, so nothing outside the range influences
   the result. [n] is the length of the whole text. */

static inline uint64_t key_word0(const unsigned char *s, intnat n, intnat pos,
                                 intnat len)
{
  uint64_t w;
  if (pos + 8 <= n)
    w = load64(s + pos);
  else if (n >= 8)
    w = load64(s + n - 8) >> ((pos - n + 8) * 8);
  else {
    intnat k;
    w = 0;
    for (k = 0; k < n - pos; k++) w |= (uint64_t)s[pos + k] << (8 * k);
  }
  if (len >= 8) return w;
  return w & ((1ULL << (8 * len)) - 1);
}

static inline uint64_t key_word1(const unsigned char *s, intnat n, intnat pos,
                                 intnat len)
{
  uint64_t tag = (uint64_t)len << 56;
  uint64_t w;
  if (len <= 8) return tag;
  if (pos + 16 <= n)
    w = load64(s + pos + 8);
  else
    w = load64(s + n - 8) >> ((pos + 16 - n) * 8);
  return tag | (w & ((1ULL << (8 * (len - 8))) - 1));
}

/* Bpe.hash_of: OCaml truncates the shifted hash to 63 bits; only 40 bits are
   set after the shift, so the two agree. The back set and front slot byte
   offsets are Bpe.set_at and Bpe.front_at on this one hash. */
static inline uint64_t hash_of(uint64_t k0, uint64_t k1)
{
  return ((k0 ^ (k1 * 0x9E3779B97F4A7C15ULL)) * 0xD6E8FEB86659FD93ULL) >> 24;
}

/* The hit test of Bpe.encode_into on the back table: way 0 then way 1.
   Returns the byte offset of the way, or -1. */
static inline intnat probe(const unsigned char *tbl, intnat set, uint64_t k0,
                           uint64_t k1)
{
  if (load64(tbl + set) == k0 && load64(tbl + set + 8) == k1) return set;
  if (load64(tbl + set + 32) == k0 && load64(tbl + set + 40) == k1)
    return set + 32;
  return -1;
}

/* Bpe.store, in the same order: way 0 shifts into way 1 and the new entry
   fills way 0, values first, then k1, then k0, so a read torn by anything at
   all comes out a miss rather than a wrong hit. */
static void cache_store(unsigned char *tbl, intnat set, uint64_t k0,
                        uint64_t k1, uint64_t v0, uint64_t v1)
{
  store64(tbl + set + 48, load64(tbl + set + 16));
  store64(tbl + set + 56, load64(tbl + set + 24));
  store64(tbl + set + 40, load64(tbl + set + 8));
  store64(tbl + set + 32, load64(tbl + set));
  store64(tbl + set + 16, v0);
  store64(tbl + set + 24, v1);
  store64(tbl + set + 8, k1);
  store64(tbl + set, k0);
}

/* Bpe.promote: a back-table hit is written into its front slot, in
   cache_store's publish order. */
static inline void front_store(unsigned char *front, intnat fslot, uint64_t k0,
                               uint64_t k1, uint64_t v0, uint64_t v1)
{
  store64(front + fslot + 16, v0);
  store64(front + fslot + 24, v1);
  store64(front + fslot + 8, k1);
  store64(front + fslot, k0);
}

/* Bpe.add_cached: a hit's four lanes are stored unconditionally, as
   Ints.add4 does; the returned count advances the id cursor. */
static inline intnat emit_lanes(value *lane, uint64_t v0, uint64_t v1)
{
  lane[0] = Val_long((intnat)(v0 & 0xFFFFFF));
  lane[1] = Val_long((intnat)((v0 >> 32) & 0xFFFFFF));
  lane[2] = Val_long((intnat)(v1 & 0xFFFFFF));
  lane[3] = Val_long((intnat)((v1 >> 32) & 0xFFFFFF));
  return (intnat)((v0 >> 24) & 0xFF);
}

/* The same four lanes as raw int32 stores, for the batch sink. Lane values
   are 24-bit, so the narrowing is exact. */
static inline intnat emit_lanes32(int32_t *lane, uint64_t v0, uint64_t v1)
{
  lane[0] = (int32_t)(v0 & 0xFFFFFF);
  lane[1] = (int32_t)((v0 >> 32) & 0xFFFFFF);
  lane[2] = (int32_t)(v1 & 0xFFFFFF);
  lane[3] = (int32_t)((v1 >> 32) & 0xFFFFFF);
  return (intnat)((v0 >> 24) & 0xFF);
}

/* The short merge — Bpe.Merge_map.find, init_word_bytes and merge_linear
   restricted to a pretoken of at most 15 bytes, on stack arrays. */

/* Merge_map.hash agrees with this even though OCaml wraps the product at 63
   bits and C at 64: the two hashes share bits 0..62, h >> 16 therefore
   shares bits 0..46, and the mask keeps well under bit 47. */
static inline intnat merge_find(const value *keys, const value *vals,
                                intnat mask, intnat key)
{
  uint64_t h64 = (uint64_t)key * 0x1B873593ULL;
  intnat h = (intnat)((h64 ^ (h64 >> 16)) & (uint64_t)mask);
  value vkey = Val_long(key);
  for (;;) {
    value k = keys[h];
    if (k == vkey) return Long_val(vals[h]);
    if (Long_val(k) < 0) return -1;
    h = (h + 1) & mask;
  }
}

#define RANK_NONE ((intnat)(((uintnat)-1) >> 1))

/* Bpe.rank_of: the packed (rank << 21) | new_id, which the minimum scan
   orders by rank and, at equal rank, by position, as HuggingFace merges. */
static inline intnat rank_of(const value *mkeys, const value *mvals,
                             intnat mmask, intnat a, intnat b)
{
  intnat v = merge_find(mkeys, mvals, mmask, (a << 21) | b);
  return v >= 0 ? v : RANK_NONE;
}

/* Bpe.merge_linear's loop and emit over [n] initialized symbols, shared by
   the byte-level and SentencePiece short merges. Writes the resulting ids to
   [out] and returns their count, or -1 when the span must go back to OCaml:
   a symbol stands for other bytes than len_table gives its id — the one case
   Bpe.emit_word would record in the opaque runs, which only OCaml appends
   to. */
static int merge_run(const value *mkeys, const value *mvals, intnat mmask,
                     const value *len_tab, intnat len_n, intnat n,
                     intnat sym_c[15], intnat sym_len[15], intnat sym_prev[15],
                     intnat sym_next[15], int32_t out[15])
{
  intnat rank[15];
  intnat k, cur;
  int m;
  for (k = 0; k + 1 < n; k++)
    rank[k] = rank_of(mkeys, mvals, mmask, sym_c[k], sym_c[k + 1]);
  rank[n - 1] = RANK_NONE;
  for (;;) {
    intnat best = RANK_NONE, bp = -1, gone, next, prev;
    for (k = 0; k < n; k++)
      if (rank[k] < best) {
        best = rank[k];
        bp = k;
      }
    if (bp < 0) break;
    gone = sym_next[bp];
    sym_c[bp] = best & 0x1FFFFF; /* merge_new_id */
    sym_len[bp] += sym_len[gone];
    sym_next[bp] = sym_next[gone];
    sym_len[gone] = 0;
    rank[gone] = RANK_NONE;
    next = sym_next[bp];
    if (next >= 0) {
      sym_prev[next] = bp;
      rank[bp] = rank_of(mkeys, mvals, mmask, sym_c[bp], sym_c[next]);
    } else
      rank[bp] = RANK_NONE;
    prev = sym_prev[bp];
    if (prev >= 0)
      rank[prev] = rank_of(mkeys, mvals, mmask, sym_c[prev], sym_c[bp]);
  }
  m = 0;
  for (cur = 0; cur >= 0; cur = sym_next[cur]) {
    intnat id = sym_c[cur];
    if (id >= len_n || Long_val(len_tab[id]) != sym_len[cur]) return -1;
    out[m++] = (int32_t)id;
  }
  return m;
}

/* Merges the [len] bytes at [s + pos] and writes the resulting ids to [out].
   Returns their count, or -1 when the span must go back to OCaml: a byte has
   no id (unk, fuse and byte-fallback semantics stay there), or the merge-run
   refusal above. Single bytes always account for exactly their byte, so the
   emit check can only fire on a merge result. */
static int merge_short(const value *mkeys, const value *mvals, intnat mmask,
                       const value *byte_ids, const value *len_tab,
                       intnat len_n, const unsigned char *s, intnat pos,
                       intnat len, int32_t out[15])
{
  intnat sym_c[15], sym_len[15], sym_prev[15], sym_next[15];
  intnat k;
  for (k = 0; k < len; k++) {
    intnat id = Long_val(byte_ids[s[pos + k]]);
    if (id < 0) return -1;
    sym_c[k] = id;
    sym_len[k] = 1;
    sym_prev[k] = k - 1;
    sym_next[k] = k + 1 < len ? k + 1 : -1;
  }
  return merge_run(mkeys, mvals, mmask, len_tab, len_n, len, sym_c, sym_len,
                   sym_prev, sym_next, out);
}

/* The entry's body. The caller validates the range and the cursor counts
   once; ids room is checked here per span, a hit storing its four lanes
   unconditionally as Ints.add4 does. [emit32] selects the ids sink — the
   int array of Kernel.byte_level_encode or the int32 Bigarray of
   Kernel.byte_level_encode_ids32 — and is a compile-time constant in each
   wrapper, so its branches cost nothing. */
static value byte_level_encode_impl(value text, value vpos, value vstop,
                                    value vspans, value vids, value vmarks,
                                    value vcursor, value vunicode, value vt,
                                    int emit32)
{
  const unsigned char *s = (const unsigned char *)String_val(text);
  intnat n = (intnat)caml_string_length(text);
  intnat stop = Long_val(vstop);
  unsigned char *spans = Bytes_val(vspans);
  intnat spans_cap = (intnat)(caml_string_length(vspans) / 8);
  /* Field lvalues are volatile for the runtime's sake; these arrays have one
     writer (the thread holding the state claim) and no GC point exists in a
     call, so the qualifier is cast away and the compiler may keep loads in
     registers. */
  value *ids_base = emit32 ? NULL : (value *)&Field(vids, 0);
  int32_t *ids32_base = emit32 ? (int32_t *)Caml_ba_data_val(vids) : NULL;
  intnat ids_cap = emit32 ? Caml_ba_array_val(vids)->dim[0]
                          : (intnat)Wosize_val(vids);
  value *marks_base = (value *)&Field(vmarks, 0);
  unsigned char *cur = Bytes_val(vcursor);

  const unsigned char *lead = Bytes_val(Field(vt, BROT_BL_LEAD));
  unsigned char *front = Bytes_val(Field(vt, BROT_BL_FRONT));
  intnat front_mask = Long_val(Field(vt, BROT_BL_FRONT_MASK));
  unsigned char *cache = Bytes_val(Field(vt, BROT_BL_CACHE));
  intnat cache_mask = Long_val(Field(vt, BROT_BL_CACHE_MASK));
  const value *byte_ids = (const value *)&Field(Field(vt, BROT_BL_BYTE_IDS), 0);
  const value *mkeys = (const value *)&Field(Field(vt, BROT_BL_MERGE_KEYS), 0);
  const value *mvals =
      (const value *)&Field(Field(vt, BROT_BL_MERGE_VALUES), 0);
  intnat mmask = Long_val(Field(vt, BROT_BL_MERGE_MASK));
  const value *len_tab =
      (const value *)&Field(Field(vt, BROT_BL_LEN_TABLE), 0);
  intnat len_n = (intnat)Wosize_val(Field(vt, BROT_BL_LEN_TABLE));
  int can_merge = Bool_val(Field(vt, BROT_BL_MERGE));

  intnat nspans = (intnat)load64(cur + CUR_SPANS);
  intnat nids = (intnat)load64(cur + CUR_IDS);
  intnat nmarks = (intnat)load64(cur + CUR_MARKS);

  walker w;
  mask_state ms;
  intnat i = Long_val(vpos);
  intnat resume;
  int reason;

  w.s = s;
  w.stop = stop;
  w.lead = lead;
  w.uni = Bytes_val(vunicode);
  w.uni_len = (intnat)caml_string_length(vunicode);
  w.cp = 0;
  mask_init(&ms, i);

  /* The loop is software-pipelined over the cache line: span k's walk, keys
     and set are computed and the set's line prefetched one iteration ahead,
     so the walk of span k+1 runs while span k's line is in flight; span k's
     probes, merge and stores — and its room checks, in the reference's order —
     resolve after that walk. The front table too is probed at resolve
     time, never ahead, so every probe sees every earlier span's stores
     and the tables stay bit for bit the reference's. The observable exit
     states are the reference's exactly: a span whose resolve exits
     discards the span walked after it,
     which the reference would never have walked, and a walked span's Class
     hand-back is reported only once every span before it has resolved and
     the span room it would need has been checked. */
  {
    int have = 0;
    intnat p_i = 0, p_e = 0, p_len = 0, p_set = -1, p_fslot = 0;
    uint64_t p_k0 = 0, p_k1 = 0;
    for (;;) {
      intnat e = -1, len = 0, set = -1, fslot = 0;
      uint64_t k0 = 0, k1 = 0;
      int walked = 0;
      if (i < stop) {
        e = mask_next_span(&ms, &w, i);
        walked = 1;
        if (e >= 0) {
          len = e - i;
          if (len <= 15 && cache_mask >= 0) {
            uint64_t hh;
            k0 = key_word0(s, n, i, len);
            k1 = key_word1(s, n, i, len);
            hh = hash_of(k0, k1);
            set = (intnat)((hh & (uint64_t)cache_mask) << 6);
            fslot = (intnat)((hh & (uint64_t)front_mask) << 5);
            BROT_PREFETCH(cache + set);
          }
        }
      }
      if (have) {
        have = 0;
        if (nspans == spans_cap) {
          reason = BROT_SPANS_FULL;
          resume = p_i;
          goto out;
        }
        if (ids_cap - nids < (p_len < 4 ? 4 : p_len)) {
          reason = BROT_IDS_FULL;
          resume = p_i;
          goto out;
        }
        if (p_len > 15) goto hand_back;
        if (p_set >= 0) {
          if (load64(front + p_fslot) == p_k0
              && load64(front + p_fslot + 8) == p_k1) {
            uint64_t v0 = load64(front + p_fslot + 16);
            uint64_t v1 = load64(front + p_fslot + 24);
            nids += emit32 ? emit_lanes32(ids32_base + nids, v0, v1)
                           : emit_lanes(ids_base + nids, v0, v1);
            goto emitted;
          }
          intnat way = probe(cache, p_set, p_k0, p_k1);
          if (way >= 0) {
            uint64_t v0 = load64(cache + way + 16);
            uint64_t v1 = load64(cache + way + 24);
            front_store(front, p_fslot, p_k0, p_k1, v0, v1);
            nids += emit32 ? emit_lanes32(ids32_base + nids, v0, v1)
                           : emit_lanes(ids_base + nids, v0, v1);
            goto emitted;
          }
        }
        if (!can_merge) goto hand_back;
        {
          int32_t ids15[15];
          int k;
          int m = merge_short(mkeys, mvals, mmask, byte_ids, len_tab, len_n,
                              s, p_i, p_len, ids15);
          if (m < 0) goto hand_back;
          if (emit32)
            for (k = 0; k < m; k++) ids32_base[nids + k] = ids15[k];
          else
            for (k = 0; k < m; k++)
              ids_base[nids + k] = Val_long((intnat)ids15[k]);
          nids += m;
          if (p_set >= 0 && m <= 4) {
            int fits = 1;
            for (k = 0; k < m; k++)
              if (ids15[k] >= 1 << 24) fits = 0;
            if (fits) {
              /* Bpe.value_lo/value_hi: two 24-bit lanes per word, the count
                 in bits 24..31 of the first, absent lanes zero. */
              uint64_t v0 = (uint64_t)ids15[0] | ((uint64_t)m << 24)
                            | (m > 1 ? (uint64_t)ids15[1] << 32 : 0);
              uint64_t v1 = (m > 2 ? (uint64_t)ids15[2] : 0)
                            | (m > 3 ? (uint64_t)ids15[3] << 32 : 0);
              cache_store(cache, p_set, p_k0, p_k1, v0, v1);
            }
          }
          goto emitted;
        }
      emitted:
        /* Spans.write's word: the start in the low 32 bits, the stop in the
           high; the mark is the id count after the span. */
        store64(spans + 8 * nspans, ((uint64_t)p_e << 32) | (uint64_t)p_i);
        marks_base[nmarks] = Val_long(nids);
        nspans++;
        nmarks++;
      }
      if (!walked) { /* i = stop and every span resolved */
        reason = BROT_DONE;
        resume = i;
        goto out;
      }
      if (e < 0) {
        if (nspans == spans_cap) {
          reason = BROT_SPANS_FULL;
          resume = i;
          goto out;
        }
        store64(cur + CUR_CP, (uint64_t)w.cp);
        reason = BROT_CLASS;
        resume = i;
        goto out;
      }
      p_i = i;
      p_e = e;
      p_len = len;
      p_k0 = k0;
      p_k1 = k1;
      p_set = set;
      p_fslot = fslot;
      have = 1;
      i = e;
    }
  hand_back:
    /* The span is written and published but its ids and mark are not: the
       driver encodes it in OCaml, appends its mark and resumes after it. */
    store64(spans + 8 * nspans, ((uint64_t)p_e << 32) | (uint64_t)p_i);
    nspans++;
    reason = BROT_ENCODE;
    resume = p_i;
  }
out:
  store64(cur + CUR_SPANS, (uint64_t)nspans);
  store64(cur + CUR_IDS, (uint64_t)nids);
  store64(cur + CUR_MARKS, (uint64_t)nmarks);
  store64(cur + CUR_RESUME, (uint64_t)resume);
  return Val_int(reason);
}

CAMLprim value brot_byte_level_encode(value text, value vpos, value vstop,
                                      value vspans, value vids, value vmarks,
                                      value vcursor, value vunicode, value vt)
{
  return byte_level_encode_impl(text, vpos, vstop, vspans, vids, vmarks,
                                vcursor, vunicode, vt, 0);
}

CAMLprim value brot_byte_level_encode_byte(value *argv, int argn)
{
  (void)argn;
  return brot_byte_level_encode(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6], argv[7], argv[8]);
}

CAMLprim value brot_byte_level_encode_ids32(value text, value vpos,
                                            value vstop, value vspans,
                                            value vids, value vmarks,
                                            value vcursor, value vunicode,
                                            value vt)
{
  return byte_level_encode_impl(text, vpos, vstop, vspans, vids, vmarks,
                                vcursor, vunicode, vt, 1);
}

CAMLprim value brot_byte_level_encode_ids32_byte(value *argv, int argn)
{
  (void)argn;
  return brot_byte_level_encode_ids32(argv[0], argv[1], argv[2], argv[3],
                                      argv[4], argv[5], argv[6], argv[7],
                                      argv[8]);
}

/* The fused SentencePiece kernel: Bpe.encode_into's unit walker, the same
   cache probe, and the short merge initialized per character rather than per
   byte. No Unicode classes come into it — the only boundary candidates are
   the three ▁ bytes and the eight punctuation split bytes — so the walk
   never fails and only Done, Ids_full and Encode are returned. An Encode
   hand-back carries the unit through the cursor alone: its start in
   CUR_RESUME, its end in the word Class code points use, and no span or
   mark buffer is touched — the caller records the whole stretch as one
   span, exactly as the OCaml walker's caller does. */

/* Kernel.sp scan values (kernel.mli pins them): 1 + slot for a punctuation
   split byte, this for the 0xE2 lead of ▁. */
#define SP_SCAN_MARK 9

/* Bpe.encode_into's unit cut, line for line: the end of the unit opening at
   [u]. [anchor] is the position this call's walk began at — the stretch
   start or a unit boundary by the resume contract — standing in for the
   walker's [pos] in the ▁-run look-behind guard. The change is
   unobservable: within [anchor, anchor+3) the look-behind window always
   covers the boundary's opening byte, which is an 0xE2 lead or an ASCII
   punct byte, never a mark continuation, so the guard resolves as the
   walker's would. */
static intnat sp_next_unit(const unsigned char *s, intnat stop, intnat anchor,
                           const unsigned char *scan, const value *safe,
                           intnat u)
{
  intnat p = u;
  while (p < stop) {
    intnat k = scan[s[p]];
    if (k == 0) {
      p++;
      continue;
    }
    if (k == SP_SCAN_MARK) {
      if (p + 3 <= stop && s[p + 1] == 0x96 && s[p + 2] == 0x81) {
        if (p > u
            && !(p >= anchor + 3 && s[p - 3] == 0xE2 && s[p - 2] == 0x96
                 && s[p - 1] == 0x81))
          return p;
        p += 3;
      } else
        p++;
    } else {
      if (p > u && Bool_val(safe[((k - 1) << 8) | s[p - 1]])) return p;
      p++;
    }
  }
  return stop;
}

/* Bpe.utf8_byte_len. */
static inline intnat utf8_len(intnat b)
{
  if ((b & 0x80) == 0) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 1;
}

/* Bpe.pack_char_key. */
static inline intnat pack_char_key(const unsigned char *p, intnat blen)
{
  switch (blen) {
  case 1: return p[0];
  case 2: return ((intnat)p[0] << 8) | p[1];
  case 3: return ((intnat)p[0] << 16) | ((intnat)p[1] << 8) | p[2];
  default:
    return ((intnat)p[0] << 24) | ((intnat)p[1] << 16) | ((intnat)p[2] << 8)
           | p[3];
  }
}

/* The short merge over a unit — Bpe.init_word_fast restricted to characters
   with a direct piece, then the shared merge run. Returns the id count, or
   -1 when the unit must go back to OCaml: a character without a piece (unk,
   fuse and byte-fallback semantics stay there, a truncated character packs
   no whole character's key and lands here too), or the merge-run refusal.
   The character map is probed with merge_find: the two maps share the
   Merge_map layout. */
static int sp_merge_short(const value *mkeys, const value *mvals, intnat mmask,
                          const value *ascii_ids, const value *ckeys,
                          const value *cvals, intnat cmask,
                          const value *len_tab, intnat len_n,
                          const unsigned char *s, intnat pos, intnat len,
                          int32_t out[15])
{
  intnat sym_c[15], sym_len[15], sym_prev[15], sym_next[15];
  intnat n = 0, k = 0;
  while (k < len) {
    intnat b = s[pos + k], id, blen;
    if (b < 128) {
      blen = 1;
      id = Long_val(ascii_ids[b]);
    } else {
      blen = utf8_len(b);
      if (blen > len - k) blen = len - k;
      id = merge_find(ckeys, cvals, cmask, pack_char_key(s + pos + k, blen));
    }
    if (id < 0) return -1;
    sym_c[n] = id;
    sym_len[n] = blen;
    sym_prev[n] = n - 1;
    sym_next[n] = n + 1;
    n++;
    k += blen;
  }
  sym_next[n - 1] = -1;
  return merge_run(mkeys, mvals, mmask, len_tab, len_n, n, sym_c, sym_len,
                   sym_prev, sym_next, out);
}

/* The entry's body. The caller validates the range and the cursor's id
   count once; ids room is checked here per unit, a hit storing its four
   lanes unconditionally as Ints.add4 does. [emit32] selects the ids sink as
   in the byte-level body. The loop is software-pipelined exactly
   as the byte-level entry's: unit k+1 is walked, keyed and its set
   prefetched while unit k's line is in flight; unit k's probes, merge and
   stores resolve after that walk, so every probe sees every earlier unit's
   stores and the tables stay bit for bit the reference's. An exit at unit
   k's resolve discards the unit walked after it, which the reference would
   never have walked. */
static value sp_encode_impl(value text, value vpos, value vstop, value vids,
                            value vcursor, value vt, int emit32)
{
  const unsigned char *s = (const unsigned char *)String_val(text);
  intnat n = (intnat)caml_string_length(text);
  intnat stop = Long_val(vstop);
  /* One writer, no GC point: the volatile qualifier is cast away as in the
     byte-level entry. */
  value *ids_base = emit32 ? NULL : (value *)&Field(vids, 0);
  int32_t *ids32_base = emit32 ? (int32_t *)Caml_ba_data_val(vids) : NULL;
  intnat ids_cap = emit32 ? Caml_ba_array_val(vids)->dim[0]
                          : (intnat)Wosize_val(vids);
  unsigned char *cur = Bytes_val(vcursor);

  unsigned char *front = Bytes_val(Field(vt, BROT_SP_FRONT));
  intnat front_mask = Long_val(Field(vt, BROT_SP_FRONT_MASK));
  unsigned char *cache = Bytes_val(Field(vt, BROT_SP_CACHE));
  intnat cache_mask = Long_val(Field(vt, BROT_SP_CACHE_MASK));
  const value *mkeys = (const value *)&Field(Field(vt, BROT_SP_MERGE_KEYS), 0);
  const value *mvals =
      (const value *)&Field(Field(vt, BROT_SP_MERGE_VALUES), 0);
  intnat mmask = Long_val(Field(vt, BROT_SP_MERGE_MASK));
  const value *len_tab =
      (const value *)&Field(Field(vt, BROT_SP_LEN_TABLE), 0);
  intnat len_n = (intnat)Wosize_val(Field(vt, BROT_SP_LEN_TABLE));
  const value *ascii_ids =
      (const value *)&Field(Field(vt, BROT_SP_ASCII_IDS), 0);
  const value *ckeys = (const value *)&Field(Field(vt, BROT_SP_CHAR_KEYS), 0);
  const value *cvals =
      (const value *)&Field(Field(vt, BROT_SP_CHAR_VALUES), 0);
  intnat cmask = Long_val(Field(vt, BROT_SP_CHAR_MASK));
  const unsigned char *scan = Bytes_val(Field(vt, BROT_SP_SCAN));
  const value *safe = (const value *)&Field(Field(vt, BROT_SP_PUNCT_SAFE), 0);

  intnat nids = (intnat)load64(cur + CUR_IDS);
  intnat anchor = Long_val(vpos);
  intnat i = anchor;
  intnat resume;
  int reason;

  {
    int have = 0;
    intnat p_i = 0, p_e = 0, p_len = 0, p_set = -1, p_fslot = 0;
    uint64_t p_k0 = 0, p_k1 = 0;
    for (;;) {
      intnat e = -1, len = 0, set = -1, fslot = 0;
      uint64_t k0 = 0, k1 = 0;
      int walked = 0;
      if (i < stop) {
        e = sp_next_unit(s, stop, anchor, scan, safe, i);
        walked = 1;
        len = e - i;
        if (len <= 15 && cache_mask >= 0) {
          uint64_t hh;
          k0 = key_word0(s, n, i, len);
          k1 = key_word1(s, n, i, len);
          hh = hash_of(k0, k1);
          set = (intnat)((hh & (uint64_t)cache_mask) << 6);
          fslot = (intnat)((hh & (uint64_t)front_mask) << 5);
          BROT_PREFETCH(cache + set);
        }
      }
      if (have) {
        have = 0;
        if (p_len > 15) goto hand_back;
        if (ids_cap - nids < (p_len < 4 ? 4 : p_len)) {
          reason = BROT_IDS_FULL;
          resume = p_i;
          goto out;
        }
        if (p_set >= 0) {
          if (load64(front + p_fslot) == p_k0
              && load64(front + p_fslot + 8) == p_k1) {
            uint64_t v0 = load64(front + p_fslot + 16);
            uint64_t v1 = load64(front + p_fslot + 24);
            nids += emit32 ? emit_lanes32(ids32_base + nids, v0, v1)
                           : emit_lanes(ids_base + nids, v0, v1);
            goto emitted;
          }
          intnat way = probe(cache, p_set, p_k0, p_k1);
          if (way >= 0) {
            uint64_t v0 = load64(cache + way + 16);
            uint64_t v1 = load64(cache + way + 24);
            front_store(front, p_fslot, p_k0, p_k1, v0, v1);
            nids += emit32 ? emit_lanes32(ids32_base + nids, v0, v1)
                           : emit_lanes(ids_base + nids, v0, v1);
            goto emitted;
          }
        }
        {
          int32_t ids15[15];
          int k;
          int m = sp_merge_short(mkeys, mvals, mmask, ascii_ids, ckeys, cvals,
                                 cmask, len_tab, len_n, s, p_i, p_len, ids15);
          if (m < 0) goto hand_back;
          if (emit32)
            for (k = 0; k < m; k++) ids32_base[nids + k] = ids15[k];
          else
            for (k = 0; k < m; k++)
              ids_base[nids + k] = Val_long((intnat)ids15[k]);
          nids += m;
          if (p_set >= 0 && m <= 4) {
            int fits = 1;
            for (k = 0; k < m; k++)
              if (ids15[k] >= 1 << 24) fits = 0;
            if (fits) {
              /* Bpe.value_lo/value_hi: two 24-bit lanes per word, the count
                 in bits 24..31 of the first, absent lanes zero. */
              uint64_t v0 = (uint64_t)ids15[0] | ((uint64_t)m << 24)
                            | (m > 1 ? (uint64_t)ids15[1] << 32 : 0);
              uint64_t v1 = (m > 2 ? (uint64_t)ids15[2] : 0)
                            | (m > 3 ? (uint64_t)ids15[3] << 32 : 0);
              cache_store(cache, p_set, p_k0, p_k1, v0, v1);
            }
          }
        }
      emitted:;
      }
      if (!walked) { /* i = stop and every unit resolved */
        reason = BROT_DONE;
        resume = i;
        goto out;
      }
      p_i = i;
      p_e = e;
      p_len = len;
      p_k0 = k0;
      p_k1 = k1;
      p_set = set;
      p_fslot = fslot;
      have = 1;
      i = e;
    }
  hand_back:
    /* Nothing was appended for the unit: the driver encodes
       [p_i, p_e) in OCaml and resumes at its end. */
    store64(cur + CUR_CP, (uint64_t)p_e);
    reason = BROT_ENCODE;
    resume = p_i;
  }
out:
  store64(cur + CUR_IDS, (uint64_t)nids);
  store64(cur + CUR_RESUME, (uint64_t)resume);
  return Val_int(reason);
}

CAMLprim value brot_sp_encode(value text, value vpos, value vstop, value vids,
                              value vcursor, value vt)
{
  return sp_encode_impl(text, vpos, vstop, vids, vcursor, vt, 0);
}

CAMLprim value brot_sp_encode_byte(value *argv, int argn)
{
  (void)argn;
  return brot_sp_encode(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value brot_sp_encode_ids32(value text, value vpos, value vstop,
                                    value vids, value vcursor, value vt)
{
  return sp_encode_impl(text, vpos, vstop, vids, vcursor, vt, 1);
}

CAMLprim value brot_sp_encode_ids32_byte(value *argv, int argn)
{
  (void)argn;
  return brot_sp_encode_ids32(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5]);
}
