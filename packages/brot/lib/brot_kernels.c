/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The fused kernels: the byte-level one — the GPT-2, cl100k or o200k
   walker, the pretoken cache probe, ignore_merges' whole-word lookup and the
   short byte-level BPE merge over a range of text, one call per chunk of
   spans — and the SentencePiece one at the end of this file, which cuts
   ▁/punctuation word units and feeds them through the same probe and merge
   machinery. Everything rare is handed back to OCaml with a resume position:
   an unclassified code point, a pretoken over 15 bytes, a byte or character
   without a vocabulary id, and a merge result or whole word that stands for
   other bytes than its entry's. The OCaml implementations in pre_tokenizer.ml,
   char_class.ml, bpe.ml and pretoken_cache.ml are the reference; every
   function here mirrors
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
#elif defined(__x86_64__) || defined(_M_X64)
/* SSE2 is part of x86-64 itself: every compiler for it has the intrinsics
   and every CPU runs them. */
#define BROT_MASK_SSE2 1
#include <emmintrin.h>
#endif

/* ABI pins, mirrored in kernel.ml — change neither without the other. */

/* Kernel.reason constructor order. */
enum { BROT_DONE, BROT_SPANS_FULL, BROT_IDS_FULL, BROT_CLASS, BROT_ENCODE };

/* Kernel.byte_level field order. */
enum {
  BROT_BL_LEAD,
  BROT_BL_CACHE,
  BROT_BL_CACHE_BASE,
  BROT_BL_CACHE_MASK,
  BROT_BL_BYTE_IDS,
  BROT_BL_MERGE_KEYS,
  BROT_BL_MERGE_VALUES,
  BROT_BL_MERGE_MASK,
  BROT_BL_LEN_TABLE,
  BROT_BL_WORDS,
  BROT_BL_WORDS_MASK
};

/* Kernel.sp field order. */
enum {
  BROT_SP_CACHE,
  BROT_SP_CACHE_BASE,
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
  BROT_SP_PUNCT_SAFE,
  BROT_SP_FALLBACK_IDS,
  BROT_SP_MARKER
};

/* Kernel.cursor byte offsets. */
enum { CUR_SPANS = 0, CUR_IDS = 8, CUR_MARKS = 16, CUR_RESUME = 24,
       CUR_CP = 32, CUR_STAGED = 40, CUR_NEXT = 48, CUR_STAGE_STOP = 56,
       CUR_ADDED = 64, CUR_STAGE = 72 };

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

/* Eight bytes in little-endian order, the first lowest: a cache key holds the
   same bytes on every machine, as Pretoken_cache's keys do. */
static inline uint64_t load64le(const void *p)
{
  uint64_t w = load64(p);
#ifdef ARCH_BIG_ENDIAN
  w = __builtin_bswap64(w);
#endif
  return w;
}

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int ctz64(uint64_t x)
{
  unsigned long i;
  _BitScanForward64(&i, x);
  return (int)i;
}
static inline int clz64(uint64_t x)
{
  unsigned long i;
  _BitScanReverse64(&i, x);
  return 63 - (int)i;
}
#else
static inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
static inline int clz64(uint64_t x) { return __builtin_clzll(x); }
#endif

/* The prefetch is advisory: GCC/Clang emit it, other compilers drop it.
   So is the forced inlining that gives each entry its own copy of a body
   specialized on its constant flags, and keeps the unit walk in the entry's
   loop. */
#if defined(__GNUC__) || defined(__clang__)
#define BROT_PREFETCH(p) __builtin_prefetch(p)
#define BROT_INLINE __attribute__((always_inline)) inline
#else
#define BROT_PREFETCH(p) ((void)(p))
#define BROT_INLINE inline
#endif

/* Hides how a flag was computed from the optimizer, which otherwise turns a
   select on it, or a test of flags it combines, back into a branch on each
   comparison — and those are the comparisons whose outcome no predictor
   learns. GCC/Clang only: elsewhere the code stays correct, only less sure
   to be branch-free. */
#if defined(__GNUC__) || defined(__clang__)
#define BROT_OPAQUE(x) __asm__("" : "+r"(x))
#else
#define BROT_OPAQUE(x) ((void)0)
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

/* The cl100k walker — Pre_tokenizer.fill_cl100k, line for line. [digits] is
   the longest group of numbers and [marks] makes the combining marks letters.
   Every function is its OCaml namesake, or -1 propagating a class
   hand-back. */

#define K_STRAY 4
#define CL100K_STRAY ((K_STRAY << 3) | 1)

/* Char_class's mark bit in a byte of the Unicode table. */
#define PROP_MARK 16

/* Char_class.at as Pre_tokenizer.decodes refines it, beyond ASCII: the code
   point at [i] with its byte length in [*len], or -1 for one stray byte —
   the sequences a decoder is to refuse (overlong, surrogates, above
   U+10FFFF) and the ones Char_class.at refuses itself. */
static inline intnat strict_decode(walker *w, intnat i, intnat *len)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat c = s[i];
  if (c < 0xC2) return -1;
  if (c < 0xE0) {
    intnat b1;
    if (i + 1 >= stop) return -1;
    b1 = s[i + 1];
    if ((b1 & 0xC0) != 0x80) return -1;
    *len = 2;
    return ((c & 0x1F) << 6) | (b1 & 0x3F);
  }
  if (c < 0xF0) {
    intnat b1, b2;
    if (i + 2 >= stop) return -1;
    b1 = s[i + 1];
    b2 = s[i + 2];
    if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) return -1;
    if ((c == 0xE0 && b1 < 0xA0) || (c == 0xED && b1 >= 0xA0)) return -1;
    *len = 3;
    return ((c & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
  }
  if (c < 0xF5) {
    intnat b1, b2, b3;
    if (i + 3 >= stop) return -1;
    b1 = s[i + 1];
    b2 = s[i + 2];
    b3 = s[i + 3];
    if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80)
      return -1;
    if ((c == 0xF0 && b1 < 0x90) || (c == 0xF4 && b1 >= 0x90)) return -1;
    *len = 4;
    return ((c & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6)
           | (b3 & 0x3F);
  }
  return -1;
}

/* Pre_tokenizer.kind_beyond_ascii. */
static intnat cl100k_kind_beyond_ascii(walker *w, intnat i, int marks)
{
  intnat len = 1;
  intnat cp = strict_decode(w, i, &len);
  if (cp < 0) return CL100K_STRAY;
  if (cp - 128 < w->uni_len && w->uni[cp - 128] != 0) {
    intnat props = w->uni[cp - 128];
    intnat cat = (marks && (props & PROP_MARK)) ? CL_LETTER : (props & 3);
    return (cat << 3) | len;
  }
  w->cp = cp;
  return -1;
}

/* Pre_tokenizer.kind. */
static inline intnat cl100k_kind(walker *w, intnat i, int marks)
{
  intnat c = w->s[i];
  if (c < 0x80) return (ascii_cat(w->lead, c) << 3) | 1;
  return cl100k_kind_beyond_ascii(w, i, marks);
}

static intnat cl100k_letter_run(walker *w, intnat i, int marks)
{
  intnat stop = w->stop;
  intnat j = letters_swar(w->s, i, stop);
  while (j < stop) {
    intnat k = cl100k_kind(w, j, marks);
    if (k < 0) return -1;
    if ((k >> 3) != CL_LETTER) break;
    j += k & 7;
  }
  return j;
}

static intnat cl100k_punctuation_run(walker *w, intnat i, int marks)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat j = i;
  while (j < stop) {
    intnat k = cl100k_kind(w, j, marks);
    if (k < 0) return -1;
    if ((k >> 3) != CL_OTHER) break;
    j += k & 7;
  }
  while (j < stop && (s[j] == '\n' || s[j] == '\r')) j++;
  return j;
}

static intnat cl100k_whitespace(walker *w, intnat i)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat j = i, last = i, newline = -1;
  while (j < stop) {
    intnat c = s[j];
    if (c == '\n' || c == '\r') {
      last = j;
      j++;
      newline = j;
    } else {
      intnat k = cl100k_kind(w, j, 0);
      if (k < 0) return -1;
      if ((k >> 3) != CL_WS) break;
      last = j;
      j += k & 7;
    }
  }
  if (newline >= 0) return newline;
  if (j == stop) return j;
  return last > i ? last : j;
}

static inline intnat ascii_lower(intnat c)
{
  return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

static intnat cl100k_contraction(const unsigned char *s, intnat i, intnat stop)
{
  intnat c1, c2;
  if (i + 1 >= stop) return i;
  c1 = ascii_lower(s[i + 1]);
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
  if (i + 2 >= stop) return i;
  c2 = ascii_lower(s[i + 2]);
  if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e')
      || (c1 == 'l' && c2 == 'l') || (c1 == 0xC5 && c2 == 0xBF))
    return i + 3;
  return i;
}

static intnat cl100k_number_run(walker *w, intnat i, intnat digits, int marks)
{
  intnat stop = w->stop;
  intnat j = i, left = digits - 1;
  while (left > 0 && j < stop) {
    intnat k = cl100k_kind(w, j, marks);
    if (k < 0) return -1;
    if ((k >> 3) != CL_NUM) break;
    j += k & 7;
    left--;
  }
  return j;
}

/* One span of Pre_tokenizer.fill_cl100k: the end of the span opening at [i],
   or -1 with [w->cp] set. */
static intnat cl100k_next_span(walker *w, intnat i, intnat digits, int marks)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat k = cl100k_kind(w, i, marks);
  intnat next, category, after = 0;
  if (k < 0) return -1;
  next = i + (k & 7);
  category = k >> 3;
  if (category == CL_LETTER) return cl100k_letter_run(w, next, marks);
  if (category == K_STRAY) {
    intnat j = next;
    while (j < stop) {
      intnat kj = cl100k_kind(w, j, marks);
      if (kj < 0) return -1;
      if ((kj >> 3) != K_STRAY) break;
      j++;
    }
    return j;
  }
  if (category == CL_NUM) return cl100k_number_run(w, next, digits, marks);
  if (next < stop) {
    after = cl100k_kind(w, next, marks);
    if (after < 0) return -1;
  }
  if (category == CL_WS) {
    intnat c = s[i];
    if (c != '\n' && c != '\r' && next < stop && (after >> 3) == CL_LETTER)
      return cl100k_letter_run(w, next, marks);
    if (c == ' ' && next < stop && (after >> 3) == CL_OTHER)
      return cl100k_punctuation_run(w, next, marks);
    return cl100k_whitespace(w, i);
  }
  if (s[i] == '\'') {
    intnat e = cl100k_contraction(s, i, stop);
    if (e > i) return e;
  }
  if (next < stop && (after >> 3) == CL_LETTER)
    return cl100k_letter_run(w, next, marks);
  return cl100k_punctuation_run(w, next, marks);
}

/* The o200k walker — Pre_tokenizer.fill_o200k, line for line. [digits] is
   the longest group of numbers and [contractions] whether a word takes one.
   Every function is its OCaml namesake, or -1 propagating a class
   hand-back. */

#define K_UPPER 5
#define K_LOWER 6
#define K_MARK 7

/* Char_class's case bits in a byte of the Unicode table. */
#define PROP_UPPER 32
#define PROP_LOWER 64

#define IN_BOTH(k) ((k) == CL_LETTER || (k) == K_MARK)

/* Pre_tokenizer.o200k_kind_of on a byte of the Unicode table. */
static inline intnat o200k_kind_of(intnat props)
{
  intnat cat = props & 3;
  if (cat == CL_LETTER) {
    if (props & PROP_UPPER) return K_UPPER;
    if (props & PROP_LOWER) return K_LOWER;
    return CL_LETTER;
  }
  if (props & PROP_MARK) return K_MARK;
  return cat;
}

/* Pre_tokenizer.o200k_ascii, from the lead table and the ASCII letters. */
static inline intnat o200k_ascii_kind(const unsigned char *lead, intnat c)
{
  if (c >= 'a' && c <= 'z') return K_LOWER;
  if (c >= 'A' && c <= 'Z') return K_UPPER;
  return ascii_cat(lead, c);
}

static intnat o200k_kind_beyond_ascii(walker *w, intnat i)
{
  intnat len = 1;
  intnat cp = strict_decode(w, i, &len);
  if (cp < 0) return CL100K_STRAY;
  if (cp - 128 < w->uni_len && w->uni[cp - 128] != 0)
    return (o200k_kind_of(w->uni[cp - 128]) << 3) | len;
  w->cp = cp;
  return -1;
}

static inline intnat o200k_kind(walker *w, intnat i)
{
  intnat c = w->s[i];
  if (c < 0x80) return (o200k_ascii_kind(w->lead, c) << 3) | 1;
  return o200k_kind_beyond_ascii(w, i);
}

static intnat o200k_lower_run(walker *w, intnat i)
{
  intnat stop = w->stop;
  intnat j = i;
  while (j < stop) {
    intnat k = o200k_kind(w, j), c;
    if (k < 0) return -1;
    c = k >> 3;
    if (c != K_LOWER && !IN_BOTH(c)) break;
    j += k & 7;
  }
  return j;
}

/* -1 is the class hand-back here and -2 no match, where the OCaml namesake
   returns -1. */
static intnat o200k_word(walker *w, intnat i, int contractions, int a2)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat u = i, last_both = -1, e;
  int lower_follows = 0;
  while (u < stop) {
    intnat k = o200k_kind(w, u), c;
    if (k < 0) return -1;
    c = k >> 3;
    if (c == K_UPPER)
      u += k & 7;
    else if (IN_BOTH(c)) {
      u += k & 7;
      last_both = u;
    } else {
      lower_follows = c == K_LOWER;
      break;
    }
  }
  if (lower_follows) {
    e = o200k_lower_run(w, u);
    if (e < 0) return -1;
  } else if (last_both >= 0)
    e = last_both;
  else if (a2 && u > i)
    e = u;
  else
    return -2;
  if (contractions && e < stop && s[e] == '\'')
    return cl100k_contraction(s, e, stop);
  return e;
}

static intnat o200k_punctuation_run(walker *w, intnat i)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat j = i;
  while (j < stop) {
    intnat k = o200k_kind(w, j), c;
    if (k < 0) return -1;
    c = k >> 3;
    if (c != CL_OTHER && c != K_MARK) break;
    j += k & 7;
  }
  while (j < stop && (s[j] == '\n' || s[j] == '\r' || s[j] == '/')) j++;
  return j;
}

/* One span of Pre_tokenizer.fill_o200k: the end of the span opening at [i],
   or -1 with [w->cp] set. */
static intnat o200k_next_span(walker *w, intnat i, intnat digits,
                              int contractions)
{
  const unsigned char *s = w->s;
  intnat stop = w->stop;
  intnat k = o200k_kind(w, i);
  intnat next, category, e, c;
  if (k < 0) return -1;
  next = i + (k & 7);
  category = k >> 3;
  if (category == K_UPPER || category == K_LOWER || category == CL_LETTER)
    return o200k_word(w, i, contractions, 1);
  if (category == K_MARK) {
    e = o200k_word(w, next, contractions, 0);
    if (e != -2) return e;
    return o200k_word(w, i, contractions, 1);
  }
  if (category == CL_NUM) return cl100k_number_run(w, next, digits, 0);
  if (category == K_STRAY) {
    intnat j = next;
    while (j < stop) {
      intnat kj = o200k_kind(w, j);
      if (kj < 0) return -1;
      if ((kj >> 3) != K_STRAY) break;
      j++;
    }
    return j;
  }
  c = s[i];
  if (c != '\n' && c != '\r') {
    e = o200k_word(w, next, contractions, 1);
    if (e != -2) return e;
  }
  if (category == CL_OTHER) return o200k_punctuation_run(w, next);
  if (c == ' ' && next < stop) {
    intnat after = o200k_kind(w, next);
    if (after < 0) return -1;
    if ((after >> 3) == CL_OTHER || (after >> 3) == K_MARK)
      return o200k_punctuation_run(w, next);
  }
  return cl100k_whitespace(w, i);
}

/* The mask scanner. Instead of walking span by span, each full 64-byte
   batch of the range is classified once into per-byte class bitmasks, the
   byte-level boundary rules are evaluated on all 64 positions at once as
   u64 shift algebra, and the walk takes every "a span starts here" bit of
   the batch at once. Bits the masks cannot decide are a bad zone that
   [next_span] re-derives — the scalar walker above stays the ground truth
   and runs the last partial batch outright, so no byte at or past [stop] is
   read; the one byte of lookahead a whitespace split needs is read only
   below [stop]. The GPT-2 masks classify the characters beyond ASCII by the
   Unicode table and leave bad only the ones they cannot (see
   [classify_uni]); the cl100k masks leave non-ASCII bytes bad, with their
   neighbours. Other bad zones are: a whitespace character at the batch edge
   whose lookahead is non-ASCII; an apostrophe whose bits are already bad,
   plus the two suffix bytes its contraction could absorb (the contraction
   is the one rule that reaches two bytes ahead); and an apostrophe opening a
   span less than three bytes before the batch edge, whose moved boundary
   the next batch could not see. */

/* Class bitmasks of one 64-byte batch: bit k describes byte [scan + k]. */
typedef struct {
  uint64_t letter; /* ASCII letters */
  uint64_t digit;  /* ASCII digits */
  uint64_t space;  /* 0x20 */
  uint64_t ws;     /* 0x09..0x0D and 0x20 */
  uint64_t hi;     /* >= 0x80 */
  uint64_t apo;    /* 0x27 */
  uint64_t nl;     /* 0x0A and 0x0D, when asked for */
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

static BROT_INLINE void batch_classify(const unsigned char *p,
                                        batch_classes *c, int newlines)
{
  uint8x16_t l[4], d[4], sp[4], ws[4], hi[4], ap[4], nl[4];
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
    nl[k] = vorrq_u8(vceqq_u8(v, vdupq_n_u8('\n')),
                     vceqq_u8(v, vdupq_n_u8('\r')));
  }
  c->letter = movemask64(l[0], l[1], l[2], l[3]);
  c->digit = movemask64(d[0], d[1], d[2], d[3]);
  c->space = movemask64(sp[0], sp[1], sp[2], sp[3]);
  c->ws = movemask64(ws[0], ws[1], ws[2], ws[3]);
  c->hi = movemask64(hi[0], hi[1], hi[2], hi[3]);
  c->apo = movemask64(ap[0], ap[1], ap[2], ap[3]);
  c->nl = newlines ? movemask64(nl[0], nl[1], nl[2], nl[3]) : 0;
}

#elif defined(BROT_MASK_SSE2)

/* Sixteen bytes at a time: an unsigned range test is a saturating minimum
   that leaves a byte unchanged, and pmovmskb gathers one bit per byte. */

static inline __m128i sse_le(__m128i x, int c)
{
  return _mm_cmpeq_epi8(_mm_min_epu8(x, _mm_set1_epi8((char)c)), x);
}

/* The 64 bits of four sixteen-lane results, lane i of [r[k]] at bit
   16k + i. */
static inline uint64_t sse_mask64(const __m128i r[4])
{
  return (uint64_t)(unsigned)_mm_movemask_epi8(r[0])
         | (uint64_t)(unsigned)_mm_movemask_epi8(r[1]) << 16
         | (uint64_t)(unsigned)_mm_movemask_epi8(r[2]) << 32
         | (uint64_t)(unsigned)_mm_movemask_epi8(r[3]) << 48;
}

/* One class at a time over the four loads, so that only the loads and one
   class's results are live: all classes at once spill to the stack. */
static BROT_INLINE void batch_classify(const unsigned char *p,
                                        batch_classes *c, int newlines)
{
  __m128i v[4], r[4], sp[4];
  int k;
  for (k = 0; k < 4; k++)
    v[k] = _mm_loadu_si128((const __m128i *)(p + 16 * k));
  for (k = 0; k < 4; k++)
    r[k] = sse_le(_mm_sub_epi8(_mm_or_si128(v[k], _mm_set1_epi8(0x20)),
                               _mm_set1_epi8('a')),
                  25);
  c->letter = sse_mask64(r);
  for (k = 0; k < 4; k++)
    r[k] = sse_le(_mm_sub_epi8(v[k], _mm_set1_epi8('0')), 9);
  c->digit = sse_mask64(r);
  for (k = 0; k < 4; k++) sp[k] = _mm_cmpeq_epi8(v[k], _mm_set1_epi8(' '));
  c->space = sse_mask64(sp);
  for (k = 0; k < 4; k++)
    r[k] = _mm_or_si128(sp[k],
                        sse_le(_mm_sub_epi8(v[k], _mm_set1_epi8(9)), 4));
  c->ws = sse_mask64(r);
  c->hi = sse_mask64(v);
  for (k = 0; k < 4; k++) r[k] = _mm_cmpeq_epi8(v[k], _mm_set1_epi8('\''));
  c->apo = sse_mask64(r);
  c->nl = 0;
  if (newlines) {
    for (k = 0; k < 4; k++)
      r[k] = _mm_or_si128(_mm_cmpeq_epi8(v[k], _mm_set1_epi8('\n')),
                          _mm_cmpeq_epi8(v[k], _mm_set1_epi8('\r')));
    c->nl = sse_mask64(r);
  }
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

static BROT_INLINE void batch_classify(const unsigned char *p,
                                        batch_classes *c, int newlines)
{
  int j;
  c->letter = c->digit = c->space = c->ws = c->hi = c->apo = c->nl = 0;
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
    if (newlines)
      c->nl |= swar_pack(((swar_ge(ascii, '\n') & swar_le(ascii, '\n'))
                          | (swar_ge(ascii, '\r') & swar_le(ascii, '\r')))
                         & ~hi)
               << sh;
  }
}

#endif

/* The classes of a batch's characters beyond ASCII, by [classify_uni]: the
   bytes of its letters, numbers and white space, the first byte of each white
   space character of two bytes and of three, and the bytes the masks leave
   to the scalar walker. */
typedef struct {
  uint64_t letter, digit, ws, ws2, ws3, bad;
} uni_classes;

/* Each character opening at a byte of [hi] in the batch at [scan], decoded
   and classified by [decode_class]: every byte of it takes its category, so
   that the masks see a character where they see a byte. A character that
   crosses the batch's end, a malformed one, one the Unicode table does not
   hold yet, white space of four bytes, and a continuation byte no character
   of the batch opens — a stray, or the tail of a character from the batch
   before — are bad. The
   loop takes one turn per character, and only batches beyond ASCII come
   here. */
static void classify_uni(walker *w, intnat scan, uint64_t hi, uni_classes *u)
{
  const unsigned char *s = w->s + scan;
  uint64_t rest = hi;
  u->letter = u->digit = u->ws = u->ws2 = u->ws3 = u->bad = 0;
  while (rest) {
    intnat k = ctz64(rest), d, len, cat;
    uint64_t one = (uint64_t)1 << k, m, is_ws;
    d = s[k] < 0xC0 ? -1 : decode_class(w, scan + k);
    len = d & 7;
    cat = (d >> 3) & 3;
    /* White space of four bytes is only ever an overlong form, whose
       give-back the masks have no lead mask for. */
    if (d < 0 || len == 1 || k + len > 64 || (len == 4 && cat == CL_WS)) {
      u->bad |= one;
      rest &= rest - 1;
      continue;
    }
    m = ((((uint64_t)1 << len) - 1) << k);
    is_ws = (uint64_t)0 - (uint64_t)(cat == CL_WS);
    u->letter |= m & ((uint64_t)0 - (uint64_t)(cat == CL_LETTER));
    u->digit |= m & ((uint64_t)0 - (uint64_t)(cat == CL_NUM));
    u->ws |= m & is_ws;
    u->ws2 |= one & is_ws & ((uint64_t)0 - (uint64_t)(len == 2));
    u->ws3 |= one & is_ws & ((uint64_t)0 - (uint64_t)(len == 3));
    rest &= ~m;
  }
}

/* The boundary masks of the batch at [scan]: bit k of [usable] is a
   trustworthy span start at [scan + k], bit k of [bad] sends byte
   [scan + k] to the scalar walker; the two never overlap. Requires
   [scan + 64 <= stop]. [anchor] is the position this call's walk began
   at — a span start by the resume protocol — so the batch there takes no
   carries and bit 0 is trivially a start; later batches read the byte
   before the batch, which is inside the range, for the four one-step
   carries the rules need. */
static void batch_masks(walker *w, intnat anchor, intnat scan,
                        uint64_t *usable, uint64_t *bad_out)
{
  const unsigned char *s = w->s;
  batch_classes c;
  uni_classes u = {0, 0, 0, 0, 0, 0};
  uint64_t pl = 0, pd = 0, psp = 0, pws = 0, po = 0;
  uint64_t letter, digit, ws, other, cont_same, after_sp, nb, split_ok, pwsb;
  uint64_t edge, boundary, bad, apo_bad, cand;
  batch_classify(s + scan, &c, 0);
  if (c.hi != 0) classify_uni(w, scan, c.hi, &u);
  letter = c.letter | u.letter;
  digit = c.digit | u.digit;
  ws = c.ws | u.ws;
  other = ~(letter | digit | ws);
  /* A bad character's neighbours take their boundaries from its class: the
     byte after it, and the white space character before it whose give-back
     looks at it, which opens up to three bytes before. */
  bad = u.bad | (u.bad << 1) | (u.bad >> 1) | (u.bad >> 2) | (u.bad >> 3);
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
     [\s+(?!\S)] give-back, which looks past the whole character. The
     apostrophe and the space sit in [other] and [ws] exactly as
     [lead_class] folds them. */
  cont_same = (letter & ((letter << 1) | pl)) | (digit & ((digit << 1) | pd))
              | (other & ((other << 1) | po));
  after_sp = (c.space << 1) | psp;
  nb = ~ws & ~cont_same & ~after_sp;
  split_ok = (c.ws & (~ws >> 1)) | (u.ws2 & (~ws >> 2)) | (u.ws3 & (~ws >> 3));
  /* The white space character ending the batch needs the byte after it for
     its give-back: a coin flip on natural text, so it is decided without a
     branch. At the end of the range the run runs out instead and keeps its
     last character. */
  edge = (c.ws & ((uint64_t)1 << 63)) | (u.ws2 & ((uint64_t)1 << 62))
         | (u.ws3 & ((uint64_t)1 << 61));
  if (scan + 64 < w->stop) {
    unsigned b = s[scan + 64];
    uint64_t hi = b >= 0x80, sp = b == ' ' || (unsigned)(b - 9) <= 4;
    bad |= ((uint64_t)0 - edge) & ((uint64_t)0 - hi); /* its bytes */
    split_ok |= edge & ((hi | sp) - 1); /* an ASCII byte that is no space */
  }
  pwsb = (ws << 1) | pws;
  boundary = nb | ((c.ws | u.ws2 | u.ws3) & (~pwsb | split_ok));
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

/* The cl100k boundary masks: [batch_masks] for [cl100k_next_span], on the
   same contract. On ASCII the pattern's alternatives come down to where a
   span starts:

   - a letter after anything but a letter, an "other" byte opening a span
     (which a word takes as its prefix) and a white space byte that is no
     newline (which always opens a span when a letter follows);
   - an "other" byte after anything but an "other" byte or a space, which
     [ ?[^\s\p{L}\p{N}]+] glues to the punctuation;
   - a digit opening its run, and every [digits]th digit after;
   - in a white space run, its first byte, unless it is a newline right after
     punctuation, which [[\r\n]*] takes along with the newlines after it,
     and then the first byte after those newlines; the byte after the run's
     last newline, the end of [\s*[\r\n]+]; and the run's last byte when it
     is no newline, which [\s+(?!\S)] gives back;
   - the byte after a contraction that an apostrophe opening a span starts.

   Bad zones: non-ASCII bytes, the byte before and the two after (a
   character's class reaches the start of the byte after it, and through an
   "other" byte that opens a span, the byte after that); digit and white
   space runs that a non-ASCII byte or the previous batch could extend
   (their starts depend on the whole run); the white space run at the batch
   edge unless the next byte ends it; an apostrophe's contraction on bad
   bits; and an apostrophe opening a span less than three bytes before the
   batch edge. */
static void cl100k_batch_masks(const walker *w, intnat anchor, intnat scan,
                               intnat digits, uint64_t *usable,
                               uint64_t *bad_out)
{
  const unsigned char *s = w->s;
  batch_classes c;
  uint64_t pl = 0, pd = 0, pw = 0, pm = 0, psp = 0, pnl = 0, po = 0;
  uint64_t ph = 0, p2h = 0, po_start = 0;
  uint64_t other, m, o_start, l_start, d_start, w_start, starts, bad;
  uint64_t absorbed, seeds, cand, gb, grown;
  int next = 0; /* the byte after the batch: 0 none, 1 white space, 2 not
                   white space, 3 non-ASCII */
  batch_classify(s + scan, &c, 1);
  other = ~(c.letter | c.digit | c.ws | c.hi);
  m = c.ws & ~c.nl;
  if (scan > anchor) {
    unsigned b = s[scan - 1], b2 = s[scan - 2];
    if (b >= 0x80)
      ph = 1;
    else {
      pl = (unsigned)((b | 0x20) - 'a') <= 25;
      pd = (unsigned)(b - '0') <= 9;
      psp = b == ' ';
      pnl = b == '\n' || b == '\r';
      pw = psp | ((unsigned)(b - 9) <= 4);
      pm = pw & !pnl;
      po = !(pl | pd | pw);
    }
    /* The "other" byte before the batch opens a span unless an "other"
       byte or a space precedes it. */
    if (b2 >= 0x80)
      p2h = 1;
    else if (po)
      po_start = (unsigned)((b2 | 0x20) - 'a') <= 25
                 || (unsigned)(b2 - '0') <= 9 || (unsigned)(b2 - 9) <= 4;
  }
  if (scan + 64 < w->stop) {
    unsigned b = s[scan + 64];
    next = b >= 0x80                            ? 3
           : b == ' ' || (unsigned)(b - 9) <= 4 ? 1
                                                : 2;
  }
  bad = c.hi | (c.hi >> 1) | (c.hi << 1) | ph | (c.hi << 2) | (ph << 1) | p2h;

  o_start = other & ~((other << 1) | po) & ~((c.space << 1) | psp);
  l_start = c.letter & ~((c.letter << 1) | pl) & ~((o_start << 1) | po_start)
            & ~((m << 1) | pm);

  /* Digit groups: each start [digits] digits after the last one. A run the
     previous batch or a non-ASCII digit could have opened earlier is bad. */
  d_start = c.digit & ~((c.digit << 1) | pd);
  if (digits <= 1)
    d_start = c.digit;
  else if (digits < 64 && (c.digit & (c.digit << 1)) != 0) {
    uint64_t g = c.digit, cur = d_start;
    intnat t;
    for (t = 1; t < digits; t++) g &= c.digit << t;
    while ((cur = (cur << digits) & g) != 0) d_start |= cur;
  }
  seeds = c.digit & ((c.hi << 1) | ph | pd);
  bad |= seeds | (((c.digit + seeds) ^ c.digit ^ seeds) & c.digit);

  /* White space. A carry chain from each seed through the run of [x] it
     sits in fills that run upward: [(x + seeds) ^ x ^ seeds] is the carry
     into each bit. */
  seeds = c.nl & ~((c.ws << 1) | pw) & ((other << 1) | po);
  absorbed = seeds | (((c.nl + seeds) ^ c.nl ^ seeds) & c.nl);
  w_start = (c.ws & ~((c.ws << 1) | pw) & ~seeds) | (m & (absorbed << 1));
  gb = m & ~(c.ws >> 1) & ~((uint64_t)1 << 63);
  if (c.ws >> 63) {
    uint64_t tail = ~(uint64_t)0 << (64 - (~c.ws ? clz64(~c.ws) : 64));
    if (next == 1 || next == 3)
      bad |= tail;
    else if (next == 2)
      gb |= m & ((uint64_t)1 << 63);
  }
  seeds = c.ws & ((c.hi << 1) | ph | pw);
  bad |= seeds | (((c.ws + seeds) ^ c.ws ^ seeds) & c.ws);
  /* After a newline, a byte opens a span when no newline follows in its
     run: when the run of non-newline white space it opens ends at a byte
     that is not white space. */
  cand = m & ((c.nl << 1) | pnl);
  while (cand) {
    intnat q = ctz64(cand);
    uint64_t rest = ~m & (~(uint64_t)0 << q);
    intnat f = rest ? ctz64(rest) : 64;
    cand &= cand - 1;
    if (f == 64 || !((c.nl >> f) & 1)) w_start |= (uint64_t)1 << q;
    if (f < 64 && ((c.hi >> f) & 1)) bad |= (uint64_t)1 << q;
  }
  w_start |= gb;
  starts = l_start | d_start | o_start | w_start;

  /* Contractions, as in [batch_masks], reaching three bytes on. */
  grown = c.apo & bad;
  bad |= (grown << 1) | (grown << 2) | (grown << 3);
  cand = c.apo & o_start & ~bad;
  while (cand) {
    intnat i = ctz64(cand);
    intnat c1, c2, k = 0;
    cand &= cand - 1;
    if (i >= 61) {
      bad |= ~(uint64_t)0 << i;
      break;
    }
    c1 = ascii_lower(s[scan + i + 1]);
    c2 = ascii_lower(s[scan + i + 2]);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
      k = 2;
    else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e')
             || (c1 == 'l' && c2 == 'l'))
      k = 3;
    if (k) starts |= (uint64_t)1 << (i + k);
  }
  *usable = starts & ~bad;
  *bad_out = bad;
}

/* The batch walk. Each call of [harvest] appends to a queue the ends of the
   spans after the pending one's start [pos], a batch at a time: a clean batch
   gives all its boundary bits at once, without a branch per span; a bad zone
   sends every span opening before the next trusted boundary (or the batch
   end) through the scalar walker. A scalar overrun keeps the 64-byte grid:
   the batch it lands in is classified again, bits at or below [pos] dead.
   The state lives for one kernel call — the masks are pure functions of the
   bytes, so a resumed call rebuilds them from its own anchor. [digits] is -1
   for the GPT-2 walker, and otherwise the cl100k walker's longest group of
   numbers, [marks] its flag. */
typedef struct {
  intnat digits;
  int marks;
  intnat anchor; /* the call's start position, carry-free by contract */
  intnat scan;   /* base of the next batch to classify */
  intnat pos;    /* the pending span's start: the last end harvested */
  int failed;    /* the scalar walker stopped at [pos] for a class */
} mask_state;

static void mask_init(mask_state *ms, intnat pos, intnat digits, int marks)
{
  ms->digits = digits;
  ms->marks = marks;
  ms->anchor = pos;
  ms->scan = pos;
  ms->pos = pos;
  ms->failed = 0;
}

/* Byte j of entry b is the position of the j-th set bit of b. */
static const uint64_t bit_pos[256] = {
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000001ULL,
    0x0000000000000100ULL, 0x0000000000000002ULL, 0x0000000000000200ULL,
    0x0000000000000201ULL, 0x0000000000020100ULL, 0x0000000000000003ULL,
    0x0000000000000300ULL, 0x0000000000000301ULL, 0x0000000000030100ULL,
    0x0000000000000302ULL, 0x0000000000030200ULL, 0x0000000000030201ULL,
    0x0000000003020100ULL, 0x0000000000000004ULL, 0x0000000000000400ULL,
    0x0000000000000401ULL, 0x0000000000040100ULL, 0x0000000000000402ULL,
    0x0000000000040200ULL, 0x0000000000040201ULL, 0x0000000004020100ULL,
    0x0000000000000403ULL, 0x0000000000040300ULL, 0x0000000000040301ULL,
    0x0000000004030100ULL, 0x0000000000040302ULL, 0x0000000004030200ULL,
    0x0000000004030201ULL, 0x0000000403020100ULL, 0x0000000000000005ULL,
    0x0000000000000500ULL, 0x0000000000000501ULL, 0x0000000000050100ULL,
    0x0000000000000502ULL, 0x0000000000050200ULL, 0x0000000000050201ULL,
    0x0000000005020100ULL, 0x0000000000000503ULL, 0x0000000000050300ULL,
    0x0000000000050301ULL, 0x0000000005030100ULL, 0x0000000000050302ULL,
    0x0000000005030200ULL, 0x0000000005030201ULL, 0x0000000503020100ULL,
    0x0000000000000504ULL, 0x0000000000050400ULL, 0x0000000000050401ULL,
    0x0000000005040100ULL, 0x0000000000050402ULL, 0x0000000005040200ULL,
    0x0000000005040201ULL, 0x0000000504020100ULL, 0x0000000000050403ULL,
    0x0000000005040300ULL, 0x0000000005040301ULL, 0x0000000504030100ULL,
    0x0000000005040302ULL, 0x0000000504030200ULL, 0x0000000504030201ULL,
    0x0000050403020100ULL, 0x0000000000000006ULL, 0x0000000000000600ULL,
    0x0000000000000601ULL, 0x0000000000060100ULL, 0x0000000000000602ULL,
    0x0000000000060200ULL, 0x0000000000060201ULL, 0x0000000006020100ULL,
    0x0000000000000603ULL, 0x0000000000060300ULL, 0x0000000000060301ULL,
    0x0000000006030100ULL, 0x0000000000060302ULL, 0x0000000006030200ULL,
    0x0000000006030201ULL, 0x0000000603020100ULL, 0x0000000000000604ULL,
    0x0000000000060400ULL, 0x0000000000060401ULL, 0x0000000006040100ULL,
    0x0000000000060402ULL, 0x0000000006040200ULL, 0x0000000006040201ULL,
    0x0000000604020100ULL, 0x0000000000060403ULL, 0x0000000006040300ULL,
    0x0000000006040301ULL, 0x0000000604030100ULL, 0x0000000006040302ULL,
    0x0000000604030200ULL, 0x0000000604030201ULL, 0x0000060403020100ULL,
    0x0000000000000605ULL, 0x0000000000060500ULL, 0x0000000000060501ULL,
    0x0000000006050100ULL, 0x0000000000060502ULL, 0x0000000006050200ULL,
    0x0000000006050201ULL, 0x0000000605020100ULL, 0x0000000000060503ULL,
    0x0000000006050300ULL, 0x0000000006050301ULL, 0x0000000605030100ULL,
    0x0000000006050302ULL, 0x0000000605030200ULL, 0x0000000605030201ULL,
    0x0000060503020100ULL, 0x0000000000060504ULL, 0x0000000006050400ULL,
    0x0000000006050401ULL, 0x0000000605040100ULL, 0x0000000006050402ULL,
    0x0000000605040200ULL, 0x0000000605040201ULL, 0x0000060504020100ULL,
    0x0000000006050403ULL, 0x0000000605040300ULL, 0x0000000605040301ULL,
    0x0000060504030100ULL, 0x0000000605040302ULL, 0x0000060504030200ULL,
    0x0000060504030201ULL, 0x0006050403020100ULL, 0x0000000000000007ULL,
    0x0000000000000700ULL, 0x0000000000000701ULL, 0x0000000000070100ULL,
    0x0000000000000702ULL, 0x0000000000070200ULL, 0x0000000000070201ULL,
    0x0000000007020100ULL, 0x0000000000000703ULL, 0x0000000000070300ULL,
    0x0000000000070301ULL, 0x0000000007030100ULL, 0x0000000000070302ULL,
    0x0000000007030200ULL, 0x0000000007030201ULL, 0x0000000703020100ULL,
    0x0000000000000704ULL, 0x0000000000070400ULL, 0x0000000000070401ULL,
    0x0000000007040100ULL, 0x0000000000070402ULL, 0x0000000007040200ULL,
    0x0000000007040201ULL, 0x0000000704020100ULL, 0x0000000000070403ULL,
    0x0000000007040300ULL, 0x0000000007040301ULL, 0x0000000704030100ULL,
    0x0000000007040302ULL, 0x0000000704030200ULL, 0x0000000704030201ULL,
    0x0000070403020100ULL, 0x0000000000000705ULL, 0x0000000000070500ULL,
    0x0000000000070501ULL, 0x0000000007050100ULL, 0x0000000000070502ULL,
    0x0000000007050200ULL, 0x0000000007050201ULL, 0x0000000705020100ULL,
    0x0000000000070503ULL, 0x0000000007050300ULL, 0x0000000007050301ULL,
    0x0000000705030100ULL, 0x0000000007050302ULL, 0x0000000705030200ULL,
    0x0000000705030201ULL, 0x0000070503020100ULL, 0x0000000000070504ULL,
    0x0000000007050400ULL, 0x0000000007050401ULL, 0x0000000705040100ULL,
    0x0000000007050402ULL, 0x0000000705040200ULL, 0x0000000705040201ULL,
    0x0000070504020100ULL, 0x0000000007050403ULL, 0x0000000705040300ULL,
    0x0000000705040301ULL, 0x0000070504030100ULL, 0x0000000705040302ULL,
    0x0000070504030200ULL, 0x0000070504030201ULL, 0x0007050403020100ULL,
    0x0000000000000706ULL, 0x0000000000070600ULL, 0x0000000000070601ULL,
    0x0000000007060100ULL, 0x0000000000070602ULL, 0x0000000007060200ULL,
    0x0000000007060201ULL, 0x0000000706020100ULL, 0x0000000000070603ULL,
    0x0000000007060300ULL, 0x0000000007060301ULL, 0x0000000706030100ULL,
    0x0000000007060302ULL, 0x0000000706030200ULL, 0x0000000706030201ULL,
    0x0000070603020100ULL, 0x0000000000070604ULL, 0x0000000007060400ULL,
    0x0000000007060401ULL, 0x0000000706040100ULL, 0x0000000007060402ULL,
    0x0000000706040200ULL, 0x0000000706040201ULL, 0x0000070604020100ULL,
    0x0000000007060403ULL, 0x0000000706040300ULL, 0x0000000706040301ULL,
    0x0000070604030100ULL, 0x0000000706040302ULL, 0x0000070604030200ULL,
    0x0000070604030201ULL, 0x0007060403020100ULL, 0x0000000000070605ULL,
    0x0000000007060500ULL, 0x0000000007060501ULL, 0x0000000706050100ULL,
    0x0000000007060502ULL, 0x0000000706050200ULL, 0x0000000706050201ULL,
    0x0000070605020100ULL, 0x0000000007060503ULL, 0x0000000706050300ULL,
    0x0000000706050301ULL, 0x0000070605030100ULL, 0x0000000706050302ULL,
    0x0000070605030200ULL, 0x0000070605030201ULL, 0x0007060503020100ULL,
    0x0000000007060504ULL, 0x0000000706050400ULL, 0x0000000706050401ULL,
    0x0000070605040100ULL, 0x0000000706050402ULL, 0x0000070605040200ULL,
    0x0000070605040201ULL, 0x0007060504020100ULL, 0x0000000706050403ULL,
    0x0000070605040300ULL, 0x0000070605040301ULL, 0x0007060504030100ULL,
    0x0000070605040302ULL, 0x0007060504030200ULL, 0x0007060504030201ULL,
    0x0706050403020100ULL
};

/* Appends base + k for every set bit k of [m] to [out], in order, and
   returns their count: eight fixed steps of one byte each, every step
   writing eight lanes at the count of the bits before it, so that the cost
   does not depend on the bits. Writes up to seven lanes past the count. */
static inline intnat flatten(uint64_t m, uint32_t base, uint32_t *out)
{
  uint64_t x = m - ((m >> 1) & 0x5555555555555555ULL);
  uint64_t incl, excl;
  int j;
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  incl = x * 0x0101010101010101ULL; /* byte j: the bits in bytes 0..j */
  excl = incl << 8;
  for (j = 0; j < 8; j++) {
    uint64_t pos = bit_pos[(m >> (8 * j)) & 0xFF];
    uint32_t *o = out + ((excl >> (8 * j)) & 0xFF);
    uint32_t b = base + 8 * (uint32_t)j;
#if defined(BROT_MASK_SSE2)
    __m128i z = _mm_setzero_si128();
    __m128i p16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128((long long)pos), z);
    __m128i add = _mm_set1_epi32((int)b);
    _mm_storeu_si128((__m128i *)o,
                     _mm_add_epi32(_mm_unpacklo_epi16(p16, z), add));
    _mm_storeu_si128((__m128i *)(o + 4),
                     _mm_add_epi32(_mm_unpackhi_epi16(p16, z), add));
#elif defined(BROT_MASK_NEON)
    uint16x8_t p16 = vmovl_u8(vcreate_u8(pos));
    uint32x4_t add = vdupq_n_u32(b);
    vst1q_u32(o, vaddq_u32(vmovl_u16(vget_low_u16(p16)), add));
    vst1q_u32(o + 4, vaddq_u32(vmovl_u16(vget_high_u16(p16)), add));
#else
    int t;
    for (t = 0; t < 8; t++) o[t] = b + (uint32_t)((pos >> (8 * t)) & 0xFF);
#endif
  }
  return (intnat)(incl >> 56);
}

/* The span opening at [i] by the scalar walker. */
static inline intnat scalar_span(const mask_state *ms, walker *w, intnat i)
{
  return ms->digits < 0 ? next_span(w, i)
                        : cl100k_next_span(w, i, ms->digits, ms->marks);
}

/* The lanes a [harvest] can write past its count: a batch's 64 boundaries, a
   scalar overrun's end, and [flatten]'s seven lanes of slack. */
#define HARVEST_LANES 72

/* Appends to [q] from [qn] the ends of the spans after [ms->pos] that the
   next batch holds — or the next span, over the tail no batch covers — and
   returns the new count; [ms->pos] becomes the last of them. The scalar
   walker stopping for a class sets [ms->failed] and appends nothing more. */
static intnat harvest(mask_state *ms, walker *w, uint32_t *q, intnat qn)
{
  intnat pos = ms->pos, base, until, p;
  uint64_t usable, bad, ulive, blive;
  if (pos >= ms->scan + 64) ms->scan += 64 * ((pos - ms->scan) / 64);
  base = ms->scan;
  if (base + 64 > w->stop) { /* the tail, scalar to the end */
    intnat e = scalar_span(ms, w, pos);
    if (e < 0)
      ms->failed = 1;
    else {
      q[qn++] = (uint32_t)e;
      ms->pos = e;
    }
    return qn;
  }
  if (ms->digits < 0)
    batch_masks(w, ms->anchor, base, &usable, &bad);
  else
    cl100k_batch_masks(w, ms->anchor, base, ms->digits, &usable, &bad);
  /* A bit at the pending span's start opens it rather than ending it, and
     bits below it lie in spans already walked; a bad bit at it stays live,
     so that a bad zone starting there is walked. */
  blive = pos >= base ? ~(uint64_t)0 << (pos - base) : ~(uint64_t)0;
  ulive = blive << (pos >= base);
  for (;;) {
    uint64_t seg_bad = bad & blive, rest;
    intnat fb;
    if (seg_bad == 0) {
      qn += flatten(usable & ulive, (uint32_t)base, q + qn);
      break;
    }
    fb = ctz64(seg_bad);
    qn += flatten(usable & ulive & (((uint64_t)1 << fb) - 1), (uint32_t)base,
                  q + qn);
    p = qn > 0 && (intnat)q[qn - 1] > pos ? (intnat)q[qn - 1] : pos;
    rest = usable & (~(uint64_t)0 << fb);
    until = rest != 0 ? base + ctz64(rest) : base + 64;
    while (p < until) {
      intnat e = scalar_span(ms, w, p);
      if (e < 0) {
        ms->failed = 1;
        ms->pos = p;
        return qn;
      }
      q[qn++] = (uint32_t)e;
      p = e;
    }
    if (p >= base + 64) { /* an overrun: on to the batch [p] lies in */
      ms->scan = base + 64 * ((p - base) / 64);
      ms->pos = p;
      return qn;
    }
    blive = ~(uint64_t)0 << (p - base);
    ulive = blive << 1;
    pos = p;
  }
  ms->scan = base + 64;
  if (qn > 0 && (intnat)q[qn - 1] > ms->pos) ms->pos = (intnat)q[qn - 1];
  return qn;
}

/* Takes the walk back to the span start [pos] it passed: the batch [pos]
   lies in is classified again, with the bits up to [pos] dead, as after an
   overrun. */
static void rewind_walk(mask_state *ms, intnat pos)
{
  if (ms->scan > pos) ms->scan = ms->anchor + 64 * ((pos - ms->anchor) / 64);
  ms->pos = pos;
  ms->failed = 0;
}

/* The cache probe — Pretoken_cache.key0/key1/hash_of and the two-way layout
   of the 64-byte back set, bit for bit; the kernels' tables have no front
   table. Keys are built from little-endian loads and every word is stored
   native-endian, so the table OCaml writes and the table written here are one
   format. A key
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
    w = load64le(s + pos);
  else if (n >= 8)
    w = load64le(s + n - 8) >> ((pos - n + 8) * 8);
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
    w = load64le(s + pos + 8);
  else
    w = load64le(s + n - 8) >> ((pos + 16 - n) * 8);
  return tag | (w & ((1ULL << (8 * (len - 8))) - 1));
}

/* Pretoken_cache.hash_of, with its fold of the key's high half into the low
   one: OCaml truncates the shifted hash to 63 bits; only 40 bits are set
   after the shift, so the two agree. The back set's byte offset is
   Pretoken_cache.set_at on this one hash. */
static inline uint64_t hash_of(uint64_t k0, uint64_t k1)
{
  uint64_t x = k0 ^ (k1 * 0x9E3779B97F4A7C15ULL);
  x ^= x >> 32;
  return (x * 0xD6E8FEB86659FD93ULL) >> 24;
}

/* Pretoken_cache.store_at, in the same order: way 0 shifts into way 1 and the
   new entry fills way 0, values first, then k1, then k0, so a read torn by
   anything at all comes out a miss rather than a wrong hit. */
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

/* Pretoken_cache.neighbour: the set beside [set], which takes the entries
   [set] evicts, or [set] itself in a table of one set. */
static inline intnat neighbour(intnat set, intnat mask)
{
  return mask > 0 ? set ^ 64 : set;
}

/* Pretoken_cache.insert_at: [cache_store], the entry way 1 held moving on into
   the neighbour set as [cache_store] puts it there. Returns whether no entry
   was dropped, which is what counts toward the table's growth. */
static int cache_insert(unsigned char *tbl, intnat set, intnat mask,
                        uint64_t k0, uint64_t k1, uint64_t v0, uint64_t v1)
{
  intnat nset = neighbour(set, mask);
  uint64_t w0 = load64(tbl + set + 32), w1 = load64(tbl + set + 40);
  uint64_t w2 = load64(tbl + set + 48), w3 = load64(tbl + set + 56);
  int dropped = w1 != 0 && (nset == set || load64(tbl + nset + 40) != 0);
  cache_store(tbl, set, k0, k1, v0, v1);
  if (w1 != 0 && nset != set) cache_store(tbl, nset, w0, w1, w2, w3);
  return !dropped;
}

/* Pretoken_cache.add_lanes: a hit's four lanes are stored unconditionally, as
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
   are 24-bit, so the narrowing is exact. A value word holds its two lanes
   at bits 0 and 32, where a little-endian int32 pair has them, so masking
   off the count makes each word one store. */
static inline intnat emit_lanes32(int32_t *lane, uint64_t v0, uint64_t v1)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  store64(lane, v0 & 0x00FFFFFF00FFFFFFULL);
  store64(lane + 2, v1 & 0x00FFFFFF00FFFFFFULL);
#else
  lane[0] = (int32_t)(v0 & 0xFFFFFF);
  lane[1] = (int32_t)((v0 >> 32) & 0xFFFFFF);
  lane[2] = (int32_t)(v1 & 0xFFFFFF);
  lane[3] = (int32_t)((v1 >> 32) & 0xFFFFFF);
#endif
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
    /* The live symbols, in order: the first is never merged away. */
    for (k = 0; k >= 0; k = sym_next[k])
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

/* The staged probe, shared by both entries. A call walks [STAGE] spans or a
   batch more ahead, keying each one and prefetching its set, then resolves
   them in order: room checks, the probe of the set, which costs a branch only
   on a miss, the neighbour set's, the merge of a miss and its store. Every
   probe and store happens in the reference's span order, so the tables stay
   bit for bit the reference's; the walk only reads the text. A stage of
   resolving covers a miss to memory while the walk itself runs as a tight
   loop, and costs the mispredicted exits of its loops once.

   The stage lives in the cursor, after its counts, so that an exit at a
   staged span keeps the spans staged after it: the call that resumes at the
   next of them, on the same range, resolves them before walking on from
   the last. Any other call starts a stage of its own; the driver also
   restarts it at every walk it begins (Kernel.restart), since a range of
   another text could start and stop at the same positions. */

#define STAGE 256

/* The spans a stage holds: the batch walk stops once it has [STAGE], and its
   last batch can add 64 more and a scalar overrun's end. */
#define STAGE_CAP (STAGE + HARVEST_LANES)

/* The stages a call walks before it returns Spans_full: OCaml's runtime can
   stop a domain for a collection only between calls, and every other domain
   waits for it, so a call ends after 1024 spans or a few batches more,
   whatever room is left. */
#define YIELD_STAGES 4

typedef struct {
  intnat i, e; /* the span */
  intnat set;  /* its set's byte offset, or -1 without a cache */
  uint64_t k0, k1;
} staged;

typedef struct {
  unsigned char *cache;
  intnat cache_mask;
  const value *mkeys, *mvals, *len_tab;
  intnat mmask, len_n;
} tables;

/* The bytes of each key word that a length keeps, and the top byte of the
   second word: the length, and for 16, which stands for every length past
   the key, a byte no entry has. */
static const uint64_t key_mask0[17] = {
    0,
    0xFF,
    0xFFFF,
    0xFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFFFFULL,
    0xFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL};
static const uint64_t key_mask1[17] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0,
    0xFF,
    0xFFFF,
    0xFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFFFFULL,
    0xFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFULL,
    0};
static const uint64_t key_top[17] = {
    0,           1ULL << 56,  2ULL << 56,  3ULL << 56,  4ULL << 56,
    5ULL << 56,  6ULL << 56,  7ULL << 56,  8ULL << 56,  9ULL << 56,
    10ULL << 56, 11ULL << 56, 12ULL << 56, 13ULL << 56, 14ULL << 56,
    15ULL << 56, 0xFFULL << 56};

/* Stages the span [i, e): its key words, set and a prefetch of the set. A
   span over 15 bytes is keyed on its first 15 with a second word no entry
   has, all ones where an entry's top byte holds a length of at most 15, so
   that its probe misses without a branch of its own. Away from the end of
   the text both key words are two loads masked to the span; [key_word0] and
   [key_word1] cover the end. [set] is -1 when there is no cache. */
static inline void stage(const tables *t, const unsigned char *s, intnat n,
                         staged *st, intnat i, intnat e)
{
  intnat len = e - i;
  intnat m = len < 15 ? len : 15;
  uint64_t k0, k1, h;
  st->i = i;
  st->e = e;
  if (t->cache_mask < 0) {
    st->set = -1;
    return;
  }
  if (i + 16 <= n) {
    k0 = load64le(s + i) & key_mask0[m];
    k1 = (load64le(s + i + 8) & key_mask1[m]) | ((uint64_t)m << 56);
  } else {
    k0 = key_word0(s, n, i, m);
    k1 = key_word1(s, n, i, m);
  }
  k1 |= (uint64_t)0 - (uint64_t)(len > 15);
  h = hash_of(k0, k1);
  st->k0 = k0;
  st->k1 = k1;
  st->set = (intnat)((h & (uint64_t)t->cache_mask) << 6);
  BROT_PREFETCH(t->cache + st->set);
}

/* [stage] over the [nb] consecutive spans after [i] that end at [ends]. Away
   from the end of the text, one test up front leaves each span's keying
   without a branch. */
static void stage_spans(const tables *t, const unsigned char *s, intnat n,
                        staged *st, intnat i, const uint32_t *ends, intnat nb)
{
  intnat j;
  if (t->cache_mask < 0 || nb == 0 || (intnat)ends[nb - 1] + 16 > n) {
    for (j = 0; j < nb; j++) {
      stage(t, s, n, &st[j], i, (intnat)ends[j]);
      i = (intnat)ends[j];
    }
    return;
  }
  for (j = 0; j < nb; j++) {
    intnat e = (intnat)ends[j], len = e - i;
    intnat m = len < 16 ? len : 16;
    uint64_t k0 = load64le(s + i) & key_mask0[m];
    uint64_t k1 = (load64le(s + i + 8) & key_mask1[m]) | key_top[m];
    intnat set = (intnat)((hash_of(k0, k1) & (uint64_t)t->cache_mask) << 6);
    BROT_PREFETCH(t->cache + set);
    st[j].i = i;
    st[j].e = e;
    st[j].set = set;
    st[j].k0 = k0;
    st[j].k1 = k1;
    i = e;
  }
}

/* The probe of a staged span: both ways of its set, compared and selected
   without a branch, so that whether a span hits, and in which way, costs no
   misprediction. Returns whether it hit, with the hit way's value words in
   [v]; on a miss [v] holds way 1's words, which the caller does not count. A
   span the cache cannot hold has a key no entry has (see [stage]). */
static inline int probe_set(const tables *t, const staged *st, uint64_t v[2])
{
  const unsigned char *b = t->cache + st->set;
  uint64_t k0 = st->k0, k1 = st->k1;
  intnat m0, m1, hit;
  const unsigned char *way;
  /* In general registers: paired into vectors, the compares cost more. */
  BROT_OPAQUE(k0);
  BROT_OPAQUE(k1);
  m0 = ((load64(b) ^ k0) | (load64(b + 8) ^ k1)) == 0;
  m1 = ((load64(b + 32) ^ k0) | (load64(b + 40) ^ k1)) == 0;
  BROT_OPAQUE(m0);
  way = b + 48 - 32 * m0;
  v[0] = load64(way);
  v[1] = load64(way + 8);
  hit = m0 | m1;
  BROT_OPAQUE(hit);
  return (int)hit;
}

/* Stores a merged miss of [m] ids when Pretoken_cache.encode would keep it:
   at most four, each within 24 bits. Pretoken_cache.value_lo/value_hi: two
   24-bit lanes per word, the count in bits 24..31 of the first, absent lanes
   zero. Returns whether the table holds one entry more, as [cache_insert]. */
static inline int remember(const tables *t, const staged *st,
                           const int32_t *ids, int m)
{
  int k;
  if (st->set < 0 || m > 4) return 0;
  for (k = 0; k < m; k++)
    if (ids[k] >= 1 << 24) return 0;
  return cache_insert(t->cache, st->set, t->cache_mask, st->k0, st->k1,
                      (uint64_t)ids[0] | ((uint64_t)m << 24)
                          | (m > 1 ? (uint64_t)ids[1] << 32 : 0),
                      (m > 2 ? (uint64_t)ids[2] : 0)
                          | (m > 3 ? (uint64_t)ids[3] << 32 : 0));
}

/* Pretoken_cache.encode past a miss in the span's own set: the neighbour set,
   where an entry the set evicted waits, and which hands a hit back to the set,
   trading places with the entry the set's way 1 held. Returns whether it hit,
   with the value words in [v]. */
static int probe_neighbour(const tables *t, const staged *st, uint64_t v[2])
{
  unsigned char *tbl = t->cache;
  intnat set = st->set, nset = neighbour(set, t->cache_mask), way;
  if (nset == set) return 0;
  if (load64(tbl + nset) == st->k0 && load64(tbl + nset + 8) == st->k1)
    way = nset;
  else if (load64(tbl + nset + 32) == st->k0
           && load64(tbl + nset + 40) == st->k1)
    way = nset + 32;
  else
    return 0;
  v[0] = load64(tbl + way + 16);
  v[1] = load64(tbl + way + 24);
  store64(tbl + way + 16, load64(tbl + set + 48));
  store64(tbl + way + 24, load64(tbl + set + 56));
  store64(tbl + way + 8, load64(tbl + set + 40));
  store64(tbl + way, load64(tbl + set + 32));
  cache_store(tbl, set, st->k0, st->k1, v[0], v[1]);
  return 1;
}

/* Bpe.whole_word_id on a staged span of at most 15 bytes, for the entries
   whose merge does not give them back: its id in the whole-word table
   [words], an open-addressing table of [words_mask + 1] 24-byte slots, each
   such an entry's two key words, as Pretoken_cache keys them, and its id,
   probed linearly from the slot its hash gives; an empty slot has a zero
   second word, which no key has. Any other entry merges to itself.
   Returns the id, -1 for no entry, or -2 when the entry's id stands for
   other bytes than len_table gives it, which only OCaml records. */
static intnat whole_word(const unsigned char *words, intnat words_mask,
                         const tables *t, const unsigned char *s, intnat n,
                         const staged *st, intnat len)
{
  uint64_t k0, k1;
  intnat at;
  if (st->set >= 0) {
    k0 = st->k0;
    k1 = st->k1;
  } else {
    k0 = key_word0(s, n, st->i, len);
    k1 = key_word1(s, n, st->i, len);
  }
  at = (intnat)(hash_of(k0, k1) & (uint64_t)words_mask);
  for (;;) {
    const unsigned char *slot = words + 24 * at;
    uint64_t q1 = load64(slot + 8);
    if (q1 == 0) return -1;
    if (q1 == k1 && load64(slot) == k0) {
      intnat id = (intnat)load64(slot + 16);
      if (id >= t->len_n || Long_val(t->len_tab[id]) != len) return -2;
      return id;
    }
    at = (at + 1) & words_mask;
  }
}

/* Appends [m] merged ids to the sink. */
static inline void emit_ids(value *ids, int32_t *ids32, int emit32, intnat at,
                            const int32_t *src, int m)
{
  int k;
  if (emit32)
    for (k = 0; k < m; k++) ids32[at + k] = src[k];
  else
    for (k = 0; k < m; k++) ids[at + k] = Val_long((intnat)src[k]);
}

static void tables_of(tables *t, value vt, int cache, int cache_base,
                      int cache_mask, int mkeys, int mvals, int mmask,
                      int len_tab)
{
  t->cache = Bytes_val(Field(vt, cache)) + Long_val(Field(vt, cache_base));
  t->cache_mask = Long_val(Field(vt, cache_mask));
  /* Field lvalues are volatile for the runtime's sake; these arrays have one
     writer (the thread holding the state claim) and no GC point exists in a
     call, so the qualifier is cast away and the compiler may keep loads in
     registers. */
  t->mkeys = (const value *)&Field(Field(vt, mkeys), 0);
  t->mvals = (const value *)&Field(Field(vt, mvals), 0);
  t->mmask = Long_val(Field(vt, mmask));
  t->len_tab = (const value *)&Field(Field(vt, len_tab), 0);
  t->len_n = (intnat)Wosize_val(Field(vt, len_tab));
}

/* The byte-level entry's body. The caller validates the range and the
   cursor counts once; ids room is checked here per span, a hit storing its
   four lanes unconditionally as Ints.add4 does. [emit32] selects the sink
   and the protocol: the int array of Kernel.byte_level_encode, with every
   span and its mark recorded, or the int32 Bigarray of
   Kernel.byte_level_encode_ids32, with neither, a hand-back span then
   carried by the cursor alone. [split] selects the walker: the GPT-2 mask
   scanner, or the cl100k mask scanner or the scalar o200k walker under the
   parameters [vwalker] packs as Kernel.split_encode documents. Both are
   compile-time constants in each wrapper. */
static BROT_INLINE value
byte_level_encode_impl(value text, value vpos, value vstop, value vspans,
                       value vids, value vmarks, value vcursor, value vunicode,
                       value vt, value vwalker, int emit32, int split)
{
  const unsigned char *s = (const unsigned char *)String_val(text);
  intnat n = (intnat)caml_string_length(text);
  intnat stop = Long_val(vstop);
  unsigned char *spans = emit32 ? NULL : Bytes_val(vspans);
  intnat spans_cap = emit32 ? 0 : (intnat)(caml_string_length(vspans) / 8);
  value *ids_base = emit32 ? NULL : (value *)&Field(vids, 0);
  int32_t *ids32_base = emit32 ? (int32_t *)Caml_ba_data_val(vids) : NULL;
  intnat ids_cap = emit32 ? Caml_ba_array_val(vids)->dim[0]
                          : (intnat)Wosize_val(vids);
  value *marks_base = emit32 ? NULL : (value *)&Field(vmarks, 0);
  unsigned char *cur = Bytes_val(vcursor);
  const value *byte_ids = (const value *)&Field(Field(vt, BROT_BL_BYTE_IDS), 0);
  const unsigned char *words = Bytes_val(Field(vt, BROT_BL_WORDS));
  intnat words_mask = Long_val(Field(vt, BROT_BL_WORDS_MASK));
  intnat nspans = emit32 ? 0 : (intnat)load64(cur + CUR_SPANS);
  intnat nids = (intnat)load64(cur + CUR_IDS);
  intnat nmarks = emit32 ? 0 : (intnat)load64(cur + CUR_MARKS);
  intnat i = Long_val(vpos);
  intnat digits = split ? Long_val(vwalker) >> 2 : 0;
  int o200k = split ? (int)((Long_val(vwalker) >> 1) & 1) : 0;
  int flag = split ? (int)(Long_val(vwalker) & 1) : 0;
  intnat resume, added = 0;
  int reason;
  tables t;
  int cached;
  walker w;
  mask_state ms;
  uint32_t ends[STAGE + HARVEST_LANES];
  intnat stages = 0;
  staged *st = (staged *)(cur + CUR_STAGE);
  intnat nb = (intnat)load64(cur + CUR_STAGED);
  intnat k = (intnat)load64(cur + CUR_NEXT);
  const staged *back = NULL;

  tables_of(&t, vt, BROT_BL_CACHE, BROT_BL_CACHE_BASE, BROT_BL_CACHE_MASK,
            BROT_BL_MERGE_KEYS, BROT_BL_MERGE_VALUES, BROT_BL_MERGE_MASK,
            BROT_BL_LEN_TABLE);
  cached = t.cache_mask >= 0;
  w.s = s;
  w.stop = stop;
  w.lead = Bytes_val(Field(vt, BROT_BL_LEAD));
  w.uni = Bytes_val(vunicode);
  w.uni_len = (intnat)caml_string_length(vunicode);
  w.cp = 0;
  if (nb <= STAGE_CAP && k < nb && st[k].i == i
      && (intnat)load64(cur + CUR_STAGE_STOP) == stop)
    i = st[nb - 1].e; /* the walk goes on after the stage */
  else
    nb = k = 0;
  store64(cur + CUR_STAGE_STOP, (uint64_t)stop);
  if (!o200k) mask_init(&ms, i, split ? digits : -1, flag);

  for (;;) {
    int class_hit = 0, roomy;
    if (k == nb) {
      intnat limit = STAGE_CAP;
      stages++;
      if (!emit32 && spans_cap - nspans < limit) limit = spans_cap - nspans;
      nb = k = 0;
      if (o200k)
        while (nb < limit && nb < STAGE && i < stop) {
          intnat e = o200k_next_span(&w, i, digits, flag);
          if (e < 0) {
            class_hit = 1;
            break;
          }
          stage(&t, s, n, &st[nb], i, e);
          nb++;
          i = e;
        }
      else {
        /* The walk harvests whole batches, so a stage can pass [STAGE]; past
           [limit], which only a short spans buffer brings below it, the walk
           is taken back; with no room at all it does not start. */
        while (nb < STAGE && nb < limit && ms.pos < stop && !ms.failed)
          nb = harvest(&ms, &w, ends, nb);
        if (nb > limit) {
          nb = limit;
          rewind_walk(&ms, (intnat)ends[nb - 1]);
        }
        stage_spans(&t, s, n, st, i, ends, nb);
        if (nb > 0) i = (intnat)ends[nb - 1];
        class_hit = ms.failed;
      }
    }
    /* A short span needs at most 15 ids of room: when the stage's worth is
       there, only a long span's room is checked. */
    roomy = ids_cap - nids >= 15 * (nb - k);
    for (; k < nb; k++) {
      const staged *p = &st[k];
      intnat len = p->e - p->i;
      uint64_t v[2] = {0, 0};
      int hit;
      intnat count;
      if ((!roomy || len > 15) && ids_cap - nids < (len < 4 ? 4 : len)) {
        reason = BROT_IDS_FULL;
        resume = p->i;
        goto out;
      }
      /* A hit's lanes are stored whether or not it hit, and counted only when
         it did: the one branch left is the miss. */
      hit = cached && probe_set(&t, p, v);
      count = emit32 ? emit_lanes32(ids32_base + nids, v[0], v[1])
                     : emit_lanes(ids_base + nids, v[0], v[1]);
      if (hit)
        nids += count;
      else if (len > 15) {
        back = p;
        goto hand_back;
      } else if (cached && probe_neighbour(&t, p, v))
        nids += emit32 ? emit_lanes32(ids32_base + nids, v[0], v[1])
                       : emit_lanes(ids_base + nids, v[0], v[1]);
      else {
        int32_t ids15[15];
        intnat word = words_mask < 0 ? -1
                                     : whole_word(words, words_mask, &t, s, n,
                                                  p, len);
        int m;
        if (word >= 0) {
          ids15[0] = (int32_t)word;
          m = 1;
        } else
          m = word == -2 ? -1
                         : merge_short(t.mkeys, t.mvals, t.mmask, byte_ids,
                                       t.len_tab, t.len_n, s, p->i, len,
                                       ids15);
        if (m < 0) {
          back = p;
          goto hand_back;
        }
        emit_ids(ids_base, ids32_base, emit32, nids, ids15, m);
        nids += m;
        added += remember(&t, p, ids15, m);
      }
      if (!emit32) {
        /* Spans.write's word: the start in the low 32 bits, the stop in the
           high; the mark is the id count after the span. */
        store64(spans + 8 * nspans, ((uint64_t)p->e << 32) | (uint64_t)p->i);
        marks_base[nmarks++] = Val_long(nids);
        nspans++;
      }
    }
    if (!emit32 && nspans == spans_cap && (class_hit || i < stop)) {
      reason = BROT_SPANS_FULL;
      resume = i;
      goto out;
    }
    if (class_hit) {
      store64(cur + CUR_CP, (uint64_t)w.cp);
      reason = BROT_CLASS;
      resume = i;
      goto out;
    }
    if (i >= stop) {
      reason = BROT_DONE;
      resume = i;
      goto out;
    }
    if (stages == YIELD_STAGES) {
      reason = BROT_SPANS_FULL;
      resume = i;
      goto out;
    }
  }
hand_back:
  /* The span needs OCaml. Recording, it is written and published but its
     ids and mark are not: the driver encodes it, appends its mark and
     resumes after it. Otherwise its end rides in the cursor. */
  if (emit32)
    store64(cur + CUR_CP, (uint64_t)back->e);
  else {
    store64(spans + 8 * nspans, ((uint64_t)back->e << 32) | (uint64_t)back->i);
    nspans++;
  }
  reason = BROT_ENCODE;
  resume = back->i;
  k++;
out:
  if (!emit32) {
    store64(cur + CUR_SPANS, (uint64_t)nspans);
    store64(cur + CUR_MARKS, (uint64_t)nmarks);
  }
  store64(cur + CUR_IDS, (uint64_t)nids);
  store64(cur + CUR_RESUME, (uint64_t)resume);
  store64(cur + CUR_STAGED, (uint64_t)nb);
  store64(cur + CUR_NEXT, (uint64_t)k);
  store64(cur + CUR_ADDED, (uint64_t)added);
  return Val_int(reason);
}

CAMLprim value brot_byte_level_encode(value text, value vpos, value vstop,
                                      value vspans, value vids, value vmarks,
                                      value vcursor, value vunicode, value vt)
{
  return byte_level_encode_impl(text, vpos, vstop, vspans, vids, vmarks,
                                vcursor, vunicode, vt, Val_long(0), 0, 0);
}

CAMLprim value brot_byte_level_encode_byte(value *argv, int argn)
{
  (void)argn;
  return brot_byte_level_encode(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6], argv[7], argv[8]);
}

CAMLprim value brot_byte_level_encode_ids32(value text, value vpos,
                                            value vstop, value vids,
                                            value vcursor, value vunicode,
                                            value vt)
{
  return byte_level_encode_impl(text, vpos, vstop, Val_unit, vids, Val_unit,
                                vcursor, vunicode, vt, Val_long(0), 1, 0);
}

CAMLprim value brot_byte_level_encode_ids32_byte(value *argv, int argn)
{
  (void)argn;
  return brot_byte_level_encode_ids32(argv[0], argv[1], argv[2], argv[3],
                                      argv[4], argv[5], argv[6]);
}

CAMLprim value brot_split_encode(value text, value vpos, value vstop,
                                  value vspans, value vids, value vmarks,
                                  value vcursor, value vunicode, value vt,
                                  value vwalker)
{
  return byte_level_encode_impl(text, vpos, vstop, vspans, vids, vmarks,
                                vcursor, vunicode, vt, vwalker, 0, 1);
}

CAMLprim value brot_split_encode_byte(value *argv, int argn)
{
  (void)argn;
  return brot_split_encode(argv[0], argv[1], argv[2], argv[3], argv[4],
                            argv[5], argv[6], argv[7], argv[8], argv[9]);
}

CAMLprim value brot_split_encode_ids32(value text, value vpos, value vstop,
                                        value vids, value vcursor,
                                        value vunicode, value vt,
                                        value vwalker)
{
  return byte_level_encode_impl(text, vpos, vstop, Val_unit, vids, Val_unit,
                                vcursor, vunicode, vt, vwalker, 1, 1);
}

CAMLprim value brot_split_encode_ids32_byte(value *argv, int argn)
{
  (void)argn;
  return brot_split_encode_ids32(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7]);
}

/* The fused SentencePiece kernel: Bpe.encode_into's unit walker, the same
   staged probe, and the short merge initialized per character rather than
   per byte. No Unicode classes come into it — the only boundary candidates
   are the three ▁ bytes and the eight punctuation split bytes — so the walk
   never fails and Class is never returned. An Encode
   hand-back carries the unit through the cursor alone: its start in
   CUR_RESUME, its end in the word Class code points use, and no span or
   mark buffer is touched — the caller records the whole stretch as one
   span, exactly as the OCaml walker's caller does. */

/* Kernel.sp scan values (kernel.mli pins them): 1 + slot for a punctuation
   split byte, this for the first byte of the marker. */
#define SP_SCAN_MARK 9

/* The unit walk finds its boundary candidates 64 bytes at a time: a bit per
   byte that the scan table can open something at, then the rules of
   Bpe.encode_into's cut on the candidates alone. The vector paths test the
   bytes Bpe.sp_split_bytes and the marker's first byte [lead] can be, a
   superset of the bytes the scan table marks (a slot can be dropped), so
   every candidate is checked against the table; the last partial block
   reads the table directly and no byte at or past [stop] is read. */

#if defined(BROT_MASK_NEON)
static uint64_t sp_block(const unsigned char *p, unsigned char lead)
{
  static const uint8_t opens[8] = {'.', ',', '"', ')', ';', ':', '!', '?'};
  uint8x16_t m[4];
  int k, j;
  for (k = 0; k < 4; k++) {
    uint8x16_t v = vld1q_u8(p + 16 * k);
    uint8x16_t a = vceqq_u8(v, vdupq_n_u8(lead));
    for (j = 0; j < 8; j++) a = vorrq_u8(a, vceqq_u8(v, vdupq_n_u8(opens[j])));
    m[k] = a;
  }
  return movemask64(m[0], m[1], m[2], m[3]);
}
#elif defined(BROT_MASK_SSE2)
static uint64_t sp_block(const unsigned char *p, unsigned char lead)
{
  static const char opens[8] = {'.', ',', '"', ')', ';', ':', '!', '?'};
  uint64_t m = 0;
  int k, j;
  for (k = 0; k < 4; k++) {
    __m128i v = _mm_loadu_si128((const __m128i *)(p + 16 * k));
    __m128i a = _mm_cmpeq_epi8(v, _mm_set1_epi8((char)lead));
    for (j = 0; j < 8; j++)
      a = _mm_or_si128(a, _mm_cmpeq_epi8(v, _mm_set1_epi8(opens[j])));
    m |= (uint64_t)(unsigned)_mm_movemask_epi8(a) << (16 * k);
  }
  return m;
}
#endif

/* The candidates among the [count] bytes at [p], from the scan table. */
static uint64_t sp_block_scalar(const unsigned char *p, intnat count,
                                const unsigned char *scan)
{
  uint64_t m = 0;
  intnat k;
  for (k = 0; k < count; k++) m |= (uint64_t)(scan[p[k]] != 0) << k;
  return m;
}

typedef struct {
  const unsigned char *s;
  intnat stop;
  intnat anchor; /* where the call began: the stretch start or a boundary */
  const unsigned char *scan;
  const value *safe;
  const unsigned char *mark; /* Kernel.sp's marker, [mlen] bytes */
  intnat mlen;
  intnat base;   /* the block the candidates are relative to */
  uint64_t cand; /* its candidates not yet passed */
} sp_walker;

static void sp_load(sp_walker *u)
{
  intnat left = u->stop - u->base;
#if defined(BROT_MASK_NEON) || defined(BROT_MASK_SSE2)
  if (left >= 64) {
    u->cand = sp_block(u->s + u->base, u->mark[0]);
    return;
  }
#endif
  u->cand = sp_block_scalar(u->s + u->base, left < 64 ? left : 64, u->scan);
}

static void sp_init(sp_walker *u, const unsigned char *s, intnat stop,
                    intnat anchor, const unsigned char *scan,
                    const value *safe, value mark)
{
  u->s = s;
  u->stop = stop;
  u->anchor = anchor;
  u->scan = scan;
  u->safe = safe;
  u->mark = (const unsigned char *)String_val(mark);
  u->mlen = (intnat)caml_string_length(mark);
  u->base = anchor;
  sp_load(u);
}

/* Bpe.marked: whether the marker is at [p], within [stop], for a [p] in
   [anchor, stop). A spaced model's marker is the one byte a space is. */
static inline int sp_marked(const sp_walker *u, intnat p)
{
  intnat k;
  if (u->mlen == 1) return u->s[p] == u->mark[0];
  if (p + u->mlen > u->stop) return 0;
  for (k = 0; k < u->mlen; k++)
    if (u->s[p + k] != u->mark[k]) return 0;
  return 1;
}

/* Bpe.encode_into's unit cut: the end of the unit opening at [b], which is
   the previous call's return. A marker ends the unit unless it extends a run
   of them; a split byte ends it after a byte the vocabulary never joins it
   to. The run test reads back to [anchor], which stands in for the walker's
   [pos]: within [anchor, anchor + mlen) the look-behind window always covers
   the boundary's opening byte, a marker's first byte or an ASCII punctuation
   byte, never a byte inside a marker, so the guard resolves as the walker's
   would. */
static BROT_INLINE intnat sp_next(sp_walker *u, intnat b)
{
  const unsigned char *s = u->s;
  for (;;) {
    while (u->cand != 0) {
      intnat p = u->base + ctz64(u->cand);
      intnat k;
      if (p > b) {
        k = u->scan[s[p]];
        if (k == SP_SCAN_MARK) {
          if (sp_marked(u, p)
              && !(p >= u->anchor + u->mlen && sp_marked(u, p - u->mlen)))
            return p;
        } else if (k != 0 && Bool_val(u->safe[((k - 1) << 8) | s[p - 1]]))
          return p;
      }
      u->cand &= u->cand - 1;
    }
    u->base += 64;
    if (u->base >= u->stop) return u->stop;
    sp_load(u);
  }
}

/* Char_class.length: the byte length of the character at [p], of the [left] bytes
   that are left: a lead byte with all the continuation bytes it asks for, or
   else that one byte, so a stray byte never takes in the bytes after it. */
static inline intnat char_len(const unsigned char *p, intnat left)
{
  intnat b = p[0], n, k;
  if (b < 0xC2) return 1;
  n = b < 0xE0 ? 2 : b < 0xF0 ? 3 : b < 0xF8 ? 4 : 1;
  if (n == 1 || n > left) return 1;
  for (k = 1; k < n; k++)
    if ((p[k] & 0xC0) != 0x80) return 1;
  return n;
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
   with a direct piece or a byte fallback for each of their bytes, then the
   shared merge run. Returns the id count, or -1 when the unit must go back
   to OCaml: a character with neither (the unknown token's semantics stay
   there; a byte that opens no whole character is a character of its own,
   whose key is no whole character's, and lands on its fallback or here
   too), or the merge-run refusal. A fallback symbol
   accounts for its one byte, so it never trips that refusal. The character
   map is probed with merge_find: the two maps share the Merge_map
   layout. */
static int sp_merge_short(const tables *t, const value *ascii_ids,
                          const value *ckeys, const value *cvals, intnat cmask,
                          const value *fallback_ids, const unsigned char *s,
                          intnat pos, intnat len, int32_t out[15])
{
  intnat sym_c[15], sym_len[15], sym_prev[15], sym_next[15];
  intnat n = 0, k = 0;
  while (k < len) {
    intnat b = s[pos + k], id, blen;
    if (b < 128) {
      blen = 1;
      id = Long_val(ascii_ids[b]);
    } else {
      blen = char_len(s + pos + k, len - k);
      id = merge_find(ckeys, cvals, cmask, pack_char_key(s + pos + k, blen));
    }
    if (id >= 0) {
      sym_c[n] = id;
      sym_len[n] = blen;
      n++;
    } else {
      intnat j;
      for (j = 0; j < blen; j++)
        if (Long_val(fallback_ids[s[pos + k + j]]) < 0) return -1;
      for (j = 0; j < blen; j++) {
        sym_c[n] = Long_val(fallback_ids[s[pos + k + j]]);
        sym_len[n] = 1;
        n++;
      }
    }
    k += blen;
  }
  for (k = 0; k < n; k++) {
    sym_prev[k] = k - 1;
    sym_next[k] = k + 1 < n ? k + 1 : -1;
  }
  return merge_run(t->mkeys, t->mvals, t->mmask, t->len_tab, t->len_n, n,
                   sym_c, sym_len, sym_prev, sym_next, out);
}

/* The SentencePiece entry's body. The caller validates the range and the
   cursor's id count once; ids room is checked here per unit, a hit storing
   its four lanes unconditionally as Ints.add4 does. [emit32] selects the
   ids sink as in the byte-level body, and units are staged as spans are
   there. */
static BROT_INLINE value sp_encode_impl(value text, value vpos,
                                            value vstop, value vids,
                                            value vcursor, value vt,
                                            int emit32)
{
  const unsigned char *s = (const unsigned char *)String_val(text);
  intnat n = (intnat)caml_string_length(text);
  intnat stop = Long_val(vstop);
  value *ids_base = emit32 ? NULL : (value *)&Field(vids, 0);
  int32_t *ids32_base = emit32 ? (int32_t *)Caml_ba_data_val(vids) : NULL;
  intnat ids_cap = emit32 ? Caml_ba_array_val(vids)->dim[0]
                          : (intnat)Wosize_val(vids);
  unsigned char *cur = Bytes_val(vcursor);
  const value *ascii_ids =
      (const value *)&Field(Field(vt, BROT_SP_ASCII_IDS), 0);
  const value *ckeys = (const value *)&Field(Field(vt, BROT_SP_CHAR_KEYS), 0);
  const value *cvals =
      (const value *)&Field(Field(vt, BROT_SP_CHAR_VALUES), 0);
  intnat cmask = Long_val(Field(vt, BROT_SP_CHAR_MASK));
  const value *fallback_ids =
      (const value *)&Field(Field(vt, BROT_SP_FALLBACK_IDS), 0);
  intnat nids = (intnat)load64(cur + CUR_IDS);
  intnat i = Long_val(vpos);
  intnat resume, added = 0;
  int reason;
  tables t;
  int cached;
  sp_walker u;
  intnat stages = 0;
  staged *st = (staged *)(cur + CUR_STAGE);
  intnat nb = (intnat)load64(cur + CUR_STAGED);
  intnat k = (intnat)load64(cur + CUR_NEXT);
  const staged *back = NULL;

  tables_of(&t, vt, BROT_SP_CACHE, BROT_SP_CACHE_BASE, BROT_SP_CACHE_MASK,
            BROT_SP_MERGE_KEYS, BROT_SP_MERGE_VALUES, BROT_SP_MERGE_MASK,
            BROT_SP_LEN_TABLE);
  cached = t.cache_mask >= 0;
  if (nb <= STAGE_CAP && k < nb && st[k].i == i
      && (intnat)load64(cur + CUR_STAGE_STOP) == stop)
    i = st[nb - 1].e; /* the walk goes on after the stage */
  else
    nb = k = 0;
  store64(cur + CUR_STAGE_STOP, (uint64_t)stop);
  sp_init(&u, s, stop, i, Bytes_val(Field(vt, BROT_SP_SCAN)),
          (const value *)&Field(Field(vt, BROT_SP_PUNCT_SAFE), 0),
          Field(vt, BROT_SP_MARKER));

  for (;;) {
    int roomy;
    if (k == nb) {
      stages++;
      nb = k = 0;
      while (nb < STAGE && i < stop) {
        intnat e = sp_next(&u, i);
        stage(&t, s, n, &st[nb], i, e);
        nb++;
        i = e;
      }
    }
    roomy = ids_cap - nids >= 15 * (nb - k); /* as in the byte-level entry */
    for (; k < nb; k++) {
      const staged *p = &st[k];
      intnat len = p->e - p->i;
      uint64_t v[2] = {0, 0};
      int hit;
      intnat count;
      if (len > 15) {
        back = p;
        goto hand_back;
      }
      if (!roomy && ids_cap - nids < (len < 4 ? 4 : len)) {
        reason = BROT_IDS_FULL;
        resume = p->i;
        goto out;
      }
      hit = cached && probe_set(&t, p, v);
      count = emit32 ? emit_lanes32(ids32_base + nids, v[0], v[1])
                     : emit_lanes(ids_base + nids, v[0], v[1]);
      if (hit)
        nids += count;
      else if (cached && probe_neighbour(&t, p, v))
        nids += emit32 ? emit_lanes32(ids32_base + nids, v[0], v[1])
                       : emit_lanes(ids_base + nids, v[0], v[1]);
      else {
        int32_t ids15[15];
        int m = sp_merge_short(&t, ascii_ids, ckeys, cvals, cmask,
                               fallback_ids, s, p->i, len, ids15);
        if (m < 0) {
          back = p;
          goto hand_back;
        }
        emit_ids(ids_base, ids32_base, emit32, nids, ids15, m);
        nids += m;
        added += remember(&t, p, ids15, m);
      }
    }
    if (i >= stop) {
      reason = BROT_DONE;
      resume = i;
      goto out;
    }
    if (stages == YIELD_STAGES) {
      reason = BROT_SPANS_FULL;
      resume = i;
      goto out;
    }
  }
hand_back:
  /* Nothing was appended for the unit: the driver encodes it in OCaml and
     resumes at its end. */
  store64(cur + CUR_CP, (uint64_t)back->e);
  reason = BROT_ENCODE;
  resume = back->i;
  k++;
out:
  store64(cur + CUR_IDS, (uint64_t)nids);
  store64(cur + CUR_RESUME, (uint64_t)resume);
  store64(cur + CUR_STAGED, (uint64_t)nb);
  store64(cur + CUR_NEXT, (uint64_t)k);
  store64(cur + CUR_ADDED, (uint64_t)added);
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

/* Kernel.cache_align. */
CAMLprim value brot_cache_align(value b)
{
  return Val_long((intnat)((0 - (uintptr_t)Bytes_val(b)) & 63));
}

/* For tests. */

/* Kernel.cache_hash: hash_of on two key words. */
CAMLprim value brot_cache_hash(value k0, value k1)
{
  return Val_long(
      (intnat)hash_of((uint64_t)Int64_val(k0), (uint64_t)Int64_val(k1)));
}
