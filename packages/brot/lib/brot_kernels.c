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
  BROT_SP_PUNCT_SAFE,
  BROT_SP_FALLBACK_IDS
};

/* Kernel.cursor byte offsets. */
enum { CUR_SPANS = 0, CUR_IDS = 8, CUR_MARKS = 16, CUR_RESUME = 24,
       CUR_CP = 32, CUR_STAGED = 40, CUR_NEXT = 48, CUR_STAGE_STOP = 56,
       CUR_STAGE = 64 };

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
#else
static inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
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

#elif defined(BROT_MASK_SSE2)

/* Sixteen bytes at a time: an unsigned range test is a saturating minimum
   that leaves a byte unchanged, and pmovmskb gathers one bit per byte. */

static inline __m128i sse_le(__m128i x, int c)
{
  return _mm_cmpeq_epi8(_mm_min_epu8(x, _mm_set1_epi8((char)c)), x);
}

static void batch_classify(const unsigned char *p, batch_classes *c)
{
  int k;
  c->letter = c->digit = c->space = c->ws = c->hi = c->apo = 0;
  for (k = 0; k < 4; k++) {
    __m128i v = _mm_loadu_si128((const __m128i *)(p + 16 * k));
    __m128i lowered = _mm_or_si128(v, _mm_set1_epi8(0x20));
    __m128i l = sse_le(_mm_sub_epi8(lowered, _mm_set1_epi8('a')), 25);
    __m128i d = sse_le(_mm_sub_epi8(v, _mm_set1_epi8('0')), 9);
    __m128i sp = _mm_cmpeq_epi8(v, _mm_set1_epi8(' '));
    __m128i ws = _mm_or_si128(sp, sse_le(_mm_sub_epi8(v, _mm_set1_epi8(9)), 4));
    __m128i ap = _mm_cmpeq_epi8(v, _mm_set1_epi8('\''));
    int sh = 16 * k;
    c->letter |= (uint64_t)(unsigned)_mm_movemask_epi8(l) << sh;
    c->digit |= (uint64_t)(unsigned)_mm_movemask_epi8(d) << sh;
    c->space |= (uint64_t)(unsigned)_mm_movemask_epi8(sp) << sh;
    c->ws |= (uint64_t)(unsigned)_mm_movemask_epi8(ws) << sh;
    c->hi |= (uint64_t)(unsigned)_mm_movemask_epi8(v) << sh;
    c->apo |= (uint64_t)(unsigned)_mm_movemask_epi8(ap) << sh;
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
   it through [next_span]; scalar overruns keep the 64-byte grid. The state
   lives for one kernel call — the masks are pure functions of the bytes,
   so a resumed call rebuilds them from its own anchor. */
typedef struct {
  intnat anchor;     /* the call's start position, carry-free by contract */
  intnat scan;       /* base of the next batch to classify */
  intnat mask_base;  /* base the rem/batch bits refer to */
  uint64_t rem;      /* pop-ready boundary bits of the current segment */
  uint64_t batch_usable;
  uint64_t batch_bad;
  intnat scalar_until; /* walk spans via next_span while below this */
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

/* [mask_next_span] once the segment is spent: the scalar walker over a bad
   zone or the tail, or the next segment or batch. */
static intnat mask_refill(mask_state *ms, walker *w, intnat i)
{
  for (;;) {
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
    batch_masks(w, ms->anchor, ms->scan, &ms->batch_usable, &ms->batch_bad);
    ms->mask_base = ms->scan;
    ms->scan += 64;
    mask_load_segment(ms, i, i > ms->mask_base ? i - ms->mask_base : 0);
  }
}

/* [next_span] through the masks: the end of the span opening at [i], or
   -1 with [w->cp] set. [i] is the previous call's return (the pending
   span's start), as the entry loop maintains. Popping a bit is the common
   case and stays inline. */
static inline intnat mask_next_span(mask_state *ms, walker *w, intnat i)
{
  if (ms->rem != 0) {
    intnat e = ms->mask_base + ctz64(ms->rem);
    ms->rem &= ms->rem - 1;
    return e;
  }
  return mask_refill(ms, w, i);
}

/* The cache probe — Pretoken_cache.key0/key1/hash_of, the direct-mapped front
   table and the two-way layout of the 64-byte back set, bit for bit: keys are
   built from little-endian loads and every word is stored native-endian, so
   the table OCaml writes and the table written here are one format. A key
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

/* Pretoken_cache.hash_of: OCaml truncates the shifted hash to 63 bits; only
   40 bits are set after the shift, so the two agree. The back set and front
   slot byte offsets are Pretoken_cache.set_at and front_at on this one
   hash. */
static inline uint64_t hash_of(uint64_t k0, uint64_t k1)
{
  return ((k0 ^ (k1 * 0x9E3779B97F4A7C15ULL)) * 0xD6E8FEB86659FD93ULL) >> 24;
}

/* The hit test of Pretoken_cache.encode on the back table: way 0 then way 1.
   Returns the byte offset of the way, or -1. */
static inline intnat probe(const unsigned char *tbl, intnat set, uint64_t k0,
                           uint64_t k1)
{
  if (load64(tbl + set) == k0 && load64(tbl + set + 8) == k1) return set;
  if (load64(tbl + set + 32) == k0 && load64(tbl + set + 40) == k1)
    return set + 32;
  return -1;
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

/* Pretoken_cache.promote: a back-table hit is written into its front slot, in
   cache_store's publish order. */
static inline void front_store(unsigned char *front, intnat fslot, uint64_t k0,
                               uint64_t k1, uint64_t v0, uint64_t v1)
{
  store64(front + fslot + 16, v0);
  store64(front + fslot + 24, v1);
  store64(front + fslot + 8, k1);
  store64(front + fslot, k0);
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

/* The staged probe, shared by both entries. A call walks up to [STAGE]
   spans ahead, keying each one and prefetching its back set, then resolves
   them in order: room checks, the front and back probes, the merge of a
   miss and its store. Every probe and store happens in the reference's span
   order, so the tables stay bit for bit the reference's; the walk only reads
   the text. Sixty-four spans of resolving cover a miss to memory while the
   walk itself runs as a tight loop.

   The stage lives in the cursor, after its counts, so that an exit at a
   staged span keeps the spans staged after it: the call that resumes at the
   next of them, on the same range, resolves them before walking on from
   the last. Any other call starts a stage of its own; the driver also
   restarts it at every walk it begins (Kernel.restart), since a range of
   another text could start and stop at the same positions. */

#define STAGE 64

/* The stages a call walks before it returns Spans_full: OCaml's runtime can
   stop a domain for a collection only between calls, and every other domain
   waits for it, so a call ends after 1024 spans whatever room is left. */
#define YIELD_STAGES 16

typedef struct {
  intnat i, e;       /* the span */
  intnat set, fslot; /* back set and front slot offsets; set is -1 uncached */
  uint64_t k0, k1;
} staged;

typedef struct {
  unsigned char *front, *cache;
  intnat front_mask, cache_mask;
  const value *mkeys, *mvals, *len_tab;
  intnat mmask, len_n;
} tables;

/* The bytes of each key word that a length keeps. */
static const uint64_t key_mask0[16] = {
    0,
    0xFF,
    0xFFFF,
    0xFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFFFFULL,
    0xFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL};
static const uint64_t key_mask1[16] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0,
    0xFF,
    0xFFFF,
    0xFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFFFFULL,
    0xFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFULL};

/* Stages the span [i, e): its key words, set and slot, and a prefetch of
   the set. Away from the end of the text both key words are two loads
   masked to the span; [key_word0] and [key_word1] cover the end. */
static inline void stage(const tables *t, const unsigned char *s, intnat n,
                         staged *st, intnat i, intnat e)
{
  intnat len = e - i;
  st->i = i;
  st->e = e;
  st->set = -1;
  if (len <= 15 && t->cache_mask >= 0) {
    uint64_t k0, k1, h;
    if (i + 16 <= n) {
      k0 = load64le(s + i) & key_mask0[len];
      k1 = (load64le(s + i + 8) & key_mask1[len]) | ((uint64_t)len << 56);
    } else {
      k0 = key_word0(s, n, i, len);
      k1 = key_word1(s, n, i, len);
    }
    h = hash_of(k0, k1);
    st->k0 = k0;
    st->k1 = k1;
    st->set = (intnat)((h & (uint64_t)t->cache_mask) << 6);
    st->fslot = (intnat)((h & (uint64_t)t->front_mask) << 5);
    BROT_PREFETCH(t->cache + st->set);
  }
}

/* The front then the back probe of a staged span, as Pretoken_cache.encode
   reads them: a back hit is promoted into the front slot. Returns whether it
   hit, with the value words in [v]. */
static inline int lookup(const tables *t, const staged *st, uint64_t v[2])
{
  const unsigned char *f = t->front + st->fslot;
  intnat way;
  if (st->set < 0) return 0;
  if (load64(f) == st->k0 && load64(f + 8) == st->k1) {
    v[0] = load64(f + 16);
    v[1] = load64(f + 24);
    return 1;
  }
  way = probe(t->cache, st->set, st->k0, st->k1);
  if (way < 0) return 0;
  v[0] = load64(t->cache + way + 16);
  v[1] = load64(t->cache + way + 24);
  front_store(t->front, st->fslot, st->k0, st->k1, v[0], v[1]);
  return 1;
}

/* Stores a merged miss of [m] ids when Pretoken_cache.encode would keep it:
   at most four, each within 24 bits. Pretoken_cache.value_lo/value_hi: two
   24-bit lanes per word, the count in bits 24..31 of the first, absent lanes
   zero. */
static inline void remember(const tables *t, const staged *st,
                            const int32_t *ids, int m)
{
  int k;
  if (st->set < 0 || m > 4) return;
  for (k = 0; k < m; k++)
    if (ids[k] >= 1 << 24) return;
  cache_store(t->cache, st->set, st->k0, st->k1,
              (uint64_t)ids[0] | ((uint64_t)m << 24)
                  | (m > 1 ? (uint64_t)ids[1] << 32 : 0),
              (m > 2 ? (uint64_t)ids[2] : 0)
                  | (m > 3 ? (uint64_t)ids[3] << 32 : 0));
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

static void tables_of(tables *t, value vt, int front, int front_mask,
                      int cache, int cache_mask, int mkeys, int mvals,
                      int mmask, int len_tab)
{
  t->front = Bytes_val(Field(vt, front));
  t->front_mask = Long_val(Field(vt, front_mask));
  t->cache = Bytes_val(Field(vt, cache));
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
   scanner, or the scalar cl100k or o200k walker under the parameters
   [vwalker] packs as Kernel.split_encode documents. Both are compile-time
   constants in each wrapper. */
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
  int can_merge = Bool_val(Field(vt, BROT_BL_MERGE));
  intnat nspans = emit32 ? 0 : (intnat)load64(cur + CUR_SPANS);
  intnat nids = (intnat)load64(cur + CUR_IDS);
  intnat nmarks = emit32 ? 0 : (intnat)load64(cur + CUR_MARKS);
  intnat i = Long_val(vpos);
  intnat digits = split ? Long_val(vwalker) >> 2 : 0;
  int o200k = split ? (int)((Long_val(vwalker) >> 1) & 1) : 0;
  int flag = split ? (int)(Long_val(vwalker) & 1) : 0;
  intnat resume;
  int reason;
  tables t;
  walker w;
  mask_state ms;
  intnat stages = 0;
  staged *st = (staged *)(cur + CUR_STAGE);
  intnat nb = (intnat)load64(cur + CUR_STAGED);
  intnat k = (intnat)load64(cur + CUR_NEXT);
  const staged *back = NULL;

  tables_of(&t, vt, BROT_BL_FRONT, BROT_BL_FRONT_MASK, BROT_BL_CACHE,
            BROT_BL_CACHE_MASK, BROT_BL_MERGE_KEYS, BROT_BL_MERGE_VALUES,
            BROT_BL_MERGE_MASK, BROT_BL_LEN_TABLE);
  w.s = s;
  w.stop = stop;
  w.lead = Bytes_val(Field(vt, BROT_BL_LEAD));
  w.uni = Bytes_val(vunicode);
  w.uni_len = (intnat)caml_string_length(vunicode);
  w.cp = 0;
  if (nb <= STAGE && k < nb && st[k].i == i
      && (intnat)load64(cur + CUR_STAGE_STOP) == stop)
    i = st[nb - 1].e; /* the walk goes on after the stage */
  else
    nb = k = 0;
  store64(cur + CUR_STAGE_STOP, (uint64_t)stop);
  if (!split) mask_init(&ms, i);

  for (;;) {
    int class_hit = 0, roomy;
    if (k == nb) {
      intnat limit = STAGE;
      stages++;
      if (!emit32 && spans_cap - nspans < limit) limit = spans_cap - nspans;
      nb = k = 0;
      while (nb < limit && i < stop) {
        intnat e = !split  ? mask_next_span(&ms, &w, i)
                   : o200k ? o200k_next_span(&w, i, digits, flag)
                           : cl100k_next_span(&w, i, digits, flag);
        if (e < 0) {
          class_hit = 1;
          break;
        }
        stage(&t, s, n, &st[nb], i, e);
        nb++;
        i = e;
      }
    }
    /* A short span needs at most 15 ids of room: when the stage's worth is
       there, only a long span's room is checked. */
    roomy = ids_cap - nids >= 15 * (nb - k);
    for (; k < nb; k++) {
      const staged *p = &st[k];
      intnat len = p->e - p->i;
      uint64_t v[2];
      if ((!roomy || len > 15) && ids_cap - nids < (len < 4 ? 4 : len)) {
        reason = BROT_IDS_FULL;
        resume = p->i;
        goto out;
      }
      if (len > 15) {
        back = p;
        goto hand_back;
      }
      if (lookup(&t, p, v))
        nids += emit32 ? emit_lanes32(ids32_base + nids, v[0], v[1])
                       : emit_lanes(ids_base + nids, v[0], v[1]);
      else {
        int32_t ids15[15];
        int m = can_merge ? merge_short(t.mkeys, t.mvals, t.mmask, byte_ids,
                                        t.len_tab, t.len_n, s, p->i, len,
                                        ids15)
                          : -1;
        if (m < 0) {
          back = p;
          goto hand_back;
        }
        emit_ids(ids_base, ids32_base, emit32, nids, ids15, m);
        nids += m;
        remember(&t, p, ids15, m);
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
   split byte, this for the 0xE2 lead of ▁. */
#define SP_SCAN_MARK 9

/* The unit walk finds its boundary candidates 64 bytes at a time: a bit per
   byte that the scan table can open something at, then the rules of
   Bpe.encode_into's cut on the candidates alone. The vector paths test the
   bytes Bpe.sp_split_bytes and ▁'s lead can be, a superset of the bytes the
   scan table marks (a slot can be dropped), so every candidate is checked
   against the table; the last partial block reads the table directly and
   no byte at or past [stop] is read. */

#if defined(BROT_MASK_NEON)
static uint64_t sp_block(const unsigned char *p)
{
  static const uint8_t opens[9] = {0xE2, '.', ',', '"', ')', ';', ':', '!', '?'};
  uint8x16_t m[4];
  int k, j;
  for (k = 0; k < 4; k++) {
    uint8x16_t v = vld1q_u8(p + 16 * k);
    uint8x16_t a = vceqq_u8(v, vdupq_n_u8(opens[0]));
    for (j = 1; j < 9; j++) a = vorrq_u8(a, vceqq_u8(v, vdupq_n_u8(opens[j])));
    m[k] = a;
  }
  return movemask64(m[0], m[1], m[2], m[3]);
}
#elif defined(BROT_MASK_SSE2)
static uint64_t sp_block(const unsigned char *p)
{
  static const char opens[9] = {'\xE2', '.', ',', '"', ')', ';', ':', '!', '?'};
  uint64_t m = 0;
  int k, j;
  for (k = 0; k < 4; k++) {
    __m128i v = _mm_loadu_si128((const __m128i *)(p + 16 * k));
    __m128i a = _mm_cmpeq_epi8(v, _mm_set1_epi8(opens[0]));
    for (j = 1; j < 9; j++)
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
  intnat base;   /* the block the candidates are relative to */
  uint64_t cand; /* its candidates not yet passed */
} sp_walker;

static void sp_load(sp_walker *u)
{
  intnat left = u->stop - u->base;
#if defined(BROT_MASK_NEON) || defined(BROT_MASK_SSE2)
  if (left >= 64) {
    u->cand = sp_block(u->s + u->base);
    return;
  }
#endif
  u->cand = sp_block_scalar(u->s + u->base, left < 64 ? left : 64, u->scan);
}

static void sp_init(sp_walker *u, const unsigned char *s, intnat stop,
                    intnat anchor, const unsigned char *scan,
                    const value *safe)
{
  u->s = s;
  u->stop = stop;
  u->anchor = anchor;
  u->scan = scan;
  u->safe = safe;
  u->base = anchor;
  sp_load(u);
}

/* Bpe.encode_into's unit cut: the end of the unit opening at [b], which is
   the previous call's return. A ▁ ends the unit unless it extends a ▁ run;
   a split byte ends it after a byte the vocabulary never joins it to. The
   run test reads back to [anchor], which stands in for the walker's [pos]:
   within [anchor, anchor + 3) the look-behind window always covers the
   boundary's opening byte, an 0xE2 lead or an ASCII punctuation byte, never
   a mark continuation, so the guard resolves as the walker's would. */
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
          if (p + 3 <= u->stop && s[p + 1] == 0x96 && s[p + 2] == 0x81
              && !(p >= u->anchor + 3 && s[p - 3] == 0xE2 && s[p - 2] == 0x96
                   && s[p - 1] == 0x81))
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
   with a direct piece or a byte fallback for each of their bytes, then the
   shared merge run. Returns the id count, or -1 when the unit must go back
   to OCaml: a character with neither (the unknown token's semantics stay
   there; a truncated character packs no whole character's key and lands
   on its fallback or here too), or the merge-run refusal. A fallback symbol
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
      blen = utf8_len(b);
      if (blen > len - k) blen = len - k;
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
  intnat resume;
  int reason;
  tables t;
  sp_walker u;
  intnat stages = 0;
  staged *st = (staged *)(cur + CUR_STAGE);
  intnat nb = (intnat)load64(cur + CUR_STAGED);
  intnat k = (intnat)load64(cur + CUR_NEXT);
  const staged *back = NULL;

  tables_of(&t, vt, BROT_SP_FRONT, BROT_SP_FRONT_MASK, BROT_SP_CACHE,
            BROT_SP_CACHE_MASK, BROT_SP_MERGE_KEYS, BROT_SP_MERGE_VALUES,
            BROT_SP_MERGE_MASK, BROT_SP_LEN_TABLE);
  if (nb <= STAGE && k < nb && st[k].i == i
      && (intnat)load64(cur + CUR_STAGE_STOP) == stop)
    i = st[nb - 1].e; /* the walk goes on after the stage */
  else
    nb = k = 0;
  store64(cur + CUR_STAGE_STOP, (uint64_t)stop);
  sp_init(&u, s, stop, i, Bytes_val(Field(vt, BROT_SP_SCAN)),
          (const value *)&Field(Field(vt, BROT_SP_PUNCT_SAFE), 0));

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
      uint64_t v[2];
      if (len > 15) {
        back = p;
        goto hand_back;
      }
      if (!roomy && ids_cap - nids < (len < 4 ? 4 : len)) {
        reason = BROT_IDS_FULL;
        resume = p->i;
        goto out;
      }
      if (lookup(&t, p, v))
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
        remember(&t, p, ids15, m);
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
