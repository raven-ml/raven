/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The fused byte-level kernel: the GPT-2 byte-level walker, the pretoken
   cache probe and the short byte-level BPE merge over a range of text, one
   call per chunk of spans. Everything rare is handed back to OCaml with a
   resume position: an unclassified code point, a pretoken over 15 bytes, a
   byte without a vocabulary id, a model whose misses may not be merged here,
   and a merge whose result stands for other bytes than its entry's. The
   OCaml implementations in pre_tokenizer.ml, char_class.ml and bpe.ml are
   the reference; every function here mirrors one there by name, and the
   native-vs-bytecode dump differential holds the two together.

   Nothing is allocated, raised or called back — the entry is [@@noalloc] —
   so no GC point exists, every derived pointer stays valid for the whole
   call and no CAMLparam rooting is needed. Stores into the ids and marks
   buffers are immediates over immediates in int arrays, which is what
   Array.unsafe_set compiles to and needs no write barrier (the OCaml 5
   deletion barrier only darkens pointer old values); Bytes stores are raw.
   All arithmetic that must wrap is unsigned. */

#include <stdint.h>
#include <string.h>

#include <caml/mlvalues.h>

/* ABI pins, mirrored in kernel.ml — change neither without the other. */

/* Kernel.reason constructor order. */
enum { BROT_DONE, BROT_SPANS_FULL, BROT_IDS_FULL, BROT_CLASS, BROT_ENCODE };

/* Kernel.byte_level field order. */
enum {
  BROT_BL_LEAD,
  BROT_BL_CACHE,
  BROT_BL_CACHE_MASK,
  BROT_BL_BYTE_IDS,
  BROT_BL_MERGE_KEYS,
  BROT_BL_MERGE_VALUES,
  BROT_BL_MERGE_MASK,
  BROT_BL_LEN_TABLE,
  BROT_BL_MERGE
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

/* The cache probe — Bpe.key0/key1/set_of and the two-way layout of the
   64-byte set, bit for bit: every word is loaded and stored native-endian,
   so the table OCaml writes and the table written here are one format. A key
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

/* Bpe.set_of: the byte offset of a key's set. OCaml truncates the Int64 hash
   to 63 bits before masking; the mask is far below bit 63, so the two
   agree. */
static inline intnat set_of(uint64_t k0, uint64_t k1, intnat mask)
{
  uint64_t h = (k0 ^ (k1 * 0x9E3779B97F4A7C15ULL)) * 0xD6E8FEB86659FD93ULL;
  return (intnat)(((h >> 24) & (uint64_t)mask) << 6);
}

/* The hit test of Bpe.encode_into: way 0 then way 1. Returns the byte offset
   of the way, or -1. */
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

/* Merges the [len] bytes at [s + pos] and writes the resulting ids to [out].
   Returns their count, or -1 when the span must go back to OCaml: a byte has
   no id (unk, fuse and byte-fallback semantics stay there), or a merged
   symbol stands for other bytes than len_table gives its id — the one case
   Bpe.emit_word would record in the opaque runs, which only OCaml appends
   to. Single bytes always account for exactly their byte, so the check can
   only fire on a merge result. */
static int merge_short(const value *mkeys, const value *mvals, intnat mmask,
                       const value *byte_ids, const value *len_tab,
                       intnat len_n, const unsigned char *s, intnat pos,
                       intnat len, int32_t out[15])
{
  intnat sym_c[15], sym_len[15], sym_prev[15], sym_next[15], rank[15];
  intnat k, cur;
  int m;
  for (k = 0; k < len; k++) {
    intnat id = Long_val(byte_ids[s[pos + k]]);
    if (id < 0) return -1;
    sym_c[k] = id;
    sym_len[k] = 1;
    sym_prev[k] = k - 1;
    sym_next[k] = k + 1 < len ? k + 1 : -1;
  }
  for (k = 0; k + 1 < len; k++)
    rank[k] = rank_of(mkeys, mvals, mmask, sym_c[k], sym_c[k + 1]);
  rank[len - 1] = RANK_NONE;
  for (;;) {
    intnat best = RANK_NONE, bp = -1, gone, next, prev;
    for (k = 0; k < len; k++)
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

/* The entry. The caller validates the range and the cursor counts once; ids
   room is checked here per span, a hit storing its four lanes
   unconditionally as Ints.add4 does. */
CAMLprim value brot_byte_level_encode(value text, value vpos, value vstop,
                                      value vspans, value vids, value vmarks,
                                      value vcursor, value vunicode, value vt)
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
  value *ids_base = (value *)&Field(vids, 0);
  intnat ids_cap = (intnat)Wosize_val(vids);
  value *marks_base = (value *)&Field(vmarks, 0);
  unsigned char *cur = Bytes_val(vcursor);

  const unsigned char *lead = Bytes_val(Field(vt, BROT_BL_LEAD));
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
  intnat i = Long_val(vpos);
  intnat resume;
  int reason;

  w.s = s;
  w.stop = stop;
  w.lead = lead;
  w.uni = Bytes_val(vunicode);
  w.uni_len = (intnat)caml_string_length(vunicode);
  w.cp = 0;

  /* The loop is software-pipelined over the cache line: span k's walk, keys
     and set are computed and the set's line prefetched one iteration ahead,
     so the walk of span k+1 runs while span k's line is in flight; span k's
     probe, merge and stores — and its room checks, in the reference's order —
     resolve after that walk. The observable exit states are the reference's
     exactly: a span whose resolve exits discards the span walked after it,
     which the reference would never have walked, and a walked span's Class
     hand-back is reported only once every span before it has resolved and
     the span room it would need has been checked. */
  {
    int have = 0;
    intnat p_i = 0, p_e = 0, p_len = 0, p_set = -1;
    uint64_t p_k0 = 0, p_k1 = 0;
    for (;;) {
      intnat e = -1, len = 0, set = -1;
      uint64_t k0 = 0, k1 = 0;
      int walked = 0;
      if (i < stop) {
        e = next_span(&w, i);
        walked = 1;
        if (e >= 0) {
          len = e - i;
          if (len <= 15 && cache_mask >= 0) {
            k0 = key_word0(s, n, i, len);
            k1 = key_word1(s, n, i, len);
            set = set_of(k0, k1, cache_mask);
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
          intnat way = probe(cache, p_set, p_k0, p_k1);
          if (way >= 0) {
            uint64_t v0 = load64(cache + way + 16);
            uint64_t v1 = load64(cache + way + 24);
            value *lane = ids_base + nids;
            lane[0] = Val_long((intnat)(v0 & 0xFFFFFF));
            lane[1] = Val_long((intnat)((v0 >> 32) & 0xFFFFFF));
            lane[2] = Val_long((intnat)(v1 & 0xFFFFFF));
            lane[3] = Val_long((intnat)((v1 >> 32) & 0xFFFFFF));
            nids += (intnat)((v0 >> 24) & 0xFF);
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

CAMLprim value brot_byte_level_encode_byte(value *argv, int argn)
{
  (void)argn;
  return brot_byte_level_encode(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6], argv[7], argv[8]);
}
