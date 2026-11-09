/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* UTF-8 validation, literal search and comparison over the bytes of text,
   for strings.ml. Bytes are uint8 bigarrays and offsets int64 bigarrays, both
   in C layout. None allocates or raises, so all are [@@noalloc]. */

#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/mlvalues.h>

#define ASCII 0x8080808080808080ULL

/* [sequence(p, end)] is the length of the valid UTF-8 sequence that starts
   at [p], or 0. The ranges of the second byte exclude overlong forms,
   surrogates and values past U+10FFFF (RFC 3629, section 4). */
static int sequence(const uint8_t *p, const uint8_t *end)
{
  uint8_t b = p[0], lo = 0x80, hi = 0xbf;
  int n;
  if (b < 0x80) return 1;
  if (b < 0xc2) return 0;
  if (b < 0xe0) n = 2;
  else if (b < 0xf0) {
    n = 3;
    if (b == 0xe0) lo = 0xa0;
    else if (b == 0xed) hi = 0x9f;
  }
  else if (b < 0xf5) {
    n = 4;
    if (b == 0xf0) lo = 0x90;
    else if (b == 0xf4) hi = 0x8f;
  }
  else return 0;
  if (end - p < n || p[1] < lo || p[1] > hi) return 0;
  for (int k = 2; k < n; k++)
    if ((p[k] & 0xc0) != 0x80) return 0;
  return n;
}

/* [invalid(p, end)] is the first byte of [p, end) that starts no valid
   sequence, or NULL. ASCII runs are skipped eight bytes at a time. */
static const uint8_t *invalid(const uint8_t *p, const uint8_t *end)
{
  while (p < end) {
    uint64_t w;
    while (end - p >= 8 && (memcpy(&w, p, 8), (w & ASCII) == 0)) p += 8;
    while (p < end && *p < 0x80) p++;
    if (p == end) return NULL;
    int n = sequence(p, end);
    if (n == 0) return p;
    p += n;
  }
  return NULL;
}

#define Bytes_ba(v) ((const uint8_t *)Caml_ba_data_val(v))
#define Offsets_ba(v) ((const int64_t *)Caml_ba_data_val(v))

/* [talon_utf_8_invalid(b, i, stop)] is the first byte of [b]'s [i, stop)
   that starts no valid sequence, or -1. */
intnat talon_utf_8_invalid(value b, intnat i, intnat stop)
{
  const uint8_t *v = Bytes_ba(b);
  const uint8_t *p = invalid(v + i, v + stop);
  return p == NULL ? -1 : p - v;
}

value talon_utf_8_invalid_byte(value b, value i, value stop)
{
  return Val_long(talon_utf_8_invalid(b, Long_val(i), Long_val(stop)));
}

/* [ascii_run(p, end)] is the first byte of [p, end) at or above 0x80, or
   [end]. */
static const uint8_t *ascii_run(const uint8_t *p, const uint8_t *end)
{
  uint64_t w;
  while (end - p >= 8 && (memcpy(&w, p, 8), (w & ASCII) == 0)) p += 8;
  while (p < end && *p < 0x80) p++;
  return p;
}

/* [talon_utf_8_row(b, o, mask, n)] is the first of the [n] rows of [b], row
   [r] being the bytes [o[r], o[r + 1]), that is not valid UTF-8, or -1. A
   row whose bit in [mask], [Some m], is clear is not read: bit [r mod 8] of
   byte [r / 8]. The offsets never decrease, so the rows lie in
   [o[0], o[n]): when its bytes are ASCII, each row is valid however they are
   cut, and no row is read. */
intnat talon_utf_8_row(value b, value o, value mask, intnat n)
{
  const uint8_t *v = Bytes_ba(b);
  const int64_t *off = Offsets_ba(o);
  const uint8_t *m = Is_block(mask) ? Bytes_ba(Field(mask, 0)) : NULL;
  if (n > 0 && ascii_run(v + off[0], v + off[n]) == v + off[n]) return -1;
  for (intnat r = 0; r < n; r++) {
    if (m != NULL && ((m[r >> 3] >> (r & 7)) & 1) == 0) continue;
    if (invalid(v + off[r], v + off[r + 1]) != NULL) return r;
  }
  return -1;
}

value talon_utf_8_row_byte(value b, value o, value mask, value n)
{
  return Val_long(talon_utf_8_row(b, o, mask, Long_val(n)));
}

/* [talon_ascii_end(b, i, stop)] is the first byte of [b]'s [i, stop) at or
   above 0x80, or [stop]. */
intnat talon_ascii_end(value b, intnat i, intnat stop)
{
  const uint8_t *v = Bytes_ba(b);
  return ascii_run(v + i, v + stop) - v;
}

value talon_ascii_end_byte(value b, value i, value stop)
{
  return Val_long(talon_ascii_end(b, Long_val(i), Long_val(stop)));
}

/* [talon_ascii_case(src, i, stop, dst, at, upper)] writes the bytes [i, stop)
   of [src], all ASCII, to [dst] from [at], their case mapped: lower when
   [upper] is 0, upper otherwise. The loop is the shape the compiler vectorises.
   [src] and [dst] do not overlap. */
void talon_ascii_case(value src, intnat i, intnat stop, value dst, intnat at,
                      intnat upper)
{
  const uint8_t *s = Bytes_ba(src) + i;
  uint8_t *d = (uint8_t *)Caml_ba_data_val(dst) + at;
  intnat n = stop - i;
  const uint8_t from = upper ? 'a' : 'A';
  const uint8_t delta = upper ? (uint8_t)(0u - 0x20) : 0x20;
  for (intnat k = 0; k < n; k++) {
    uint8_t c = s[k];
    d[k] = c + ((uint8_t)(c - from) < 26 ? delta : 0);
  }
}

value talon_ascii_case_byte(value *argv, int argn)
{
  (void)argn;
  talon_ascii_case(argv[0], Long_val(argv[1]), Long_val(argv[2]), argv[3],
                   Long_val(argv[4]), Long_val(argv[5]));
  return Val_unit;
}

/* [talon_find(b, i, stop, s)] is the first byte of [b]'s [i, stop) at which
   the bytes of [s] lie whole, or -1. */
intnat talon_find(value b, intnat i, intnat stop, value s)
{
  const uint8_t *v = Bytes_ba(b);
  const uint8_t *str = (const uint8_t *)String_val(s);
  size_t n = caml_string_length(s);
  if (stop - i < (intnat)n) return -1;
  if (n == 0) return i;
  const uint8_t *p = v + i, *last = v + stop - n;
  while (p <= last) {
    p = memchr(p, str[0], (size_t)(last - p) + 1);
    if (p == NULL) return -1;
    if (memcmp(p + 1, str + 1, n - 1) == 0) return p - v;
    p++;
  }
  return -1;
}

value talon_find_byte(value b, value i, value stop, value s)
{
  return Val_long(talon_find(b, Long_val(i), Long_val(stop), s));
}

/* [talon_dfa_scan(t, c, v, s, i, stop, out)] runs the table [t] over the bytes
   [i, stop) of [v], from state [s] (its table row): a byte's class comes from
   [c], and the walk stops at the first entry that is not a state. It stores
   the resting state in [out[0]] and returns the byte it stopped at. */
intnat talon_dfa_scan(value t, value c, value v, intnat s, intnat i,
                      intnat stop, value out)
{
  const intnat *table = (const intnat *)Caml_ba_data_val(t);
  const uint8_t *classes = (const uint8_t *)Caml_ba_data_val(c);
  const uint8_t *b = Bytes_ba(v);
  while (i < stop) {
    intnat next = table[s + classes[b[i]]];
    if (next < 0) break;
    s = next;
    i++;
  }
  *(intnat *)Caml_ba_data_val(out) = s;
  return i;
}

value talon_dfa_scan_byte(value *argv, int argn)
{
  (void)argn;
  return Val_long(talon_dfa_scan(argv[0], argv[1], argv[2], Long_val(argv[3]),
                                 Long_val(argv[4]), Long_val(argv[5]), argv[6]));
}

/* Comparison with texts

   A row is compared with each text by its first eight bytes, read as one
   little-endian word, and calls memcmp only past them: past 16 bytes for
   equality, which reads no byte of a row of another length than the text's,
   and past 8 for order. */

/* [le(w)] is the word whose byte [i] in memory is its byte of weight [i]. */
static inline uint64_t le(uint64_t w)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  w = __builtin_bswap64(w);
#endif
  return w;
}

/* [head(p, n, end)] is the first [n] bytes from [p], at most 8, as a word that
   is 0 above them. It reads past [p + n] only before [end]. */
static inline uint64_t head(const uint8_t *p, size_t n, const uint8_t *end)
{
  uint64_t w = 0;
  if (end - p < 8) {
    memcpy(&w, p, n);
    return le(w);
  }
  memcpy(&w, p, 8);
  w = le(w);
  return n >= 8 ? w : w & ((1ULL << (8 * n)) - 1);
}

/* [tail(p, n)] is the last 8 of the [n] bytes from [p], [n] at least 8. */
static inline uint64_t tail(const uint8_t *p, size_t n)
{
  uint64_t w;
  memcpy(&w, p + n - 8, 8);
  return w;
}

typedef struct {
  const uint8_t *s;
  size_t len;
  uint64_t head;
} text;

/* [same(p, m, h, t)] is true iff the [m] bytes from [p], whose head is [h],
   are those of [t]. */
static inline int same(const uint8_t *p, size_t m, uint64_t h, const text *t)
{
  if (m != t->len || h != t->head) return 0;
  if (m <= 8) return 1;
  if (m <= 16) return tail(p, m) == tail(t->s, m);
  return memcmp(p + 8, t->s + 8, m - 8) == 0;
}

/* [order(p, m, h, t)] is -1, 0 or 1 where the [m] bytes from [p], whose head
   is [h], order before, as or after [t]: bytes compare as unsigned numbers,
   and a text orders before every text it is a prefix of. The first byte where
   the heads differ, below both lengths, decides. */
static inline int order(const uint8_t *p, size_t m, uint64_t h, const text *t)
{
  size_t k = m < t->len ? m : t->len;
  uint64_t x = h ^ t->head;
  if (k < 8) x &= (1ULL << (8 * k)) - 1;
  int c = 0;
  if (x != 0) {
    int i = __builtin_ctzll(x) >> 3;
    c = p[i] < t->s[i] ? -1 : 1;
  }
  else if (k > 8)
    c = memcmp(p + 8, t->s + 8, k - 8);
  if (c == 0) return (m > t->len) - (m < t->len);
  return (c > 0) - (c < 0);
}

/* [texts(a, first, u, t)] puts in [t] the [u] strings of the array [a] from
   [first], and is the mask of their lengths: bit [l] for a length [l] below
   63, bit 63 for 63 and longer. */
static uint64_t texts(value a, intnat first, int u, text *t)
{
  uint64_t lengths = 0;
  for (int j = 0; j < u; j++) {
    value s = Field(a, first + j);
    t[j].s = (const uint8_t *)String_val(s);
    t[j].len = caml_string_length(s);
    t[j].head = head(t[j].s, t[j].len < 8 ? t[j].len : 8, t[j].s + t[j].len);
    lengths |= 1ULL << (t[j].len < 63 ? t[j].len : 63);
  }
  return lengths;
}

/* At most this many texts are read per pass over the rows. */
#define PASS 8

/* [member(v, end, off, n, t, u, lengths, bits)] sets bit [r] of [bits] where
   row [r] of the [n] rows of [v], cut by [off], is one of the [u] texts [t],
   whose lengths are the mask [lengths]. A row of a length no text has is not
   read. */
static void member(const uint8_t *v, const uint8_t *end, const int64_t *off,
                   intnat n, const text *t, int u, uint64_t lengths,
                   uint8_t *bits)
{
  for (intnat base = 0; base < n; base += 8) {
    int rows = n - base < 8 ? (int)(n - base) : 8;
    unsigned byte = 0;
    for (int q = 0; q < rows; q++) {
      const uint8_t *p = v + off[base + q];
      size_t m = (size_t)(off[base + q + 1] - off[base + q]);
      if (((lengths >> (m < 63 ? m : 63)) & 1) == 0) continue;
      uint64_t h = head(p, m < 8 ? m : 8, end);
      unsigned hit = 0;
      for (int j = 0; j < u; j++) hit |= (unsigned)same(p, m, h, &t[j]);
      byte |= hit << q;
    }
    bits[base >> 3] |= (uint8_t)byte;
  }
}

/* [talon_member(b, o, n, a, out)] sets bit [r] of the bit array [out], bit
   [r mod 8] of byte [r / 8], iff row [r] of the [n] rows of [b], cut by [o],
   is one of the strings of the array [a], and clears it otherwise. */
value talon_member(value b, value o, intnat n, value a, value out)
{
  const uint8_t *v = Bytes_ba(b);
  const uint8_t *end = v + Caml_ba_array_val(b)->dim[0];
  uint8_t *bits = (uint8_t *)Caml_ba_data_val(out);
  intnat k = Wosize_val(a);
  memset(bits, 0, (size_t)((n + 7) / 8));
  for (intnat first = 0; first < k; first += PASS) {
    text t[PASS];
    int u = k - first < PASS ? (int)(k - first) : PASS;
    uint64_t lengths = texts(a, first, u, t);
    member(v, end, Offsets_ba(o), n, t, u, lengths, bits);
  }
  return Val_unit;
}

value talon_member_byte(value b, value o, value n, value a, value out)
{
  return talon_member(b, o, Long_val(n), a, out);
}

/* The sets of orders [talon_compare] accepts: bit [1 + c] for order [c]. */
enum { BEFORE = 1, SAME = 2, AFTER = 4 };

/* [talon_compare(b, o, n, s, accept, out)] sets bit [r] of the bit array
   [out], bit [r mod 8] of byte [r / 8], iff the order of row [r] of the [n]
   rows of [b], cut by [o], against [s] is in the set [accept], and clears it
   otherwise. Equality and inequality read no byte of a row of another
   length. */
value talon_compare(value b, value o, intnat n, value s, intnat accept,
                    value out)
{
  const uint8_t *v = Bytes_ba(b);
  const uint8_t *end = v + Caml_ba_array_val(b)->dim[0];
  const int64_t *off = Offsets_ba(o);
  uint8_t *bits = (uint8_t *)Caml_ba_data_val(out);
  size_t bytes = (size_t)((n + 7) / 8);
  text t;
  t.s = (const uint8_t *)String_val(s);
  t.len = caml_string_length(s);
  t.head = head(t.s, t.len < 8 ? t.len : 8, t.s + t.len);
  if (accept == SAME || accept == (BEFORE | AFTER)) {
    memset(bits, 0, bytes);
    member(v, end, off, n, &t, 1, 1ULL << (t.len < 63 ? t.len : 63), bits);
    if (accept == SAME) return Val_unit;
    for (size_t i = 0; i < bytes; i++) bits[i] = (uint8_t)~bits[i];
    if (n % 8 != 0) bits[bytes - 1] &= (uint8_t)((1u << (n % 8)) - 1);
    return Val_unit;
  }
  for (intnat base = 0; base < n; base += 8) {
    int rows = n - base < 8 ? (int)(n - base) : 8;
    unsigned byte = 0;
    for (int q = 0; q < rows; q++) {
      const uint8_t *p = v + off[base + q];
      size_t m = (size_t)(off[base + q + 1] - off[base + q]);
      int c = order(p, m, head(p, m < 8 ? m : 8, end), &t);
      byte |= (unsigned)((accept >> (1 + c)) & 1) << q;
    }
    bits[base >> 3] = (uint8_t)byte;
  }
  return Val_unit;
}

value talon_compare_byte(value *argv, int argn)
{
  (void)argn;
  return talon_compare(argv[0], argv[1], Long_val(argv[2]), argv[3],
                       Long_val(argv[4]), argv[5]);
}
