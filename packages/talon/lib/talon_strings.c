/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* UTF-8 validation and literal search over the bytes of text, for
   strings.ml. Bytes are uint8 bigarrays and offsets int64 bigarrays, both
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

/* [talon_utf_8_row(b, o, mask, n)] is the first of the [n] rows of [b], row
   [r] being the bytes [o[r], o[r + 1]), that is not valid UTF-8, or -1. A
   row whose bit in [mask], [Some m], is clear is not read: bit [r mod 8] of
   byte [r / 8]. */
intnat talon_utf_8_row(value b, value o, value mask, intnat n)
{
  const uint8_t *v = Bytes_ba(b);
  const int64_t *off = Offsets_ba(o);
  const uint8_t *m = Is_block(mask) ? Bytes_ba(Field(mask, 0)) : NULL;
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

/* [talon_compare(b, o, n, s, out)] writes to [out] -1, 0 or 1 where each of
   the [n] rows of [b], cut by [o], orders before, as or after [s]: bytes
   compare as unsigned numbers, and a row orders before every row it is a
   prefix of. */
value talon_compare(value b, value o, intnat n, value s, value out)
{
  const uint8_t *v = Bytes_ba(b);
  const int64_t *off = Offsets_ba(o);
  const uint8_t *str = (const uint8_t *)String_val(s);
  size_t len = caml_string_length(s);
  int8_t *signs = (int8_t *)Caml_ba_data_val(out);
  for (intnat r = 0; r < n; r++) {
    size_t m = (size_t)(off[r + 1] - off[r]);
    int c = memcmp(v + off[r], str, m < len ? m : len);
    if (c == 0) c = (m > len) - (m < len);
    signs[r] = (int8_t)((c > 0) - (c < 0));
  }
  return Val_unit;
}

value talon_compare_byte(value b, value o, value n, value s, value out)
{
  return talon_compare(b, o, Long_val(n), s, out);
}
