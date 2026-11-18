/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  FLAC frames (RFC 9639, section 9): a header checked by its CRC-8, one
  subframe a channel, zero bits to a byte and a CRC-16 over the frame. A
  subframe is a constant, its samples verbatim, or warm-up samples and a
  fixed or linear predictor whose residual is Rice coded in partitions.

  Every sum runs in 64 bits, wide enough for a 33-bit side sample times a
  15-bit coefficient over 32 terms (Appendix A.3). A residual must be below
  2^31 in magnitude (9.2.7.3) and a decoded sample must fit its subframe's
  depth, then the frame's after the stereo channels are restored, so a
  sample out of range raises where it is decoded and never feeds a later
  prediction.

  A decode starts at a frame, the cursor's byte and first row, and runs to
  the rows it is asked for: it drops the first [skip] rows, writes the
  next [rows] into [dst] from row [at], each sample in the top bits of an
  element of [width] bytes, and the rest of the last frame into [rest].
  Asked for no rows, it reads every frame to the stream's end, the stated
  length or, without one, the file's. It moves the cursor past the last
  frame it read.

  The stubs capture each bigarray's pointer and the stream's facts before
  they release the runtime lock, and then touch only bigarray storage,
  their locals and memory they allocated with malloc.
  --------------------------------------------------------------------------*/

#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HOST_BIG (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)

/* Streams */

/* A stream's facts, from STREAMINFO and its first frame: in [Fields],
   the order of the OCaml array [stream] in flac.ml. [samples] is whether a
   frame's coded number is its first row (RFC 9639, 9.1.5 and B.1), and
   [length] is -1 when STREAMINFO states none. [end] is the file's
   length. */
enum Fields {
  CHANNELS,
  BITS,
  RATE,
  MIN_BLOCK,
  MAX_BLOCK,
  MIN_FRAME,
  MAX_FRAME,
  LENGTH,
  STRATEGY,
  SAMPLES,
  END,
  WIDTH,
  FIELDS
};

typedef struct {
  int channels, bits, rate, min_block, max_block, strategy, samples, width;
  int64_t min_frame, max_frame, length, end;
} stream;

static void stream_of(value v, stream *s) {
  s->channels = Int_val(Field(v, CHANNELS));
  s->bits = Int_val(Field(v, BITS));
  s->rate = Int_val(Field(v, RATE));
  s->min_block = Int_val(Field(v, MIN_BLOCK));
  s->max_block = Int_val(Field(v, MAX_BLOCK));
  s->min_frame = Long_val(Field(v, MIN_FRAME));
  s->max_frame = Long_val(Field(v, MAX_FRAME));
  s->length = Long_val(Field(v, LENGTH));
  s->strategy = Int_val(Field(v, STRATEGY));
  s->samples = Int_val(Field(v, SAMPLES));
  s->end = Long_val(Field(v, END));
  s->width = Int_val(Field(v, WIDTH));
}

/* CRCs (9.1.8, 9.3): CRC-8 of polynomial x^8 + x^2 + x + 1 and CRC-16 of
   x^16 + x^15 + x^2 + 1, most significant bit first, from 0. */

typedef struct {
  uint8_t crc8[256];
  uint16_t crc16[256];
} crcs;

static void crcs_init(crcs *t) {
  for (int i = 0; i < 256; i++) {
    uint8_t c = (uint8_t)i;
    uint16_t d = (uint16_t)(i << 8);
    for (int k = 0; k < 8; k++) {
      c = (uint8_t)(c & 0x80 ? (c << 1) ^ 0x07 : c << 1);
      d = (uint16_t)(d & 0x8000 ? (d << 1) ^ 0x8005 : d << 1);
    }
    t->crc8[i] = c;
    t->crc16[i] = d;
  }
}

static uint8_t crc8(const crcs *t, const uint8_t *p, size_t n) {
  uint8_t c = 0;
  for (size_t i = 0; i < n; i++) c = t->crc8[c ^ p[i]];
  return c;
}

static uint16_t crc16(const crcs *t, const uint8_t *p, size_t n) {
  uint16_t c = 0;
  for (size_t i = 0; i < n; i++)
    c = (uint16_t)((c << 8) ^ t->crc16[(c >> 8) ^ p[i]]);
  return c;
}

/* Bits, most significant first */

/* [acc] holds the next [n] bits at its top and zeros below. A read past
   [end] sets [over] and gives zeros, so a malformed frame ends its loops
   at their bounds and is refused after them. */
typedef struct {
  const uint8_t *p, *end;
  uint64_t acc;
  int n, over;
} reader;

static inline void fill(reader *r) {
  while (r->n <= 56 && r->p < r->end) {
    r->acc |= (uint64_t)*r->p++ << (56 - r->n);
    r->n += 8;
  }
}

/* The next [k] bits, 0 to 56, unsigned. */
static inline uint64_t take(reader *r, int k) {
  if (k == 0) return 0;
  if (r->n < k) {
    fill(r);
    if (r->n < k) {
      r->over = 1;
      r->acc = 0;
      r->n = 0;
      return 0;
    }
  }
  uint64_t v = r->acc >> (64 - k);
  r->acc <<= k;
  r->n -= k;
  return v;
}

/* The next [k] bits, 0 to 56, two's complement. */
static inline int64_t take_signed(reader *r, int k) {
  if (k == 0) return 0;
  uint64_t sign = UINT64_C(1) << (k - 1);
  return (int64_t)(take(r, k) ^ sign) - (int64_t)sign;
}

/* The zeros before the next one, which it reads. */
static inline uint64_t zeros(reader *r) {
  uint64_t q = 0;
  for (;;) {
    if (r->n == 0) {
      fill(r);
      if (r->n == 0) {
        r->over = 1;
        return q;
      }
    }
    if (r->acc == 0) {
      q += (uint64_t)r->n;
      r->n = 0;
      continue;
    }
    int z = __builtin_clzll(r->acc);
    q += (uint64_t)z;
    r->acc = (r->acc << z) << 1;
    r->n -= z + 1;
    return q;
  }
}

/* The byte the reader is at, once its bits are aligned. */
static inline const uint8_t *at_byte(const reader *r) { return r->p - r->n / 8; }

/* Subframes (9.2) */

/* [floor_shift v s] is [v] divided by [2^s], rounded down: the right shift
   of a linear prediction (9.2.6), which C leaves to the compiler for a
   negative [v]. */
static inline int64_t floor_shift(int64_t v, int s) {
  return v >= 0 ? v >> s : -((-v - 1) >> s) - 1;
}

/* The residual of a subframe of [bs] samples after [order] warm-up
   samples, into [x] from [order] (9.2.7). */
static const char *residual(reader *r, int64_t *x, int bs, int order) {
  int method = (int)take(r, 2);
  if (method > 1) return "a residual's coding method is reserved";
  int pbits = method == 0 ? 4 : 5, escape = method == 0 ? 15 : 31;
  int po = (int)take(r, 4);
  int parts = 1 << po;
  if (bs % parts != 0)
    return "a residual's partition order does not divide its block";
  int per = bs >> po;
  if (per <= order)
    return "a residual's partitions are not longer than its predictor's order";
  int i = order;
  for (int p = 0; p < parts; p++) {
    int n = per - (p == 0 ? order : 0);
    int k = (int)take(r, pbits);
    if (k == escape) {
      int raw = (int)take(r, 5);
      for (int j = 0; j < n; j++) x[i++] = take_signed(r, raw);
    } else {
      for (int j = 0; j < n; j++) {
        /* A folded residual of 2^32 - 1 would be -2^31. */
        uint64_t q = zeros(r);
        if (q > (UINT64_C(0xFFFFFFFE) >> k))
          return "a residual is 2^31 or more in magnitude";
        uint64_t folded = (q << k) | take(r, k);
        if (folded > UINT64_C(0xFFFFFFFE))
          return "a residual is 2^31 or more in magnitude";
        x[i++] = (int64_t)(folded >> 1) ^ -(int64_t)(folded & 1);
      }
    }
    if (r->over) return "the frame ends past the file";
  }
  return NULL;
}

/* [fits v bits] is whether [v] is a two's complement number of [bits]. */
static inline int fits(int64_t v, int bits) {
  int64_t half = INT64_C(1) << (bits - 1);
  return v >= -half && v < half;
}

/* The fixed predictors of orders 0 to 4 (9.2.5). */
static int64_t fixed(const int64_t *x, int i, int order) {
  switch (order) {
  case 0:
    return 0;
  case 1:
    return x[i - 1];
  case 2:
    return 2 * x[i - 1] - x[i - 2];
  case 3:
    return 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3];
  default:
    return 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4];
  }
}

/* One subframe of [bs] samples of [bits] bits into [x]. */
static const char *subframe(reader *r, int64_t *x, int bs, int bits) {
  if (take(r, 1)) return "a subframe's first bit is set";
  int type = (int)take(r, 6);
  int wasted = 0;
  if (take(r, 1)) {
    uint64_t k = zeros(r) + 1;
    if (k >= (uint64_t)bits) return "a subframe's wasted bits leave it none";
    wasted = (int)k;
  }
  bits -= wasted;
  int order;
  if (type == 0) {
    int64_t v = take_signed(r, bits);
    for (int i = 0; i < bs; i++) x[i] = v;
  } else if (type == 1) {
    for (int i = 0; i < bs; i++) x[i] = take_signed(r, bits);
  } else if (type >= 8 && type <= 12) {
    order = type - 8;
    if (order > bs) return "a subframe's predictor order is past its block";
    for (int i = 0; i < order; i++) x[i] = take_signed(r, bits);
    const char *e = residual(r, x, bs, order);
    if (e) return e;
    for (int i = order; i < bs; i++) {
      x[i] += fixed(x, i, order);
      if (!fits(x[i], bits)) return "a predicted sample is past its depth";
    }
  } else if (type >= 32) {
    order = type - 31;
    if (order > bs) return "a subframe's predictor order is past its block";
    for (int i = 0; i < order; i++) x[i] = take_signed(r, bits);
    int precision = (int)take(r, 4) + 1;
    if (precision == 16) return "a predictor's precision is the forbidden 0b1111";
    int shift = (int)take_signed(r, 5);
    if (shift < 0) return "a predictor's shift is negative";
    int64_t coef[32];
    for (int j = 0; j < order; j++) coef[j] = take_signed(r, precision);
    const char *e = residual(r, x, bs, order);
    if (e) return e;
    for (int i = order; i < bs; i++) {
      int64_t sum = 0;
      for (int j = 0; j < order; j++) sum += coef[j] * x[i - 1 - j];
      x[i] += floor_shift(sum, shift);
      if (!fits(x[i], bits)) return "a predicted sample is past its depth";
    }
  } else
    return "a subframe's type is reserved";
  if (r->over) return "the frame ends past the file";
  if (wasted)
    for (int i = 0; i < bs; i++) x[i] = (int64_t)((uint64_t)x[i] << wasted);
  return NULL;
}

/* Frames (9.1) */

/* A frame's header: its block size, its coded number and where its
   subframes start. */
typedef struct {
  int bs, assignment;
  int64_t number;
  size_t body;
} header;

static const int rates[12] = {0,     88200, 176400, 192000, 8000,  16000,
                              22050, 24000, 32000,  44100,  48000, 96000};
static const int depths[8] = {0, 8, 12, 0, 16, 20, 24, 32};

/* The header of the frame at byte [pos] of [src], checked against [s] and
   by its CRC-8. */
static const char *frame_header(const uint8_t *src, const stream *s,
                                const crcs *t, int64_t pos, header *h) {
  const uint8_t *f = src + pos;
  int64_t avail = s->end - pos;
  if (avail < 5) return "the file ends inside a frame header";
  if (f[0] != 0xFF || (f[1] & 0xFE) != 0xF8) return "no frame sync code";
  if ((f[1] & 1) != s->strategy)
    return "the blocking strategy differs from the first frame's";
  int bs_code = f[2] >> 4, rate_code = f[2] & 15;
  int ch_code = f[3] >> 4, depth_code = (f[3] >> 1) & 7;
  if (f[3] & 1) return "a frame header's reserved bit is set";
  /* The coded number, a UTF-8-like code of up to 7 bytes (9.1.5), a frame
     number of at most 6. */
  int lead = f[4];
  int len = lead < 0x80   ? 1
            : lead < 0xC0 ? 0
            : lead < 0xE0 ? 2
            : lead < 0xF0 ? 3
            : lead < 0xF8 ? 4
            : lead < 0xFC ? 5
            : lead < 0xFE ? 6
            : lead == 0xFE ? 7
                           : 0;
  if (len == 0 || len > (s->samples ? 7 : 6))
    return "a frame's coded number is malformed";
  int extra = (bs_code == 6) + 2 * (bs_code == 7) + (rate_code == 12) +
              2 * (rate_code == 13 || rate_code == 14);
  if (avail < 4 + len + extra + 1) return "the file ends inside a frame header";
  int64_t number = len == 1 ? lead : lead & (0xFF >> (len + 1));
  int i = 5;
  for (int k = 1; k < len; k++, i++) {
    if ((f[i] & 0xC0) != 0x80) return "a frame's coded number is malformed";
    number = (number << 6) | (f[i] & 0x3F);
  }
  int bs;
  if (bs_code == 0) return "a frame's block size code is reserved";
  else if (bs_code == 1) bs = 192;
  else if (bs_code <= 5) bs = 144 << bs_code;
  else if (bs_code == 6) bs = f[i++] + 1;
  else if (bs_code == 7) {
    bs = ((f[i] << 8) | f[i + 1]) + 1;
    i += 2;
    if (bs == 65536) return "a frame's block size is the forbidden 65536";
  } else bs = 1 << bs_code;
  int rate;
  if (rate_code == 0) rate = s->rate;
  else if (rate_code <= 11) rate = rates[rate_code];
  else if (rate_code == 12) rate = f[i++] * 1000;
  else if (rate_code == 13) {
    rate = (f[i] << 8) | f[i + 1];
    i += 2;
  } else if (rate_code == 14) {
    rate = ((f[i] << 8) | f[i + 1]) * 10;
    i += 2;
  } else return "a frame's sample rate code is forbidden";
  if (crc8(t, f, (size_t)i) != f[i]) return "the frame header's CRC-8 differs";
  int channels = ch_code < 8 ? ch_code + 1 : 2;
  if (ch_code > 10) return "a frame's channel code is reserved";
  int bits = depth_code == 0 ? s->bits : depths[depth_code];
  if (depth_code == 3) return "a frame's bit depth code is reserved";
  if (rate != s->rate) return "a frame's sample rate differs from STREAMINFO's";
  if (channels != s->channels)
    return "a frame's channels differ from STREAMINFO's";
  if (bits != s->bits) return "a frame's bit depth differs from STREAMINFO's";
  if (bs > s->max_block)
    return "a frame's block is longer than STREAMINFO's longest";
  h->bs = bs;
  h->assignment = ch_code;
  h->number = number;
  h->body = (size_t)pos + (size_t)i + 1;
  return NULL;
}

/* The frame at the cursor [pos, row], decoded into [x], [channels] runs of
   [max_block] samples; [next] is the byte after it. */
static const char *frame(const uint8_t *src, const stream *s, const crcs *t,
                         int64_t pos, int64_t row, int64_t *x, int *bs,
                         int64_t *next) {
  header h;
  const char *e = frame_header(src, s, t, pos, &h);
  if (e) return e;
  int64_t expected = s->samples ? row : row / s->max_block;
  if (h.number != expected)
    return s->samples ? "a frame's coded row is not its first row"
                      : "a frame's coded number is not its number";
  if (s->length >= 0 && row + h.bs > s->length)
    return "the frame runs past STREAMINFO's length";
  if (h.bs < s->min_block && s->length >= 0 && row + h.bs < s->length)
    return "a frame shorter than STREAMINFO's shortest is not the last";
  reader r = {src + h.body, src + s->end, 0, 0, 0};
  for (int c = 0; c < s->channels; c++) {
    /* A side channel has a bit more (4.2). */
    int side = (h.assignment == 8 && c == 1) || (h.assignment == 9 && c == 0) ||
               (h.assignment == 10 && c == 1);
    e = subframe(&r, x + (size_t)c * s->max_block, h.bs, s->bits + side);
    if (e) return e;
  }
  int64_t *a = x, *b = x + s->max_block;
  for (int i = 0; i < h.bs && h.assignment >= 8; i++) {
    if (h.assignment == 8) b[i] = a[i] - b[i];
    else if (h.assignment == 9) a[i] = a[i] + b[i];
    else {
      int64_t mid = (int64_t)((uint64_t)a[i] << 1) | (b[i] & 1), side = b[i];
      a[i] = floor_shift(mid + side, 1);
      b[i] = floor_shift(mid - side, 1);
    }
  }
  for (int c = 0; c < s->channels && h.assignment >= 8; c++)
    for (int i = 0; i < h.bs; i++)
      if (!fits(x[(size_t)c * s->max_block + i], s->bits))
        return "a sample is past its depth";
  if (take(&r, r.n & 7)) return "a frame's padding bits are not zero";
  const uint8_t *footer = at_byte(&r);
  if (r.over || footer + 2 > src + s->end) return "the frame ends past the file";
  uint16_t stored = (uint16_t)((footer[0] << 8) | footer[1]);
  if (crc16(t, src + pos, (size_t)(footer - (src + pos))) != stored)
    return "the frame's CRC-16 differs";
  int64_t size = footer + 2 - (src + pos);
  if ((s->min_frame && size < s->min_frame) ||
      (s->max_frame && size > s->max_frame))
    return "a frame's size is outside STREAMINFO's sizes";
  *bs = h.bs;
  *next = pos + size;
  return NULL;
}

/* [store x channels from n width shift dst] writes rows [from] to
   [from + n - 1] of the frame [x] into [dst], each sample shifted into the
   top of an element of [width] bytes in the host's order. */
static void store(const int64_t *x, const stream *s, int from, int n,
                  int shift, uint8_t *dst) {
  for (int i = from; i < from + n; i++)
    for (int c = 0; c < s->channels; c++) {
      uint64_t v = (uint64_t)x[(size_t)c * s->max_block + i] << shift;
      if (s->width == 1) *dst = (uint8_t)v;
      else if (s->width == 2) {
        uint16_t h = (uint16_t)v;
        memcpy(dst, &h, 2);
      } else {
        uint32_t w = (uint32_t)v;
        memcpy(dst, &w, 4);
      }
      dst += s->width;
    }
}

/* [caml_nx_io_flac_decode src stream cursor skip dst at rows rest] decodes
   as the preamble states, [cursor] the array [|byte; row|], [rows] -1 for
   none, and is the rows it wrote into [rest]. It raises [Failure] with the
   reason alone, the cursor at the frame that failed. */
CAMLprim value caml_nx_io_flac_decode(value vsrc, value vstream,
                                      value vcursor, value vskip, value vdst,
                                      value vat, value vrows, value vrest) {
  CAMLparam5(vsrc, vstream, vcursor, vdst, vrest);
  stream s;
  stream_of(vstream, &s);
  const uint8_t *src = Caml_ba_data_val(vsrc);
  uint8_t *dst = Caml_ba_data_val(vdst), *rest = Caml_ba_data_val(vrest);
  int64_t pos = Long_val(Field(vcursor, 0)), row = Long_val(Field(vcursor, 1));
  int64_t skip = Long_val(vskip), rows = Long_val(vrows);
  size_t stride = (size_t)s.channels * s.width;
  int shift = 8 * s.width - s.bits;
  dst += (size_t)Long_val(vat) * stride;
  int64_t *x = malloc((size_t)s.channels * s.max_block * sizeof(int64_t));
  crcs *t = malloc(sizeof(crcs));
  if (!x || !t) {
    free(x);
    free(t);
    caml_raise_out_of_memory();
  }
  const char *e = NULL;
  int64_t kept = 0, written = 0;
  /* Without a length, a frame shorter than the shortest must be the
     last. */
  int short_seen = 0;
  caml_release_runtime_system();
  crcs_init(t);
  for (;;) {
    if (rows >= 0 && written == rows) break;
    if (s.length >= 0 && row == s.length) {
      if (rows >= 0) e = "the stream ends before the rows asked for";
      break;
    }
    if (s.length < 0 && pos == s.end) break;
    if (pos >= s.end) {
      e = "the file ends before STREAMINFO's length";
      break;
    }
    if (short_seen) {
      e = "a frame follows one shorter than STREAMINFO's shortest";
      break;
    }
    int bs;
    int64_t next;
    e = frame(src, &s, t, pos, row, x, &bs, &next);
    if (e) break;
    short_seen = bs < s.min_block;
    int from = (int)(skip < bs ? skip : bs);
    skip -= from;
    if (rows >= 0) {
      int64_t n = bs - from;
      if (n > rows - written) n = rows - written;
      store(x, &s, from, (int)n, shift, dst + (size_t)written * stride);
      written += n;
      kept = bs - from - n;
      store(x, &s, from + (int)n, (int)kept, shift, rest);
    }
    pos = next;
    row += bs;
  }
  caml_acquire_runtime_system();
  free(x);
  free(t);
  /* The cursor names the frame that failed. */
  Store_field(vcursor, 0, Val_long(pos));
  Store_field(vcursor, 1, Val_long(row));
  if (e) caml_failwith(e);
  CAMLreturn(Val_long(kept));
}

CAMLprim value caml_nx_io_flac_decode_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_flac_decode(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6], argv[7]);
}
