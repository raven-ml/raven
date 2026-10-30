/* The sequences section (RFC 8878 section 3.1.1.3.2) and the execution of
   its sequences (section 3.1.1.4).

   Each sequence is decoded from the three FSE states and executed at
   once. The fast path copies literals and matches in 16-byte steps that
   may write up to LZ_ZSTD_SLACK bytes past their end and read as far past
   the literals, so it runs while those bytes stay in the buffers, the
   output within the block's maximum and, when the literals are the
   destination's last bytes within the block's reach, before the literals
   not yet read; a match closer than 16 bytes is first spread to a period
   of at least 8. Everything else, a match into the window's earlier
   segment, the end of the buffer and every error, takes the exact path.

   An error in a sequence is at the byte holding its first bit, the first
   of the state bits that select its codes, found by reading the bitstream
   again from its end. */

#include "zstd_dec.h"

static const char header_past[] = "Sequences_Section_Header past the block";
static const char after_none[] = "bytes after Number_of_Sequences 0";
static const char reserved[] = "reserved Symbol_Compression_Modes bits";
static const char no_table[] = "Repeat_Mode with no previous table";
static const char rle_past[] = "RLE_Mode symbol past the block";
static const char rle_big[] = "RLE_Mode symbol above the maximum";
static const char stream_zero[] = "sequences bitstream ends in a 0 byte";
static const char stream_inexact[] = "sequences bitstream not read exactly";
static const char zero_offset[] = "offset of 0";
static const char far_offset[] = "offset above Window_Size";
static const char early_offset[] = "offset before the frame's start";
static const char lit_past[] = "Literals_Length past the literals";
static const char too_large[] = "block output above Block_Maximum_Size";

/* Codes */

/* Section 3.1.1.3.2.1.1, Tables 16 and 17: the baseline and extra bits of
   each literals length and match length code. An offset code c reads c
   extra bits; from c = 2 its baseline is the offset itself, 2^c - 3
   (Offset_Value - 3), and codes 0 and 1 select repeat offsets. Offset
   codes up to 31 are supported, as the reference decoder does. */
#define LL_MAX 35
#define ML_MAX 52
#define OF_MAX 31

static const uint32_t ll_base[LL_MAX + 1] = {
    0,  1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11,
    12, 13, 14, 15, 16, 18, 20, 22,  24,  28,  32,   40,
    48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

static const uint8_t ll_extra[LL_MAX + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  1,
    1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const uint32_t ml_base[ML_MAX + 1] = {
    3,  4,  5,  6,  7,  8,  9,  10,  11,  12,   13,   14,   15,   16,
    17, 18, 19, 20, 21, 22, 23, 24,  25,  26,   27,   28,   29,   30,
    31, 32, 33, 34, 35, 37, 39, 41,  43,  47,   51,   59,   67,   83,
    99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};

static const uint8_t ml_extra[ML_MAX + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const uint32_t of_base[OF_MAX + 1] = {
    0,         1,         1,          5,          13,        29,
    61,        125,       253,        509,        1021,      2045,
    4093,      8189,      16381,      32765,      65533,     131069,
    262141,    524285,    1048573,    2097149,    4194301,   8388605,
    16777213,  33554429,  67108861,   134217725,  268435453, 536870909,
    1073741821, 2147483645};

static const uint8_t of_extra[OF_MAX + 1] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

/* Section 3.1.1.3.2.2: the predefined distributions, with accuracy logs 6,
   6 and 5. They are the specification's. */
static const int16_t ll_default[LL_MAX + 1] = {
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2,  2,
    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};

static const int16_t ml_default[ML_MAX + 1] = {
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};

static const int16_t of_default[29] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, -1, -1, -1, -1, -1};

static void build_defaults(struct lz_zstd_dec *st) {
  lz_fse_build(st->ll_d, ll_default, LL_MAX, 6, ll_base, ll_extra);
  lz_fse_build(st->ml_d, ml_default, ML_MAX, 6, ml_base, ml_extra);
  lz_fse_build(st->of_d, of_default, 28, 5, of_base, of_extra);
  st->defaults = true;
}

/* Tables */

/* [table st mode s n max log_max own deflt dlog base extra tp lp f] sets
   [*tp] and [*lp] to the table of [mode] (Table 15), reading its
   description from [s]; [own] is where a table of this block goes. It is
   the description's size, or -1 with [f] set at an offset from [s].
   Repeat_Mode keeps the table in place. */
static long table(struct lz_zstd_dec *st, unsigned mode, const uint8_t *s,
                  size_t n, unsigned max, unsigned log_max,
                  struct lz_zstd_cell *own, const struct lz_zstd_cell *deflt,
                  unsigned dlog, const uint32_t *base, const uint8_t *extra,
                  const struct lz_zstd_cell **tp, uint8_t *lp,
                  struct lz_zstd_fail *f) {
  switch (mode) {
  case 0:
    if (!st->defaults) build_defaults(st);
    *tp = deflt;
    *lp = (uint8_t)dlog;
    return 0;
  case 1:
    if (n == 0) return lz_zstd_fail(f, LZ_MALFORMED, 0, rle_past);
    if (s[0] > max) return lz_zstd_fail(f, LZ_MALFORMED, 0, rle_big);
    own[0].next = 0;
    own[0].bits = 0;
    own[0].extra = extra[s[0]];
    own[0].base = base[s[0]];
    *tp = own;
    *lp = 0;
    return 1;
  case 2: {
    int16_t norm[256];
    unsigned log;
    size_t used = lz_fse_ncount(norm, &max, &log, log_max, s, n, f);
    if (used == 0) return -1;
    lz_fse_build(own, norm, max, log, base, extra);
    *tp = own;
    *lp = (uint8_t)log;
    return (long)used;
  }
  default:
    return 0;
  }
}

/* Copies */

static inline void copy8(uint8_t *d, const uint8_t *s) { memcpy(d, s, 8); }

static inline void copy16(uint8_t *d, const uint8_t *s) { memcpy(d, s, 16); }

/* [wild d s n] copies [n] bytes in 16-byte steps, [s] at least 16 bytes
   from [d], writing up to 32 bytes past [d + n]: one step up to 16 bytes,
   two up to 32, then two a round. */
LZ_INLINE void wild(uint8_t *d, const uint8_t *s, size_t n) {
  copy16(d, s);
  if (LZ_LIKELY(n <= 16)) return;
  copy16(d + 16, s + 16);
  for (size_t i = 32; i < n; i += 32) {
    copy16(d + i, s + i);
    copy16(d + i + 16, s + i + 16);
  }
}

/* [match op off n] copies [n] bytes from [off] bytes back, writing up to
   32 bytes past [op + n]. Below 16 bytes the copy overlaps itself: the
   first 8 bytes are written so that the source can then stay a multiple
   of [off] of at least 8 bytes behind. Those of 4 to 7 come from the
   smallest multiple of [off] of at least 4 back. */
LZ_INLINE void match(uint8_t *op, size_t off, size_t n) {
  static const uint8_t back4[8] = {0, 4, 4, 6, 4, 5, 6, 7};
  static const uint8_t back8[8] = {0, 8, 8, 9, 8, 10, 12, 14};
  const uint8_t *m = op - off;
  if (LZ_LIKELY(off >= 16)) {
    wild(op, m, n);
    return;
  }
  uint8_t *e = op + n;
  if (off < 8) {
    op[0] = m[0];
    op[1] = m[1];
    op[2] = m[2];
    op[3] = m[3];
    memcpy(op + 4, op + 4 - back4[off], 4);
    m = op + 8 - back8[off];
    op += 8;
    if (op >= e) return;
  }
  do {
    copy8(op, m);
    op += 8;
    m += 8;
  } while (op < e);
}

/* [exact op m n] copies [n] bytes from [m] before [op], byte for byte
   where they overlap, without writing past [op + n]. */
static void exact(uint8_t *op, const uint8_t *m, size_t n) {
  while (n > 0) {
    size_t k = (size_t)(op - m);
    if (k > n) k = n;
    memcpy(op, m, k);
    op += k;
    n -= k;
  }
}

/* Execution */

/* The block being executed: its start, the hard end of its output (the
   buffer's or the block limit's, whichever is first), the end of the
   output wide copies may reach, the literals' end, the bytes a wide
   literal copy needs after the literals it copies, and whether the
   literals lie within the wide copies' reach. */
struct run {
  uint8_t *ostart;
  uint8_t *hard;
  uint8_t *fast;
  const uint8_t *lit_end;
  size_t lit_slack;
  bool near;
  const struct lz_zstd_out *o;
};

/* [wide r op lit ll ml] is whether a sequence of [ll] literals at [lit]
   and a match of [ml] may be copied wide at [op]. */
LZ_INLINE bool wide(const struct run *r, const uint8_t *op, const uint8_t *lit,
                    size_t ll, size_t ml) {
  return ll + r->lit_slack <= (size_t)(r->lit_end - lit) &&
         (ptrdiff_t)(ll + ml) <= r->fast - op &&
         !(r->near && (ptrdiff_t)(ml + LZ_ZSTD_SLACK) > lit - op);
}

/* [overflow r op k f] reports [k] bytes at [op] that pass the hard end:
   past the block limit the block is malformed, else the destination is
   too short. */
static long overflow(const struct run *r, uint8_t *op, size_t k,
                     struct lz_zstd_fail *f) {
  size_t total = (size_t)(op - r->ostart) + k;
  int kind = total > r->o->block_max ? LZ_MALFORMED : LZ_TOO_LONG;
  return lz_zstd_fail(f, kind, 0, too_large);
}

static uint8_t *refuse(struct lz_zstd_fail *f, const char *msg) {
  lz_zstd_fail(f, LZ_MALFORMED, 0, msg);
  return NULL;
}

/* The exact path of one sequence. It is the new output position, or NULL
   with [f]'s kind and message set. */
LZ_NOINLINE static uint8_t *slow(const struct run *r, uint8_t *op,
                                 const uint8_t **lit, size_t ll, size_t ml,
                                 size_t off, struct lz_zstd_fail *f) {
  const struct lz_zstd_out *o = r->o;
  size_t hist = (size_t)(op - o->base) + ll;
  if (ll > (size_t)(r->lit_end - *lit)) return refuse(f, lit_past);
  if (ll + ml > (size_t)(r->hard - op)) {
    overflow(r, op, ll + ml, f);
    return NULL;
  }
  if (off == 0) return refuse(f, zero_offset);
  if (off > o->window) return refuse(f, far_offset);
  if (off > hist + o->ext) return refuse(f, early_offset);

  /* A sequence with the fast path's room reached here for a match into
     the earlier segment: its source lies at least LZ_ZSTD_SLACK bytes
     ahead of the output (the ring wraps past the window and the slack),
     so wide copies read it before overwriting anything. */
  bool room = wide(r, op, *lit, ll, ml);
  if (room)
    wild(op, *lit, ll);
  else
    memmove(op, *lit, ll);
  *lit += ll;
  op += ll;
  if (off > hist) {
    size_t back = off - hist;
    size_t k = back < ml ? back : ml;
    if (room)
      wild(op, o->ext_end - back, k);
    else
      memmove(op, o->ext_end - back, k);
    op += k;
    ml -= k;
    if (ml == 0) return op;
  }
  if (room)
    match(op, off, ml);
  else
    exact(op, op - off, ml);
  return op + ml;
}

/* [seq_byte st s len i] is the offset in the bitstream [s] of the byte
   holding sequence [i]'s first bit: the bitstream is read again as
   run_body reads it, up to the states that select the sequence's codes:
   for the first sequence, the bits after the end mark. */
LZ_NOINLINE static size_t seq_byte(const struct lz_zstd_dec *st,
                                   const uint8_t *s, size_t len, size_t i) {
  struct lz_bits b;
  if (!lz_bits_init(&b, s, len)) return len - 1;
  if (i == 0) return lz_bits_byte(&b);
  size_t ls = lz_bits_read(&b, st->ll_log);
  size_t os = lz_bits_read(&b, st->of_log);
  size_t ms = lz_bits_read(&b, st->ml_log);
  for (size_t k = 1;; k++) {
    const struct lz_zstd_cell *L = &st->ll[ls], *M = &st->ml[ms],
                              *O = &st->of[os];
    lz_bits_reload(&b);
    lz_bits_read(&b, O->extra);
    lz_bits_reload(&b);
    lz_bits_read(&b, M->extra);
    lz_bits_read(&b, L->extra);
    lz_bits_reload(&b);
    if (k == i) return lz_bits_byte(&b);
    ls = L->next + lz_bits_read(&b, L->bits);
    ms = M->next + lz_bits_read(&b, M->bits);
    os = O->next + lz_bits_read(&b, O->bits);
  }
}

/* Section 3.1.1.3.2.1.2 and 3.1.1.5: [count] sequences from the bitstream
   [s] of [len] bytes, which starts [at] bytes into the block. Without
   [guard] the literals are out of the wide copies' way, with slack after
   them, and the loop skips those checks. */
LZ_INLINE long run_body(struct lz_zstd_dec *st, const struct run *r,
                        size_t count, const uint8_t *s, size_t len,
                        size_t at, const uint8_t *lit,
                        struct lz_zstd_fail *f, const bool guard) {
  struct lz_bits b;
  if (!lz_bits_init(&b, s, len))
    return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)(at + len) - (len > 0),
                        stream_zero);

  const struct lz_zstd_cell *llt = st->ll, *oft = st->of, *mlt = st->ml;
  size_t lls = lz_bits_read(&b, st->ll_log);
  size_t ofs = lz_bits_read(&b, st->of_log);
  size_t mls = lz_bits_read(&b, st->ml_log);
  lz_bits_reload(&b);
  uint64_t rep0 = st->rep[0], rep1 = st->rep[1], rep2 = st->rep[2];
  const struct lz_zstd_out *o = r->o;
  uint8_t *const base = o->base;
  const uint8_t *const lit_end = r->lit_end;
  const size_t lit_slack = guard ? r->lit_slack : 0;
  uint8_t *const fast = r->fast;
  const bool near = guard && r->near;
  const uint64_t window = o->window;
  uint8_t *op = o->op;

  for (size_t n = count; n > 0; n--) {
    const struct lz_zstd_cell L = llt[lls], M = mlt[mls], O = oft[ofs];
    size_t ll = L.base, ml = M.base;
    uint64_t off;
    if (O.extra > 1) {
      off = O.base + lz_bits_read(&b, O.extra);
      rep2 = rep1;
      rep1 = rep0;
      rep0 = off;
    } else {
      /* A literals length of 0 shifts the repeat offsets by one. */
      unsigned i = O.base + (ll == 0) + (unsigned)lz_bits_read(&b, O.extra);
      if (i == 0) {
        off = rep0;
      } else if (i == 1) {
        off = rep1;
        rep1 = rep0;
        rep0 = off;
      } else {
        off = i == 2 ? rep2 : rep0 - 1;
        rep2 = rep1;
        rep1 = rep0;
        rep0 = off;
      }
    }
    if (M.extra) ml += lz_bits_read(&b, M.extra);
    if (O.extra + M.extra + L.extra > 30) lz_bits_reload(&b);
    if (L.extra) ll += lz_bits_read(&b, L.extra);
    if (n > 1) {
      lls = L.next + lz_bits_read(&b, L.bits);
      mls = M.next + lz_bits_read(&b, M.bits);
      ofs = O.next + lz_bits_read(&b, O.bits);
      lz_bits_reload(&b);
    }

    uint8_t *olit = op + ll;
    if (LZ_UNLIKELY(ll + lit_slack > (size_t)(lit_end - lit) ||
                    (ptrdiff_t)(ll + ml) > fast - op ||
                    off > (uint64_t)(olit - base) || off > window || off == 0 ||
                    (near && (ptrdiff_t)(ml + LZ_ZSTD_SLACK) > lit - op))) {
      op = slow(r, op, &lit, ll, ml, (size_t)off, f);
      if (op != NULL) continue;
      if (f->kind == LZ_MALFORMED)
        f->at = (int64_t)(at + seq_byte(st, s, len, count - n));
      return -1;
    }
    wild(op, lit, ll);
    lit += ll;
    match(olit, (size_t)off, ml);
    op = olit + ml;
  }

  if (!lz_bits_done(&b))
    return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)at, stream_inexact);
  st->rep[0] = rep0;
  st->rep[1] = rep1;
  st->rep[2] = rep2;

  size_t rest = (size_t)(lit_end - lit);
  if (rest > (size_t)(r->hard - op)) return overflow(r, op, rest, f);
  memmove(op, lit, rest);
  return (long)(op + rest - r->ostart);
}

static long run_base(struct lz_zstd_dec *st, const struct run *r, size_t n,
                     const uint8_t *s, size_t len, size_t at,
                     const uint8_t *lit, struct lz_zstd_fail *f) {
  if (r->near || r->lit_slack > 0)
    return run_body(st, r, n, s, len, at, lit, f, true);
  return run_body(st, r, n, s, len, at, lit, f, false);
}

#if LZ_ZSTD_BMI2
LZ_TARGET_BMI2 static long run_bmi2(struct lz_zstd_dec *st, const struct run *r,
                                    size_t n, const uint8_t *s, size_t len,
                                    size_t at, const uint8_t *lit,
                                    struct lz_zstd_fail *f) {
  if (r->near || r->lit_slack > 0)
    return run_body(st, r, n, s, len, at, lit, f, true);
  return run_body(st, r, n, s, len, at, lit, f, false);
}
#endif

static long run_seqs(struct lz_zstd_dec *st, const struct run *r, size_t n,
                     const uint8_t *s, size_t len, size_t at,
                     const uint8_t *lit, struct lz_zstd_fail *f) {
#if LZ_ZSTD_BMI2
  if (st->path == LZ_ZSTD_WITH_BMI2)
    return run_bmi2(st, r, n, s, len, at, lit, f);
#endif
  return run_base(st, r, n, s, len, at, lit, f);
}

/* Section 3.1.1.3.2.1: the header, the three tables, the bitstream. A
   header cut by the block's end is at the section's first byte. */
long lz_zstd_sequences(struct lz_zstd_dec *st, const uint8_t *s, size_t n,
                       size_t at, const struct lz_zstd_lits *l,
                       const struct lz_zstd_out *o, struct lz_zstd_fail *f) {
  struct run r;
  size_t room = (size_t)(o->end - o->op);
  size_t hard = room < o->block_max ? room : o->block_max;
  size_t fast = lz_zstd_reach(o);
  r.ostart = o->op;
  r.hard = o->op + hard;
  r.fast = o->op + fast;
  r.lit_end = l->p + l->len;
  r.o = o;
  /* The wide copies end LZ_ZSTD_SLACK past [fast] at most: literals in
     the destination before there are in their way. */
  r.lit_slack = l->tight ? LZ_ZSTD_SLACK : 0;
  r.near = l->dest && (size_t)(l->p - o->op) < fast + LZ_ZSTD_SLACK;

  if (n == 0) return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)at, header_past);
  size_t count = s[0], h = 1;
  if (count >= 128) {
    h = count == 255 ? 3 : 2;
    if (n < h) return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)at, header_past);
    count = count == 255 ? lz_le16(s + 1) + 0x7F00u : ((count - 128) << 8) + s[1];
  }

  if (count == 0) {
    if (h != n)
      return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)(at + h), after_none);
    if (l->len > hard) return overflow(&r, o->op, l->len, f);
    memmove(o->op, l->p, l->len);
    return (long)l->len;
  }

  if (h >= n) return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)at, header_past);
  if ((s[h] & 3) != 0)
    return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)(at + h), reserved);
  size_t modes_at = h;
  unsigned modes = s[h++];
  struct {
    unsigned mode, max, log_max, dlog;
    struct lz_zstd_cell *own;
    const struct lz_zstd_cell *deflt;
    const uint32_t *base;
    const uint8_t *extra;
    const struct lz_zstd_cell **tp;
    uint8_t *lp;
  } t[3] = {
      {modes >> 6, LL_MAX, 9, 6, st->ll_t, st->ll_d, ll_base, ll_extra,
       &st->ll, &st->ll_log},
      {(modes >> 4) & 3, OF_MAX, 8, 5, st->of_t, st->of_d, of_base, of_extra,
       &st->of, &st->of_log},
      {(modes >> 2) & 3, ML_MAX, 9, 6, st->ml_t, st->ml_d, ml_base, ml_extra,
       &st->ml, &st->ml_log},
  };
  for (int i = 0; i < 3; i++) {
    if (t[i].mode == 3 && !st->seq_tables)
      return lz_zstd_fail(f, LZ_MALFORMED, (int64_t)(at + modes_at), no_table);
    long k = table(st, t[i].mode, s + h, n - h, t[i].max, t[i].log_max,
                   t[i].own, t[i].deflt, t[i].dlog, t[i].base, t[i].extra,
                   t[i].tp, t[i].lp, f);
    if (k < 0) {
      f->at += (int64_t)(at + h);
      return -1;
    }
    h += (size_t)k;
  }
  st->seq_tables = true;

  return run_seqs(st, &r, count, s + h, n - h, at + h, l->p, f);
}
