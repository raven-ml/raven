/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Sorts and numberings, with their words of kernels.h.

   A sort orders each slice of x by its elements' keys, unsigned integers
   in the contract's order (Keys), by radix of 8-bit digits. Three paths,
   the host's choice by the slice's length n and the elements it keeps:

   - sort_block: slices of at most a block's cap of elements, [per] of
     them a group, held in registers and ranked in threadgroup memory.
   - sort_hist, then once a key byte sort_count and sort_scatter: longer
     slices, by LSD radix through workspace. sort_hist counts every digit
     of every slice in one read of x. A pass cuts a slice's tiles into
     parts of consecutive tiles: sort_count counts each part's digits, and
     the slice's last part to take its ticket turns the counts into each
     part's first place in each bin (Satish, Harris and Garland 2009, Fig.
     2); sort_scatter then ranks each tile of a part in turn and moves its
     items there. A slice of one part takes its places from sort_hist and
     no sort_count. A pass whose digit every key of a slice shares is
     skipped for that slice.
   - sort_round once a digit, sort_gather, then sort_block: a long slice
     that keeps at most a quarter of the cap. Rounds of an MSD radix
     select on (key, position) find the kept elements' least prefix that
     leaves at most a cap of candidates, which sort_block sorts.

   A numbering sorts its rows stably, one word at a time from the last, by
   the radix passes, then numbers them: unique_count, unique_heads,
   unique_rank, unique_ids.

   A launch takes the units, blocks, chunks or parts, from [first] to
   [end] - 1, one a group, so that the host splits long work into
   launches by ranges of units. A slice's units are consecutive; a count
   whose slice's last unit lies in the launch takes the slice's ticket
   among its units in the launch, and the last to take it combines every
   unit of the slice, those of earlier launches included, which ran
   before (slice_last).

   Determinism. No float arithmetic. Atomics are integer adds into counts,
   candidate slots (sorted by a distinct key after), ORs of bits, a
   sub-byte element's store of its own bits, and tickets. Every count of
   digits adds once a distinct digit of a SIMD-group, by the lowest lane
   of the digit's peers (device.h's nx_peers).

   Keys. An element's key is an unsigned integer in the contract's order:
   an unsigned integer its bits; a signed one its bits with the sign
   flipped; a boolean 1 where not zero; a float its bits with every bit
   flipped where the sign is set and the sign flipped elsewhere, so -0 sits
   below +0, and every NaN, a magnitude past [nan_abs], all ones, above
   +inf and equal; a complex number its real part's key above its
   imaginary part's, all ones where either is a NaN. A descending sort
   complements the keys: an ascending stable sort of them gives the
   reversed order with equal elements in increasing position. The format
   is the launch's words, the same for every element, so its branches are
   uniform. */

#include "device.h"
#include "values.h"

using namespace metal;

#define ST NX_SORT_THREADS

struct sort_words {
  device const uchar *x;
  device uchar *values;
  device long *positions;
  device atomic_uint *tickets;
  device uint *hist;
  device uint *meta;
  device uint *counts;
  device uchar *e0;
  device uchar *p0;
  device uchar *e1;
  device uchar *p1;
  device uchar *cand;
  device ulong *cand_p;
  device const long *order;
  constant long *axes;
  ulong keep;
  ulong flags;
  ulong pass;
  ulong first;
  ulong end;
  ulong units;
  ulong units_div;
  ulong tiles;
  ulong span;
  ulong stride;
  ulong per;
  ulong pbytes;
  ulong carry;
  ulong skew;
  ulong kind;
  ulong ebits;
  ulong kbytes;
  ulong nan_abs;
  ulong kmask;
  ulong flip;
  ulong flip_signed;
  ulong dmask_lo;
  ulong dmask_hi;
  ulong n;
  ulong n_div;
  ulong count;
  long sx;
  long sv;
  long sp;
  long off_x;
  long off_v;
  long off_p;
  ulong kept;
  long held[NX_SORT_AXIS_WORDS * NX_SORT_HOLDS];
};

NX_FIELD(sort_words, x, NX_SORT_X);
NX_FIELD(sort_words, values, NX_SORT_VALUES);
NX_FIELD(sort_words, positions, NX_SORT_POSITIONS);
NX_FIELD(sort_words, tickets, NX_SORT_TICKETS);
NX_FIELD(sort_words, hist, NX_SORT_HIST);
NX_FIELD(sort_words, meta, NX_SORT_META);
NX_FIELD(sort_words, counts, NX_SORT_COUNTS);
NX_FIELD(sort_words, e0, NX_SORT_E0);
NX_FIELD(sort_words, p0, NX_SORT_P0);
NX_FIELD(sort_words, e1, NX_SORT_E1);
NX_FIELD(sort_words, p1, NX_SORT_P1);
NX_FIELD(sort_words, cand, NX_SORT_CAND);
NX_FIELD(sort_words, cand_p, NX_SORT_CAND_P);
NX_FIELD(sort_words, order, NX_SORT_ORDER);
NX_FIELD(sort_words, axes, NX_SORT_AXES);
NX_FIELD(sort_words, keep, NX_SORT_KEEP);
NX_FIELD(sort_words, flags, NX_SORT_FLAGS);
NX_FIELD(sort_words, pass, NX_SORT_PASS);
NX_FIELD(sort_words, first, NX_SORT_FIRST);
NX_FIELD(sort_words, end, NX_SORT_END);
NX_FIELD(sort_words, units, NX_SORT_UNITS);
NX_FIELD(sort_words, units_div, NX_SORT_UNITS_DIV);
NX_FIELD(sort_words, tiles, NX_SORT_TILES);
NX_FIELD(sort_words, span, NX_SORT_SPAN);
NX_FIELD(sort_words, stride, NX_SORT_STRIDE);
NX_FIELD(sort_words, per, NX_SORT_PER);
NX_FIELD(sort_words, pbytes, NX_SORT_PBYTES);
NX_FIELD(sort_words, carry, NX_SORT_CARRY);
NX_FIELD(sort_words, skew, NX_SORT_SKEW);
NX_FIELD(sort_words, kind, NX_SORT_KIND);
NX_FIELD(sort_words, ebits, NX_SORT_EBITS);
NX_FIELD(sort_words, kbytes, NX_SORT_KBYTES);
NX_FIELD(sort_words, nan_abs, NX_SORT_NAN_ABS);
NX_FIELD(sort_words, kmask, NX_SORT_KMASK);
NX_FIELD(sort_words, flip, NX_SORT_FLIP);
NX_FIELD(sort_words, flip_signed, NX_SORT_FLIP_SIGNED);
NX_FIELD(sort_words, dmask_lo, NX_SORT_DMASK_LO);
NX_FIELD(sort_words, dmask_hi, NX_SORT_DMASK_HI);
NX_FIELD(sort_words, n, NX_SORT_N);
NX_FIELD(sort_words, n_div, NX_SORT_N_DIV);
NX_FIELD(sort_words, count, NX_SORT_COUNT);
NX_FIELD(sort_words, sx, NX_SORT_SX);
NX_FIELD(sort_words, sv, NX_SORT_SV);
NX_FIELD(sort_words, sp, NX_SORT_SP);
NX_FIELD(sort_words, off_x, NX_SORT_OFF_X);
NX_FIELD(sort_words, off_v, NX_SORT_OFF_V);
NX_FIELD(sort_words, off_p, NX_SORT_OFF_P);
NX_FIELD(sort_words, kept, NX_SORT_KEPT);
NX_FIELD(sort_words, held, NX_SORT_HELD);
NX_SIZE(sort_words, NX_SORT_WORDS);

/* Elements and keys

   A kernel of width E holds an element, and its key, in a register of
   its width: a uint for elements of at most 32 bits, a sub-byte one's
   code in its low bits; a ulong for those of 64 bits, a complex64's real
   part in its low half; two ulongs for a complex128, its real part in
   [lo]. */

struct u128 {
  ulong lo, hi;
};

template <int E> struct el;
template <> struct el<4> {
  typedef uint T;
};
template <> struct el<8> {
  typedef ulong T;
};
template <> struct el<16> {
  typedef u128 T;
};

/* The elements a block's group holds of width E: NX_SORT_CAP, or
   NX_SORT_CAP16 of 16 bytes; a select's candidates of a slice. */
template <int E> constexpr uint cap_of() {
  return E == 16 ? NX_SORT_CAP16 : NX_SORT_CAP;
}

static inline uint zero_of(uint) { return 0; }
static inline ulong zero_of(ulong) { return 0; }
static inline u128 zero_of(u128) { return u128{0, 0}; }

static inline uint ones_of(uint) { return ~0u; }
static inline ulong ones_of(ulong) { return ~0ul; }
static inline u128 ones_of(u128) { return u128{~0ul, ~0ul}; }

static inline uint and_of(uint a, uint b) { return a & b; }
static inline ulong and_of(ulong a, ulong b) { return a & b; }
static inline u128 and_of(u128 a, u128 b) { return u128{a.lo & b.lo, a.hi & b.hi}; }

static inline uint or_of(uint a, uint b) { return a | b; }
static inline ulong or_of(ulong a, ulong b) { return a | b; }
static inline u128 or_of(u128 a, u128 b) { return u128{a.lo | b.lo, a.hi | b.hi}; }

/* 32-bit word [q] of [e], from the least significant. */
static inline uint word_of(uint e, uint) { return e; }
static inline uint word_of(ulong e, uint q) { return uint(e >> (32 * q)); }
static inline uint word_of(u128 e, uint q) {
  return uint((q < 2 ? e.lo : e.hi) >> (32 * (q & 1)));
}

static inline void set_word(thread uint &e, uint, uint v) { e = v; }
static inline void set_word(thread ulong &e, uint q, uint v) {
  const ulong m = 0xFFFFFFFFul << (32 * q);
  e = (e & ~m) | (ulong(v) << (32 * q));
}
static inline void set_word(thread u128 &e, uint q, uint v) {
  thread ulong &h = q < 2 ? e.lo : e.hi;
  set_word(h, q & 1, v);
}

/* Byte [p] of [k], from the least significant. */
static inline uint kbyte(uint k, uint p) { return k >> (8 * p) & 255; }
static inline uint kbyte(ulong k, uint p) { return uint(k >> (8 * p)) & 255; }
static inline uint kbyte(u128 k, uint p) {
  return uint((p < 8 ? k.lo : k.hi) >> (8 * (p & 7))) & 255;
}

static inline bool fnan32(uint b, uint nan_abs) {
  return (b & 0x7FFFFFFFu) > nan_abs;
}
static inline uint fkey32(uint b) {
  return (b & 0x80000000u) != 0 ? ~b : b | 0x80000000u;
}
static inline bool fnan64(ulong b, ulong nan_abs) {
  return (b & 0x7FFFFFFFFFFFFFFFul) > nan_abs;
}
static inline ulong fkey64(ulong b) {
  return (b >> 63) != 0 ? ~b : b | (1ul << 63);
}

/* A real element's key is its bits xored with [flip_signed] where its
   sign is set and [flip] where not, or [kmask] where its magnitude is past
   [nan_abs]: the flips are the sign and all ones for a float, the sign
   twice for a signed integer, zero for the others, and [nan_abs] all ones
   for every non-float. A boolean's bits are first 1 where not zero. */
static inline uint key_of(constant sort_words &w, uint b) {
  if (w.kind == NX_SORT_BOOL) b = b != 0 ? 1u : 0u;
  const uint sign = 1u << (uint(w.ebits) - 1);
  const uint k = (b & ~sign) > uint(w.nan_abs)
                     ? uint(w.kmask)
                     : b ^ ((b & sign) != 0 ? uint(w.flip_signed) : uint(w.flip));
  return k ^ uint(w.dmask_lo);
}

static inline ulong key_of(constant sort_words &w, ulong b) {
  ulong k;
  if (w.kind == NX_SORT_COMPLEX) {
    const uint re = uint(b), im = uint(b >> 32), nan = uint(w.nan_abs);
    k = fnan32(re, nan) || fnan32(im, nan)
            ? ~0ul
            : ulong(fkey32(re)) << 32 | ulong(fkey32(im));
  } else {
    const ulong sign = 1ul << 63;
    k = (b & ~sign) > w.nan_abs ? w.kmask
                                : b ^ ((b & sign) != 0 ? w.flip_signed : w.flip);
  }
  return k ^ w.dmask_lo;
}

static inline u128 key_of(constant sort_words &w, u128 e) {
  u128 k;
  if (fnan64(e.lo, w.nan_abs) || fnan64(e.hi, w.nan_abs)) {
    k = u128{~0ul, ~0ul};
  } else {
    k.hi = fkey64(e.lo);
    k.lo = fkey64(e.hi);
  }
  k.lo ^= w.dmask_lo;
  k.hi ^= w.dmask_hi;
  return k;
}

/* Element [off] of [base], [ebits] wide: a sub-byte one's code from the
   low bits of its byte first. A complex number is read a part at a time:
   its array promises only a part's alignment. */
static inline void load_el(constant sort_words &w, device const uchar *base,
                           long off, thread uint &e) {
  switch (w.ebits) {
  case 1:
  case 4: {
    const long bit = off * long(w.ebits);
    e = uint(base[bit >> 3]) >> uint(bit & 7) & ((1u << uint(w.ebits)) - 1);
    break;
  }
  case 8: e = base[off]; break;
  case 16: e = ((device const ushort *)base)[off]; break;
  default: e = ((device const uint *)base)[off]; break;
  }
}

static inline void load_el(constant sort_words &w, device const uchar *base,
                           long off, thread ulong &e) {
  if (w.kind == NX_SORT_COMPLEX) {
    device const uint *c = (device const uint *)base + 2 * off;
    e = ulong(c[1]) << 32 | ulong(c[0]);
    return;
  }
  e = ((device const ulong *)base)[off];
}

static inline void load_el(constant sort_words &, device const uchar *base,
                           long off, thread u128 &e) {
  device const ulong *c = (device const ulong *)base + 2 * off;
  e = u128{c[0], c[1]};
}

/* Stores [e] at element [off] of the values, a complex number a part at a
   time. A sub-byte element shares its byte, maybe its 32-bit word, with
   others, of other slices or other arrays: store_bits changes its own
   bits of the word, from the word below the values' first byte, which
   lies [skew] bits into it. A thread that reads another array's bytes of
   such a word sees them unchanged. */
static inline void store_el(constant sort_words &w, long off, uint e) {
  device uchar *base = w.values;
  switch (w.ebits) {
  case 1:
  case 4:
    store_bits(base - w.skew / 8, w.skew + ulong(off) * w.ebits,
               uint(w.ebits), e);
    break;
  case 8: base[off] = uchar(e); break;
  case 16: ((device ushort *)base)[off] = ushort(e); break;
  default: ((device uint *)base)[off] = e; break;
  }
}

static inline void store_el(constant sort_words &w, long off, ulong e) {
  if (w.kind == NX_SORT_COMPLEX) {
    device uint *c = (device uint *)w.values + 2 * off;
    c[0] = uint(e);
    c[1] = uint(e >> 32);
    return;
  }
  ((device ulong *)w.values)[off] = e;
}

static inline void store_el(constant sort_words &w, long off, u128 e) {
  device ulong *c = (device ulong *)w.values + 2 * off;
  c[0] = e.lo;
  c[1] = e.hi;
}

/* Element [i] of a pass's buffer [b], elements of [eb] bytes: an
   element's own bytes, a sub-byte one's code in one. */
static inline void get_el(device const uchar *b, uint eb, ulong i,
                          thread uint &e) {
  switch (eb) {
  case 1: e = b[i]; break;
  case 2: e = ((device const ushort *)b)[i]; break;
  default: e = ((device const uint *)b)[i]; break;
  }
}
static inline void get_el(device const uchar *b, uint, ulong i,
                          thread ulong &e) {
  e = ((device const ulong *)b)[i];
}
static inline void get_el(device const uchar *b, uint, ulong i,
                          thread u128 &e) {
  device const ulong *c = (device const ulong *)b + 2 * i;
  e = u128{c[0], c[1]};
}

static inline void put_el(device uchar *b, uint eb, ulong i, uint e) {
  switch (eb) {
  case 1: b[i] = uchar(e); break;
  case 2: ((device ushort *)b)[i] = ushort(e); break;
  default: ((device uint *)b)[i] = e; break;
  }
}
static inline void put_el(device uchar *b, uint, ulong i, ulong e) {
  ((device ulong *)b)[i] = e;
}
static inline void put_el(device uchar *b, uint, ulong i, u128 e) {
  device ulong *c = (device ulong *)b + 2 * i;
  c[0] = e.lo;
  c[1] = e.hi;
}

/* The bytes of a pass's buffer element: the element's, one for a
   sub-byte one. */
static inline uint ebytes(constant sort_words &w) {
  return max(1u, uint(w.ebits) / 8);
}

/* Slices */

/* A slice's first elements of x, the values and the positions. */
struct offs {
  long x, v, p;
};

/* Slice [s]'s offsets: its coordinates along the kept axes, from the
   innermost, each by its divisor where the slices' indices lie below
   2^32 and by a 64-bit division under NX_SORT_HUGE; the outermost is
   what remains. */
static inline offs slice_at(constant sort_words &w, ulong s) {
  offs o = {w.off_x, w.off_v, w.off_p};
  const int kept = int(w.kept);
  constant long *a = kept <= NX_SORT_HOLDS ? w.held : w.axes;
  const bool huge = (w.flags & NX_SORT_HUGE) != 0;
  for (int j = kept - 1; j >= 0; j--) {
    constant long *ax = a + NX_SORT_AXIS_WORDS * j;
    ulong c = s;
    if (j > 0) {
      const ulong e = ulong(ax[NX_SORT_AXIS_EXTENT]);
      const ulong q =
          huge ? s / e : ulong(divide(uint(s), ulong(ax[NX_SORT_AXIS_DIV])));
      c = s - q * e;
      s = q;
    }
    o.x += long(c) * ax[NX_SORT_AXIS_SX];
    o.v += long(c) * ax[NX_SORT_AXIS_SV];
    o.p += long(c) * ax[NX_SORT_AXIS_SP];
  }
  return o;
}

/* [u] / [units], by [units]' divisor where both lie below 2^32. */
static inline ulong unit_slice(constant sort_words &w, ulong u) {
  if (u >> 32 == 0 && w.units >> 32 == 0)
    return divide(uint(u), w.units_div);
  return u / w.units;
}

/* Whether the calling group, of unit [u] of slice [s], is the last of its
   slice's units in the launch to take the slice's ticket, which counts
   them, in every thread of it: none where the slice's last unit lies past
   the launch, whose later launch combines it. The tickets are the
   launch's slices' from its first. */
static inline bool slice_last(constant sort_words &w, ulong s, uint t,
                              threadgroup uint *flag) {
  const ulong first = s * w.units, last = first + w.units - 1;
  if (last >= w.end) return false;
  const ulong from = max(w.first, first);
  const ulong s0 = unit_slice(w, w.first);
  return nx_last(w.tickets + (s - s0), uint(last + 1 - from), t, flag);
}

/* Where slice [s]'s pass reads and writes: its [q]th of [qs] passes, x
   for the first, the last writing the values and positions. The passes
   alternate buffers so that the last reads buf1. Slice [s]'s run of a
   buffer, its elements then its positions, lies [s * stride] bytes into
   it: its own, so that slices whose passes differ never share one. */
struct turn {
  bool on, first, final;
  device uchar *se, *sp, *de, *dp;
};

static inline turn turn_of(constant sort_words &w, ulong s) {
  turn r = {false, false, false, nullptr, nullptr, nullptr, nullptr};
  const uint mask = w.meta[s], p = uint(w.pass);
  if ((mask >> p & 1) == 0) return r;
  const uint q = popcount(mask & ((1u << p) - 1)), qs = popcount(mask);
  r.on = true;
  r.first = q == 0;
  r.final = q == qs - 1;
  const bool s1 = ((qs - q) & 1) != 0, d1 = ((qs - 1 - q) & 1) != 0;
  const ulong at = s * w.stride;
  r.se = (s1 ? w.e1 : w.e0) + at;
  r.sp = (s1 ? w.p1 : w.p0) + at;
  r.de = (d1 ? w.e1 : w.e0) + at;
  r.dp = (d1 ? w.p1 : w.p0) + at;
  return r;
}

/* Item [l] of slice [s]'s sequence that a first pass reads: x's element
   at the position [order] gives there, or at [l], and that position. */
template <typename T>
static inline void first_item(constant sort_words &w, long ox, ulong s,
                              ulong l, thread T &e, thread ulong &pos) {
  pos = (w.flags & NX_SORT_ORDERED) != 0 ? ulong(w.order[s * w.n + l]) : l;
  load_el(w, w.x, ox + long(pos) * w.sx, e);
}

/* Slice [s]'s bin [b] of byte [p] in sort_hist's counts. */
static inline device uint *bin_at(constant sort_words &w, ulong s, uint p,
                                  uint b) {
  const bool wide = (w.flags & NX_SORT_WIDE) != 0;
  const uint bw = wide ? NX_SORT_BIN_WIDE : NX_SORT_BIN;
  return w.hist + ((s * w.kbytes + p) * 256 + b) * bw;
}

/* A bin's first place in its slice, once sort_hist combined it. */
static inline ulong bin_place(constant sort_words &w, device uint *bin) {
  if ((w.flags & NX_SORT_WIDE) != 0)
    return ulong(bin[2]) | ulong(bin[3]) << 32;
  return bin[1];
}

/* Block sorts */

/* Sorts slices of at most a cap of elements in a group: [per] slices from
   x, or under NX_SORT_CANDIDATES one slice's candidates (sort_gather).
   Each thread holds N elements in registers and its items' 16-bit
   ordinals in the group; a pass ranks one 8-bit digit (device.h's
   nx_rank), then moves each element's bits to its rank through
   threadgroup memory a 32-bit word at a time, with its ordinal, and reads
   back the elements now at its own items. Passes: the candidates'
   position bytes, from the least, read by ordinal from the candidates;
   then the key bytes from the least, but those every key of the group
   shares; then the slice, where the group holds several.

   Threadgroup memory: the exchange (a word a cap), the ordinals (16 bits
   a cap), the SIMD-groups' counts (NX_SIMDS rows of 257 16-bit counts),
   the bins' starts and counts (257 words each), NX_SIMDS sums, and the
   keys' AND and OR (2 E / 4 words). Before the first pass the exchange
   holds each slice's offset of x, and after the last each slice's of the
   values and positions. */
template <int E, int N>
static inline void block_sort(constant sort_words &w, threadgroup uchar *mem,
                              uint g, uint t) {
  typedef typename el<E>::T T;
  constexpr uint CAP = ST * N, KW = E / 4;
  threadgroup uint *xch = (threadgroup uint *)mem;
  threadgroup long *offsets = (threadgroup long *)mem;
  threadgroup ushort *ords = (threadgroup ushort *)(xch + CAP);
  threadgroup ushort *cnt = ords + CAP;
  threadgroup uint *start = (threadgroup uint *)(cnt + NX_SIMDS * 257);
  threadgroup uint *count = start + 257;
  threadgroup uint *sums = count + 257;
  threadgroup atomic_uint *shared = (threadgroup atomic_uint *)(sums + NX_SIMDS);
  const uint lane = t & 31, sg = t / 32;
  const bool cands = (w.flags & NX_SORT_CANDIDATES) != 0;
  const ulong unit = w.first + g;
  const uint n = uint(w.n);
  const uint eb = ebytes(w);

  /* The group's slices and items. */
  ulong s0;
  uint slices, v;
  if (cands) {
    const bool wide = (w.flags & NX_SORT_WIDE) != 0;
    const uint sw = wide ? NX_SORT_SELECT_WORDS_WIDE : NX_SORT_SELECT_WORDS;
    s0 = unit;
    slices = 1;
    v = w.hist[s0 * sw + NX_SORT_SELECT_CNT];
  } else {
    s0 = unit * w.per;
    slices = uint(min(w.per, w.count - s0));
    v = slices * n;
  }
  device const uchar *ce = w.cand + s0 * CAP * eb;
  device const ulong *cp = w.cand_p + s0 * CAP;

  if (!cands && t < slices) offsets[t] = slice_at(w, s0 + t).x;
  if (t < 2 * KW)
    atomic_store_explicit(&shared[t], t < KW ? ~0u : 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  T b[N];
  ushort o[N];
  T kand = ones_of(zero_of(T())), kor = zero_of(T());
#pragma unroll
  for (int j = 0; j < N; j++) {
    const uint l = sg * N * 32 + j * 32 + lane;
    o[j] = ushort(l);
    b[j] = zero_of(T());
    if (l >= v) continue;
    if (cands) get_el(ce, eb, l, b[j]);
    else {
      const uint sl = divide(l, w.n_div);
      ulong pos;
      first_item(w, offsets[sl], s0 + sl, l - sl * n, b[j], pos);
    }
    const T k = key_of(w, b[j]);
    kand = and_of(kand, k);
    kor = or_of(kor, k);
  }

  /* The key bytes every key of the group shares: where the AND of the
     keys equals their OR. */
#pragma unroll
  for (uint q = 0; q < KW; q++) {
    const uint a = simd_and(word_of(kand, q)), r = simd_or(word_of(kor, q));
    if (lane == 0) {
      atomic_fetch_and_explicit(&shared[q], a, memory_order_relaxed);
      atomic_fetch_or_explicit(&shared[KW + q], r, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  T diff = zero_of(T());
#pragma unroll
  for (uint q = 0; q < KW; q++)
    set_word(diff, q,
             atomic_load_explicit(&shared[q], memory_order_relaxed) ^
                 atomic_load_explicit(&shared[KW + q], memory_order_relaxed));

  const uint pb = cands ? uint(w.pbytes) : 0, kb = uint(w.kbytes);
  const uint passes = !cands && n == 1 ? 0 : pb + kb + (slices > 1 ? 1 : 0);
  for (uint p = 0; p < passes; p++) {
    if (p >= pb && p < pb + kb && kbyte(diff, p - pb) == 0) continue;
    uint d[N], r[N];
#pragma unroll
    for (int j = 0; j < N; j++) {
      const uint l = sg * N * 32 + j * 32 + lane;
      if (l >= v) d[j] = 256;
      else if (p < pb) d[j] = uint(cp[o[j]] >> (8 * p)) & 255;
      else if (p < pb + kb) d[j] = kbyte(key_of(w, b[j]), p - pb);
      else d[j] = divide(o[j], w.n_div);
    }
    nx_rank<N, ushort>(d, r, t, cnt, start, count, sums);
#pragma unroll
    for (uint q = 0; q < KW; q++) {
#pragma unroll
      for (int j = 0; j < N; j++) {
        xch[r[j]] = word_of(b[j], q);
        if (q == 0) ords[r[j]] = o[j];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (int j = 0; j < N; j++) {
        const uint l = sg * N * 32 + j * 32 + lane;
        set_word(b[j], q, xch[l]);
        if (q == 0) o[j] = ords[l];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  /* Item at group order l is rank l: of slice l / n, its rank l % n. */
  if (t < slices) {
    const offs at = slice_at(w, s0 + t);
    offsets[2 * t] = at.v;
    offsets[2 * t + 1] = at.p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool store = (w.flags & NX_SORT_STORE) != 0;
  const bool ordered = (w.flags & NX_SORT_ORDERED) != 0;
#pragma unroll
  for (int j = 0; j < N; j++) {
    const uint l = sg * N * 32 + j * 32 + lane;
    if (l >= v) continue;
    const uint sl = cands ? 0 : divide(l, w.n_div);
    const uint r = l - sl * n;
    if (r >= w.keep) continue;
    /* The item's place in its slice's sequence, x's position there. */
    const uint i = o[j] - sl * n;
    const long pos = cands     ? long(cp[o[j]])
                     : ordered ? w.order[(s0 + sl) * w.n + i]
                               : long(i);
    w.positions[offsets[2 * sl + 1] + long(r) * w.sp] = pos;
    if (store) store_el(w, offsets[2 * sl] + long(r) * w.sv, b[j]);
  }
}

#define BLOCK(E, N)                                                            \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_block_##E(constant sort_words &w [[buffer(0)]],                    \
                      threadgroup uchar *mem [[threadgroup(0)]],               \
                      uint g [[threadgroup_position_in_grid]],                 \
                      uint t [[thread_index_in_threadgroup]]) {                \
    block_sort<E, N>(w, mem, g, t);                                            \
  }
BLOCK(4, NX_SORT_BLOCK_ITEMS)
BLOCK(8, NX_SORT_BLOCK_ITEMS)
BLOCK(16, NX_SORT_BLOCK_ITEMS16)

/* Radix passes */

/* Counts every digit of every key of a chunk of its slice into the
   slice's bins, by leader adds into one threadgroup copy of the key
   bytes' counts, then one device add a nonzero bin; the slice's last
   chunk makes them each bin's first place and finds the passes the slice
   takes: those whose digit not every key shares, or the last pass where
   every one is.

   Threadgroup memory: kbytes 256 counts, then the combine's 256 64-bit
   sums and two words. */
template <int E>
static inline void hist_pass(constant sort_words &w, threadgroup uchar *mem,
                             uint g, uint t) {
  typedef typename el<E>::T T;
  const uint kb = uint(w.kbytes);
  threadgroup atomic_uint *h = (threadgroup atomic_uint *)mem;
  threadgroup ulong *buf = (threadgroup ulong *)(mem + 4 * 256 * kb);
  threadgroup uint *flag = (threadgroup uint *)(buf + ST);
  const uint below = nx_below(t & 31);
  const ulong u = w.first + g, s = unit_slice(w, u), c = u - s * w.units;
  for (uint i = t; i < kb * 256; i += ST)
    atomic_store_explicit(&h[i], 0u, memory_order_relaxed);
  const long ox = slice_at(w, s).x;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong lo = c * NX_SORT_CHUNK, hi = min(w.n, lo + NX_SORT_CHUNK);
  for (ulong i0 = lo; i0 < hi; i0 += ST) {
    const ulong i = i0 + t;
    const bool valid = i < hi;
    T e = zero_of(T());
    ulong pos;
    if (valid) first_item(w, ox, s, i, e, pos);
    const T k = key_of(w, e);
    for (uint p = 0; p < kb; p++) {
      const uint d = valid ? kbyte(k, p) : 256;
      const uint m = nx_peers<9>(d);
      if (d < 256 && (m & below) == 0)
        atomic_fetch_add_explicit(&h[p * 256 + d], popcount(m),
                                  memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool wide = (w.flags & NX_SORT_WIDE) != 0;
  for (uint i = t; i < kb * 256; i += ST) {
    const uint v = atomic_load_explicit(&h[i], memory_order_relaxed);
    if (v == 0) continue;
    device atomic_uint *bin =
        (device atomic_uint *)bin_at(w, s, i / 256, i % 256);
    if (wide) nx_carry_add(bin, bin + 1, v, uint(w.carry));
    else atomic_fetch_add_explicit(bin, v, memory_order_relaxed);
  }
  if (!slice_last(w, s, t, flag)) return;

  uint mask = 0;
  for (uint p = 0; p < kb; p++) {
    coherent(device) device uint *bin =
        (coherent(device) device uint *)bin_at(w, s, p, t);
    const ulong k = wide ? nx_carried(bin[0], bin[1], uint(w.carry)) : bin[0];
    if (t == 0) flag[1] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == w.n) flag[1] = 1;
    ulong all;
    const ulong place = nx_group_excl64(k, t, buf, all);
    if (wide) {
      bin[2] = uint(place);
      bin[3] = uint(place >> 32);
    } else bin[1] = uint(place);
    if (flag[1] == 0) mask |= 1u << p;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) w.meta[s] = mask != 0 ? mask : 1u << (kb - 1);
}

#define HIST(E)                                                                \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_hist_##E(constant sort_words &w [[buffer(0)]],                     \
                     threadgroup uchar *mem [[threadgroup(0)]],                \
                     uint g [[threadgroup_position_in_grid]],                  \
                     uint t [[thread_index_in_threadgroup]]) {                 \
    hist_pass<E>(w, mem, g, t);                                                \
  }
HIST(4)
HIST(8)
HIST(16)

/* A part's counts, then places, in [counts]: a 32-bit word a bin, two
   under NX_SORT_WIDE, for the part [q] of slice [s]. */
static inline device uint *part_at(constant sort_words &w, ulong s, ulong q,
                                   uint b) {
  const bool wide = (w.flags & NX_SORT_WIDE) != 0;
  return w.counts + ((s * w.units + q) * 256 + b) * (wide ? 2 : 1);
}

/* Counts the digits of each part's items into the part's counts, by
   leader adds into a copy of the bins a SIMD-group: a SIMD-group's
   leaders write distinct bins of its own copy, so each adds by a plain
   write, ordered across steps by a SIMD-group barrier. The slice's last
   part makes the counts each part's first places in each bin, after the
   slice's earlier parts, from the bin's first place in sort_hist's
   counts: a thread a bin walks the parts, loading 8 parts' counts ahead.

   Threadgroup memory: NX_SIMDS copies of 256 counts, then two words. */
#define PARTS_AHEAD 8

template <int E, int TN>
static inline void count_pass(constant sort_words &w, threadgroup uchar *mem,
                              uint g, uint t) {
  typedef typename el<E>::T T;
  constexpr uint TILE = ST * TN;
  threadgroup uint *cnt = (threadgroup uint *)mem;
  threadgroup uint *flag = cnt + NX_SIMDS * 256;
  const uint sg = t / 32, below = nx_below(t & 31);
  const ulong u = w.first + g, s = unit_slice(w, u), part = u - s * w.units;
  const turn tn = turn_of(w, s);
  if (!tn.on) return;
  const uint p = uint(w.pass), eb = ebytes(w);
  for (uint i = t; i < NX_SIMDS * 256; i += ST) cnt[i] = 0;
  const long ox = tn.first ? slice_at(w, s).x : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong lo = part * w.span * TILE;
  const ulong hi = min(w.n, (part + 1) * w.span * TILE);
  for (ulong i0 = lo; i0 < hi; i0 += ST) {
    const ulong i = i0 + t;
    const bool valid = i < hi;
    T e = zero_of(T());
    if (valid) {
      ulong pos;
      if (tn.first) first_item(w, ox, s, i, e, pos);
      else get_el(tn.se, eb, i, e);
    }
    const uint d = valid ? kbyte(key_of(w, e), p) : 256;
    const uint m = nx_peers<9>(d);
    if (d < 256 && (m & below) == 0)
      cnt[sg * 256 + d] = cnt[sg * 256 + d] + popcount(m);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint k = 0;
#pragma unroll
  for (uint i = 0; i < NX_SIMDS; i++) k += cnt[i * 256 + t];
  const bool wide = (w.flags & NX_SORT_WIDE) != 0;
  coherent(device) device uint *mine =
      (coherent(device) device uint *)part_at(w, s, part, t);
  mine[0] = k;
  if (wide) mine[1] = 0;
  if (!slice_last(w, s, t, flag)) return;

  ulong run = bin_place(w, bin_at(w, s, p, t));
  const ulong parts = w.units;
  coherent(device) device uint *c =
      (coherent(device) device uint *)part_at(w, s, 0, t);
  const uint step = 256 * (wide ? 2 : 1);
  for (ulong q0 = 0; q0 < parts; q0 += PARTS_AHEAD) {
    uint ahead[PARTS_AHEAD];
#pragma unroll
    for (uint j = 0; j < PARTS_AHEAD; j++)
      ahead[j] = q0 + j < parts ? c[(q0 + j) * step] : 0;
#pragma unroll
    for (uint j = 0; j < PARTS_AHEAD; j++) {
      if (q0 + j >= parts) break;
      coherent(device) device uint *at = c + (q0 + j) * step;
      at[0] = uint(run);
      if (wide) at[1] = uint(run >> 32);
      run += ahead[j];
    }
  }
}

#define COUNT(E, TN)                                                           \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_count_##E(constant sort_words &w [[buffer(0)]],                    \
                      threadgroup uchar *mem [[threadgroup(0)]],               \
                      uint g [[threadgroup_position_in_grid]],                 \
                      uint t [[thread_index_in_threadgroup]]) {                \
    count_pass<E, TN>(w, mem, g, t);                                           \
  }
COUNT(4, NX_SORT_TILE_ITEMS4)
COUNT(8, NX_SORT_TILE_ITEMS8)
COUNT(16, NX_SORT_TILE_ITEMS16)

/* A pass's position of an item: 32 bits, or 64 under NX_SORT_WIDE. */
template <bool WIDE> struct pos_t;
template <> struct pos_t<false> {
  typedef uint T;
};
template <> struct pos_t<true> {
  typedef ulong T;
};

static inline void get_pos(device const uchar *b, ulong i, thread uint &p) {
  p = ((device const uint *)b)[i];
}
static inline void get_pos(device const uchar *b, ulong i, thread ulong &p) {
  p = ((device const ulong *)b)[i];
}
static inline void put_pos(device uchar *b, ulong i, uint p) {
  ((device uint *)b)[i] = p;
}
static inline void put_pos(device uchar *b, ulong i, ulong p) {
  ((device ulong *)b)[i] = p;
}

/* A part's first place in bin [b]: sort_count's. */
static inline ulong part_place(constant sort_words &w, ulong s, ulong q,
                               uint b) {
  device uint *at = part_at(w, s, q, b);
  if ((w.flags & NX_SORT_WIDE) != 0) return ulong(at[0]) | ulong(at[1]) << 32;
  return at[0];
}

/* The 32-bit words a pass moves of an item, an element's then its
   position's, and how many the exchange holds at once: as many as fit
   beside the scatter's other memory, NX_SORT_SCATTER_REST bytes. */
template <int E, bool WIDE> constexpr uint fields_of() {
  return E / 4 + (WIDE ? 2 : 1);
}
template <int E, int TN, bool WIDE> constexpr uint held_of() {
  return (32768 - NX_SORT_SCATTER_REST) / (ST * TN * 4) <
                 fields_of<E, WIDE>()
             ? (32768 - NX_SORT_SCATTER_REST) / (ST * TN * 4)
             : fields_of<E, WIDE>();
}

/* Moves each part's items to their places, from the part's first place
   in each bin: sort_hist's where the slice is one part, else
   sort_count's. Each tile of the part in turn: its items ranked (nx_rank),
   their words moved to their ranks through the exchange, a group of
   fields at a time, then each thread stores the items at ranks t + j
   NX_SORT_THREADS, so that consecutive threads write consecutive
   addresses within a bin's run; each bin's next place carries to the
   next tile.

   Threadgroup memory: the exchange (TILE words a field it holds at once),
   the SIMD-groups' counts (NX_SIMDS rows of 257 16-bit counts), the bins'
   places (256 64-bit words), their starts and counts (257 words each),
   and NX_SIMDS sums. */
template <int E, int TN, bool WIDE>
static inline void scatter_pass(constant sort_words &w, threadgroup uchar *mem,
                                uint g, uint t) {
  typedef typename el<E>::T T;
  typedef typename pos_t<WIDE>::T P;
  constexpr uint TILE = ST * TN, KW = E / 4;
  constexpr uint F = fields_of<E, WIDE>(), H = held_of<E, TN, WIDE>();
  threadgroup uint *xch = (threadgroup uint *)mem;
  threadgroup ushort *cnt = (threadgroup ushort *)(xch + TILE * H);
  threadgroup ulong *place = (threadgroup ulong *)(cnt + NX_SIMDS * 257);
  threadgroup uint *start = (threadgroup uint *)(place + 256);
  threadgroup uint *count = start + 257;
  threadgroup uint *sums = count + 257;
  const uint lane = t & 31, sg = t / 32;
  const ulong u = w.first + g, s = unit_slice(w, u), part = u - s * w.units;
  const turn tn = turn_of(w, s);
  if (!tn.on) return;
  const uint p = uint(w.pass), eb = ebytes(w);
  place[t] = w.units == 1 ? bin_place(w, bin_at(w, s, p, t))
                          : part_place(w, s, part, t);
  const offs o = tn.first || tn.final ? slice_at(w, s) : offs{0, 0, 0};
  const bool store = (w.flags & NX_SORT_STORE) != 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong t0 = part * w.span, t1 = min(w.tiles, (part + 1) * w.span);
  for (ulong tile = t0; tile < t1; tile++) {
    const ulong l0 = tile * TILE;
    const uint valid = uint(min(ulong(TILE), w.n - l0));
    T e[TN];
    P pos[TN];
    uint d[TN], r[TN];
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const uint l = sg * TN * 32 + j * 32 + lane;
      e[j] = zero_of(T());
      pos[j] = 0;
      d[j] = 256;
      if (l >= valid) continue;
      if (tn.first) {
        ulong at;
        first_item(w, o.x, s, l0 + l, e[j], at);
        pos[j] = P(at);
      } else {
        get_el(tn.se, eb, l0 + l, e[j]);
        get_pos(tn.sp, l0 + l, pos[j]);
      }
      d[j] = kbyte(key_of(w, e[j]), p);
    }
    nx_rank<TN, ushort>(d, r, t, cnt, start, count, sums);

    /* Field f of an item: word f of its element, then word f - KW of its
       position. */
#pragma unroll
    for (uint f0 = 0; f0 < F; f0 += H) {
#pragma unroll
      for (uint f = f0; f < f0 + H && f < F; f++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          xch[(f - f0) * TILE + r[j]] =
              f < KW ? word_of(e[j], f) : uint(ulong(pos[j]) >> (32 * (f - KW)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (uint f = f0; f < f0 + H && f < F; f++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const uint v = xch[(f - f0) * TILE + t + j * ST];
          if (f < KW) set_word(e[j], f, v);
          else if (f == KW) pos[j] = WIDE ? P((ulong(pos[j]) & ~0xFFFFFFFFul) | v) : P(v);
          else pos[j] = P((ulong(pos[j]) & 0xFFFFFFFFul) | ulong(v) << 32);
        }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma unroll
    for (int j = 0; j < TN; j++) {
      const uint i = t + j * ST;
      if (i >= valid) continue;
      const uint b = kbyte(key_of(w, e[j]), p);
      const ulong at = place[b] + i - start[b];
      if (!tn.final) {
        put_el(tn.de, eb, at, e[j]);
        put_pos(tn.dp, at, pos[j]);
      } else if (at < w.keep) {
        w.positions[o.p + long(at) * w.sp] = long(pos[j]);
        if (store) store_el(w, o.v + long(at) * w.sv, e[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    place[t] += count[t];
  }
}

#define SCATTER(E, TN)                                                         \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_scatter_##E(constant sort_words &w [[buffer(0)]],                  \
                        threadgroup uchar *mem [[threadgroup(0)]],             \
                        uint g [[threadgroup_position_in_grid]],               \
                        uint t [[thread_index_in_threadgroup]]) {              \
    scatter_pass<E, TN, false>(w, mem, g, t);                                  \
  }                                                                            \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_scatter_wide_##E(constant sort_words &w [[buffer(0)]],             \
                             threadgroup uchar *mem [[threadgroup(0)]],        \
                             uint g [[threadgroup_position_in_grid]],          \
                             uint t [[thread_index_in_threadgroup]]) {         \
    scatter_pass<E, TN, true>(w, mem, g, t);                                   \
  }
SCATTER(4, NX_SORT_TILE_ITEMS4)
SCATTER(8, NX_SORT_TILE_ITEMS8)
SCATTER(16, NX_SORT_TILE_ITEMS16)

/* Selects */

/* Slice [s]'s select state. */
static inline device uint *select_at(constant sort_words &w, ulong s) {
  const bool wide = (w.flags & NX_SORT_WIDE) != 0;
  return w.hist + s * (wide ? NX_SORT_SELECT_WORDS_WIDE : NX_SORT_SELECT_WORDS);
}

/* Digit [j] of (key, position) from the most significant, a select's,
   positions of [pbytes] bytes. */
template <typename T>
static inline uint cdigit(uint kb, uint pb, T k, ulong pos, uint j) {
  if (j < kb) return kbyte(k, kb - 1 - j);
  return uint(pos >> (8 * (kb + pb - 1 - j))) & 255;
}

/* -1, 0 or 1 as the first [rounds] digits of (k, pos) are below, equal to
   or above [prefix]. */
template <typename T>
static inline int against(uint kb, uint pb, T k, ulong pos,
                          threadgroup const uchar *prefix, uint rounds) {
  for (uint j = 0; j < rounds; j++) {
    const uint d = cdigit(kb, pb, k, pos, j);
    if (d != prefix[j]) return d < prefix[j] ? -1 : 1;
  }
  return 0;
}

/* Round [pass] of each slice's select, unless done: counts that digit of
   the elements whose earlier digits match the prefix, by leader adds into
   one threadgroup copy, then one device add a nonzero bin. The slice's
   last chunk finds the bin of the [need]th of them, which extends the
   prefix, and ends the select where the elements below or matching it
   are at most a cap, or no digit is left.

   Threadgroup memory: 256 counts, the prefix's 32 bytes, the combine's
   256 64-bit sums and two words. */
template <int E>
static inline void round_pass(constant sort_words &w, threadgroup uchar *mem,
                              uint g, uint t) {
  typedef typename el<E>::T T;
  threadgroup atomic_uint *h = (threadgroup atomic_uint *)mem;
  threadgroup uchar *prefix = (threadgroup uchar *)(mem + 4 * 256);
  threadgroup ulong *buf = (threadgroup ulong *)(prefix + 32);
  threadgroup uint *flag = (threadgroup uint *)(buf + ST);
  const uint below = nx_below(t & 31);
  const ulong u = w.first + g, s = unit_slice(w, u), c = u - s * w.units;
  device uint *st = select_at(w, s);
  if (st[NX_SORT_SELECT_DONE] != 0) return;
  atomic_store_explicit(&h[t], 0u, memory_order_relaxed);
  if (t < 32) prefix[t] = ((device const uchar *)st)[t];
  const long ox = slice_at(w, s).x;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint round = uint(w.pass), kb = uint(w.kbytes), pb = uint(w.pbytes);
  const ulong lo = c * NX_SORT_CHUNK, hi = min(w.n, lo + NX_SORT_CHUNK);
  for (ulong i0 = lo; i0 < hi; i0 += ST) {
    const ulong i = i0 + t;
    uint d = 256;
    if (i < hi) {
      T e;
      ulong pos;
      first_item(w, ox, s, i, e, pos);
      const T k = key_of(w, e);
      if (against(kb, pb, k, pos, prefix, round) == 0)
        d = cdigit(kb, pb, k, pos, round);
    }
    const uint m = nx_peers<9>(d);
    if (d < 256 && (m & below) == 0)
      atomic_fetch_add_explicit(&h[d], popcount(m), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool wide = (w.flags & NX_SORT_WIDE) != 0;
  device uint *bins = st + NX_SORT_SELECT_STATE;
  const uint v = atomic_load_explicit(&h[t], memory_order_relaxed);
  if (v != 0) {
    if (wide)
      nx_carry_add((device atomic_uint *)(bins + 2 * t),
                   (device atomic_uint *)(bins + 2 * t + 1), v,
                   uint(w.carry));
    else
      atomic_fetch_add_explicit((device atomic_uint *)(bins + t), v,
                                memory_order_relaxed);
  }
  if (!slice_last(w, s, t, flag)) return;

  coherent(device) device uint *cst = (coherent(device) device uint *)st;
  coherent(device) device uint *cb = cst + NX_SORT_SELECT_STATE;
  const ulong need =
      round == 0 ? w.keep
                 : ulong(cst[NX_SORT_SELECT_NEED]) |
                       ulong(cst[NX_SORT_SELECT_NEED + 1]) << 32;
  const ulong x = wide ? nx_carried(cb[2 * t], cb[2 * t + 1], uint(w.carry))
                       : ulong(cb[t]);
  ulong all;
  const ulong e = nx_group_excl64(x, t, buf, all);
  if (wide) {
    cb[2 * t] = 0;
    cb[2 * t + 1] = 0;
  } else cb[t] = 0;
  if (e < need && need <= e + x) {
    const ulong left = need - e, under = w.keep - left;
    ((coherent(device) device uchar *)cst)[NX_SORT_SELECT_PREFIX * 4 + round] =
        uchar(t);
    cst[NX_SORT_SELECT_NEED] = uint(left);
    cst[NX_SORT_SELECT_NEED + 1] = uint(left >> 32);
    cst[NX_SORT_SELECT_ROUNDS] = round + 1;
    cst[NX_SORT_SELECT_DONE] =
        under + x <= cap_of<E>() || round + 1 == kb + pb ? 1u : 0u;
  }
}

#define ROUND(E)                                                               \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_round_##E(constant sort_words &w [[buffer(0)]],                    \
                      threadgroup uchar *mem [[threadgroup(0)]],               \
                      uint g [[threadgroup_position_in_grid]],                 \
                      uint t [[thread_index_in_threadgroup]]) {                \
    round_pass<E>(w, mem, g, t);                                               \
  }
ROUND(4)
ROUND(8)
ROUND(16)

/* Gathers each slice's elements below or matching its select's prefix
   into its candidates, a SIMD-group's at once by one add to the slice's
   count. Their order there depends on the order groups run in;
   sort_block orders them by (key, position), which every candidate holds
   a distinct one of.

   Threadgroup memory: the prefix's 32 bytes. */
template <int E>
static inline void gather_pass(constant sort_words &w, threadgroup uchar *mem,
                               uint g, uint t) {
  typedef typename el<E>::T T;
  constexpr uint CAP = cap_of<E>();
  threadgroup uchar *prefix = (threadgroup uchar *)mem;
  const uint lane = t & 31, below = nx_below(lane);
  const ulong u = w.first + g, s = unit_slice(w, u), c = u - s * w.units;
  device uint *st = select_at(w, s);
  if (t < 32) prefix[t] = ((device const uchar *)st)[t];
  const long ox = slice_at(w, s).x;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint rounds = st[NX_SORT_SELECT_ROUNDS];
  const uint kb = uint(w.kbytes), pb = uint(w.pbytes), eb = ebytes(w);
  device atomic_uint *cnt = (device atomic_uint *)(st + NX_SORT_SELECT_CNT);
  device uchar *ce = w.cand + s * CAP * eb;
  device ulong *cp = w.cand_p + s * CAP;
  const ulong lo = c * NX_SORT_CHUNK, hi = min(w.n, lo + NX_SORT_CHUNK);
  for (ulong i0 = lo; i0 < hi; i0 += ST) {
    const ulong i = i0 + t;
    T e = zero_of(T());
    ulong pos = 0;
    bool take = false;
    if (i < hi) {
      first_item(w, ox, s, i, e, pos);
      take = against(kb, pb, key_of(w, e), pos, prefix, rounds) <= 0;
    }
    const uint m = uint(ulong(simd_ballot(take)));
    if (m == 0) continue;
    const uint leader = ctz(m);
    uint at = 0;
    if (lane == leader)
      at = atomic_fetch_add_explicit(cnt, popcount(m), memory_order_relaxed);
    at = simd_shuffle(at, ushort(leader));
    if (!take) continue;
    const uint slot = at + popcount(m & below);
    put_el(ce, eb, slot, e);
    cp[slot] = pos;
  }
}

#define GATHER(E)                                                              \
  NX_KERNEL(NX_SORT_THREADS)                                                   \
  void sort_gather_##E(constant sort_words &w [[buffer(0)]],                   \
                       threadgroup uchar *mem [[threadgroup(0)]],              \
                       uint g [[threadgroup_position_in_grid]],                \
                       uint t [[thread_index_in_threadgroup]]) {               \
    gather_pass<E>(w, mem, g, t);                                              \
  }
GATHER(4)
GATHER(8)
GATHER(16)

/* Numbering */

struct unique_words {
  device long *ids;
  device long *first;
  device long *counts;
  device long *count;
  device const ulong *keys;
  device const ulong *skeys;
  device const long *order;
  device ulong *heads;
  device atomic_uint *tickets;
  device long *gid;
  device long *gfirst;
  device long *gstart;
  device long *k;
  device atomic_uint *bits;
  device ulong *below;
  device uint *set;
  ulong m;
  ulong n;
  ulong w;
  long s0;
  long s1;
  ulong span;
  ulong wspan;
  ulong flags;
  ulong first_unit;
  ulong end;
};

NX_FIELD(unique_words, ids, NX_UNIQUE_IDS);
NX_FIELD(unique_words, first, NX_UNIQUE_FIRST);
NX_FIELD(unique_words, counts, NX_UNIQUE_COUNTS);
NX_FIELD(unique_words, count, NX_UNIQUE_COUNT);
NX_FIELD(unique_words, keys, NX_UNIQUE_KEYS);
NX_FIELD(unique_words, skeys, NX_UNIQUE_SKEYS);
NX_FIELD(unique_words, order, NX_UNIQUE_ORDER);
NX_FIELD(unique_words, heads, NX_UNIQUE_HEADS);
NX_FIELD(unique_words, tickets, NX_UNIQUE_TICKETS);
NX_FIELD(unique_words, gid, NX_UNIQUE_GID);
NX_FIELD(unique_words, gfirst, NX_UNIQUE_GFIRST);
NX_FIELD(unique_words, gstart, NX_UNIQUE_GSTART);
NX_FIELD(unique_words, k, NX_UNIQUE_K);
NX_FIELD(unique_words, bits, NX_UNIQUE_BITS);
NX_FIELD(unique_words, below, NX_UNIQUE_BELOW);
NX_FIELD(unique_words, set, NX_UNIQUE_SET);
NX_FIELD(unique_words, m, NX_UNIQUE_M);
NX_FIELD(unique_words, n, NX_UNIQUE_N);
NX_FIELD(unique_words, w, NX_UNIQUE_W);
NX_FIELD(unique_words, s0, NX_UNIQUE_S0);
NX_FIELD(unique_words, s1, NX_UNIQUE_S1);
NX_FIELD(unique_words, span, NX_UNIQUE_SPAN);
NX_FIELD(unique_words, wspan, NX_UNIQUE_WSPAN);
NX_FIELD(unique_words, flags, NX_UNIQUE_FLAGS);
NX_FIELD(unique_words, first_unit, NX_UNIQUE_FIRST_UNIT);
NX_FIELD(unique_words, end, NX_UNIQUE_END);
NX_SIZE(unique_words, NX_UNIQUE_WORDS);

#define UNIQUE_ITEMS (NX_UNIQUE_TILE / ST)

/* Whether sorted rows [i - 1] and [i] differ. */
static inline bool differs(constant unique_words &u, ulong i) {
  if ((u.flags & NX_UNIQUE_BY_SKEYS) != 0) return u.skeys[i] != u.skeys[i - 1];
  const bool ordered = (u.flags & NX_UNIQUE_BY_ORDER) != 0;
  const long a = ordered ? u.order[i] : long(i);
  const long b = ordered ? u.order[i - 1] : long(i) - 1;
  for (ulong j = 0; j < u.w; j++)
    if (u.keys[a * u.s0 + long(j) * u.s1] != u.keys[b * u.s0 + long(j) * u.s1])
      return true;
  return false;
}

static inline bool head(constant unique_words &u, ulong i) {
  return i == 0 || differs(u, i);
}

/* Whether the calling group is the last of the call's [parts] parts in
   the launch to take the call's ticket, in every thread of it: none where
   the last part lies past the launch. */
static inline bool call_last(constant unique_words &u, ulong parts, uint t,
                             threadgroup uint *flag) {
  if (parts - 1 >= u.end) return false;
  return nx_last(u.tickets, uint(parts - u.first_unit), t, flag);
}

/* Replaces the [count] 64-bit words at [c] by their exclusive sum, and is
   their sum. Every thread calls it. */
static inline ulong scan_words(coherent(device) device ulong *c, ulong count,
                               uint t, threadgroup ulong *buf) {
  ulong run = 0;
  for (ulong q0 = 0; q0 < count; q0 += ST) {
    const ulong q = q0 + t;
    const ulong v = q < count ? c[q] : 0;
    ulong all;
    const ulong e = nx_group_excl64(v, t, buf, all);
    if (q < count) c[q] = run + e;
    run += all;
  }
  return run;
}

/* Counts the keys that start in each part of the sorted rows into
   [heads]; the last part makes them each part's first key, and stores the
   keys counted and the end of the last key's sorted rows.

   Threadgroup memory: 256 64-bit sums and two words. */
NX_KERNEL(NX_SORT_THREADS)
void unique_count(constant unique_words &u [[buffer(0)]],
                  threadgroup uchar *mem [[threadgroup(0)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]]) {
  threadgroup ulong *buf = (threadgroup ulong *)mem;
  threadgroup uint *flag = (threadgroup uint *)(buf + ST);
  const ulong parts = (u.n + u.span - 1) / u.span;
  const ulong q = u.first_unit + g;
  const ulong hi = min(u.n, (q + 1) * u.span);
  ulong c = 0;
  for (ulong i = q * u.span + t; i < hi; i += ST) c += head(u, i) ? 1 : 0;
  ulong all;
  nx_group_excl64(c, t, buf, all);
  coherent(device) device ulong *heads = (coherent(device) device ulong *)u.heads;
  if (t == 0) heads[q] = all;
  if (!call_last(u, parts, t, flag)) return;
  const ulong k = scan_words(heads, parts, t, buf);
  if (t == 0) {
    *u.k = long(k);
    u.gstart[k] = long(u.n);
  }
}

/* Each sorted row's key's number in sorted order, gid, from its part's
   first key; each key's first row, gfirst (the stable sort puts it
   first), and first sorted row, gstart; a bit at each first row, counted
   into [set] for its part of the bit words. The last part makes those
   counts each part's first bit.

   Threadgroup memory: NX_SIMDS sums, 256 64-bit sums and two words. */
NX_KERNEL(NX_SORT_THREADS)
void unique_heads(constant unique_words &u [[buffer(0)]],
                  threadgroup uchar *mem [[threadgroup(0)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint t [[thread_index_in_threadgroup]]) {
  threadgroup ulong *buf = (threadgroup ulong *)mem;
  threadgroup uint *flag = (threadgroup uint *)(buf + ST);
  threadgroup uint *sums = flag + 2;
  const bool ordered = (u.flags & NX_UNIQUE_BY_ORDER) != 0;
  const ulong parts = (u.n + u.span - 1) / u.span;
  const ulong q = u.first_unit + g;
  const ulong hi = min(u.n, (q + 1) * u.span);
  ulong before = u.heads[q];
  for (ulong c0 = q * u.span; c0 < hi; c0 += NX_UNIQUE_TILE) {
    const ulong i0 = c0 + t * UNIQUE_ITEMS;
    uint hs = 0;
    for (uint j = 0; j < UNIQUE_ITEMS; j++) {
      const ulong i = i0 + j;
      if (i < hi && head(u, i)) hs |= 1u << j;
    }
    uint all;
    long k = long(before + nx_group_excl(popcount(hs), t, sums, all)) - 1;
    for (uint j = 0; j < UNIQUE_ITEMS; j++) {
      const ulong i = i0 + j;
      if (i >= hi) break;
      if ((hs >> j & 1) != 0) {
        const long f = ordered ? u.order[i] : long(i);
        k++;
        u.gfirst[k] = f;
        u.gstart[k] = long(i);
        atomic_fetch_or_explicit(&u.bits[f >> 5], 1u << (f & 31),
                                 memory_order_relaxed);
        atomic_fetch_add_explicit(
            (device atomic_uint *)&u.set[2 * ((ulong(f) >> 5) / u.wspan)], 1u,
            memory_order_relaxed);
      }
      u.gid[i] = k;
    }
    before += all;
  }
  if (!call_last(u, parts, t, flag)) return;
  const ulong words = (u.n + 31) / 32, wparts = (words + u.wspan - 1) / u.wspan;
  coherent(device) device uint *set = (coherent(device) device uint *)u.set;
  ulong run = 0;
  for (ulong q0 = 0; q0 < wparts; q0 += ST) {
    const ulong w = q0 + t;
    const ulong v = w < wparts ? set[2 * w] : 0;
    ulong all;
    const ulong e = nx_group_excl64(v, t, buf, all);
    if (w < wparts) {
      set[2 * w] = uint(run + e);
      set[2 * w + 1] = uint((run + e) >> 32);
    }
    run += all;
  }
}

/* The first rows set before each bit word: each part's from its first
   bit, by a scan.

   Threadgroup memory: NX_SIMDS sums. */
NX_KERNEL(NX_SORT_THREADS)
void unique_rank(constant unique_words &u [[buffer(0)]],
                 threadgroup uchar *mem [[threadgroup(0)]],
                 uint g [[threadgroup_position_in_grid]],
                 uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint *sums = (threadgroup uint *)mem;
  device const uint *bits = (device const uint *)u.bits;
  const ulong words = (u.n + 31) / 32;
  const ulong q = u.first_unit + g;
  const ulong hi = min(words, (q + 1) * u.wspan);
  ulong before = ulong(u.set[2 * q]) | ulong(u.set[2 * q + 1]) << 32;
  for (ulong c0 = q * u.wspan; c0 < hi; c0 += NX_UNIQUE_TILE) {
    const ulong i0 = c0 + t * UNIQUE_ITEMS;
    uint c = 0;
    for (uint j = 0; j < UNIQUE_ITEMS; j++)
      if (i0 + j < hi) c += popcount(bits[i0 + j]);
    uint all;
    ulong at = before + nx_group_excl(c, t, sums, all);
    for (uint j = 0; j < UNIQUE_ITEMS && i0 + j < hi; j++) {
      u.below[i0 + j] = at;
      at += popcount(bits[i0 + j]);
    }
    before += all;
  }
}

/* Each row's id, the rank of its key's first row among the first rows;
   each key's first row and count, and the entries past the keys: a
   thread a row, and an entry. */
NX_KERNEL(NX_SORT_THREADS)
void unique_ids(constant unique_words &u [[buffer(0)]],
                uint g [[threadgroup_position_in_grid]],
                uint t [[thread_index_in_threadgroup]]) {
  device const uint *bits = (device const uint *)u.bits;
  const bool ordered = (u.flags & NX_UNIQUE_BY_ORDER) != 0;
  const ulong i = (u.first_unit + g) * ST + t;
  const long k = u.n == 0 ? 0 : *u.k;
  if (i < u.n) {
    const long gi = u.gid[i], f = u.gfirst[gi];
    const uint under = bits[f >> 5] & ((1u << (f & 31)) - 1);
    const long num = long(u.below[f >> 5]) + long(popcount(under));
    u.ids[ordered ? u.order[i] : long(i)] = num;
    if (u.gstart[gi] == long(i) && ulong(num) < u.m) {
      u.first[num] = f;
      u.counts[num] = u.gstart[gi + 1] - long(i);
    }
  }
  const ulong j = ulong(k) + i;
  if (j < u.m) {
    u.first[j] = long(u.n);
    u.counts[j] = 0;
  }
  if (i == 0) *u.count = k;
}
