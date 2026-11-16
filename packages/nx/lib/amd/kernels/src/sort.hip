/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Sorts and numberings, with their words of kernels.h.

   A sort orders each slice of x by its elements' keys, unsigned integers
   in the contract's order (Keys), by radix of 8-bit digits. Three paths,
   the host's choice by the slice's length n and the elements it keeps:

   - sort_block: slices of at most [cap] elements, [per] of them a group,
     sorted in LDS.
   - sort_hist, then sort_count and sort_scatter once a key byte: longer
     slices, by LSD radix over global memory. sort_hist counts every digit
     of every slice in one read. A pass cuts a slice's tiles into parts of
     consecutive tiles: sort_count counts each part's digits, and the
     slice's last part to take its ticket turns the counts into each
     part's first place in each bin; sort_scatter then ranks each tile of
     a part in turn and moves its items there. A slice of one part takes
     its places from sort_hist and no sort_count. A pass whose digit every
     key of a slice shares is skipped for that slice.
   - sort_round once a digit, sort_gather, then sort_block: a long slice
     that keeps at most [cap] / 4 elements. Rounds of an MSD radix select
     on (key, position) find the kept elements' least prefix that leaves
     at most [cap] candidates, which sort_block then sorts.

   A numbering sorts its rows stably, one word at a time from the last, by
   the radix passes, then numbers them: unique_count, unique_heads,
   unique_rank, unique_ids.

   Every kernel's groups take its work's units, tiles, chunks or parts,
   [groups] apart, so the host sizes a grid by the work alone.

   Determinism. No float arithmetic. Atomics are integer adds into counts,
   candidate slots (sorted by a distinct key after), ORs of bits, a
   sub-byte element's store of its own bits, and tickets.

   Progress. No group waits on another: a slice's or a call's last group
   to take a ticket ({!last_of}) finishes what the others counted.

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
   uniform.

   Ranks. A group ranks its items' 8-bit digits stably in the order wave,
   then item, then lane (rank_digits): within a wave by the lanes of its
   digit, nine ballots, then each bin's waves one after another, then the
   bins. Items past the slice take digit 256, after every real one. */

#include "device.h"

#define ST NX_SORT_THREADS
#define WAVES (ST / 32)

/* Arithmetic */

DEVICE int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
DEVICE int popc(uint32_t x) { return __builtin_popcount(x); }
DEVICE int popc64(uint64_t x) { return __builtin_popcountll(x); }

/* Group scans */

/* The exclusive sum of [v] over the group's work-items in item order, and
   in [total] their sum. Every work-item calls it. */
template <class T> DEVICE T sum_excl(T v, T &total) {
  return group_excl<ST>(v, (T)0, Add_op<T>{}, total);
}

/* Whether [p] holds in some work-item of the group. Every work-item calls
   it. */
DEVICE bool group_any(bool p) {
  static SHARED uint32_t any[WAVES];
  const uint32_t b = __builtin_amdgcn_ballot_w32(p);
  barrier();
  if (lane() == 0) any[wave()] = b;
  barrier();
  uint32_t a = 0;
#pragma unroll
  for (int i = 0; i < WAVES; i++) a |= any[i];
  return a != 0;
}

/* Keys */

/* A key's format (NX_SORT_KEY_*). */
struct Form {
  int kind, ebits, kbytes;
  uint64_t nan_abs, kmask, flip, flip_signed, dmask_lo, dmask_hi;
};

DEVICE Form form_of(const int64_t *k) {
  Form f;
  f.kind = (int)uniform(k[NX_SORT_KEY_KIND]);
  f.ebits = (int)uniform(k[NX_SORT_KEY_EBITS]);
  f.kbytes = (int)uniform(k[NX_SORT_KEY_KBYTES]);
  f.nan_abs = (uint64_t)uniform(k[NX_SORT_KEY_NAN_ABS]);
  f.kmask = (uint64_t)uniform(k[NX_SORT_KEY_KMASK]);
  f.flip = (uint64_t)uniform(k[NX_SORT_KEY_FLIP]);
  f.flip_signed = (uint64_t)uniform(k[NX_SORT_KEY_FLIP_SIGNED]);
  f.dmask_lo = (uint64_t)uniform(k[NX_SORT_KEY_DMASK_LO]);
  f.dmask_hi = (uint64_t)uniform(k[NX_SORT_KEY_DMASK_HI]);
  return f;
}

struct El {
  uint64_t lo, hi;
};

struct Key {
  uint64_t lo, hi;
};

DEVICE bool fnan(uint64_t b, int bits, uint64_t nan_abs) {
  return (b & ((1ull << (bits - 1)) - 1)) > nan_abs;
}

DEVICE uint64_t fkey(uint64_t b, int bits) {
  const uint64_t sign = 1ull << (bits - 1), mask = sign | (sign - 1);
  return b & sign ? ~b & mask : b | sign;
}

/* A real element's key is its bits xored with [flip_signed] where its
   sign is set and [flip] where not, or [kmask] where its magnitude is past
   [nan_abs]: the flips are the sign and all ones for a float, the sign
   twice for a signed integer, zero for the others, and [nan_abs] all ones
   for every non-float. A boolean's bits are first 1 where not zero. */
DEVICE Key key_of(const Form &F, El e) {
  Key k = {0, 0};
  const int bits = F.ebits;
  if (F.kind != NX_SORT_COMPLEX) {
    const uint64_t b = F.kind == NX_SORT_BOOL ? e.lo != 0 : e.lo;
    const uint64_t sign = 1ull << (bits - 1);
    k.lo = (b & ~sign) > F.nan_abs ? F.kmask
                                    : b ^ (b & sign ? F.flip_signed : F.flip);
  } else if (bits == 64) {
    const uint64_t re = e.lo & 0xFFFFFFFFu, im = e.lo >> 32;
    k.lo = fnan(re, 32, F.nan_abs) || fnan(im, 32, F.nan_abs)
               ? ~0ull
               : fkey(re, 32) << 32 | fkey(im, 32);
  } else if (fnan(e.lo, 64, F.nan_abs) || fnan(e.hi, 64, F.nan_abs)) {
    k.lo = k.hi = ~0ull;
  } else {
    k.hi = fkey(e.lo, 64);
    k.lo = fkey(e.hi, 64);
  }
  k.lo ^= F.dmask_lo;
  k.hi ^= F.dmask_hi;
  return k;
}

/* Byte [p] of [k], from the least significant. */
DEVICE int kbyte(Key k, int p) {
  return (int)((p < 8 ? k.lo >> (8 * p) : k.hi >> (8 * (p - 8))) & 255);
}

/* Digit [j] of (key, position) from the most significant, a select's,
   positions of [pbytes] bytes. */
DEVICE int cdigit(const Form &F, int pbytes, Key k, int64_t pos, int j) {
  if (j < F.kbytes) return kbyte(k, F.kbytes - 1 - j);
  return (int)((uint64_t)pos >> (8 * (F.kbytes + pbytes - 1 - j)) & 255);
}

/* Elements */

/* Element [off] of [base], [ebits] wide; a sub-byte one's code from the
   low bits of its byte first. A complex number is read a part at a time:
   its array promises only a part's alignment. */
DEVICE El load_el(const Form &F, const void *base, int64_t off) {
  El e = {0, 0};
  if (F.kind == NX_SORT_COMPLEX && F.ebits == 64) {
    const uint32_t *c = (const uint32_t *)base + 2 * off;
    e.lo = (uint64_t)c[1] << 32 | c[0];
    return e;
  }
  switch (F.ebits) {
  case 1:
  case 4: {
    const int64_t bit = off * F.ebits;
    const uint32_t b = ((const uint8_t *)base)[bit >> 3];
    e.lo = (b >> (bit & 7)) & ((1u << F.ebits) - 1);
    break;
  }
  case 8: e.lo = ((const uint8_t *)base)[off]; break;
  case 16: e.lo = ((const uint16_t *)base)[off]; break;
  case 32: e.lo = ((const uint32_t *)base)[off]; break;
  case 64: e.lo = ((const uint64_t *)base)[off]; break;
  default:
    e.lo = ((const uint64_t *)base)[2 * off];
    e.hi = ((const uint64_t *)base)[2 * off + 1];
  }
  return e;
}

/* Stores [e] at element [off] of [base], a complex number a part at a
   time. A sub-byte element shares its byte with others, maybe of other
   slices: it clears and sets its own bits by atomics on their word, which
   commute with its neighbours'. */
DEVICE void store_el(const Form &F, void *base, int64_t off, El e) {
  if (F.kind == NX_SORT_COMPLEX && F.ebits == 64) {
    uint32_t *c = (uint32_t *)base + 2 * off;
    c[0] = (uint32_t)e.lo;
    c[1] = (uint32_t)(e.lo >> 32);
    return;
  }
  switch (F.ebits) {
  case 1:
  case 4: store_bits(base, off, F.ebits, (uint32_t)e.lo); break;
  case 8: ((uint8_t *)base)[off] = (uint8_t)e.lo; break;
  case 16: ((uint16_t *)base)[off] = (uint16_t)e.lo; break;
  case 32: ((uint32_t *)base)[off] = (uint32_t)e.lo; break;
  case 64: ((uint64_t *)base)[off] = e.lo; break;
  default:
    ((uint64_t *)base)[2 * off] = e.lo;
    ((uint64_t *)base)[2 * off + 1] = e.hi;
  }
}

/* Slices */

/* A sort's slices (NX_SORT_SLICES_*), [w] their staged words. */
struct Slices {
  const int64_t *w;
  int64_t n, count, sx, sv, sp, off_x, off_v, off_p;
  uint64_t n_div;
  bool small;
  int kept;
};

DEVICE Slices slices_of(const int64_t *s) {
  Slices S;
  S.w = s;
  S.n = uniform(s[NX_SORT_SLICES_N]);
  S.n_div = (uint64_t)uniform(s[NX_SORT_SLICES_N_DIV]);
  S.small = uniform(s[NX_SORT_SLICES_SMALL]);
  S.count = uniform(s[NX_SORT_SLICES_COUNT]);
  S.sx = uniform(s[NX_SORT_SLICES_SX]);
  S.sv = uniform(s[NX_SORT_SLICES_SV]);
  S.sp = uniform(s[NX_SORT_SLICES_SP]);
  S.off_x = uniform(s[NX_SORT_SLICES_OFF_X]);
  S.off_v = uniform(s[NX_SORT_SLICES_OFF_V]);
  S.off_p = uniform(s[NX_SORT_SLICES_OFF_P]);
  S.kept = (int)uniform(s[NX_SORT_SLICES_KEPT]);
  return S;
}

/* Slice [s]'s first elements of x, the values and the positions, into
   [o]. */
DEVICE void slice_at(const Slices &S, int64_t s, int64_t *o) {
  int64_t at[3] = {S.off_x, S.off_v, S.off_p};
  place(S.w + NX_SORT_SLICES_AXES, NX_SORT_SLICES_AXIS, S.kept, s, S.small,
        at);
  o[0] = at[0];
  o[1] = at[1];
  o[2] = at[2];
}

/* Element [i] of a group's [n]-element slices, below 2^32: its slice's
   index among them, [i] becoming its index in the slice. */
DEVICE int slice_of(const Slices &S, int64_t &i) {
  int64_t b = i;
  i = split(b, S.n, S.n_div, true);
  return (int)b;
}

/* Stages the launch's words, which end with its slices' at [slices], but
   those from [gap] to [gap_end] (device.h's stage). */
DEVICE void stage_sort(int64_t *w, int slices, int64_t gap = 0,
                       int64_t gap_end = 0) {
  const int64_t kept = words()[slices + NX_SORT_SLICES_KEPT];
  stage(w, slices + NX_SORT_SLICES_AXES + NX_SORT_SLICES_AXIS * kept, ST, gap,
        gap_end);
}

/* Where a sort writes: [values] under NX_SORT_VALUES and [positions],
   [keep] of each slice's. */
struct Out {
  void *values;
  int64_t *positions;
  int64_t keep;
  bool store_values;
};

/* Writes the element [e] at position [pos] of its slice, rank [r] of the
   sorted order, where the slice keeps that rank. */
DEVICE void emit(const Form &F, const Slices &S, const Out &O,
                 const int64_t *o, int64_t r, El e, int64_t pos) {
  if (r >= O.keep) return;
  O.positions[o[2] + r * S.sp] = pos;
  if (O.store_values) store_el(F, O.values, o[1] + r * S.sv, e);
}

/* Selects */

/* A slice's select, zero before its first round: the digits of (key,
   position) resolved in [prefix], the kept elements still to find among
   those that match them in [need], the candidates gathered in [cnt], the
   rounds resolved, whether it is done, and the round's counts. */
struct sort_select {
  uint64_t need, cnt;
  uint32_t rounds, done;
  uint8_t prefix[32];
  uint64_t hist[256];
};

static_assert(sizeof(sort_select) == NX_SORT_SELECT_BYTES,
              "NX_SORT_SELECT_BYTES is a select's bytes");

/* -1, 0 or 1 as the first [rounds] digits of (k, pos) are below, equal to
   or above [prefix]. */
DEVICE int against(const Form &F, int pbytes, Key k, int64_t pos,
                   const uint8_t *prefix, int rounds) {
  for (int j = 0; j < rounds; j++) {
    const int d = cdigit(F, pbytes, k, pos, j);
    if (d != prefix[j]) return d < prefix[j] ? -1 : 1;
  }
  return 0;
}

/* Block sorts */

#define BLOCK_ITEMS (NX_SORT_CAP / ST)

/* Sorts slices of at most [cap] elements in LDS: [per] slices from x a
   group, or, under NX_SORT_CANDIDATES, one slice's candidates
   (sort_gather). Each element's key waits in LDS, kbytes rounded up to
   words; the items' order moves as 16-bit indices, by LSD passes over the
   candidates' positions, then the keys' bytes, then the slices in the
   group. */
KERNEL SIZED(ST) void sort_block(Words<NX_SORT_BLOCK_WORDS>) {
  extern SHARED uint32_t dyn[];
  static SHARED int64_t w[NX_SORT_BLOCK_WORDS];
  static SHARED Rank<ST> R;
  static SHARED int64_t so[256][3];
  stage_sort(w, NX_SORT_BLOCK_SLICES);
  const Form F = form_of(w + NX_SORT_BLOCK_KEY);
  const Slices S = slices_of(w + NX_SORT_BLOCK_SLICES);
  const int flags = (int)uniform(w[NX_SORT_BLOCK_FLAGS]);
  const Out O = {address<void>(w, NX_SORT_BLOCK_VALUES),
                 address<int64_t>(w, NX_SORT_BLOCK_POSITIONS),
                 uniform(w[NX_SORT_BLOCK_KEEP]),
                 (flags & NX_SORT_VALUES) != 0};
  const uint8_t *x = address<const uint8_t>(w, NX_SORT_BLOCK_X);
  const sort_select *sel = address<const sort_select>(w, NX_SORT_BLOCK_SEL);
  const uint64_t *cand = address<const uint64_t>(w, NX_SORT_BLOCK_CAND);
  const int cap = (int)uniform(w[NX_SORT_BLOCK_CAP]);
  const int per = (int)uniform(w[NX_SORT_BLOCK_PER]);
  const int64_t groups = uniform(w[NX_SORT_BLOCK_GROUPS]);
  const bool cands = flags & NX_SORT_CANDIDATES;
  const int pb = cands ? (int)uniform(w[NX_SORT_BLOCK_PBYTES]) : 0;
  const int nb = cap / ST, kw = (F.kbytes + 3) / 4;
  const int ln = (int)lane(), wv = (int)wave();
  uint32_t *keys = dyn;
  uint16_t *ord = (uint16_t *)(keys + cap * kw);
  const int64_t n = S.n;
  const int64_t blocks = cands ? S.count : ceil_div(S.count, per);
  for (int64_t g = group_x(); g < blocks; g += groups) {
    int64_t s0;
    int b_slices, v;
    if (cands) {
      s0 = g;
      b_slices = 1;
      v = (int)sel[s0].cnt;
    } else {
      s0 = g * per;
      b_slices = (int)imin(per, S.count - s0);
      v = b_slices * (int)n;
    }
    if (item() < (uint32_t)b_slices) slice_at(S, s0 + item(), so[item()]);
    barrier();
    const uint64_t *cb = cand + s0 * cap * 3;
    for (int i = item(); i < v; i += ST) {
      El e;
      if (cands) e = {cb[3 * i], cb[3 * i + 1]};
      else {
        int64_t at = i;
        const int b = slice_of(S, at);
        e = load_el(F, x, so[b][0] + at * S.sx);
      }
      const Key k = key_of(F, e);
      for (int q = 0; q < kw; q++)
        keys[i * kw + q] = (uint32_t)((q < 2 ? k.lo : k.hi) >> (32 * (q & 1)));
    }
    barrier();
    uint16_t idx[BLOCK_ITEMS];
#pragma unroll
    for (int j = 0; j < BLOCK_ITEMS; j++)
      idx[j] = (uint16_t)(wv * nb * 32 + j * 32 + ln);
    const int passes = pb + F.kbytes + (b_slices > 1);
    for (int p = 0; p < passes; p++) {
      int d[BLOCK_ITEMS];
      uint32_t r[BLOCK_ITEMS];
#pragma unroll
      for (int j = 0; j < BLOCK_ITEMS; j++) {
        const int i = idx[j];
        if (j >= nb || i >= v) d[j] = 256;
        else if (p < pb) d[j] = (int)(cb[3 * i + 2] >> (8 * p) & 255);
        else if (p < pb + F.kbytes) {
          const int q = p - pb;
          d[j] = (int)(keys[i * kw + q / 4] >> (8 * (q & 3)) & 255);
        } else {
          int64_t at = i;
          d[j] = slice_of(S, at);
        }
      }
      rank_digits(d, nb, r, R);
#pragma unroll
      for (int j = 0; j < BLOCK_ITEMS; j++)
        if (j < nb) ord[r[j]] = idx[j];
      barrier();
#pragma unroll
      for (int j = 0; j < BLOCK_ITEMS; j++)
        if (j < nb) idx[j] = ord[wv * nb * 32 + j * 32 + ln];
    }
    for (int r = item(); r < v; r += ST) {
      const int i = ord[r];
      int64_t at = i;
      const int b = cands ? 0 : slice_of(S, at);
      const int64_t pos = cands ? (int64_t)cb[3 * i + 2] : at;
      const El e = cands ? El{cb[3 * i], cb[3 * i + 1]}
                         : load_el(F, x, so[b][0] + pos * S.sx);
      emit(F, S, O, so[b], r - b * n, e, pos);
    }
    barrier();
  }
}

/* Radix passes */

/* Counts every digit of every key of a chunk of its slice into the
   slice's counts; the last chunk of the slice makes them each bin's first
   place and finds the passes the slice takes: those whose digit not every
   key shares, or the last pass where every one is. */
KERNEL SIZED(ST) void sort_hist(Words<NX_SORT_HIST_WORDS>) {
  static SHARED int64_t w[NX_SORT_HIST_WORDS];
  static SHARED uint32_t h[16 * 256];
  static SHARED int64_t o[3];
  stage_sort(w, NX_SORT_HIST_SLICES);
  const Form F = form_of(w + NX_SORT_HIST_KEY);
  const Slices S = slices_of(w + NX_SORT_HIST_SLICES);
  const uint8_t *x = address<const uint8_t>(w, NX_SORT_HIST_X);
  uint64_t *hist = address<uint64_t>(w, NX_SORT_HIST_HIST);
  uint32_t *meta = address<uint32_t>(w, NX_SORT_HIST_META);
  uint32_t *tickets = address<uint32_t>(w, NX_SORT_HIST_TICKETS);
  const int64_t hblocks = uniform(w[NX_SORT_HIST_HBLOCKS]);
  const uint64_t hblocks_div = (uint64_t)uniform(w[NX_SORT_HIST_HBLOCKS_DIV]);
  const int64_t groups = uniform(w[NX_SORT_HIST_GROUPS]);
  const int kb = F.kbytes;
  for (int64_t g = group_x(); g < S.count * hblocks; g += groups) {
    int64_t s = g;
    const int64_t c = split(s, hblocks, hblocks_div, S.small);
    for (int i = item(); i < kb * 256; i += ST) h[i] = 0;
    if (item() == 0) slice_at(S, s, o);
    barrier();
    const int64_t lo = c * NX_SORT_CHUNK, hi = imin(S.n, lo + NX_SORT_CHUNK);
    const int64_t ox = o[0];
    for (int64_t i = lo + item(); i < hi; i += ST) {
      const Key k = key_of(F, load_el(F, x, ox + i * S.sx));
      for (int p = 0; p < kb; p++) lds_add(&h[p * 256 + kbyte(k, p)], 1u);
    }
    barrier();
    uint64_t *hs = hist + s * kb * 256;
    for (int i = item(); i < kb * 256; i += ST)
      if (h[i]) a_add<uint64_t>(&hs[i], h[i]);
    if (!last_of(tickets + s, (uint32_t)hblocks)) continue;
    uint32_t mask = 0;
    for (int p = 0; p < kb; p++) {
      const uint64_t c = hs[p * 256 + item()];
      uint64_t all;
      hs[p * 256 + item()] = sum_excl<uint64_t>(c, all);
      if (!group_any(c == (uint64_t)S.n)) mask |= 1u << p;
    }
    if (item() == 0) meta[s] = mask ? mask : 1u << (kb - 1);
  }
}

/* An item: an element and its position in its slice. */
struct It {
  uint64_t lo, hi;
  int64_t pos;
};

/* Items as a pass's buffers hold them, slice s's at s n of [total]:
   8 bytes, an element of at most 32 bits above its 32-bit position; 12,
   an element of at most 64 bits, then apart its 32-bit position; 24, an
   element of at most 128 bits and its 64-bit position. */
template <int B> struct Items;

template <> struct Items<8> {
  typedef uint64_t T;
  static DEVICE_MEMBER T pack(It e) { return e.lo << 32 | (uint32_t)e.pos; }
  static DEVICE_MEMBER It unpack(T t) {
    return {t >> 32, 0, (int64_t)(t & 0xFFFFFFFFu)};
  }
  static DEVICE_MEMBER T get(const void *b, int64_t, int64_t i) {
    return ((const T *)b)[i];
  }
  static DEVICE_MEMBER void put(void *b, int64_t, int64_t i, T t) {
    ((T *)b)[i] = t;
  }
};

struct Two {
  uint64_t a, b;
};

template <> struct Items<12> {
  typedef Two T;
  static DEVICE_MEMBER T pack(It e) { return {e.lo, (uint64_t)e.pos}; }
  static DEVICE_MEMBER It unpack(T t) { return {t.a, 0, (int64_t)t.b}; }
  static DEVICE_MEMBER T get(const void *b, int64_t total, int64_t i) {
    const uint64_t *bits = (const uint64_t *)b;
    return {bits[i], ((const uint32_t *)(bits + total))[i]};
  }
  static DEVICE_MEMBER void put(void *b, int64_t total, int64_t i, T t) {
    uint64_t *bits = (uint64_t *)b;
    bits[i] = t.a;
    ((uint32_t *)(bits + total))[i] = (uint32_t)t.b;
  }
};

struct Three {
  uint64_t a, b, c;
};

template <> struct Items<24> {
  typedef Three T;
  static DEVICE_MEMBER T pack(It e) { return {e.lo, e.hi, (uint64_t)e.pos}; }
  static DEVICE_MEMBER It unpack(T t) { return {t.a, t.b, (int64_t)t.c}; }
  static DEVICE_MEMBER T get(const void *b, int64_t, int64_t i) {
    return ((const T *)b)[i];
  }
  static DEVICE_MEMBER void put(void *b, int64_t, int64_t i, T t) {
    ((T *)b)[i] = t;
  }
};

/* A pass's words and its slice's state. */
struct Pass {
  Form F;
  Slices S;
  const uint8_t *x;
  const int64_t *order;
  const uint64_t *hist;
  const uint32_t *meta;
  uint64_t *counts;
  void *buf0, *buf1;
  int flags, p;
  int64_t tiles, parts, span, groups;
  uint64_t parts_div;
};

DEVICE Pass pass_of(const int64_t *w) {
  Pass P;
  P.F = form_of(w + NX_SORT_PASS_KEY);
  P.S = slices_of(w + NX_SORT_PASS_SLICES);
  P.x = address<const uint8_t>(w, NX_SORT_PASS_X);
  P.order = address<const int64_t>(w, NX_SORT_PASS_ORDER);
  P.hist = address<const uint64_t>(w, NX_SORT_PASS_HIST);
  P.meta = address<const uint32_t>(w, NX_SORT_PASS_META);
  P.counts = address<uint64_t>(w, NX_SORT_PASS_COUNTS);
  P.buf0 = address<void>(w, NX_SORT_PASS_BUF0);
  P.buf1 = address<void>(w, NX_SORT_PASS_BUF1);
  P.flags = (int)uniform(w[NX_SORT_PASS_FLAGS]);
  P.p = (int)uniform(w[NX_SORT_PASS_PASS]);
  P.tiles = uniform(w[NX_SORT_PASS_TILES]);
  P.parts = uniform(w[NX_SORT_PASS_PARTS]);
  P.parts_div = (uint64_t)uniform(w[NX_SORT_PASS_PARTS_DIV]);
  P.span = uniform(w[NX_SORT_PASS_SPAN]);
  P.groups = uniform(w[NX_SORT_PASS_GROUPS]);
  return P;
}

/* Where slice [s]'s pass reads and writes its items: [q] passes of its
   [qs] before this one, x for the first, the last writing the values and
   positions. The passes alternate buffers so that the last reads buf1. */
struct Turn {
  bool on, first, final;
  void *src, *dst;
};

DEVICE Turn turn_of(const Pass &P, int64_t s) {
  Turn t = {false, false, false, nullptr, nullptr};
  const uint32_t mask = (uint32_t)uniform(P.meta[s]);
  if (!(mask >> P.p & 1)) return t;
  const int q = popc(mask & ((1u << P.p) - 1)), qs = popc(mask);
  t.on = true;
  t.first = q == 0;
  t.final = q == qs - 1;
  t.src = (qs - q) & 1 ? P.buf1 : P.buf0;
  t.dst = (qs - 1 - q) & 1 ? P.buf1 : P.buf0;
  return t;
}

/* Item [l] of the slice's sequence a first pass reads: x's element at the
   position [order] gives there, or at [l]. */
DEVICE It first_item(const Pass &P, int64_t ox, int64_t base, int64_t l) {
  const int64_t e = P.flags & NX_SORT_ORDER ? P.order[base + l] : l;
  const El x = load_el(P.F, P.x, ox + e * P.S.sx);
  return {x.lo, x.hi, e};
}

/* Slice [s]'s pass as radix_count and radix_move take it: a first pass
   reads x, staging a tile's elements in LDS in a loop that inlines
   load_el once; the last writes the values and positions it keeps; the
   others move items between buf0 and buf1. */
template <int B> struct Slice_pass {
  typedef Items<B> M;
  typedef typename M::T T;
  const Pass &P;
  const Turn Tn;
  const Form F;
  const int p;
  const Out &O;
  const int64_t *o;
  int64_t base, total;

  DEVICE_MEMBER int digit(T t) const {
    const It e = M::unpack(t);
    return kbyte(key_of(F, El{e.lo, e.hi}), p);
  }
  DEVICE_MEMBER bool fetch(int64_t l, T &t) const {
    t = Tn.first ? M::pack(first_item(P, o[0], base, l))
                 : M::get(Tn.src, total, base + l);
    return true;
  }
  DEVICE_MEMBER bool stage(int64_t l0, int valid, T *held) const {
    if (!Tn.first) return false;
    for (int l = item(); l < valid; l += ST)
      held[l] = M::pack(first_item(P, o[0], base, l0 + l));
    return true;
  }
  DEVICE_MEMBER void store(int64_t at, T t) const {
    if (!Tn.final) {
      M::put(Tn.dst, total, base + at, t);
      return;
    }
    const It e = M::unpack(t);
    emit(F, P.S, O, o, at, El{e.lo, e.hi}, e.pos);
  }
};

/* Counts the digits of each part's items into [counts]; the slice's last
   part makes them each part's first places in each bin, after the
   slice's earlier parts and from the bin's first place [hist]. */
template <int B, int N>
DEVICE void count_pass(void) {
  constexpr int TILE = ST * N;
  static SHARED int64_t w[NX_SORT_PASS_WORDS];
  static SHARED uint32_t h[256];
  static SHARED int64_t o[3];
  stage_sort(w, NX_SORT_PASS_SLICES, NX_SORT_PASS_VALUES,
             NX_SORT_PASS_TICKETS);
  const Pass P = pass_of(w);
  const Out O = {nullptr, nullptr, 0, false};
  uint32_t *tickets = address<uint32_t>(w, NX_SORT_PASS_TICKETS);
  const int64_t n = P.S.n, total = P.S.count * n;
  for (int64_t g = group_x(); g < P.S.count * P.parts; g += P.groups) {
    int64_t s = g;
    const int64_t part = split(s, P.parts, P.parts_div, P.S.small);
    const Turn T = turn_of(P, s);
    if (!T.on) continue;
    h[item()] = 0;
    if (item() == 0) slice_at(P.S, s, o);
    barrier();
    const Slice_pass<B> sp = {P, T, P.F, P.p, O, o, s * n, total};
    radix_count<ST>(sp, part * P.span * TILE,
                    imin(n, (part + 1) * P.span * TILE), h);
    barrier();
    P.counts[g * 256 + item()] = h[item()];
    if (!last_of(tickets + s, (uint32_t)P.parts)) continue;
    uint64_t run = P.hist[(s * P.F.kbytes + P.p) * 256 + item()];
    uint64_t *c = P.counts + s * P.parts * 256 + item();
    for (int64_t q = 0; q < P.parts; q++) {
      const uint64_t k = c[q * 256];
      c[q * 256] = run;
      run += k;
    }
  }
}

/* Moves each part's items to their places, from the part's first place
   in each bin: [hist]'s where the slice is one part, else [counts]'. */
template <int B, int N>
DEVICE void scatter_pass(void) {
  typedef typename Items<B>::T T;
  constexpr int TILE = ST * N;
  static SHARED int64_t w[NX_SORT_PASS_WORDS];
  static SHARED Rank<ST> R;
  static SHARED T items[TILE];
  static SHARED uint64_t gbase[256];
  static SHARED int64_t o[3];
  stage_sort(w, NX_SORT_PASS_SLICES, NX_SORT_PASS_TICKETS,
             NX_SORT_PASS_TICKETS + 1);
  const Pass P = pass_of(w);
  const Out O = {address<void>(w, NX_SORT_PASS_VALUES),
                 address<int64_t>(w, NX_SORT_PASS_POSITIONS),
                 uniform(w[NX_SORT_PASS_KEEP]),
                 (P.flags & NX_SORT_VALUES) != 0};
  const int64_t n = P.S.n, total = P.S.count * n;
  for (int64_t g = group_x(); g < P.S.count * P.parts; g += P.groups) {
    int64_t s = g;
    const int64_t part = split(s, P.parts, P.parts_div, P.S.small);
    const Turn Tn = turn_of(P, s);
    if (!Tn.on) continue;
    gbase[item()] = P.parts == 1
                        ? P.hist[(s * P.F.kbytes + P.p) * 256 + item()]
                        : P.counts[g * 256 + item()];
    if (item() == 0) slice_at(P.S, s, o);
    barrier();
    const Slice_pass<B> sp = {P, Tn, P.F, P.p, O, o, s * n, total};
    radix_move<ST, N>(sp, part * P.span, imin(P.tiles, (part + 1) * P.span), n,
                      gbase, R, items);
  }
}

KERNEL SIZED(ST) void sort_count8(Words<NX_SORT_PASS_WORDS>) {
  count_pass<8, NX_SORT_ITEMS8>();
}

KERNEL SIZED(ST) void sort_count12(Words<NX_SORT_PASS_WORDS>) {
  count_pass<12, NX_SORT_ITEMS12>();
}

KERNEL SIZED(ST) void sort_count24(Words<NX_SORT_PASS_WORDS>) {
  count_pass<24, NX_SORT_ITEMS24>();
}

KERNEL SIZED(ST) void sort_scatter8(Words<NX_SORT_PASS_WORDS>) {
  scatter_pass<8, NX_SORT_ITEMS8>();
}

KERNEL SIZED(ST) void sort_scatter12(Words<NX_SORT_PASS_WORDS>) {
  scatter_pass<12, NX_SORT_ITEMS12>();
}

KERNEL SIZED(ST) void sort_scatter24(Words<NX_SORT_PASS_WORDS>) {
  scatter_pass<24, NX_SORT_ITEMS24>();
}

/* Selects */

/* Round [pass] of each slice's select, unless done: counts that digit of
   the elements whose earlier digits match the prefix. The slice's last
   group finds the bin of the [need]th of them, which extends the prefix,
   and ends the select where the elements below or matching it are at
   most [cap], or no digit is left. */
KERNEL SIZED(ST) void sort_round(Words<NX_SORT_SELECT_WORDS>) {
  static SHARED int64_t w[NX_SORT_SELECT_WORDS];
  static SHARED uint32_t h[256];
  static SHARED uint8_t prefix[32];
  static SHARED int64_t o[3];
  stage_sort(w, NX_SORT_SELECT_SLICES, NX_SORT_SELECT_CAND,
             NX_SORT_SELECT_CAND + 1);
  const Form F = form_of(w + NX_SORT_SELECT_KEY);
  const Slices S = slices_of(w + NX_SORT_SELECT_SLICES);
  const uint8_t *x = address<const uint8_t>(w, NX_SORT_SELECT_X);
  sort_select *sel = address<sort_select>(w, NX_SORT_SELECT_SEL);
  uint32_t *tickets = address<uint32_t>(w, NX_SORT_SELECT_TICKETS);
  const int64_t keep = uniform(w[NX_SORT_SELECT_KEEP]);
  const int round = (int)uniform(w[NX_SORT_SELECT_PASS]);
  const int64_t groups = uniform(w[NX_SORT_SELECT_GROUPS]);
  const int64_t hblocks = uniform(w[NX_SORT_SELECT_HBLOCKS]);
  const uint64_t hblocks_div =
      (uint64_t)uniform(w[NX_SORT_SELECT_HBLOCKS_DIV]);
  const int64_t cap = uniform(w[NX_SORT_SELECT_CAP]);
  const int pbytes = (int)uniform(w[NX_SORT_SELECT_PBYTES]);
  for (int64_t g = group_x(); g < S.count * hblocks; g += groups) {
    int64_t s = g;
    const int64_t c = split(s, hblocks, hblocks_div, S.small);
    sort_select *st = sel + s;
    if (uniform(st->done)) continue;
    h[item()] = 0;
    if (item() < 32) prefix[item()] = st->prefix[item()];
    if (item() == 0) slice_at(S, s, o);
    barrier();
    const int64_t lo = c * NX_SORT_CHUNK, hi = imin(S.n, lo + NX_SORT_CHUNK);
    const int64_t ox = o[0];
    for (int64_t i = lo + item(); i < hi; i += ST) {
      const Key k = key_of(F, load_el(F, x, ox + i * S.sx));
      if (against(F, pbytes, k, i, prefix, round) == 0)
        lds_add(&h[cdigit(F, pbytes, k, i, round)], 1u);
    }
    barrier();
    if (h[item()]) a_add<uint64_t>(&st->hist[item()], h[item()]);
    if (!last_of(tickets + s, (uint32_t)hblocks)) continue;
    const uint64_t need = round == 0 ? (uint64_t)keep : st->need;
    const uint64_t x_ = st->hist[item()];
    uint64_t all;
    const uint64_t e = sum_excl<uint64_t>(x_, all);
    st->hist[item()] = 0;
    if (e < need && need <= e + x_) {
      const uint64_t left = need - e, below = (uint64_t)keep - left;
      st->prefix[round] = (uint8_t)item();
      st->need = left;
      st->rounds = round + 1;
      st->done =
          below + x_ <= (uint64_t)cap || round + 1 == F.kbytes + pbytes;
    }
  }
}

/* Gathers each slice's elements below or matching its select's prefix
   into its candidates. Their order there depends on the order groups run
   in; sort_block orders them by (key, position), which every candidate
   holds a distinct one of. */
KERNEL SIZED(ST) void sort_gather(Words<NX_SORT_SELECT_WORDS>) {
  static SHARED int64_t w[NX_SORT_SELECT_WORDS];
  static SHARED uint8_t prefix[32];
  static SHARED int64_t o[3];
  stage_sort(w, NX_SORT_SELECT_SLICES, NX_SORT_SELECT_TICKETS,
             NX_SORT_SELECT_CAND);
  const Form F = form_of(w + NX_SORT_SELECT_KEY);
  const Slices S = slices_of(w + NX_SORT_SELECT_SLICES);
  const uint8_t *x = address<const uint8_t>(w, NX_SORT_SELECT_X);
  sort_select *sel = address<sort_select>(w, NX_SORT_SELECT_SEL);
  uint64_t *cand = address<uint64_t>(w, NX_SORT_SELECT_CAND);
  const int64_t groups = uniform(w[NX_SORT_SELECT_GROUPS]);
  const int64_t hblocks = uniform(w[NX_SORT_SELECT_HBLOCKS]);
  const uint64_t hblocks_div =
      (uint64_t)uniform(w[NX_SORT_SELECT_HBLOCKS_DIV]);
  const int64_t cap = uniform(w[NX_SORT_SELECT_CAP]);
  const int pbytes = (int)uniform(w[NX_SORT_SELECT_PBYTES]);
  for (int64_t g = group_x(); g < S.count * hblocks; g += groups) {
    int64_t s = g;
    const int64_t c = split(s, hblocks, hblocks_div, S.small);
    sort_select *st = sel + s;
    if (item() < 32) prefix[item()] = st->prefix[item()];
    if (item() == 0) slice_at(S, s, o);
    barrier();
    const int rounds = (int)uniform(st->rounds);
    uint64_t *cb = cand + s * cap * 3;
    const int64_t lo = c * NX_SORT_CHUNK, hi = imin(S.n, lo + NX_SORT_CHUNK);
    const int64_t ox = o[0];
    for (int64_t i = lo + item(); i < hi; i += ST) {
      const El e = load_el(F, x, ox + i * S.sx);
      if (against(F, pbytes, key_of(F, e), i, prefix, rounds) > 0) continue;
      const uint64_t at = a_add<uint64_t>(&st->cnt, 1ull);
      cb[3 * at] = e.lo;
      cb[3 * at + 1] = e.hi;
      cb[3 * at + 2] = (uint64_t)i;
    }
    barrier();
  }
}

/* Numbering */

#define UNIQUE_ITEMS (NX_UNIQUE_TILE / ST)

/* A numbering's words. */
struct Numbering {
  const uint64_t *keys, *skeys;
  const int64_t *order;
  int64_t n, w, s0, s1, span, wspan, groups;
  int flags;
};

DEVICE Numbering numbering_of(const int64_t *w) {
  Numbering U;
  U.keys = address<const uint64_t>(w, NX_UNIQUE_KEYS);
  U.skeys = address<const uint64_t>(w, NX_UNIQUE_SKEYS);
  U.order = address<const int64_t>(w, NX_UNIQUE_ORDER);
  U.n = uniform(w[NX_UNIQUE_N]);
  U.w = uniform(w[NX_UNIQUE_W]);
  U.s0 = uniform(w[NX_UNIQUE_S0]);
  U.s1 = uniform(w[NX_UNIQUE_S1]);
  U.span = uniform(w[NX_UNIQUE_SPAN]);
  U.wspan = uniform(w[NX_UNIQUE_WSPAN]);
  U.groups = uniform(w[NX_UNIQUE_GROUPS]);
  U.flags = (int)uniform(w[NX_UNIQUE_FLAGS]);
  return U;
}

/* Whether sorted rows [i - 1] and [i] differ. */
DEVICE bool differs(const Numbering &U, int64_t i) {
  if (U.flags & NX_UNIQUE_BY_SKEYS) return U.skeys[i] != U.skeys[i - 1];
  const bool ordered = U.flags & NX_UNIQUE_BY_ORDER;
  const int64_t a = ordered ? U.order[i] : i;
  const int64_t b = ordered ? U.order[i - 1] : i - 1;
  for (int64_t j = 0; j < U.w; j++)
    if (U.keys[a * U.s0 + j * U.s1] != U.keys[b * U.s0 + j * U.s1])
      return true;
  return false;
}

DEVICE bool head(const Numbering &U, int64_t i) {
  return i == 0 || differs(U, i);
}

/* Replaces the [count] words at [c] by their exclusive sum, and is their
   sum. Every work-item calls it. */
DEVICE uint64_t scan_words(uint64_t *c, int64_t count) {
  uint64_t run = 0;
  for (int64_t q0 = 0; q0 < count; q0 += ST) {
    const int64_t q = q0 + item();
    const uint64_t v = q < count ? c[q] : 0;
    uint64_t all;
    const uint64_t e = sum_excl<uint64_t>(v, all);
    if (q < count) c[q] = run + e;
    run += all;
  }
  return run;
}

/* Counts the keys that start in each part of the sorted rows into
   [heads]; the last part makes them each part's first key, and stores the
   keys counted and the end of the last key's sorted rows. */
KERNEL SIZED(ST) void unique_count(Words<NX_UNIQUE_WORDS>) {
  static SHARED int64_t w[NX_UNIQUE_WORDS];
  stage(w, NX_UNIQUE_WORDS, ST, NX_UNIQUE_IDS, NX_UNIQUE_M + 1);
  const Numbering U = numbering_of(w);
  uint64_t *heads = address<uint64_t>(w, NX_UNIQUE_HEADS);
  uint32_t *ticket = address<uint32_t>(w, NX_UNIQUE_TICKETS);
  const int64_t parts = ceil_div(U.n, U.span);
  for (int64_t g = group_x(); g < parts; g += U.groups) {
    const int64_t hi = imin(U.n, (g + 1) * U.span);
    uint64_t c = 0;
    for (int64_t i = g * U.span + item(); i < hi; i += ST) c += head(U, i);
    uint64_t all;
    sum_excl<uint64_t>(c, all);
    if (item() == 0) heads[g] = all;
    if (!last_of(ticket, (uint32_t)parts)) continue;
    const uint64_t k = scan_words(heads, parts);
    if (item() == 0) {
      *address<int64_t>(w, NX_UNIQUE_K) = (int64_t)k;
      address<int64_t>(w, NX_UNIQUE_GSTART)[k] = U.n;
    }
  }
}

/* Each sorted row's key's number in sorted order, gid, from its part's
   first key; each key's first row, gfirst (the stable sort puts it
   first), and first sorted row, gstart; a bit at each first row, counted
   into [set] for its part of the bit words. The last part makes those
   counts each part's first bit. */
KERNEL SIZED(ST) void unique_heads(Words<NX_UNIQUE_WORDS>) {
  static SHARED int64_t w[NX_UNIQUE_WORDS];
  stage(w, NX_UNIQUE_WORDS, ST, NX_UNIQUE_IDS, NX_UNIQUE_M + 1);
  const Numbering U = numbering_of(w);
  const uint64_t *heads = address<const uint64_t>(w, NX_UNIQUE_HEADS);
  uint32_t *ticket = address<uint32_t>(w, NX_UNIQUE_TICKETS);
  int64_t *gid = address<int64_t>(w, NX_UNIQUE_GID);
  int64_t *gfirst = address<int64_t>(w, NX_UNIQUE_GFIRST);
  int64_t *gstart = address<int64_t>(w, NX_UNIQUE_GSTART);
  uint64_t *bits = address<uint64_t>(w, NX_UNIQUE_BITS);
  uint64_t *set = address<uint64_t>(w, NX_UNIQUE_SET);
  const bool ordered = U.flags & NX_UNIQUE_BY_ORDER;
  const int64_t parts = ceil_div(U.n, U.span);
  for (int64_t g = group_x(); g < parts; g += U.groups) {
    const int64_t hi = imin(U.n, (g + 1) * U.span);
    uint64_t before = heads[g];
    for (int64_t c0 = g * U.span; c0 < hi; c0 += NX_UNIQUE_TILE) {
      const int64_t i0 = c0 + item() * UNIQUE_ITEMS;
      uint32_t hs = 0;
      for (int j = 0; j < UNIQUE_ITEMS; j++) {
        const int64_t i = i0 + j;
        if (i < hi && head(U, i)) hs |= 1u << j;
      }
      uint64_t all;
      int64_t k = (int64_t)(before + sum_excl<uint64_t>(popc(hs), all)) - 1;
      for (int j = 0; j < UNIQUE_ITEMS; j++) {
        const int64_t i = i0 + j;
        if (i >= hi) break;
        if (hs >> j & 1) {
          const int64_t f = ordered ? U.order[i] : i;
          k++;
          gfirst[k] = f;
          gstart[k] = i;
          a_or<uint64_t>(&bits[f >> 6], 1ull << (f & 63));
          a_add<uint64_t>(&set[(f >> 6) / U.wspan], 1ull);
        }
        gid[i] = k;
      }
      before += all;
    }
    if (!last_of(ticket, (uint32_t)parts)) continue;
    scan_words(set, ceil_div(ceil_div(U.n, 64), U.wspan));
  }
}

/* The first rows set before each bit word: each part's from its first
   bit, by a scan. */
KERNEL SIZED(ST) void unique_rank(Words<NX_UNIQUE_WORDS>) {
  static SHARED int64_t w[NX_UNIQUE_WORDS];
  stage(w, NX_UNIQUE_WORDS, ST, NX_UNIQUE_IDS, NX_UNIQUE_BITS);
  const uint64_t *bits = address<const uint64_t>(w, NX_UNIQUE_BITS);
  uint64_t *below = address<uint64_t>(w, NX_UNIQUE_BELOW);
  const uint64_t *set = address<const uint64_t>(w, NX_UNIQUE_SET);
  const int64_t n = uniform(w[NX_UNIQUE_N]);
  const int64_t groups = uniform(w[NX_UNIQUE_GROUPS]);
  const int64_t wspan = uniform(w[NX_UNIQUE_WSPAN]);
  const int64_t words = ceil_div(n, 64), parts = ceil_div(words, wspan);
  for (int64_t g = group_x(); g < parts; g += groups) {
    const int64_t hi = imin(words, (g + 1) * wspan);
    uint64_t before = set[g];
    for (int64_t c0 = g * wspan; c0 < hi; c0 += NX_UNIQUE_TILE) {
      const int64_t i0 = c0 + item() * UNIQUE_ITEMS;
      uint64_t c = 0;
      for (int j = 0; j < UNIQUE_ITEMS; j++)
        if (i0 + j < hi) c += popc64(bits[i0 + j]);
      uint64_t all;
      uint64_t at = before + sum_excl<uint64_t>(c, all);
      for (int j = 0; j < UNIQUE_ITEMS && i0 + j < hi; j++) {
        below[i0 + j] = at;
        at += popc64(bits[i0 + j]);
      }
      before += all;
    }
  }
}

/* Each row's id, the rank of its key's first row among the first rows;
   each key's first row and count, and the entries past the keys. */
KERNEL SIZED(ST) void unique_ids(Words<NX_UNIQUE_WORDS>) {
  static SHARED int64_t w[NX_UNIQUE_WORDS];
  stage(w, NX_UNIQUE_WORDS, ST, NX_UNIQUE_KEYS, NX_UNIQUE_ORDER);
  int64_t *ids = address<int64_t>(w, NX_UNIQUE_IDS);
  int64_t *first = address<int64_t>(w, NX_UNIQUE_FIRST);
  int64_t *counts = address<int64_t>(w, NX_UNIQUE_COUNTS);
  const int64_t *order = address<const int64_t>(w, NX_UNIQUE_ORDER);
  const int64_t *gid = address<const int64_t>(w, NX_UNIQUE_GID);
  const int64_t *gfirst = address<const int64_t>(w, NX_UNIQUE_GFIRST);
  const int64_t *gstart = address<const int64_t>(w, NX_UNIQUE_GSTART);
  const uint64_t *bits = address<const uint64_t>(w, NX_UNIQUE_BITS);
  const uint64_t *below = address<const uint64_t>(w, NX_UNIQUE_BELOW);
  const int64_t n = uniform(w[NX_UNIQUE_N]), m = uniform(w[NX_UNIQUE_M]);
  const int64_t groups = uniform(w[NX_UNIQUE_GROUPS]);
  const bool ordered = uniform(w[NX_UNIQUE_FLAGS]) & NX_UNIQUE_BY_ORDER;
  const int64_t k =
      n == 0 ? 0 : uniform(*address<const int64_t>(w, NX_UNIQUE_K));
  const int64_t stride = groups * ST;
  for (int64_t i = (int64_t)group_x() * ST + item(); i < n; i += stride) {
    const int64_t g = gid[i], f = gfirst[g];
    const uint64_t under = bits[f >> 6] & ((1ull << (f & 63)) - 1);
    const int64_t num = (int64_t)(below[f >> 6] + popc64(under));
    ids[ordered ? order[i] : i] = num;
    if (gstart[g] == i && num < m) {
      first[num] = f;
      counts[num] = gstart[g + 1] - i;
    }
  }
  for (int64_t j = k + (int64_t)group_x() * ST + item(); j < m; j += stride) {
    first[j] = n;
    counts[j] = 0;
  }
  if (group_x() == 0 && item() == 0) *address<int64_t>(w, NX_UNIQUE_COUNT) = k;
}
