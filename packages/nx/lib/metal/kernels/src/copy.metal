/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Copies: a source's elements, bits for bits, into consecutive elements of
   a destination; widenings, which cast a source's elements narrower than
   a byte into consecutive bytes; and the zeroing of fresh tickets.

   A copy's thread copies one element, the one whose number is its index
   in the grid past the launch's first, and finds its source position
   from the extents and strides of the copy's axes, which its parameters
   hold up to NX_COPY_HOLDS axes and a constant buffer holds past them.
   Its position is 64-bit, so a copy reaches any element of any layout.
   Where a launch's indices fit 32 bits, its coordinates come from 32-bit
   divisions by the host's divisors, in a kernel of its unit (copy_8 to
   copy_64), whose code holds no 64-bit division and no other unit's
   copy. Past them, copy takes them from 64-bit divisions, which the GPU
   emulates at thousands of cycles, and its unit from the parameters.

   A source whose elements are consecutive is a run of bytes, which
   copy_run copies 16 bytes a thread: a run's copy needs no position, the
   work of a thread an element. A thread writes one aligned block of 16
   bytes of the destination. Its source bytes lie at one shift from a
   multiple of 16 bytes in every block, and the thread reads the two
   aligned words that hold them and joins them: a vector load from an
   address off its width would drop the address's low bits.

   A source of rows of consecutive bytes, each of 16 bytes or more, is
   copied 16 bytes a thread: a block's bytes come from at most two rows,
   each read as copy_run reads a run, by its own shift. copy_rows finds a
   block's rows by 32-bit divisions where the launch's byte indices fit
   32 bits, and copy_rows_wide by 64-bit ones past them. */

#include "device.h"
#include "nx_dtype.h"
#include "nx_kinds.h"
#include "values.h"

struct copy_words {
  device uchar *dst;
  device const uchar *src;
  constant long *axes;
  ulong first;
  ulong count;
  ulong rank;
  ulong unit;
  ulong units;
  long held[NX_AXIS_WORDS * NX_COPY_HOLDS];
};

NX_FIELD(copy_words, dst, NX_COPY_DST);
NX_FIELD(copy_words, src, NX_COPY_SRC);
NX_FIELD(copy_words, axes, NX_COPY_AXES);
NX_FIELD(copy_words, first, NX_COPY_FIRST);
NX_FIELD(copy_words, count, NX_COPY_COUNT);
NX_FIELD(copy_words, rank, NX_COPY_RANK);
NX_FIELD(copy_words, unit, NX_COPY_UNIT);
NX_FIELD(copy_words, units, NX_COPY_UNITS);
NX_FIELD(copy_words, held, NX_COPY_HELD);
NX_SIZE(copy_words, NX_COPY_WORDS);

/* The source's position of index [k], in elements: its coordinates from
   the innermost axis, each times its stride. The outermost coordinate is
   what remains, since [k] is below the product of the extents. Where the
   launch's indices fit 32 bits, so do [k] and each quotient: a
   coordinate takes a 32-bit divide by its extent's divisor, or is the
   whole rest where the extent passes 32 bits (divisor 0). */
static inline long copy_position32(constant copy_words &w, uint k) {
  constant long *axes = w.rank <= NX_COPY_HOLDS ? w.held : w.axes;
  long at = 0;
  uint rest = k;
  for (int j = int(w.rank) - 1; j >= 1; j--) {
    constant long *axis = axes + NX_AXIS_WORDS * j;
    const ulong md = ulong(axis[NX_AXIS_DIVISOR]);
    const uint q = md == 0 ? 0 : divide(rest, md);
    const uint i = rest - q * uint(axis[NX_AXIS_EXTENT]);
    at += long(i) * axis[NX_AXIS_STRIDE];
    rest = q;
  }
  return at + long(rest) * axes[NX_AXIS_STRIDE];
}

/* copy_position32's position past 32-bit indices: a 64-bit [%] and [/]
   an axis. */
static inline long copy_position64(constant copy_words &w, ulong k) {
  constant long *axes = w.rank <= NX_COPY_HOLDS ? w.held : w.axes;
  long at = 0;
  ulong rest = k;
  for (int j = int(w.rank) - 1; j >= 1; j--) {
    constant long *axis = axes + NX_AXIS_WORDS * j;
    const ulong e = ulong(axis[NX_AXIS_EXTENT]);
    at += long(rest % e) * axis[NX_AXIS_STRIDE];
    rest /= e;
  }
  return at + long(rest) * axes[NX_AXIS_STRIDE];
}

/* The position of index [k] of a launch of [w], by the launch's
   indices' width. */
static inline long copy_position(constant copy_words &w, ulong k) {
  return w.count <= 0x100000000ul ? copy_position32(w, uint(k))
                                  : copy_position64(w, k);
}

/* Copies the [units] units of type [T] of element [at] of [src] into
   element [k] of [dst]. */
template <typename T>
static inline void copy_element(device uchar *dst, device const uchar *src,
                                ulong k, long at, uint units) {
  device T *d = (device T *)dst + k * units;
  device const T *s = (device const T *)src + at * long(units);
  for (uint u = 0; u < units; u++) d[u] = s[u];
}

NX_KERNEL(NX_COPY_THREADS)
void copy(constant copy_words &w [[buffer(0)]],
          uint g [[threadgroup_position_in_grid]],
          uint t [[thread_index_in_threadgroup]]) {
  const ulong k = w.first + nx_thread(g, t, NX_COPY_THREADS);
  if (k >= w.count) return;

  const long at = copy_position64(w, k);
  const uint units = uint(w.units);
  switch (w.unit) {
  case 1: copy_element<uchar>(w.dst, w.src, k, at, units); break;
  case 2: copy_element<ushort>(w.dst, w.src, k, at, units); break;
  case 4: copy_element<uint>(w.dst, w.src, k, at, units); break;
  default: copy_element<ulong>(w.dst, w.src, k, at, units); break;
  }
}

/* A copy of 32-bit indices whose unit is [T]. */
#define NX_COPY_KERNEL(name, T)                                                \
  NX_KERNEL(NX_COPY_THREADS)                                                   \
  void name(constant copy_words &w [[buffer(0)]],                              \
            uint g [[threadgroup_position_in_grid]],                           \
            uint t [[thread_index_in_threadgroup]]) {                          \
    const uint k = uint(w.first) + g * NX_COPY_THREADS + t;                    \
    if (k >= w.count) return;                                                  \
    copy_element<T>(w.dst, w.src, k, copy_position32(w, k), uint(w.units));    \
  }

NX_COPY_KERNEL(copy_8, uchar)
NX_COPY_KERNEL(copy_16, ushort)
NX_COPY_KERNEL(copy_32, uint)
NX_COPY_KERNEL(copy_64, ulong)

struct copy_run_words {
  device uchar *dst;
  device const uchar *src;
  ulong first;
  ulong end;
  ulong head;
  ulong shift;
};

NX_FIELD(copy_run_words, dst, NX_COPY_RUN_DST);
NX_FIELD(copy_run_words, src, NX_COPY_RUN_SRC);
NX_FIELD(copy_run_words, first, NX_COPY_RUN_FIRST);
NX_FIELD(copy_run_words, end, NX_COPY_RUN_END);
NX_FIELD(copy_run_words, head, NX_COPY_RUN_HEAD);
NX_FIELD(copy_run_words, shift, NX_COPY_RUN_SHIFT);
NX_SIZE(copy_run_words, NX_COPY_RUN_WORDS);

/* The bytes [s] to [s] + 15 of the 32 bytes [lo] then [hi], for [s] below
   16: whole words by [s] / 4, then a shift by the bytes left. */
static inline uint4 copy_join(uint4 lo, uint4 hi, uint s) {
  uint4 a, b;
  switch (s >> 2) {
  case 0: a = lo; b = hi; break;
  case 1: a = uint4(lo.yzw, hi.x); b = uint4(hi.yzw, 0); break;
  case 2: a = uint4(lo.zw, hi.xy); b = uint4(hi.zw, 0, 0); break;
  default: a = uint4(lo.w, hi.xyz); b = uint4(hi.w, 0, 0, 0); break;
  }
  const uint bits = 8 * (s & 3);
  if (bits == 0) return a;
  return (a >> bits) | (uint4(a.yzw, b.x) << (32 - bits));
}

NX_KERNEL(NX_COPY_RUN_THREADS)
void copy_run(constant copy_run_words &w [[buffer(0)]],
              uint g [[threadgroup_position_in_grid]],
              uint t [[thread_index_in_threadgroup]]) {
  /* The run's bytes [at] to [at] + 15 fill one aligned block of [dst]. */
  const ulong block = w.first + nx_thread(g, t, NX_COPY_RUN_THREADS);
  const long at = long(block * NX_COPY_RUN_BYTES) - long(w.head);
  const long end = long(w.end);
  if (at >= end) return;

  /* The aligned words from byte [at] - [shift] of the run: one where the
     source lies at the destination's place, two otherwise. A block they
     would take past either end of the run copies its bytes one at a
     time: at most two at each end. */
  const uint s = uint(w.shift);
  const long reach = s == 0 ? NX_COPY_RUN_BYTES : 2 * NX_COPY_RUN_BYTES - s;
  if (at < long(s) || at + reach > end) {
    const long stop = at + NX_COPY_RUN_BYTES;
    for (long i = at < 0 ? 0 : at; i < stop && i < end; i++)
      w.dst[i] = w.src[i];
    return;
  }

  device const uint4 *words = (device const uint4 *)(w.src + (at - s));
  const uint4 lo = words[0];
  const uint4 hi = s == 0 ? lo : words[1];
  *(device uint4 *)(w.dst + at) = copy_join(lo, hi, s);
}

struct copy_rows_words {
  device uchar *dst;
  device const uchar *src;
  constant long *axes;
  ulong first;
  ulong end;
  ulong head;
  ulong shift;
  ulong rank;
  long lo;
  long hi;
  long held[NX_AXIS_WORDS * NX_COPY_ROWS_HOLDS];
};

NX_FIELD(copy_rows_words, dst, NX_COPY_ROWS_DST);
NX_FIELD(copy_rows_words, src, NX_COPY_ROWS_SRC);
NX_FIELD(copy_rows_words, axes, NX_COPY_ROWS_AXES);
NX_FIELD(copy_rows_words, first, NX_COPY_ROWS_FIRST);
NX_FIELD(copy_rows_words, end, NX_COPY_ROWS_END);
NX_FIELD(copy_rows_words, head, NX_COPY_ROWS_HEAD);
NX_FIELD(copy_rows_words, shift, NX_COPY_ROWS_SHIFT);
NX_FIELD(copy_rows_words, rank, NX_COPY_ROWS_RANK);
NX_FIELD(copy_rows_words, lo, NX_COPY_ROWS_LO);
NX_FIELD(copy_rows_words, hi, NX_COPY_ROWS_HI);
NX_FIELD(copy_rows_words, held, NX_COPY_ROWS_HELD);
NX_SIZE(copy_rows_words, NX_COPY_ROWS_WORDS);

/* [n] over [axis]'s extent: by its divisor for a 32-bit [n], 0 where the
   extent passes 32 bits (divisor 0), and by a 64-bit division, which the
   GPU emulates, for a 64-bit [n]. */
static inline uint rows_over(uint n, constant long *axis) {
  const ulong md = ulong(axis[NX_AXIS_DIVISOR]);
  return md == 0 ? 0 : divide(n, md);
}

static inline ulong rows_over(ulong n, constant long *axis) {
  return n / ulong(axis[NX_AXIS_EXTENT]);
}

/* The byte place of row [q]'s first byte about [src], from its
   coordinates on the axes before the row, innermost first; and in [next]
   row [q] + 1's, from the same coordinates: a step along the innermost
   axis whose coordinate is not its last, less the steps back of the
   axes inside it. */
template <typename I>
static inline long rows_place(constant long *axes, int rank, I q,
                              thread long &next) {
  long at = 0, back = 0;
  bool found = false;
  I rest = q;
  for (int j = rank - 2; j >= 1; j--) {
    constant long *axis = axes + NX_AXIS_WORDS * j;
    const I p = rows_over(rest, axis);
    const I e = I(axis[NX_AXIS_EXTENT]);
    const I i = rest - p * e;
    const long stride = axis[NX_AXIS_STRIDE];
    at += long(i) * stride;
    if (!found) {
      if (i + 1 < e) {
        next = stride - back;
        found = true;
      } else
        back += long(e - 1) * stride;
    }
    rest = p;
  }
  at += long(rest) * axes[NX_AXIS_STRIDE];
  if (!found) next = axes[NX_AXIS_STRIDE] - back;
  next += at;
  return at;
}

/* Whether the aligned words that hold the 16 bytes from byte [a] about
   [src] lie in the source's span: one word where [a] lies on 16 bytes,
   two otherwise. */
static inline bool rows_within(constant copy_rows_words &w, long a) {
  const long s = long(uint(a + long(w.shift)) & (NX_COPY_RUN_BYTES - 1));
  const long n = s == 0 ? NX_COPY_RUN_BYTES : 2 * NX_COPY_RUN_BYTES;
  return a - s >= w.lo && a - s + n <= w.hi;
}

/* The 16 bytes from byte [a] about [src]: the aligned word that holds
   byte [a], and where [a] lies off 16 bytes the next, joined by words
   selected by index. A switch on the shift would diverge across a SIMD
   group, whose threads' rows lie at different shifts. */
static inline uint4 rows_run(constant copy_rows_words &w, long a) {
  const uint s = uint(a + long(w.shift)) & (NX_COPY_RUN_BYTES - 1);
  device const uint4 *words = (device const uint4 *)(w.src + (a - long(s)));
  const uint4 lo = words[0];
  if (s == 0) return lo;
  const uint4 hi = words[1];
  const uint v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  const uint q = s >> 2, bits = 8 * (s & 3);
  if (bits == 0) return uint4(v[q], v[q + 1], v[q + 2], v[q + 3]);
  uint4 r;
  for (uint k = 0; k < 4; k++)
    r[k] = (v[q + k] >> bits) | (v[q + k + 1] << (32 - bits));
  return r;
}

/* Copies the destination's bytes [from] to [to] - 1 one at a time, each
   from its row, stepping along the row and then to the next. */
template <typename I>
static inline void rows_bytes(constant copy_rows_words &w,
                              constant long *axes, int rank, long from,
                              long to) {
  constant long *row = axes + NX_AXIS_WORDS * (rank - 1);
  const I e = I(row[NX_AXIS_EXTENT]);
  I q = rows_over(I(from), row);
  I i = I(from) - q * e;
  long next;
  long at = rows_place(axes, rank, q, next);
  for (long k = from; k < to; k++) {
    w.dst[k] = w.src[at + long(i)];
    if (++i == e && k + 1 < to) {
      i = 0;
      at = rows_place(axes, rank, ++q, next);
    }
  }
}

/* A thread's block of copy_rows, its byte indices of type [I]. */
template <typename I>
static inline void copy_rows_block(constant copy_rows_words &w, I block) {
  /* The destination's bytes [at] to [at] + 15 fill one aligned block. */
  const long at = long(block) * NX_COPY_RUN_BYTES - long(w.head);
  const long end = long(w.end);
  if (at >= end) return;

  constant long *axes = w.rank <= NX_COPY_ROWS_HOLDS ? w.held : w.axes;
  const int rank = int(w.rank);
  if (at < 0 || at + NX_COPY_RUN_BYTES > end) {
    rows_bytes<I>(w, axes, rank, at < 0 ? 0 : at,
                  metal::min(at + NX_COPY_RUN_BYTES, end));
    return;
  }

  /* Row [q] holds the block's first [c] bytes, from its byte [i]; where
     [c] is under 16, row [q] + 1 the rest, read as the 16 bytes that end
     at its byte 16 - [c]. */
  constant long *row = axes + NX_AXIS_WORDS * (rank - 1);
  const I e = I(row[NX_AXIS_EXTENT]);
  const I q = rows_over(I(at), row);
  const I i = I(at) - q * e;
  const I c = e - i;
  long next;
  const long a = rows_place(axes, rank, q, next) + long(i);
  device uint4 *d = (device uint4 *)(w.dst + at);
  if (c >= NX_COPY_RUN_BYTES) {
    if (!rows_within(w, a)) {
      rows_bytes<I>(w, axes, rank, at, at + NX_COPY_RUN_BYTES);
      return;
    }
    *d = rows_run(w, a);
    return;
  }
  const long b = next - long(c);
  if (!rows_within(w, a) || !rows_within(w, b)) {
    rows_bytes<I>(w, axes, rank, at, at + NX_COPY_RUN_BYTES);
    return;
  }
  const uint4 x = rows_run(w, a), y = rows_run(w, b);
  uint4 r;
  for (uint k = 0; k < 4; k++) {
    const int n = int(c) - int(4 * k);
    const uint m =
        n >= 4 ? 0xffffffffu : n <= 0 ? 0u : 0xffffffffu >> (32 - 8 * n);
    r[k] = (x[k] & m) | (y[k] & ~m);
  }
  *d = r;
}

NX_KERNEL(NX_COPY_ROWS_THREADS)
void copy_rows(constant copy_rows_words &w [[buffer(0)]],
               uint g [[threadgroup_position_in_grid]],
               uint t [[thread_index_in_threadgroup]]) {
  copy_rows_block<uint>(w, uint(w.first) + g * NX_COPY_ROWS_THREADS + t);
}

NX_KERNEL(NX_COPY_ROWS_THREADS)
void copy_rows_wide(constant copy_rows_words &w [[buffer(0)]],
                    uint g [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]]) {
  copy_rows_block<ulong>(w, w.first + nx_thread(g, t, NX_COPY_ROWS_THREADS));
}

NX_KERNEL(NX_COPY_THREADS)
void widen(constant copy_words &w [[buffer(0)]],
           uint g [[threadgroup_position_in_grid]],
           uint t [[thread_index_in_threadgroup]]) {
  const ulong k = w.first + nx_thread(g, t, NX_COPY_THREADS);
  if (k >= w.count) return;

  const long at = copy_position(w, k);
  const int from = int(w.unit & 0xff), to = int(w.unit >> 8);
  const long b = long(bits_of(from));
  const u64 bit = u64(long(w.units) + b * at);
  const held h = read(w.src, bit / u64(b), uint(bit % u64(b)), from);
  write(w.dst, k, 0, to, cast(from, to, h));
}

struct zero_words {
  device uint *dst;
  ulong count;
};

NX_FIELD(zero_words, dst, NX_ZERO_DST);
NX_FIELD(zero_words, count, NX_ZERO_COUNT);
NX_SIZE(zero_words, NX_ZERO_WORDS);

NX_KERNEL(NX_ZERO_THREADS)
void zero(constant zero_words &w [[buffer(0)]],
          uint g [[threadgroup_position_in_grid]],
          uint t [[thread_index_in_threadgroup]]) {
  const ulong k = nx_thread(g, t, NX_ZERO_THREADS);
  if (k < w.count) w.dst[k] = 0;
}
