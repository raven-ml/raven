/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Gathers and assemblies: moves of units, a thread a unit, each copied,
   gathered or filled with a pattern.

   A move's thread finds its unit's coordinates along the move's axes from
   its index, by the host's 32-bit divisors below 2^32 units a launch and
   by nx_divide past them, then the unit's place in the destination, the
   source and the positions, each a sum of coordinates times strides. A
   unit is the widest of 16, 8, 4, 2 and 1 bytes that every address the
   move reaches lies on and that divides the bytes the innermost axis
   moves as one run: a row of a gather whose positions step 0 along it is
   one thread's 16 bytes, an element of a gather along the innermost axis
   one thread's element. A vector load lies on its own width, since Metal
   merges loads into 16-byte ones that drop low address bits otherwise.

   An element of 1 or 4 bits is a unit of its own, read from the byte that
   holds it and stored by store_bits, which changes the element's bits of
   its 32-bit word alone, so other threads' stores to the word's other
   elements, and reads of bytes another array holds in it, see their own
   bits. */

#include "device.h"
#include "values.h"

/* The places along a move's axes of a unit: its destination's and its
   source's, in units, and its position's, in elements. */
struct places {
  long dst, src, idx;
};

/* Adds coordinate [c] along the axis whose words are [a] to [p]. */
static inline void add(thread places &p, constant long *a, long c) {
  p.dst += c * a[NX_MOVE_AXIS_SD];
  p.src += c * a[NX_MOVE_AXIS_SS];
  p.idx += c * a[NX_MOVE_AXIS_SI];
}

/* The places of unit [k] along the [rank] axes [axes], its coordinates
   taken from the innermost axis, each as the quotient by the extent's
   divisor; the outermost's is what remains. */
static inline places places_of(constant long *axes, int rank, ulong k,
                               bool wide) {
  places p = {0, 0, 0};
  if (rank == 0) return p;
  if (!wide) {
    uint rest = uint(k);
    for (int j = rank - 1; j >= 1; j--) {
      constant long *a = axes + NX_MOVE_AXIS_WORDS * j;
      const uint q = divide(rest, ulong(a[NX_MOVE_AXIS_DIVISOR]));
      add(p, a, long(rest - q * uint(a[NX_MOVE_AXIS_EXTENT])));
      rest = q;
    }
    k = rest;
  } else {
    for (int j = rank - 1; j >= 1; j--) {
      constant long *a = axes + NX_MOVE_AXIS_WORDS * j;
      const ulong e = ulong(a[NX_MOVE_AXIS_EXTENT]);
      const ulong q = nx_divide(k, e, ulong(a[NX_MOVE_AXIS_DIVISOR]));
      add(p, a, long(k - q * e));
      k = q;
    }
  }
  add(p, axes, long(k));
  return p;
}

/* A field of [how] of [bits] bits at bit [at]. */
static inline uint field(ulong how, uint at, uint bits) {
  return uint(how >> at) & ((1u << bits) - 1);
}

/* Moves */

struct move_words {
  device uchar *dst;
  device const uchar *src;
  device const long *idx;
  constant long *axes;
  ulong first;
  ulong count;
  ulong how;
  long d;
  long sa;
  long from;
  long held[NX_MOVE_AXIS_WORDS * NX_MOVE_HOLDS];
};

NX_FIELD(move_words, dst, NX_MOVE_DST);
NX_FIELD(move_words, src, NX_MOVE_SRC);
NX_FIELD(move_words, idx, NX_MOVE_IDX);
NX_FIELD(move_words, axes, NX_MOVE_AXES);
NX_FIELD(move_words, first, NX_MOVE_FIRST);
NX_FIELD(move_words, count, NX_MOVE_COUNT);
NX_FIELD(move_words, how, NX_MOVE_HOW);
NX_FIELD(move_words, d, NX_MOVE_D);
NX_FIELD(move_words, sa, NX_MOVE_SA);
NX_FIELD(move_words, from, NX_MOVE_FROM);
NX_FIELD(move_words, held, NX_MOVE_HELD);
NX_SIZE(move_words, NX_MOVE_WORDS);

/* Stores the unit [s] of [src], or zero bits where [zero], as unit [d] of
   [dst]. */
template <typename T>
static inline void move_unit(device uchar *dst, device const uchar *src,
                             long d, long s, bool zero) {
  ((device T *)dst)[d] = zero ? T(0) : ((device const T *)src)[s];
}

/* Stores the bytes of [pattern] at unit [d]'s offset modulo 16 as unit [d]
   of [dst], a unit of 2^([unit] - 3) bytes, or an element of 2^[unit]
   bits from bit [bit] of [dst]'s word. */
static inline void fill_unit(device uchar *dst, long d, uint unit, uint bit,
                             uint4 pattern) {
  if (unit < 3) {
    const uint bits = 1u << unit;
    const long b = long(bit) + long(bits) * d;
    store_bits(dst + 4 * (b >> 5), u64(b & 31), bits,
               pattern.x & ((1u << bits) - 1));
    return;
  }
  const uint at = uint(ulong(d) << (unit - 3)) & 15;
  const uint word = pattern[at >> 2] >> (8 * (at & 3));
  switch (unit) {
  case 3: ((device uchar *)dst)[d] = uchar(word); return;
  case 4: ((device ushort *)dst)[d] = ushort(word); return;
  case 5: ((device uint *)dst)[d] = word; return;
  case 6: ((device uint2 *)dst)[d] = at == 0 ? pattern.xy : pattern.zw; return;
  default: ((device uint4 *)dst)[d] = pattern; return;
  }
}

/* A move's thread for unit [k]: a fill's where [FILL], a copy's or a
   gather's otherwise. A fill is a kernel of its own, so that no copy or
   gather carries its branch. */
template <bool FILL>
static inline void move_of(constant move_words &w, ulong k) {
  const ulong how = w.how;
  const int rank = int(field(how, NX_HOW_RANK, 8));
  constant long *axes = rank <= NX_MOVE_HOLDS ? w.held : w.axes;
  places p = places_of(axes, rank, k, field(how, NX_HOW_WIDE, 1) != 0);

  if (FILL) {
    const ulong lo = ulong(w.d), hi = ulong(w.sa);
    fill_unit(w.dst, p.dst, field(how, NX_HOW_UNIT, 5),
              field(how, NX_HOW_DST_BIT, 5),
              uint4(uint(lo), uint(lo >> 32), uint(hi), uint(hi >> 32)));
    return;
  }

  bool zero = false;
  if (field(how, 0, 2) == NX_MOVE_GATHER) {
    const long i = p.idx - w.from;
    const long q = field(how, NX_HOW_NARROW, 1) != 0
                       ? long(((device const int *)w.idx)[i])
                       : w.idx[i];
    zero = q < 0 || q >= w.d;
    p.src += zero ? 0 : q * w.sa;
  }

  const uint unit = field(how, NX_HOW_UNIT, 5);
  switch (unit) {
  case 0:
  case 2: {
    /* An element of 1 or 4 bits: positions count elements from the bit
       of their position 0 in its word. */
    const uint bits = 1u << unit;
    uint v = 0;
    if (!zero) {
      const long bit = long(field(how, NX_HOW_SRC_BIT, 5)) + long(bits) * p.src;
      v = uint(w.src[bit >> 3] >> uint(bit & 7)) & ((1u << bits) - 1);
    }
    /* A piece's region may step back from its first element. */
    const long bit = long(field(how, NX_HOW_DST_BIT, 5)) + long(bits) * p.dst;
    store_bits(w.dst + 4 * (bit >> 5), u64(bit & 31), bits, v);
    return;
  }
  case 3: move_unit<uchar>(w.dst, w.src, p.dst, p.src, zero); return;
  case 4: move_unit<ushort>(w.dst, w.src, p.dst, p.src, zero); return;
  case 5: move_unit<uint>(w.dst, w.src, p.dst, p.src, zero); return;
  case 6: move_unit<uint2>(w.dst, w.src, p.dst, p.src, zero); return;
  default: move_unit<uint4>(w.dst, w.src, p.dst, p.src, zero); return;
  }
}

NX_KERNEL(NX_MOVE_THREADS)
void move(constant move_words &w [[buffer(0)]],
          uint g [[threadgroup_position_in_grid]],
          uint t [[thread_index_in_threadgroup]]) {
  const ulong k = w.first + nx_thread(g, t, NX_MOVE_THREADS);
  if (k < w.count) move_of<false>(w, k);
}

NX_KERNEL(NX_MOVE_THREADS)
void move_fill(constant move_words &w [[buffer(0)]],
               uint g [[threadgroup_position_in_grid]],
               uint t [[thread_index_in_threadgroup]]) {
  const ulong k = w.first + nx_thread(g, t, NX_MOVE_THREADS);
  if (k < w.count) move_of<true>(w, k);
}

/* Runs */

struct runs_words {
  device long *dst;
  device const long *src;
  device long *top;
  constant long *axes;
  ulong first;
  ulong count;
  ulong how;
  ulong n;
  ulong chunks;
  ulong divisor;
  long stride;
  long held[NX_MOVE_AXIS_WORDS * NX_MOVE_HOLDS];
};

NX_FIELD(runs_words, dst, NX_RUNS_DST);
NX_FIELD(runs_words, src, NX_RUNS_SRC);
NX_FIELD(runs_words, top, NX_RUNS_TOP);
NX_FIELD(runs_words, axes, NX_RUNS_AXES);
NX_FIELD(runs_words, first, NX_RUNS_FIRST);
NX_FIELD(runs_words, count, NX_RUNS_COUNT);
NX_FIELD(runs_words, how, NX_RUNS_HOW);
NX_FIELD(runs_words, n, NX_RUNS_N);
NX_FIELD(runs_words, chunks, NX_RUNS_CHUNKS);
NX_FIELD(runs_words, divisor, NX_RUNS_DIVISOR);
NX_FIELD(runs_words, stride, NX_RUNS_STRIDE);
NX_FIELD(runs_words, held, NX_RUNS_HELD);
NX_SIZE(runs_words, NX_RUNS_WORDS);

/* The maximum of [a] and the value of lane [l - d], for lanes [d] or more
   past the first: a step of a SIMD-group's running maximum. */
static inline long up_max(long a, uint l, ushort d) {
  const uint2 h = metal::simd_shuffle_up(as_type<uint2>(a), d);
  return l >= d ? metal::max(a, as_type<long>(h)) : a;
}

/* A group's chunk of a row: each thread's 16 entries' running maximum,
   then the SIMD-group's running maximum of the threads', then the
   group's of the SIMD-groups', each step carried into the next. Entries
   past the row's end read as the least int64. */
NX_KERNEL(NX_RUNS_THREADS)
void runs(constant runs_words &w [[buffer(0)]],
          uint g [[threadgroup_position_in_grid]],
          uint t [[thread_index_in_threadgroup]],
          uint lane [[thread_index_in_simdgroup]],
          uint simd [[simdgroup_index_in_threadgroup]],
          threadgroup long *shared [[threadgroup(0)]]) {
  const ulong group = w.first + g;
  const uint row = divide(uint(group), w.divisor);
  const ulong chunk = group - ulong(row) * w.chunks;

  const ulong how = w.how;
  const int rank = int(field(how, NX_HOW_RANK, 8));
  constant long *axes = rank <= NX_MOVE_HOLDS ? w.held : w.axes;
  const bool first_level = field(how, NX_HOW_UNIT, 1) != 0;
  const places p =
      places_of(axes, rank, row, field(how, NX_HOW_WIDE, 1) != 0);
  device long *dst = w.dst + p.dst + (first_level ? 2 : 0);
  if (first_level && chunk == 0 && t == 0) {
    w.dst[p.dst] = p.idx;
    w.dst[p.dst + 1] = p.src;
  }

  /* The chunk's entries, of 2^c, and this thread's first. */
  const uint c = field(how, NX_HOW_CHUNK, 5);
  const ulong end = metal::min(w.n, (chunk + 1) << c);
  const ulong at = (chunk << c) + ulong(t) * NX_RUNS_EACH;
  long v[NX_RUNS_EACH];
  long m = LONG_MIN;
  for (int i = 0; i < NX_RUNS_EACH; i++) {
    const ulong j = at + ulong(i);
    if (j < end) m = metal::max(m, w.src[p.src + long(j) * w.stride]);
    v[i] = m;
  }

  /* The running maximum of the lanes' maxima, and of the SIMD-groups'. */
  long run = m;
  for (ushort d = 1; d < 32; d *= 2) run = up_max(run, lane, d);
  if (lane == 31) shared[simd] = run;
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  long before = LONG_MIN;
  for (uint s = 0; s < simd; s++) before = metal::max(before, shared[s]);
  const uint2 prior = metal::simd_shuffle_up(as_type<uint2>(run), ushort(1));
  if (lane > 0) before = metal::max(before, as_type<long>(prior));

  for (int i = 0; i < NX_RUNS_EACH; i++) {
    const ulong j = at + ulong(i);
    if (j < end) dst[j] = metal::max(before, v[i]);
  }
  if (t == NX_RUNS_THREADS - 1)
    w.top[ulong(row) * w.chunks + chunk] = metal::max(before, v[NX_RUNS_EACH - 1]);
}

NX_KERNEL(NX_RUNS_THREADS)
void runs_carry(constant runs_words &w [[buffer(0)]],
                uint g [[threadgroup_position_in_grid]],
                uint t [[thread_index_in_threadgroup]]) {
  const ulong e = w.first + nx_thread(g, t, NX_RUNS_THREADS);
  if (e >= w.count) return;
  const bool wide = field(w.how, NX_HOW_WIDE, 1) != 0;
  const ulong row =
      wide ? nx_divide(e, w.n, w.divisor) : ulong(divide(uint(e), w.divisor));
  const ulong j = e - row * w.n;
  const ulong chunk = j >> field(w.how, NX_HOW_CHUNK, 5);
  if (chunk == 0) return;
  const bool first_level = field(w.how, NX_HOW_UNIT, 1) != 0;
  device long *d =
      w.dst + long(row) * w.stride + (first_level ? 2 : 0) + long(j);
  *d = metal::max(*d, w.top[row * w.chunks + chunk - 1]);
}

/* Range gathers' positions */

struct positions_words {
  device uchar *dst;
  device const long *maxima;
  device const long *starts;
  device const long *ends;
  ulong first;
  ulong count;
  ulong how;
  ulong k;
  ulong size;
  ulong divisor;
  long ss;
  long es;
  long d;
  ulong from;
};

NX_FIELD(positions_words, dst, NX_POSITIONS_DST);
NX_FIELD(positions_words, maxima, NX_POSITIONS_MAXIMA);
NX_FIELD(positions_words, starts, NX_POSITIONS_STARTS);
NX_FIELD(positions_words, ends, NX_POSITIONS_ENDS);
NX_FIELD(positions_words, first, NX_POSITIONS_FIRST);
NX_FIELD(positions_words, count, NX_POSITIONS_COUNT);
NX_FIELD(positions_words, how, NX_POSITIONS_HOW);
NX_FIELD(positions_words, k, NX_POSITIONS_K);
NX_FIELD(positions_words, size, NX_POSITIONS_SIZE);
NX_FIELD(positions_words, divisor, NX_POSITIONS_DIVISOR);
NX_FIELD(positions_words, ss, NX_POSITIONS_SS);
NX_FIELD(positions_words, es, NX_POSITIONS_ES);
NX_FIELD(positions_words, d, NX_POSITIONS_D);
NX_FIELD(positions_words, from, NX_POSITIONS_FROM);
NX_SIZE(positions_words, NX_POSITIONS_WORDS);

NX_KERNEL(NX_RUNS_THREADS)
void runs_positions(constant positions_words &w [[buffer(0)]],
                    uint g [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]]) {
  const ulong e = w.first + nx_thread(g, t, NX_RUNS_THREADS);
  if (e >= w.count) return;
  const bool wide = field(w.how, NX_HOW_WIDE, 1) != 0;
  const ulong r =
      wide ? nx_divide(e, w.size, w.divisor) : ulong(divide(uint(e), w.divisor));
  const long p = long(e - r * w.size);

  const ulong k = w.k;
  device const long *row = w.maxima + r * (k + 2);
  device const long *m = row + 2;
  ulong lo = 0, hi = k;
  while (lo < hi) {
    const ulong mid = (lo + hi) / 2;
    if (m[mid] > p) hi = mid;
    else lo = mid + 1;
  }
  long q = -1;
  if (lo < k) {
    const long end = lo == 0 ? 0 : w.ends[row[1] + long(lo - 1) * w.es];
    const long start = w.starts[row[0] + long(lo) * w.ss];
    q = long(ulong(start) + ulong(p) - ulong(end));
    if (q < 0 || q >= w.d) q = -1;
  }
  const ulong at = e - w.from;
  if (field(w.how, NX_HOW_NARROW, 1) != 0) ((device int *)w.dst)[at] = int(q);
  else ((device long *)w.dst)[at] = q;
}

/* Folds */

struct fold_words {
  device uchar *dst;
  device const uchar *x;
  constant long *axes;
  ulong first;
  ulong count;
  ulong how;
  long t0;
  long t1;
  long windows;
  long seek;
};

NX_FIELD(fold_words, dst, NX_FOLD_DST);
NX_FIELD(fold_words, x, NX_FOLD_X);
NX_FIELD(fold_words, axes, NX_FOLD_AXES);
NX_FIELD(fold_words, first, NX_FOLD_FIRST);
NX_FIELD(fold_words, count, NX_FOLD_COUNT);
NX_FIELD(fold_words, how, NX_FOLD_HOW);
NX_FIELD(fold_words, t0, NX_FOLD_T0);
NX_FIELD(fold_words, t1, NX_FOLD_T1);
NX_FIELD(fold_words, windows, NX_FOLD_WINDOWS);
NX_FIELD(fold_words, seek, NX_FOLD_SEEK);
NX_SIZE(fold_words, NX_FOLD_WORDS);

/* The quotient of [v] >= 0 by word [at] of the axis record [a], whose
   divisor is word [div]. */
template <typename T>
static inline T over(constant long *a, T v, bool wide, int at, int div) {
  const T d = T(a[at]);
  const ulong m = ulong(a[div]);
  if (d == 1) return v;
  if (!wide) return T(divide(uint(v), m));
  return T(nx_divide(ulong(v), ulong(d), m));
}

/* The quotient of [v] >= 0 by the step of the axis record [a], or by its
   dilation where [dil]. */
template <typename T>
static inline T quotient(constant long *a, T v, bool wide, bool dil) {
  return dil ? over<T>(a, v, wide, NX_FOLD_AXIS_DIL, NX_FOLD_AXIS_DIL_DIVISOR)
             : over<T>(a, v, wide, NX_FOLD_AXIS_STEP, NX_FOLD_AXIS_STEP_DIVISOR);
}

/* The first tap at or after [j] of the window [a] that reads padded
   position [P], its window in [q]; the window's taps for none. A tap j
   reads P at window (P - j dil) / step where that lies in [0, nwin) and
   step divides P - j dil: j lies between (P - (nwin - 1) step) / dil
   and r = P / dil, and the taps that read P repeat every [apart]. Where
   dil divides step, they are those congruent to r modulo [apart], where
   dil divides P: the first at or after j is r less the multiples of
   [apart] that fit in r - j. Otherwise a scan of at most [apart] taps
   from the least finds it. */
template <typename T>
static inline T next_tap(constant long *a, T P, T j, thread T &q, bool wide) {
  const T taps = T(a[NX_FOLD_AXIS_TAPS]), dil = T(a[NX_FOLD_AXIS_DIL]);
  const T step = T(a[NX_FOLD_AXIS_STEP]), nwin = T(a[NX_FOLD_AXIS_NWIN]);
  if (P < 0 || nwin <= 0) return taps;
  const T top = (nwin - 1) * step;
  const T r = quotient<T>(a, P, wide, true);
  const T hi = metal::min(taps - 1, r);
  if (P > top) j = metal::max(j, quotient<T>(a, P - top + dil - 1, wide, true));
  if (a[NX_FOLD_AXIS_DIRECT] != 0) {
    if (j > hi || r * dil != P) return taps;
    const T apart = T(a[NX_FOLD_AXIS_APART]);
    const T t =
        r - apart * over<T>(a, r - j, wide, NX_FOLD_AXIS_APART,
                            NX_FOLD_AXIS_APART_DIVISOR);
    if (t > hi) return taps;
    q = quotient<T>(a, P - t * dil, wide, false);
    return t;
  }
  const T last = metal::min(hi, j + T(a[NX_FOLD_AXIS_APART]) - 1);
  for (; j <= last; j++) {
    const T v = P - j * dil;
    const T k = quotient<T>(a, v, wide, false);
    if (k * step == v) {
      q = k;
      return j;
    }
  }
  return taps;
}

/* [a] + [b] by Prog.Add at [dt]: a float's in float32, a narrow float's
   rounded once to it, each part of a complex64 apart, an integer's
   wrapping at its width. */
static inline held fold_add(int dt, held a, held b) {
  switch (dt) {
  case NX_FLOAT32:
    return held_of(as_type<uint>(as_type<float>(uint(a.lo)) +
                                 as_type<float>(uint(b.lo))),
                   0);
  case NX_COMPLEX64: {
    const float2 s = as_type<float2>(a.lo) + as_type<float2>(b.lo);
    return held_of(as_type<ulong>(s), 0);
  }
  case NX_FLOAT16:
  case NX_BFLOAT16:
  case NX_FLOAT8_E4M3FN:
  case NX_FLOAT8_E5M2:
  case NX_FLOAT4_E2M1FN:
    return held_of(narrow_code(dt, narrow_value(dt, uint(a.lo)) +
                                       narrow_value(dt, uint(b.lo))),
                   0);
  default: return held_of(of_raw(dt, a.lo + b.lo), 0);
  }
}

/* Sets [v]'s element [k] to [e]; [k] unrolled for few windows, so that
   [v] stays in registers. */
template <int W, typename T>
static inline void put(thread T *v, int k, T e) {
  if (W < NX_FOLD_MOST) {
    for (int u = 0; u < W; u++)
      if (u == k) v[u] = e;
  } else
    v[k] = e;
}

/* A fold of [W] odometer windows, or of [w.windows] up to NX_FOLD_MOST,
   its positions, taps and tap numbers in [T]: int below 2^31, which the
   host checks, long otherwise. */
template <int W, typename T>
static inline void fold_of(constant fold_words &w, ulong k) {
  const ulong how = w.how;
  /* 32-bit instances compile no 64-bit division. */
  const bool wide = sizeof(T) == 8 && field(how, NX_HOW_WIDE, 1) != 0;
  const int rank = int(field(how, NX_HOW_RANK, 8));
  const int dt = int(field(how, NX_HOW_DTYPE, 5));
  const int nw = W < NX_FOLD_MOST ? W : int(w.windows);
  constant long *axes = w.axes;
  constant long *wins = axes + NX_FOLD_AXIS_WORDS * rank;
  const int S = W == 0 ? 1 : W;

  /* The element's padded position on each axis: an odometer window's
     kept, every other axis's window read at once. */
  T P[S], tap[S], q[S];
  T base = 0;
  bool none = false;
  ulong rest = k;
  for (int a = rank - 1, u = nw - 1; a >= 0; a--) {
    constant long *ax = axes + NX_FOLD_AXIS_WORDS * a;
    T c;
    if (a == 0) c = T(rest);
    else if (!wide) {
      const uint r = uint(rest);
      const uint d = divide(r, ulong(ax[NX_FOLD_AXIS_DIVISOR]));
      c = T(r - d * uint(ax[NX_FOLD_AXIS_EXTENT]));
      rest = d;
    } else {
      const ulong e = ulong(ax[NX_FOLD_AXIS_EXTENT]);
      const ulong d = nx_divide(rest, e, ulong(ax[NX_FOLD_AXIS_DIVISOR]));
      c = T(rest - d * e);
      rest = d;
    }
    const T pp = T(ax[NX_FOLD_AXIS_LO]) + c * T(ax[NX_FOLD_AXIS_SPACE]);
    if (ax[NX_FOLD_AXIS_TAPS] > 1) {
      put<S, T>(P, u, pp);
      u--;
    } else if (pp < 0) none = true;
    else {
      const T r = quotient<T>(ax, pp, wide, false);
      if (r * T(ax[NX_FOLD_AXIS_STEP]) != pp || r >= T(ax[NX_FOLD_AXIS_NWIN]))
        none = true;
      else base += r * T(ax[NX_FOLD_AXIS_SW]);
    }
  }

  /* Each window's first tap, kept with its window for the odometer's
     carries. */
  T first[S], fq[S];
  T number = 0;
  for (int u = 0; u < nw && !none; u++) {
    constant long *x = wins + NX_FOLD_AXIS_WORDS * u;
    tap[u] = next_tap<T>(x, P[u], 0, q[u], wide);
    first[u] = tap[u];
    fq[u] = q[u];
    none = tap[u] == T(x[NX_FOLD_AXIS_TAPS]);
    number += tap[u] * T(x[NX_FOLD_AXIS_SPAN]);
  }
  const uint phase = field(how, NX_HOW_DST_BIT, 5);
  if (none) {
    write(w.dst, k, phase, dt, held_of(0, 0));
    return;
  }
  const T t0 = T(w.t0), t1 = T(w.t1);
  if (number >= t1) return;

  /* An element begun in an earlier range takes the smallest tap number
     at or past t0 that reads it. Of t0's digits d along the windows, it
     keeps the longest prefix that reads it; then takes the next reading
     tap on the window that ends the prefix, or failing that on the
     latest earlier window that has one; then each later window's first
     reading tap. */
  held acc = held_of(0, 0);
  bool fresh = number >= t0;
  if (!fresh) {
    acc = read(w.dst, k, phase, dt);
    constant long *d = axes + w.seek;
    T at[S];
    int pre = nw;
    for (int u = 0; u < nw && pre == nw; u++) {
      constant long *x = wins + NX_FOLD_AXIS_WORDS * u;
      at[u] = next_tap<T>(x, P[u], T(d[u]), q[u], wide);
      if (at[u] != T(d[u])) pre = u;
    }
    int from = nw;
    if (pre < nw) {
      from = -1;
      for (int u = pre; u >= 0; u--) {
        constant long *x = wins + NX_FOLD_AXIS_WORDS * u;
        const T j =
            u == pre ? at[u] : next_tap<T>(x, P[u], T(d[u]) + 1, q[u], wide);
        if (j < T(x[NX_FOLD_AXIS_TAPS])) {
          at[u] = j;
          from = u;
          break;
        }
      }
      if (from < 0) return;
    }
    number = 0;
    for (int u = 0; u < nw; u++) {
      tap[u] = u <= from ? at[u] : first[u];
      if (u > from) q[u] = fq[u];
      number += tap[u] * T(wins[NX_FOLD_AXIS_WORDS * u + NX_FOLD_AXIS_SPAN]);
    }
    if (number >= t1) return;
  }

  /* The odometer, the last window fastest. A window's next reading tap
     lies [apart] past the one it read, [drop] windows back: one exists
     while that tap is a tap and that window is not negative. */
  const uint xbit = field(how, NX_HOW_SRC_BIT, 5);
  while (true) {
    T p = base;
    for (int u = 0; u < nw; u++) {
      constant long *x = wins + NX_FOLD_AXIS_WORDS * u;
      p += q[u] * T(x[NX_FOLD_AXIS_SW]) + tap[u] * T(x[NX_FOLD_AXIS_ST]);
    }
    const held y = read(w.x, ulong(long(p)), xbit, dt);
    acc = fresh ? y : fold_add(dt, acc, y);
    fresh = false;
    bool carry = true;
    for (int u = nw - 1; u >= 0 && carry; u--) {
      constant long *x = wins + NX_FOLD_AXIS_WORDS * u;
      const T taps = T(x[NX_FOLD_AXIS_TAPS]), span = T(x[NX_FOLD_AXIS_SPAN]);
      const T j = tap[u] + T(x[NX_FOLD_AXIS_APART]);
      const T v = q[u] - T(x[NX_FOLD_AXIS_DROP]);
      number -= tap[u] * span;
      carry = j >= taps || v < 0;
      tap[u] = carry ? first[u] : j;
      q[u] = carry ? fq[u] : v;
      number += tap[u] * span;
    }
    if (carry || number >= t1) break;
  }
  write(w.dst, k, phase, dt, acc);
}

#define NX_FOLD_KERNEL(name, W, T)                                             \
  NX_KERNEL(NX_FOLD_THREADS)                                                   \
  void name(constant fold_words &w [[buffer(0)]],                              \
            uint g [[threadgroup_position_in_grid]],                           \
            uint t [[thread_index_in_threadgroup]]) {                          \
    const ulong k = w.first + nx_thread(g, t, NX_FOLD_THREADS);                \
    if (k < w.count) fold_of<W, T>(w, k);                                      \
  }

NX_FOLD_KERNEL(fold_0, 0, int)
NX_FOLD_KERNEL(fold_1, 1, int)
NX_FOLD_KERNEL(fold_2, 2, int)
NX_FOLD_KERNEL(fold_3, 3, int)
NX_FOLD_KERNEL(fold_n, NX_FOLD_MOST, int)
NX_FOLD_KERNEL(fold_w, NX_FOLD_MOST, long)
