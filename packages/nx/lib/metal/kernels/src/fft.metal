/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fourier transforms' moves (kernels.h, Fourier transforms): the transposes
   fft.ml plans, one function for each of what it reads, multiplies by and
   writes, and its 64-bit twin; and the split and the pre-pass of a long
   halved line, between its points and its bins.

   A group takes one tile of 32 rows by 32 columns of its transpose's grid
   through threadgroup memory, 32 rows of 33 points so that both a row's and
   a column's 32 points fall in distinct banks. Its 256 threads read 4 points
   each into the tile, lanes along the side's points or along its lines as
   the launch's mode says, so that a SIMD group's 32 loads touch adjacent
   elements on a rows side and adjacent lines on a columns one; then, after
   a barrier, write 4 each the same way. A twiddle's lanes lie along r at one
   c, so their high-table entries are few and adjacent.

   A point's position is its line's first point, from the line's index over
   the side's axes by their divisors, plus its index times the point stride:
   32-bit where every offset within a line fits, which the 64-bit twin does
   not assume. A move computes its tile's 32 lines' first points once a side
   into threadgroup memory past the tile. Every
   access is a float's, 4-byte aligned, since a complex64 array lies on 4
   bytes. A complex product is two products and two fused multiply-adds, the
   steps' own.

   Every table is the forward transform's. An inverse launch runs the
   forward operation with NX_FFT_CONJ, which conjugates each point it loads
   and each it stores; the launches compose, since conjugating twice is
   exact, so an inverse's bits are the conjugate of the forward computation
   of its conjugated operand. */

#include "fft.h"

struct fft_move_words {
  device uchar *src;
  device uchar *dst;
  constant float2 *lo;
  constant float2 *hi;
  constant long *tail;
  ulong first;
  ulong rows_divisor;
  ulong tiles_divisor;
  ulong a;
  ulong b;
  ulong n;
  ulong mode;
  fft_side_words in;
  fft_side_words out;
};

NX_FIELD(fft_move_words, src, NX_FFT_MOVE_SRC);
NX_FIELD(fft_move_words, dst, NX_FFT_MOVE_DST);
NX_FIELD(fft_move_words, lo, NX_FFT_MOVE_LO);
NX_FIELD(fft_move_words, hi, NX_FFT_MOVE_HI);
NX_FIELD(fft_move_words, tail, NX_FFT_MOVE_TAIL);
NX_FIELD(fft_move_words, first, NX_FFT_MOVE_FIRST);
NX_FIELD(fft_move_words, rows_divisor, NX_FFT_MOVE_ROWS_DIVISOR);
NX_FIELD(fft_move_words, tiles_divisor, NX_FFT_MOVE_TILES_DIVISOR);
NX_FIELD(fft_move_words, a, NX_FFT_MOVE_A);
NX_FIELD(fft_move_words, b, NX_FFT_MOVE_B);
NX_FIELD(fft_move_words, n, NX_FFT_MOVE_N);
NX_FIELD(fft_move_words, mode, NX_FFT_MOVE_MODE);
NX_FIELD(fft_move_words, in, NX_FFT_MOVE_IN);
NX_FIELD(fft_move_words, out, NX_FFT_MOVE_OUT);
NX_SIZE(fft_move_words, NX_FFT_MOVE_WORDS);

struct fft_pair_words {
  device uchar *src;
  device uchar *dst;
  constant float2 *lo;
  constant float2 *hi;
  constant long *tail;
  ulong first;
  ulong blocks_divisor;
  ulong h;
  ulong mode;
  fft_side_words in;
  fft_side_words out;
};

NX_FIELD(fft_pair_words, src, NX_FFT_PAIR_SRC);
NX_FIELD(fft_pair_words, dst, NX_FFT_PAIR_DST);
NX_FIELD(fft_pair_words, lo, NX_FFT_PAIR_LO);
NX_FIELD(fft_pair_words, hi, NX_FFT_PAIR_HI);
NX_FIELD(fft_pair_words, tail, NX_FFT_PAIR_TAIL);
NX_FIELD(fft_pair_words, first, NX_FFT_PAIR_FIRST);
NX_FIELD(fft_pair_words, blocks_divisor, NX_FFT_PAIR_BLOCKS_DIVISOR);
NX_FIELD(fft_pair_words, h, NX_FFT_PAIR_H);
NX_FIELD(fft_pair_words, mode, NX_FFT_PAIR_MODE);
NX_FIELD(fft_pair_words, in, NX_FFT_PAIR_IN);
NX_FIELD(fft_pair_words, out, NX_FFT_PAIR_OUT);
NX_SIZE(fft_pair_words, NX_FFT_PAIR_WORDS);

/* fft.ml sets a pair's buffers, tables and tail at a transpose's words. */
static_assert(NX_FFT_PAIR_SRC == NX_FFT_MOVE_SRC &&
                  NX_FFT_PAIR_DST == NX_FFT_MOVE_DST &&
                  NX_FFT_PAIR_LO == NX_FFT_MOVE_LO &&
                  NX_FFT_PAIR_HI == NX_FFT_MOVE_HI &&
                  NX_FFT_PAIR_TAIL == NX_FFT_MOVE_TAIL,
              "a pair's words as a transpose's");

/* Tables and positions */

/* The root w_l^e from the twiddle's two tables of l. */
static inline float2 fft_root(constant float2 *lo, constant float2 *hi,
                              ulong e) {
  const ulong low = (1u << NX_FFT_LOW_BITS) - 1u;
  return fft_cmul(lo[e & low], hi[e >> NX_FFT_LOW_BITS]);
}

/* The float at which the line whose first point is element [at] of side
   [s] starts. */
static inline long fft_first(constant fft_side_words &s, long at) {
  return s.width == 8 ? 2 * at : at;
}

/* An offset within a line: 32-bit, or 64-bit for a wide side. A point is
   reached from its line's first float by this offset alone: from the
   buffer's start, a 64-bit sum a point, the 4096-point step ran 1.7 times
   slower on an M1 Max. */
template <bool Wide> using fft_index = metal::conditional_t<Wide, long, int>;

/* The offset of point [j] from its line's first, of stride [ps]. */
template <bool Wide> static inline fft_index<Wide> fft_off(ulong j, long ps) {
  return fft_index<Wide>(j) * fft_index<Wide>(ps);
}

/* Point [j] of the line that starts at [line]: a complex element, or the
   floats 2 j and 2 j + 1 of a 4-byte side. */
template <bool Wide>
static inline float2 fft_load(device const float *line,
                              constant fft_side_words &s, ulong j) {
  if (s.width == 8) {
    const fft_c64 z = ((device const fft_c64 *)line)[fft_off<Wide>(j, s.ps)];
    return float2(z.re, z.im);
  }
  return float2(line[fft_off<Wide>(2 * j, s.ps)],
                line[fft_off<Wide>(2 * j + 1, s.ps)]);
}

template <bool Wide>
static inline void fft_store(device float *line, constant fft_side_words &s,
                             ulong k, float2 z) {
  if (s.width == 8) {
    ((device fft_c64 *)line)[fft_off<Wide>(k, s.ps)] = fft_c64{z.x, z.y};
    return;
  }
  line[fft_off<Wide>(2 * k, s.ps)] = z.x;
  line[fft_off<Wide>(2 * k + 1, s.ps)] = z.y;
}

/* Transposes */

/* Point [j] of the line whose first float is [at] as the transpose reads
   it, scaled by the chirp where it scales on reading. */
template <int In, int Scale, bool Wide>
static inline float2 fft_read(constant fft_move_words &w, long at, ulong j) {
  device const float *p = (device const float *)w.src + at;
  if (Scale == NX_FFT_CHIRP_IN && j >= w.n) return float2(0.0f, 0.0f);
  const bool c = (w.mode & NX_FFT_CONJ) != 0;
  float2 z;
  if (In == NX_FFT_IN_REAL) {
    z = float2(p[fft_off<Wide>(j, w.in.ps)], 0.0f);
  } else if (In == NX_FFT_IN_BINS) {
    if (j == 0) {
      z = float2(fft_load<Wide>(p, w.in, 0).x, 0.0f);
    } else if (j <= w.n / 2) {
      z = fft_flip(fft_load<Wide>(p, w.in, j), c);
    } else {
      z = fft_conj(fft_flip(fft_load<Wide>(p, w.in, w.n - j), c));
    }
  } else {
    z = fft_flip(fft_load<Wide>(p, w.in, j), c);
  }
  if (Scale == NX_FFT_CHIRP_IN) z = fft_cmul(z, w.lo[j]);
  return z;
}

/* Writes [z], at row [r] and column [c], as point [k] of the line whose
   first float is [at], scaled where it scales on writing. */
template <int Scale, int Out, bool Wide>
static inline void fft_write(constant fft_move_words &w, long at, ulong k,
                             ulong r, ulong c, float2 z) {
  if (Scale == NX_FFT_CHIRP_OUT && k >= w.n) return;
  if (Out == NX_FFT_OUT_HALF && k > w.n / 2) return;
  if (Scale == NX_FFT_TWIDDLE) {
    z = fft_cmul(z, fft_root(w.lo, w.hi, r * c));
  } else if (Scale == NX_FFT_KERNEL) {
    z = fft_conj(fft_cmul(z, w.lo[k]));
  } else if (Scale == NX_FFT_CHIRP_OUT) {
    z = fft_cmul(fft_conj(z), w.lo[k]);
  }
  z = fft_flip(z, (w.mode & NX_FFT_CONJ) != 0);
  device float *p = (device float *)w.dst + at;
  if (Out == NX_FFT_OUT_REAL)
    p[fft_off<Wide>(k, w.out.ps)] = z.x;
  else
    fft_store<Wide>(p, w.out, k, z);
}

template <int In, int Scale, int Out, bool Wide>
static inline void fft_move(constant fft_move_words &w,
                            threadgroup float2 *tile, uint g, uint t) {
  constexpr uint T = NX_FFT_TILE, P = NX_FFT_TILE + 1;
  const ulong rows = (w.a + T - 1) / T, columns = (w.b + T - 1) / T;
  const ulong tiles = rows * columns;
  const bool move = (w.mode & NX_FFT_MOVE_ONE) != 0;

  /* The block's tile, dividing only where the quotient is not known: a move
     is one run, and a grid of one column of tiles has its row at once. */
  const ulong block = w.first + g;
  const ulong run = move ? 0 : nx_divide(block, tiles, w.tiles_divisor);
  const ulong at = block - run * tiles;
  const ulong column = columns == 1 ? 0 : nx_divide(at, rows, w.rows_divisor);
  const ulong row = at - column * rows;
  const ulong r0 = row * T, c0 = column * T;
  constant long *in_axes = w.in.rank <= NX_FFT_SIDE_HOLDS ? w.in.axes : w.tail;
  constant long *out_axes = w.out.rank <= NX_FFT_SIDE_HOLDS
                                ? w.out.axes
                                : w.tail + NX_FFT_AXIS_WORDS * w.in.rank;
  const uint tx = t % T, ty = t / T;

  /* A split's run is one line of each side. A move's rows are its lines:
     the tile's 32 lines' positions on each side, one a thread of the first
     64, so that no point divides. */
  const long in_run = move ? 0 : fft_first(w.in, fft_line(w.in, in_axes, run));
  const long out_run =
      move ? 0 : fft_first(w.out, fft_line(w.out, out_axes, run));
  threadgroup long *lines = (threadgroup long *)(tile + T * P);
  if (move) {
    const ulong r = r0 + t % T;
    if (t < 2 * T && r < w.a)
      lines[t] = t < T ? fft_first(w.in, fft_line(w.in, in_axes, r))
                       : fft_first(w.out, fft_line(w.out, out_axes, r));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  /* The tile's points, each at (row r0 + i, column c0 + j) in slot i P + j. */
  const bool read_r = (w.mode & NX_FFT_READ_ALONG_R) != 0;
  for (uint q = 0; q < 4; q++) {
    const uint i = read_r ? tx : ty + 8 * q, j = read_r ? ty + 8 * q : tx;
    const ulong r = r0 + i, c = c0 + j;
    if (r < w.a && c < w.b) {
      const long at = move ? lines[i] : in_run;
      const ulong point = move ? c : r * w.b + c;
      tile[i * P + j] = fft_read<In, Scale, Wide>(w, at, point);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool write_r = (w.mode & NX_FFT_WRITE_ALONG_R) != 0;
  for (uint q = 0; q < 4; q++) {
    const uint i = write_r ? tx : ty + 8 * q, j = write_r ? ty + 8 * q : tx;
    const ulong r = r0 + i, c = c0 + j;
    if (r < w.a && c < w.b) {
      const long at = move ? lines[T + i] : out_run;
      const ulong point = move ? c : c * w.a + r;
      fft_write<Scale, Out, Wide>(w, at, point, r, c,
                                  tile[i * P + j]);
    }
  }
}

#define NX_FFT_MOVE_KERNEL_(name, input, scale, output, wide)                 \
  NX_KERNEL(NX_FFT_MOVE_THREADS)                                              \
  void name(constant fft_move_words &w [[buffer(0)]],                         \
            threadgroup float2 *tile [[threadgroup(0)]],                      \
            uint g [[threadgroup_position_in_grid]],                          \
            uint t [[thread_index_in_threadgroup]]) {                         \
    fft_move<input, scale, output, wide>(w, tile, g, t);                      \
  }

#define NX_FFT_MOVE_KERNELS_(X, code, input, scale, output)                   \
  NX_FFT_MOVE_KERNEL_(fft_t_##code, input, scale, output, false)              \
  NX_FFT_MOVE_KERNEL_(fft_t_##code##_w, input, scale, output, true)

NX_METAL_FFT_MOVES(NX_FFT_MOVE_KERNELS_, _)

/* Splits and pre-passes */

template <bool Split, bool Wide>
static inline void fft_pairs(constant fft_pair_words &w, uint g, uint t) {
  constexpr uint B = NX_FFT_PAIR_BLOCK, T = NX_FFT_PAIR_THREADS;
  const ulong blocks = (w.h / 2 + B) / B;
  const ulong block = w.first + g;
  const ulong line = nx_divide(block, blocks, w.blocks_divisor);
  const ulong k0 = (block - line * blocks) * B;
  constant long *in_axes = w.in.rank <= NX_FFT_SIDE_HOLDS ? w.in.axes : w.tail;
  constant long *out_axes = w.out.rank <= NX_FFT_SIDE_HOLDS
                                ? w.out.axes
                                : w.tail + NX_FFT_AXIS_WORDS * w.in.rank;
  device float *d =
      (device float *)w.dst + fft_first(w.out, fft_line(w.out, out_axes, line));
  constant fft_side_words &from = Split ? w.out : w.in;
  device const float *s =
      Split ? d
            : (device const float *)w.src +
                  fft_first(w.in, fft_line(w.in, in_axes, line));
  const ulong h = w.h;
  const bool c = (w.mode & NX_FFT_CONJ) != 0;
  for (uint q = 0; q < B / T; q++) {
    const ulong k = k0 + q * T + t;
    if (k > h / 2) return;
    const float2 r = fft_root(w.lo, w.hi, k);
    float2 a = fft_flip(fft_load<Wide>(s, from, k), c);
    float2 b =
        fft_flip(fft_load<Wide>(s, from, Split && k == 0 ? 0 : h - k), c);
    if (!Split && k == 0) {
      a.y = 0.0f;
      b.y = 0.0f;
    }
    const float2 bc = fft_conj(b);
    const float2 e = a + bc, dd = a - bc;
    const float u = fma(r.x, dd.y, r.y * dd.x);
    const float v = fma(-r.x, dd.x, r.y * dd.y);
    if (Split) {
      fft_store<Wide>(d, w.out, k,
                      fft_flip(float2(0.5f * (e.x + u), 0.5f * (e.y + v)), c));
      fft_store<Wide>(d, w.out, h - k,
                      fft_flip(float2(0.5f * (e.x - u), 0.5f * (v - e.y)), c));
    } else {
      fft_store<Wide>(d, w.out, k, fft_flip(float2(e.x + u, e.y + v), c));
      if (k != 0 && 2 * k != h)
        fft_store<Wide>(d, w.out, h - k,
                        fft_flip(float2(e.x - u, v - e.y), c));
    }
  }
}

#define NX_FFT_PAIR_KERNEL_(name, split, wide)                                \
  NX_KERNEL(NX_FFT_PAIR_THREADS)                                              \
  void name(constant fft_pair_words &w [[buffer(0)]],                         \
            uint g [[threadgroup_position_in_grid]],                          \
            uint t [[thread_index_in_threadgroup]]) {                         \
    fft_pairs<split, wide>(w, g, t);                                          \
  }

NX_FFT_PAIR_KERNEL_(fft_split, true, false)
NX_FFT_PAIR_KERNEL_(fft_split_w, true, true)
NX_FFT_PAIR_KERNEL_(fft_prepass, false, false)
NX_FFT_PAIR_KERNEL_(fft_prepass_w, false, true)

/* Steps */

struct fft_step_words {
  device uchar *src;
  device uchar *dst;
  constant float2 *lo;
  constant float2 *hi;
  constant long *tail;
  ulong first;
  ulong mode;
  fft_side_words in;
  fft_side_words out;
};

NX_FIELD(fft_step_words, src, NX_FFT_STEP_SRC);
NX_FIELD(fft_step_words, dst, NX_FFT_STEP_DST);
NX_FIELD(fft_step_words, lo, NX_FFT_STEP_LO);
NX_FIELD(fft_step_words, hi, NX_FFT_STEP_HI);
NX_FIELD(fft_step_words, tail, NX_FFT_STEP_TAIL);
NX_FIELD(fft_step_words, first, NX_FFT_STEP_FIRST);
NX_FIELD(fft_step_words, mode, NX_FFT_STEP_MODE);
NX_FIELD(fft_step_words, in, NX_FFT_STEP_IN);
NX_FIELD(fft_step_words, out, NX_FFT_STEP_OUT);
NX_SIZE(fft_step_words, NX_FFT_STEP_WORDS);

/* fft.ml sets a step's buffers, tables and tail at a transpose's words. */
static_assert(NX_FFT_STEP_SRC == NX_FFT_MOVE_SRC &&
                  NX_FFT_STEP_DST == NX_FFT_MOVE_DST &&
                  NX_FFT_STEP_LO == NX_FFT_MOVE_LO &&
                  NX_FFT_STEP_HI == NX_FFT_MOVE_HI &&
                  NX_FFT_STEP_TAIL == NX_FFT_MOVE_TAIL,
              "a step's words as a transpose's");

/* w_4096^e, e < 4096, from the inner table's first half: one load at an
   index chosen first. A choice between two loads, [e < 2048 ? lo[e] :
   -lo[e - 2048]], takes both. */
static inline __attribute__((always_inline)) float2
fft_inner(constant float2 *lo, uint e) {
  const uint upper = NX_FFT_STEP_POINTS / 2;
  const bool up = e >= upper;
  const float2 w = lo[up ? e - upper : e];
  return up ? -w : w;
}

/* The plane's slot of point x. */
static inline __attribute__((always_inline)) uint fft_slot(uint x) {
  return x ^ ((x >> 4) & 15u) ^ ((x >> 8) & 15u);
}

/* A butterfly past the first pass: v[r] times w_4096^(e r), then the DFT
   of 16. */
static inline __attribute__((always_inline)) void
fft_turn16(thread float2 *v, constant float2 *lo, uint e) {
  NX_FFT_UNROLL for (uint r = 1; r < 16; r++) {
    v[r] = fft_cmul(v[r], fft_inner(lo, e * r));
  }
  fft_dft16(v);
}

/* A pass of 16 past the first, on points of span [Ns]: reads v[r] from the
   plane, multiplies it by its twiddle and takes the DFT. A pass that writes
   the plane again waits for every thread's reads first. */
template <uint Ns, bool Rewrites>
static inline __attribute__((always_inline)) void
fft_pass16(thread float2 *v, threadgroup float2 *plane, constant float2 *lo,
           uint t) {
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    v[r] = plane[fft_slot(t + 256 * r)];
  }
  if (Rewrites) threadgroup_barrier(mem_flags::mem_threadgroup);
  fft_turn16(v, lo, (NX_FFT_STEP_POINTS / (16 * Ns)) * (t % Ns));
}

NX_KERNEL(NX_FFT_STEP_THREADS)
void fft_s4096(constant fft_step_words &w [[buffer(0)]],
               threadgroup float2 *plane [[threadgroup(0)]],
               uint g [[threadgroup_position_in_grid]],
               uint t [[thread_index_in_threadgroup]]) {
  const ulong line = w.first + g;
  constant long *in_axes = w.in.rank <= NX_FFT_SIDE_HOLDS ? w.in.axes : w.tail;
  constant long *out_axes = w.out.rank <= NX_FFT_SIDE_HOLDS
                                ? w.out.axes
                                : w.tail + NX_FFT_AXIS_WORDS * w.in.rank;
  const long in = fft_line(w.in, in_axes, line);
  const long out = fft_line(w.out, out_axes, line);
  /* Each point a complex element at a 32-bit offset from its line's first,
     as the step's sides are complex and not wide, which emit_step checks
     (fft_index). */
  device const fft_c64 *s = (device const fft_c64 *)w.src + in;
  device fft_c64 *d = (device fft_c64 *)w.dst + out;
  const int ips = int(w.in.ps), ops = int(w.out.ps);
  const bool c = (w.mode & NX_FFT_CONJ) != 0;

  float2 v[16];
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    const fft_c64 z = s[int(t + 256 * r) * ips];
    v[r] = fft_flip(float2(z.re, z.im), c);
  }
  fft_dft16(v);
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    plane[fft_slot(16 * t + r)] = v[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  fft_pass16<16, true>(v, plane, w.lo, t);
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    plane[fft_slot((t / 16) * 256 + t % 16 + 16 * r)] = v[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  fft_pass16<256, false>(v, plane, w.lo, t);
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    const float2 z = fft_flip(v[r], c);
    d[int(t + 256 * r) * ops] = fft_c64{z.x, z.y};
  }
}

/* Column steps */

struct fft_cols_words {
  device uchar *src;
  device uchar *dst;
  constant float2 *lo;
  constant float2 *hi;
  constant long *tail;
  ulong first;
  ulong mode;
  ulong blocks;
  ulong blocks_divisor;
  ulong lines;
  fft_side_words in;
  fft_side_words out;
};

NX_FIELD(fft_cols_words, src, NX_FFT_COLS_SRC);
NX_FIELD(fft_cols_words, dst, NX_FFT_COLS_DST);
NX_FIELD(fft_cols_words, lo, NX_FFT_COLS_LO);
NX_FIELD(fft_cols_words, hi, NX_FFT_COLS_HI);
NX_FIELD(fft_cols_words, tail, NX_FFT_COLS_TAIL);
NX_FIELD(fft_cols_words, first, NX_FFT_COLS_FIRST);
NX_FIELD(fft_cols_words, mode, NX_FFT_COLS_MODE);
NX_FIELD(fft_cols_words, blocks, NX_FFT_COLS_BLOCKS);
NX_FIELD(fft_cols_words, blocks_divisor, NX_FFT_COLS_BLOCKS_DIVISOR);
NX_FIELD(fft_cols_words, lines, NX_FFT_COLS_LINES);
NX_FIELD(fft_cols_words, in, NX_FFT_COLS_IN);
NX_FIELD(fft_cols_words, out, NX_FFT_COLS_OUT);
NX_SIZE(fft_cols_words, NX_FFT_COLS_WORDS);

/* fft.ml sets a column step's buffers, tables and tail at a transpose's
   words. */
static_assert(NX_FFT_COLS_SRC == NX_FFT_MOVE_SRC &&
                  NX_FFT_COLS_DST == NX_FFT_MOVE_DST &&
                  NX_FFT_COLS_LO == NX_FFT_MOVE_LO &&
                  NX_FFT_COLS_HI == NX_FFT_MOVE_HI &&
                  NX_FFT_COLS_TAIL == NX_FFT_MOVE_TAIL,
              "a column step's words as a transpose's");

/* A lane's line of a column step's group (kernels.h): its block's line
   [col] of run [run], whether the run holds it, and the group's part. */
struct fft_cols_lane {
  ulong line;
  bool held;
  uint part;
};

static inline fft_cols_lane fft_cols_at(constant fft_cols_words &w, uint g,
                                        uint i) {
  const ulong at = w.first + g;
  const ulong q = nx_divide(at, w.blocks, w.blocks_divisor);
  const ulong col = 16 * (at - q * w.blocks) + i % 16;
  return {(q / 16) * w.lines + col, col < w.lines, uint(q % 16)};
}

NX_KERNEL(NX_FFT_STEP_THREADS)
void fft_c4096_first(constant fft_cols_words &w [[buffer(0)]],
                     uint g [[threadgroup_position_in_grid]],
                     uint i [[thread_index_in_threadgroup]]) {
  const fft_cols_lane at = fft_cols_at(w, g, i);
  if (!at.held) return;
  constant long *in_axes = w.in.rank <= NX_FFT_SIDE_HOLDS ? w.in.axes : w.tail;
  constant long *out_axes = w.out.rank <= NX_FFT_SIDE_HOLDS
                                ? w.out.axes
                                : w.tail + NX_FFT_AXIS_WORDS * w.in.rank;
  /* Complex points at 32-bit offsets from each line's first, as a column
     step's sides are complex and not wide, which emit checks (fft_index). */
  device const fft_c64 *s =
      (device const fft_c64 *)w.src + fft_line(w.in, in_axes, at.line);
  device fft_c64 *d = (device fft_c64 *)w.dst + fft_line(w.out, out_axes, at.line);
  const int ips = int(w.in.ps), ops = int(w.out.ps);
  const bool c = (w.mode & NX_FFT_CONJ) != 0;
  const uint t = 16 * at.part + i / 16;

  float2 v[16];
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    const fft_c64 z = s[int(t + 256 * r) * ips];
    v[r] = fft_flip(float2(z.re, z.im), c);
  }
  fft_dft16(v);
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    d[int(256 * r + t) * ops] = fft_c64{v[r].x, v[r].y};
  }
}

NX_KERNEL(NX_FFT_STEP_THREADS)
void fft_c4096_last(constant fft_cols_words &w [[buffer(0)]],
                    threadgroup float2 *plane [[threadgroup(0)]],
                    uint g [[threadgroup_position_in_grid]],
                    uint i [[thread_index_in_threadgroup]]) {
  const fft_cols_lane at = fft_cols_at(w, g, i);
  constant long *in_axes = w.in.rank <= NX_FFT_SIDE_HOLDS ? w.in.axes : w.tail;
  constant long *out_axes = w.out.rank <= NX_FFT_SIDE_HOLDS
                                ? w.out.axes
                                : w.tail + NX_FFT_AXIS_WORDS * w.in.rank;
  const int ips = int(w.in.ps), ops = int(w.out.ps);
  const bool c = (w.mode & NX_FFT_CONJ) != 0;
  const uint m = at.part, u = i / 16, col = i % 16;

  /* A lane past its run's lines loads and stores nothing; its plane slots
     hold its own column alone, which no other lane reads. */
  float2 v[16];
  if (at.held) {
    device const fft_c64 *s =
        (device const fft_c64 *)w.src + fft_line(w.in, in_axes, at.line);
    NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
      const fft_c64 z = s[int(256 * m + u + 16 * r) * ips];
      v[r] = float2(z.re, z.im);
    }
  } else {
    NX_FFT_UNROLL for (uint r = 0; r < 16; r++) { v[r] = float2(0.0f); }
  }
  fft_turn16(v, w.lo, 16 * m);
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    plane[16 * (16 * u + r) + col] = v[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    v[r] = plane[16 * (u + 16 * r) + col];
  }
  fft_turn16(v, w.lo, m + 16 * u);
  if (!at.held) return;
  device fft_c64 *d = (device fft_c64 *)w.dst + fft_line(w.out, out_axes, at.line);
  NX_FFT_UNROLL for (uint r = 0; r < 16; r++) {
    const float2 z = fft_flip(v[r], c);
    d[int(m + 16 * u + 256 * r) * ops] = fft_c64{z.x, z.y};
  }
}
