/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Copies and casts.

   A copy moves bits, a block of planes at a time through nx_copy_box:
   rows of bytes by memcpy, a transposed source in square blocks (on
   x86-64 through a buffer where its rows lie 4 KiB apart), sub-byte rows
   as runs of whole bytes with compare-and-swap ends, other strides
   element by element. A cast is unstage ∘ stage, plane by plane:
   the stage brings the source's block to its carrier, in place where the
   source is already one in contiguous rows, and the unstage converts the
   carrier into the destination's dtype. A cast to the source's own dtype,
   and a bitcast, is a copy, NaN payloads kept. A transposed copy's square
   blocks store past the caches where its result gains (streams). */

#include <string.h>

#include <caml/mlvalues.h>

#include "cpu.h"

/* Boxes */

/* Rows of a copy are read ahead: a row that starts apart from the one before,
   as a box's rows do in a wider array, is a stream the hardware prefetcher has
   not seen, and its copy would wait on memory. On the M1, 4 MiB in rows of
   1 KiB, 4 KiB apart, gather in 470 us without it and in 104 us reading two
   rows ahead, where one memcpy of the 4 MiB takes 90. Only a row's first 4 KiB
   are asked for: the prefetcher follows a longer row once it has started. A
   read ahead is a hint: past the buffer's end it loads nothing and faults
   nothing. */
#define READ_AHEAD_ROWS 2
#define READ_AHEAD_BYTES 4096
#define CACHE_LINE 64

/* Asks for the first bytes, at most [READ_AHEAD_BYTES], of the [n] at
   [base + off] in cache. The address is formed as an integer, since it may
   lie past the buffer. */
static inline __attribute__((always_inline)) void read_ahead(
    const uint8_t *base, int64_t off, int64_t n) {
  if (n > READ_AHEAD_BYTES) n = READ_AHEAD_BYTES;
  for (int64_t b = 0; b < n; b += CACHE_LINE)
    __builtin_prefetch((const void *)((uintptr_t)base + (uintptr_t)(off + b)));
}

/* Copies [n] elements of [w] bytes, the [k]th from [s + k·bs] to
   [d + k·bd]. Called with a constant [w], the copies are loads and stores. */
static inline __attribute__((always_inline)) void strided(
    uint8_t *d, int64_t bd, const uint8_t *s, int64_t bs, int64_t n,
    size_t w) {
  for (int64_t k = 0; k < n; k++, d += bd, s += bs) memcpy(d, s, w);
}

/* Copies the 4x4 block whose column q is the [4·w] bytes at [s + q·sc]
   into the rows at [d + p·dr], [4·w] bytes each: a transpose. */
static inline __attribute__((always_inline)) void block(
    uint8_t *d, int64_t dr, const uint8_t *s, int64_t sc, size_t w) {
  uint8_t x[4][4 * 16];
  for (int q = 0; q < 4; q++) memcpy(x[q], s + q * sc, 4 * w);
  for (int p = 0; p < 4; p++)
    for (int q = 0; q < 4; q++) memcpy(d + p * dr + q * w, x[q] + p * w, w);
}

/* The 8x8 block of 4-byte elements whose column q is the 32 bytes at
   [s + q·sc] into the rows at [d + p·dr], in registers: each column read
   and each row written as two 16-byte vectors, four 4x4 transposes
   between. 32 bytes of each line at a time beat the 16 of 4x4 blocks: a
   transposed 512x512 float32 copy on one thread takes 36.4 us against
   59.2 in 4x4 blocks on the M1, where Accelerate's vDSP_mtrans takes 35.4,
   and 35.5 against 61.7 on kimchi, where OpenBLAS takes 41.2. */
#if defined(__aarch64__)
static inline __attribute__((always_inline)) void transpose4(
    uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t e, uint32x4_t *o) {
  uint64x2_t t0 = vreinterpretq_u64_u32(vtrn1q_u32(a, b));
  uint64x2_t t1 = vreinterpretq_u64_u32(vtrn2q_u32(a, b));
  uint64x2_t t2 = vreinterpretq_u64_u32(vtrn1q_u32(c, e));
  uint64x2_t t3 = vreinterpretq_u64_u32(vtrn2q_u32(c, e));
  o[0] = vreinterpretq_u32_u64(vtrn1q_u64(t0, t2));
  o[1] = vreinterpretq_u32_u64(vtrn1q_u64(t1, t3));
  o[2] = vreinterpretq_u32_u64(vtrn2q_u64(t0, t2));
  o[3] = vreinterpretq_u32_u64(vtrn2q_u64(t1, t3));
}

static inline __attribute__((always_inline)) void block8(uint8_t *d, int64_t dr,
                                                         const uint8_t *s,
                                                         int64_t sc) {
  uint32x4_t x[8][2], o[4];
  for (int q = 0; q < 8; q++)
    for (int h = 0; h < 2; h++)
      x[q][h] = vld1q_u32((const uint32_t *)(s + q * sc + 16 * h));
  for (int h = 0; h < 2; h++)
    for (int v = 0; v < 2; v++) {
      transpose4(x[4 * v][h], x[4 * v + 1][h], x[4 * v + 2][h],
                 x[4 * v + 3][h], o);
      for (int p = 0; p < 4; p++)
        vst1q_u32((uint32_t *)(d + (4 * h + p) * dr + 16 * v), o[p]);
    }
}
#elif defined(__x86_64__)
/* x86-64 writes each row's 32 bytes with two adjacent stores: a transposed
   512x512 float32 copy on one kimchi thread takes 35.5 us against 43.8
   with the halves of four rows interleaved. arm64 gains nothing from it. */
static inline __attribute__((always_inline)) void block8(uint8_t *d, int64_t dr,
                                                         const uint8_t *s,
                                                         int64_t sc) {
  for (int h = 0; h < 2; h++) {
    __m128 t[2][4];
    for (int v = 0; v < 2; v++) {
      const uint8_t *b = s + 4 * v * sc + 16 * h;
      __m128 x0 = _mm_loadu_ps((const float *)b);
      __m128 x1 = _mm_loadu_ps((const float *)(b + sc));
      __m128 x2 = _mm_loadu_ps((const float *)(b + 2 * sc));
      __m128 x3 = _mm_loadu_ps((const float *)(b + 3 * sc));
      _MM_TRANSPOSE4_PS(x0, x1, x2, x3);
      t[v][0] = x0, t[v][1] = x1, t[v][2] = x2, t[v][3] = x3;
    }
    for (int p = 0; p < 4; p++) {
      uint8_t *o = d + (4 * h + p) * dr;
      _mm_storeu_ps((float *)o, t[0][p]);
      _mm_storeu_ps((float *)(o + 16), t[1][p]);
    }
  }
}
#else
static inline __attribute__((always_inline)) void block8(uint8_t *d, int64_t dr,
                                                         const uint8_t *s,
                                                         int64_t sc) {
  for (int h = 0; h < 2; h++)
    for (int v = 0; v < 2; v++)
      block(d + 4 * h * dr + 16 * v, dr, s + 4 * v * sc + 16 * h, sc, 4);
}
#endif

/* The 4x4 block of 8-byte elements whose column q is the 32 bytes at
   [s + q·sc] into the rows at [d + p·dr], in registers: each column read
   and each row written as two 16-byte vectors, the pairs of columns
   interleaved. block's round trip through memory made a transposed
   float64 pack of 96 rows by 256 7.5% slower than moving its elements one
   by one on six kimchi cores. */
#if defined(__aarch64__)
static inline __attribute__((always_inline)) void block4x8(uint8_t *d,
                                                           int64_t dr,
                                                           const uint8_t *s,
                                                           int64_t sc) {
  uint64x2_t x[4][2];
  for (int q = 0; q < 4; q++)
    for (int h = 0; h < 2; h++)
      x[q][h] = vld1q_u64((const uint64_t *)(s + q * sc + 16 * h));
  for (int h = 0; h < 2; h++)
    for (int v = 0; v < 2; v++) {
      uint64x2_t a = x[2 * v][h], b = x[2 * v + 1][h];
      vst1q_u64((uint64_t *)(d + 2 * h * dr + 16 * v), vtrn1q_u64(a, b));
      vst1q_u64((uint64_t *)(d + (2 * h + 1) * dr + 16 * v),
                vtrn2q_u64(a, b));
    }
}
#elif defined(__x86_64__)
static inline __attribute__((always_inline)) void block4x8(uint8_t *d,
                                                           int64_t dr,
                                                           const uint8_t *s,
                                                           int64_t sc) {
  for (int h = 0; h < 2; h++)
    for (int v = 0; v < 2; v++) {
      __m128d a = _mm_loadu_pd((const double *)(s + 2 * v * sc + 16 * h));
      __m128d b = _mm_loadu_pd((const double *)(s + (2 * v + 1) * sc + 16 * h));
      _mm_storeu_pd((double *)(d + 2 * h * dr + 16 * v), _mm_unpacklo_pd(a, b));
      _mm_storeu_pd((double *)(d + (2 * h + 1) * dr + 16 * v),
                    _mm_unpackhi_pd(a, b));
    }
}
#else
static inline __attribute__((always_inline)) void block4x8(uint8_t *d,
                                                           int64_t dr,
                                                           const uint8_t *s,
                                                           int64_t sc) {
  block(d, dr, s, sc, 8);
}
#endif

/* The square blocks past the caches: block8's and block4x8's rows, each
   row's 32 bytes in one nx_stream32. They are functions of their own: one
   stream takes a row's two halves together, and the plain blocks rewritten
   so, their stores reordered, made a transposed 512x512 float32 copy 18%
   slower on one fudge core under gcc. A target without such stores stores
   plainly. */
#if defined(__aarch64__)
static inline __attribute__((always_inline)) void block8_nt(uint8_t *d,
                                                            int64_t dr,
                                                            const uint8_t *s,
                                                            int64_t sc) {
  uint32x4_t x[8][2], r[4][2];
  for (int q = 0; q < 8; q++)
    for (int h = 0; h < 2; h++)
      x[q][h] = vld1q_u32((const uint32_t *)(s + q * sc + 16 * h));
  for (int h = 0; h < 2; h++) {
    uint32x4_t lo[4], hi[4];
    transpose4(x[0][h], x[1][h], x[2][h], x[3][h], lo);
    transpose4(x[4][h], x[5][h], x[6][h], x[7][h], hi);
    for (int p = 0; p < 4; p++) {
      r[p][0] = lo[p];
      r[p][1] = hi[p];
      nx_stream32(d + (4 * h + p) * dr, r[p]);
    }
  }
}

static inline __attribute__((always_inline)) void block4x8_nt(
    uint8_t *d, int64_t dr, const uint8_t *s, int64_t sc) {
  uint64x2_t x[4][2];
  for (int q = 0; q < 4; q++)
    for (int h = 0; h < 2; h++)
      x[q][h] = vld1q_u64((const uint64_t *)(s + q * sc + 16 * h));
  for (int h = 0; h < 2; h++) {
    uint64x2_t r[2][2] = {
        {vtrn1q_u64(x[0][h], x[1][h]), vtrn1q_u64(x[2][h], x[3][h])},
        {vtrn2q_u64(x[0][h], x[1][h]), vtrn2q_u64(x[2][h], x[3][h])}};
    nx_stream32(d + 2 * h * dr, r[0]);
    nx_stream32(d + (2 * h + 1) * dr, r[1]);
  }
}
#elif defined(__x86_64__)
static inline __attribute__((always_inline)) void block8_nt(uint8_t *d,
                                                            int64_t dr,
                                                            const uint8_t *s,
                                                            int64_t sc) {
  for (int h = 0; h < 2; h++) {
    __m128 t[4][2];
    for (int v = 0; v < 2; v++) {
      const uint8_t *b = s + 4 * v * sc + 16 * h;
      __m128 x0 = _mm_loadu_ps((const float *)b);
      __m128 x1 = _mm_loadu_ps((const float *)(b + sc));
      __m128 x2 = _mm_loadu_ps((const float *)(b + 2 * sc));
      __m128 x3 = _mm_loadu_ps((const float *)(b + 3 * sc));
      _MM_TRANSPOSE4_PS(x0, x1, x2, x3);
      t[0][v] = x0, t[1][v] = x1, t[2][v] = x2, t[3][v] = x3;
    }
    for (int p = 0; p < 4; p++) nx_stream32(d + (4 * h + p) * dr, t[p]);
  }
}

static inline __attribute__((always_inline)) void block4x8_nt(
    uint8_t *d, int64_t dr, const uint8_t *s, int64_t sc) {
  for (int h = 0; h < 2; h++) {
    __m128d x[4];
    for (int q = 0; q < 4; q++)
      x[q] = _mm_loadu_pd((const double *)(s + q * sc + 16 * h));
    __m128d r[2][2] = {
        {_mm_unpacklo_pd(x[0], x[1]), _mm_unpacklo_pd(x[2], x[3])},
        {_mm_unpackhi_pd(x[0], x[1]), _mm_unpackhi_pd(x[2], x[3])}};
    nx_stream32(d + 2 * h * dr, r[0]);
    nx_stream32(d + (2 * h + 1) * dr, r[1]);
  }
}
#else
#define block8_nt block8
#define block4x8_nt block4x8
#endif

/* Interleaves the [n] elements of [w] bytes at [a] and at [b] into [d]:
   a's i-th element at d's 2i-th, b's at the 2i+1-th, as a window two
   elements wide reads its rows. 4- and 8-byte elements zip in registers,
   four or two pairs at a time. */
static inline __attribute__((always_inline)) void zip2(uint8_t *d,
                                                       const uint8_t *a,
                                                       const uint8_t *b,
                                                       int64_t n, size_t w) {
  int64_t i = 0;
#if defined(__aarch64__)
  if (w == 4)
    for (; i + 4 <= n; i += 4) {
      uint32x4x2_t x = {{vld1q_u32((const uint32_t *)(a + 4 * i)),
                         vld1q_u32((const uint32_t *)(b + 4 * i))}};
      vst2q_u32((uint32_t *)(d + 8 * i), x);
    }
  if (w == 8)
    for (; i + 2 <= n; i += 2) {
      uint64x2x2_t x = {{vld1q_u64((const uint64_t *)(a + 8 * i)),
                         vld1q_u64((const uint64_t *)(b + 8 * i))}};
      vst2q_u64((uint64_t *)(d + 16 * i), x);
    }
#elif defined(__x86_64__)
  if (w == 4)
    for (; i + 4 <= n; i += 4) {
      __m128 x = _mm_loadu_ps((const float *)(a + 4 * i));
      __m128 y = _mm_loadu_ps((const float *)(b + 4 * i));
      _mm_storeu_ps((float *)(d + 8 * i), _mm_unpacklo_ps(x, y));
      _mm_storeu_ps((float *)(d + 8 * i + 16), _mm_unpackhi_ps(x, y));
    }
  if (w == 8)
    for (; i + 2 <= n; i += 2) {
      __m128d x = _mm_loadu_pd((const double *)(a + 8 * i));
      __m128d y = _mm_loadu_pd((const double *)(b + 8 * i));
      _mm_storeu_pd((double *)(d + 16 * i), _mm_unpacklo_pd(x, y));
      _mm_storeu_pd((double *)(d + 16 * i + 16), _mm_unpackhi_pd(x, y));
    }
#endif
  for (; i < n; i++) {
    memcpy(d + 2 * i * w, a + i * w, w);
    memcpy(d + (2 * i + 1) * w, b + i * w, w);
  }
}

/* A block of elements of [w] bytes, steps in bytes: rows of adjacent
   elements are memcpy, a source that steps one element across rows into
   adjacent elements moves in square blocks, 8x8 of 4-byte elements and 4x4
   of the others, two adjacent runs of the source into adjacent elements
   zip, other steps element by element. The square blocks of 4- and 8-byte
   elements store past the caches where [nt]. */
static inline __attribute__((always_inline)) void bytes(
    uint8_t *d, int64_t dr, int64_t dc, const uint8_t *s, int64_t sr,
    int64_t sc, int64_t rows, int64_t cols, size_t w, int nt) {
  int64_t i = 0, sw = (int64_t)w, side = w == 4 ? 8 : 4;
  if (rows == 2 && dr == sw && dc == 2 * sw && sc == sw) {
    zip2(d, s, s + sr, cols, w);
    return;
  }
  if (cols == 2 && dc == sw && dr == 2 * sw && sr == sw) {
    zip2(d, s, s + sc, rows, w);
    return;
  }
  if (dc == sw && sr == sw)
    for (; i + side <= rows; i += side) {
      int64_t j = 0;
      for (; j + side <= cols; j += side)
        if (w == 4 && nt)
          block8_nt(d + i * dr + j * sw, dr, s + i * sr + j * sc, sc);
        else if (w == 4)
          block8(d + i * dr + j * sw, dr, s + i * sr + j * sc, sc);
        else if (w == 8 && nt)
          block4x8_nt(d + i * dr + j * sw, dr, s + i * sr + j * sc, sc);
        else if (w == 8)
          block4x8(d + i * dr + j * sw, dr, s + i * sr + j * sc, sc);
        else
          block(d + i * dr + j * sw, dr, s + i * sr + j * sc, sc, w);
      for (int p = 0; p < side; p++)
        strided(d + (i + p) * dr + j * sw, sw, s + (i + p) * sr + j * sc, sc,
                cols - j, w);
    }
  for (; i < rows; i++)
    if (dc == sw && sc == sw) {
      if (sr != cols * sw && i + READ_AHEAD_ROWS < rows)
        read_ahead(s, (i + READ_AHEAD_ROWS) * sr, cols * sw);
      memcpy(d + i * dr, s + i * sr, (size_t)(cols * sw));
    } else
      strided(d + i * dr, dc, s + i * sr, sc, cols, w);
}

/* A plane of nx_copy_box, inlined into its planes loop; of nx_stream_box
   where [nt]. */
static inline __attribute__((always_inline)) void plane(
    uint8_t *dst, int64_t pd, int64_t dst_row, int64_t dst_col,
    const uint8_t *src, int64_t ps, int64_t src_row, int64_t src_col,
    int64_t rows, int64_t cols, int bits, int nt) {
  if (bits < 8) {
    for (int64_t i = 0; i < rows; i++) {
      int64_t d = pd + i * dst_row, s = ps + i * src_row;
      if (dst_col == 1) {
        nx_sub_copy_run(dst, d, src, s, src_col, cols, bits);
        continue;
      }
      for (int64_t j = 0; j < cols; j++, d += dst_col, s += src_col)
        nx_sub_store(dst, bits, d, nx_sub_load(src, bits, s));
    }
    return;
  }
  int64_t w = bits / 8;
  uint8_t *d = dst + pd * w;
  const uint8_t *s = src + ps * w;
  if (dst_col == 1 && src_col == 1 && dst_row == cols && src_row == cols) {
    memcpy(d, s, (size_t)(rows * cols * w));
    return;
  }
  int64_t dr = dst_row * w, dc = dst_col * w;
  int64_t sr = src_row * w, sc = src_col * w;
  /* A plane streams where its pieces of rows are whole lines. */
  nt = nt && ((uintptr_t)d | (uint64_t)dr | (uint64_t)(cols * w)) % 64 == 0;
  switch (w) {
    case 1: bytes(d, dr, dc, s, sr, sc, rows, cols, 1, 0); return;
    case 2: bytes(d, dr, dc, s, sr, sc, rows, cols, 2, 0); return;
    case 4: bytes(d, dr, dc, s, sr, sc, rows, cols, 4, nt); return;
    case 8: bytes(d, dr, dc, s, sr, sc, rows, cols, 8, nt); return;
    default: bytes(d, dr, dc, s, sr, sc, rows, cols, 16, 0); return;
  }
}

/* The planes loop below plane's dispatch, inlined into it, so a box of
   small planes chooses its case once. A destination that steps one element
   across rows, its source one along them, swaps the two axes on both sides:
   the same copy, as a transposed source, which the square blocks take. */
static inline __attribute__((always_inline)) void box(
    uint8_t *dst, const uint8_t *src, const nx_box *b, int bits, int nt) {
  int64_t pd = b->first[0], ps = b->first[1];
  int64_t dp = b->step[0][0], dr = b->step[0][1], dc = b->step[0][2];
  int64_t sp = b->step[1][0], sr = b->step[1][1], sc = b->step[1][2];
  int64_t rows = b->extent[1], cols = b->extent[2];
  if (dr == 1 && sc == 1 && !(dc == 1 && sr == 1)) {
    int64_t x = dr;
    dr = dc;
    dc = x;
    x = sr;
    sr = sc;
    sc = x;
    x = rows;
    rows = cols;
    cols = x;
  }
  /* One plane, the walk's common block, takes no loop. */
  if (b->extent[0] == 1) {
    plane(dst, pd, dr, dc, src, ps, sr, sc, rows, cols, bits, nt);
    return;
  }
  for (int64_t p = 0; p < b->extent[0]; p++, pd += dp, ps += sp)
    plane(dst, pd, dr, dc, src, ps, sr, sc, rows, cols, bits, nt);
}

void nx_copy_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                 int bits) {
  box(dst, src, b, bits, 0);
}

void nx_stream_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                   int bits) {
  box(dst, src, b, bits, NX_STREAMS);
  nx_stream_drain();
}

/* Elements of a copy's block: at 64 Ki a block's call costs next to
   nothing beside its memcpy, and a job still has units to share. */
#define COPY_MOST (64 * 1024)

/* The widest form of a cast from [s] to [d]: its source, its carrier or its
   destination. */
static int64_t cast_most(int s, int d) {
  int w = nx_cpu_width(s), c = nx_cpu_width(nx_cpu_carrier(s));
  int x = nx_cpu_width(d);
  if (c > w) w = c;
  if (x > w) w = x;
  return NX_CPU_SLOT / w;
}

/* The bytes of a piece of a row a plane casts through where the thread's
   kept scratch cannot be had. */
#define PIECE 1024

/* A plane staged into a slot of the carrier [c], then unstaged: the slot
   in the thread's kept scratch, which a thread's stack should not hold,
   else a piece of a row at a time through a small one. */
static void cast_slot(const nx_cpu_block *b, const nx_array *a, int c) {
  int64_t w = nx_cpu_width(c);
  uint8_t *slot = nx_cpu_take(NX_CPU_SLOT);
  if (slot != NULL) {
    nx_cpu_stage(&a[1], b, 1, slot, b->n0 * w);
    nx_cpu_unstage(&a[0], b, 0, slot, b->n0 * w, c);
    nx_cpu_give(slot);
    return;
  }
  _Alignas(64) uint8_t piece[PIECE];
  nx_cpu_block p = *b;
  p.n1 = 1;
  for (int64_t j = 0; j < b->n1; j++)
    for (int64_t i = 0; i < b->n0; i += PIECE / w) {
      p.n0 = b->n0 - i < PIECE / w ? b->n0 - i : PIECE / w;
      for (int k = 0; k < 2; k++)
        p.at[k] = b->at[k] + j * b->s1[k] + i * b->s0[k];
      nx_cpu_stage(&a[1], &p, 1, piece, p.n0 * w);
      nx_cpu_unstage(&a[0], &p, 0, piece, p.n0 * w, c);
    }
}

/* A plane of a cast. */
static void cast_plane(const nx_cpu_block *b, const nx_array *a) {
  int s = a[1].dtype, c = nx_cpu_carrier(s), w = nx_cpu_width(c);
  /* A source that is its carrier in rows of contiguous elements is read in
     place. */
  if (s == c && b->s0[1] == 1) {
    nx_cpu_unstage(&a[0], b, 0, a[1].base + b->at[1] * w, b->s1[1] * w, c);
    return;
  }
  /* A destination that is the carrier, in rows of contiguous elements, is
     staged into. */
  if (a[0].dtype == c && b->s0[0] == 1) {
    nx_cpu_stage(&a[1], b, 1, a[0].base + b->at[0] * w, b->s1[0] * w);
    return;
  }
  cast_slot(b, a, c);
}

/* A cast stages plane by plane: its slot holds one. */
static void cast_block(const nx_cpu_block *b, void *ctx) {
  if (b->n2 == 1) {
    cast_plane(b, ctx);
    return;
  }
  nx_cpu_block p = *b;
  p.n2 = 1;
  for (int64_t q = 0; q < b->n2; q++) {
    cast_plane(&p, ctx);
    for (int k = 0; k < 2; k++) p.at[k] += b->s2[k];
  }
}

/* A copy's operands, destination first, and the copy of its boxes:
   nx_copy_box, or nx_stream_box past the caches. */
typedef struct {
  const nx_array *a;
  void (*box)(uint8_t *, const uint8_t *, const nx_box *, int);
} copying;

#if defined(__x86_64__)
/* The bytes of a tile buffered on x86-64. There a load waits for an earlier
   store whose address has the same low 12 bits, and the rows of a tile
   whose input rows lie a multiple of 4 KiB apart all share theirs with the
   output's: such a tile is first copied row by row into a buffer, then
   transposed from it. A transposed 4096x4096 float32 copy on kimchi takes
   2.91 ms against 3.21; on the M1, which has no such wait, staging took it
   from 1.7 to 2.6 ms. */
#define TILE_BUFFER (32 * 1024)

/* A tile through the thread's kept scratch, which a thread's stack should
   not hold: 0, or 1 where the scratch cannot be had, the tile then
   copied as any other. */
static int via_buffer(const nx_cpu_block *b, const copying *c) {
  const nx_array *a = c->a;
  uint8_t *tile = nx_cpu_take(TILE_BUFFER);
  if (tile == NULL) return 1;
  nx_copy_box(tile, a[1].base,
              &(nx_box){{1, b->n0, b->n1},
                        {0, b->at[1]},
                        {{0, b->n1, 1}, {0, b->s0[1], 1}}},
              a[0].bits);
  c->box(a[0].base, tile,
         &(nx_box){{1, b->n1, b->n0},
                   {b->at[0], 0},
                   {{0, b->s1[0], 1}, {0, 1, b->n1}}},
         a[0].bits);
  nx_cpu_give(tile);
  return 0;
}
#endif

static void copy_block(const nx_cpu_block *b, void *ctx) {
  const copying *c = ctx;
  const nx_array *a = c->a;
#if defined(__x86_64__)
  int64_t w = a[0].bits / 8;
  if (b->n2 == 1 && b->s0[0] == 1 && b->s1[1] == 1 && a[0].bits >= 8 &&
      b->s0[1] * w % 4096 == 0 && b->n0 * b->n1 * w <= TILE_BUFFER &&
      via_buffer(b, c) == 0)
    return;
#endif
  c->box(a[0].base, a[1].base,
         &(nx_box){{b->n2, b->n1, b->n0},
                   {b->at[0], b->at[1]},
                   {{b->s2[0], b->s1[0], b->s0[0]},
                    {b->s2[1], b->s1[1], b->s0[1]}}},
         a[0].bits);
}

/* Whether a copy of [bytes] on [threads] threads streams its square blocks
   (nx_stream_box): where its result exceeds what a cache keeps for its
   reader. On x86-64, past a core's L2 on one thread and past the largest
   cache on several: on Arrow Lake a transposed 4096x4096 float32 copy took
   5.34 ms streamed against 8.67 on one core and 1.83 against 2.70 on six,
   a float64 2048x2048 one 2.83 against 3.32 and 0.79 against 0.90, while
   results of 4 to 8 MiB took 0.95 to 1.04 times as long streamed on six.
   On arm64, past half of a core's L2 on one thread and past four times
   the largest cache on several: macOS reports the M1 Max's 12 MiB cluster
   L2, not its 48 MiB system cache, and on eight of its cores float32
   results of 4 to 16 MiB took 1.05 to 1.15 times as long streamed, of 32
   MiB 0.97 to 1.01, of 62 MiB 0.72 to 0.88. */
static int streams(int64_t bytes, int threads) {
#if defined(__x86_64__)
  int64_t one = nx_cpu_l2, several = nx_cpu_llc;
#else
  int64_t one = nx_cpu_l2 / 2, several = 4 * nx_cpu_llc;
#endif
  return NX_STREAMS && bytes > (threads == 1 ? one : several);
}

/* Copies the loop [l] of the operands [a], destination first, [per] an
   element, on the threads its walk takes: only a transposed copy, which
   the walk tiles, streams. */
static void copy_walk(nx_array *a, const nx_loop *l, double per) {
  int64_t n = 1;
  for (int i = 0; i < l->rank; i++) n *= l->extent[i];
  int64_t bytes = n * a[0].bits / 8;
  int threads = nx_cpu_walk_threads(2, a, l, per);
  copying c = {a, streams(bytes, threads) ? nx_stream_box : nx_copy_box};
  nx_cpu_walk(2, a, l, COPY_MOST, per, copy_block, &c);
}

/* Casts the loop [l] of the operands [a], destination first, [per] an
   element. */
static void cast_walk(nx_array *a, const nx_loop *l, double per) {
  nx_cpu_walk(2, a, l, cast_most(a[1].dtype, a[0].dtype), per, cast_block,
              a);
}

void nx_cpu_copy_loop(uint8_t *dst, const uint8_t *src, int bits,
                      const nx_loop *l, double per) {
  /* The walk and the block copy read the operands' bases and widths. */
  nx_array pair[2] = {{.base = dst, .bits = bits},
                      {.base = (uint8_t *)src, .bits = bits}};
  copy_walk(pair, l, per);
}

void nx_cpu_cast_loop(const nx_array *dst, const nx_array *src,
                      const nx_loop *l, double per) {
  nx_array pair[2] = {*dst, *src};
  if (dst->dtype == src->dtype)
    nx_cpu_walk(2, pair, l, COPY_MOST, per, copy_block, pair);
  else
    nx_cpu_walk(2, pair, l, cast_most(src->dtype, dst->dtype), per,
                cast_block, pair);
}

/* Reads [vd], written, and [vs] of the dtypes [d] and [s] through the door,
   and walks them with [walk], whose element is [kind]'s (nx_kernel.h's
   code): a cast's from [s] to [d], a copy's or a bitcast's at [s], of
   [d]'s width, which it computes in. */
static value walk_operands(value vd, value vs, int d, int s, int kind,
                           void (*walk)(nx_array *, const nx_loop *,
                                        double)) {
  nx_operand in[2] = {{vd, d, 1}, {vs, s, 0}};
  nx_array a[2];
  nx_loop l;
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  if (!(e = nx_coalesce(2, a, &l))) {
    double per = kind == NX_OP1_CAST ? nx_cpu_cast_cost(s, d)
                                     : nx_cpu_kind_cost(1, kind, s, s);
    int released = nx_cpu_release(nx_cpu_loop_cost(2, a, &l, per));
    walk(a, &l, per);
    nx_cpu_reacquire(released);
  }
  nx_done(2, a);
  return Val_int(e);
}

value nx_cpu_copy(value vd, value vs) {
  int d = nx_array_dtype(vd);
  return walk_operands(vd, vs, d, d, NX_OP1_COPY, copy_walk);
}

/* A bitcast copies the bits of a dtype of the destination's width. */
value nx_cpu_bitcast(value vd, value vs) {
  int d = nx_array_dtype(vd), s = nx_array_dtype(vs);
  if (nx_dtype_row_of(d).bits != nx_dtype_row_of(s).bits)
    return Val_int(NX_DTYPE);
  return walk_operands(vd, vs, d, s, NX_OP1_BITCAST, copy_walk);
}

value nx_cpu_cast(value vd, value vs) {
  int d = nx_array_dtype(vd), s = nx_array_dtype(vs);
  if (d == s) return nx_cpu_copy(vd, vs);
  return walk_operands(vd, vs, d, s, NX_OP1_CAST, cast_walk);
}
