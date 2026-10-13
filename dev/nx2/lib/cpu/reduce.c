/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reductions and scans of one operand by Sum, Prod, Max or Min, its
   reductions by Logsumexp, Arg and Moments, and its scan by Moments.

   Association. An output's terms are numbered t = 0, 1, … in C order of the
   reduced axes' indices. They fall into blocks of NX_CPU_LANE_BLOCK
   consecutive terms and, within a block, into NX_CPU_LANES lanes from the
   identity, term t into lane t mod 16. Lane i takes lane i + 8 for i < 8,
   then lane i + 4, i + 2 and i + 1: lane 0 is the block's value. The blocks
   combine by the binary tree whose left part holds the largest power of two
   of blocks below their count. A binary counter computes it in one pass: it
   combines the two newest values once per trailing zero of the number of
   blocks done, then what remains, newest first.

   A scan cuts each slice along its axis into chunks of SCAN_CHUNK terms from
   its start. A chunk's total is its terms reduced in the order above. The
   carry into chunk 0 is the identity, into chunk c + 1 the carry into c
   combined with c's total, and result i of chunk c is the carry combined
   with the chunk's terms up to i, left to right.

   Only float sums and products depend on this order: integers wrap and
   extremes are exact. Neither the layout, the path nor the thread count
   changes it. A thread takes a group of blocks, a power of two of them from
   a multiple of that power, and such a group's value is a subtree of the
   blocks' tree: the groups' values combine by the same counter.

   NaN. A float result that is NaN is replaced by its first NaN term in
   index order, found by a second walk taken only then, or by nx_kinds.h's
   NaN where no term is NaN. A scan's results stay NaN once they are, since
   adding, multiplying and the extremes keep a NaN: where a slice's last
   result is NaN, every result from its first NaN term on is that term, and
   every NaN result before it nx_kinds.h's.

   Paths. The operand and the destination, read at the operand's shape with
   step 0 along the reduced axes, coalesce: reduced and kept axes never
   merge, and the axes keep their order. A unit of work reduces W outputs:
   - one output, its runs of terms along the innermost reduced axis into
     lanes, where a reduced axis steps least through the operand;
   - W outputs along the kept axis that steps least, each term a row of W
     elements combined into one of 16 accumulator rows, elsewhere and where
     outputs have few terms. A 2x2 window's maximum takes this path.

   States. Logsumexp, Arg and Moments reduce by the same blocks, tree and
   paths, an output's value a state of its terms in place of a monoid's
   element: a kernel makes a block's state from its terms, contiguous in
   the operand or copied so, and merges two states as the tree combines
   values. A float result that is NaN is replaced as a monoid's is, but
   Arg's, whose state holds its first NaN term.

   Moments. A Moments scan computes nx_kernel.mli's network on each slice,
   op by op with the kinds' arithmetic, as Nx's expansion does, so that the
   two give the same bits. It holds O(log n) states per slice.

   The kernels come first, compiled once per target: as reduce.c for base
   and as its copy reduce_v3.c for v3. The entries follow, compiled once. */

#include "cpu.h"

#if !defined(NX_CPU_V3) || defined(__x86_64__)

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nx_kinds.h"

/* Kernels */

/* A monoid at a dtype is its identity, a lane step that adds terms
   into NX_CPU_LANES lanes by their index, and a combine that adds a row of
   terms into a row of accumulators, element by element. Integers compute in
   their own width, unsigned, so they wrap as a wider sum stored back would;
   booleans reduce as any and all of their bytes' being non-zero. A float sum
   or product uses the instruction: its NaN may differ from nx_kinds.h's, and
   the entries replace every NaN result. Maxima and minima are nx_kinds.h's,
   whose ties order -0 below +0. Contiguous terms take a loop of all 16
   lanes, which the compiler vectorises. */

#define BLOCK NX_CPU_LANE_BLOCK

/* The outputs whose lanes a few-term reduction holds side by side: 1 KiB of
   each lane. */
#define STRIP(T) (1024 / (int64_t)sizeof(T))

/* How far a scan's sequences run behind one another, in elements. */
#define LAG 16

/* A lane step and a combine of [F] over elements of [T]. */
#define MONOID(NAME, T, F)                                                     \
  static void lanes_##NAME(const uint8_t *x_, int64_t s, int64_t n,          \
                           uint8_t *l_, int first) {                        \
    const T *x = (const T *)x_;                                              \
    T *l = (T *)l_;                                                          \
    int64_t i = 0;                                                           \
    for (; i < n && ((first + i) & 15) != 0; i++)                            \
      l[(first + i) & 15] = F(l[(first + i) & 15], x[i * s]);                \
    if (s == 1 && n - i >= 16) {                                             \
      T v[16];                                                               \
      memcpy(v, l, sizeof v);                                                \
      for (; i + 16 <= n; i += 16)                                           \
        for (int k = 0; k < 16; k++) v[k] = F(v[k], x[i + k]);               \
      memcpy(l, v, sizeof v);                                                \
    }                                                                        \
    for (; i < n; i++) l[(first + i) & 15] = F(l[(first + i) & 15], x[i * s]); \
  }                                                                          \
  static void combine_##NAME(uint8_t *a_, const uint8_t *x_, int64_t s,      \
                             int64_t n) {                                    \
    T *a = (T *)a_;                                                          \
    const T *x = (const T *)x_;                                              \
    /* Steps of 2 and 4, a window's or a short row's, as constants the     \
       compiler loads interleaved. */                                        \
    if (s == 1)                                                              \
      for (int64_t i = 0; i < n; i++) a[i] = F(a[i], x[i]);                  \
    else if (s == 2)                                                         \
      for (int64_t i = 0; i < n; i++) a[i] = F(a[i], x[2 * i]);              \
    else if (s == 4)                                                         \
      for (int64_t i = 0; i < n; i++) a[i] = F(a[i], x[4 * i]);              \
    else                                                                     \
      for (int64_t i = 0; i < n; i++) a[i] = F(a[i], x[i * s]);              \
  }                                                                          \
  static void scan_##NAME(uint8_t *a_, const uint8_t *x_, int64_t s,         \
                          int64_t xs, uint8_t *y_, int64_t sy, int64_t ys,   \
                          int64_t n, int k) {                                \
    const T *x = (const T *)x_;                                              \
    T *y = (T *)y_, a[4];                                                    \
    memcpy(a, a_, k * sizeof(T));                                            \
    /* Sequence j runs LAG·j elements behind sequence 0: sequences a       \
       multiple of 4 KiB apart would otherwise touch addresses x86 takes   \
       for one another's (4K aliasing). */                                   \
    int64_t i = 0;                                                           \
    if (k == 4 && n > 3 * LAG) {                                             \
      for (int j = 0; j < 4; j++)                                            \
        for (int64_t m = 0; m < (3 - j) * LAG; m++)                          \
          y[j * ys + m * sy] = a[j] = F(a[j], x[j * xs + m * s]);            \
      /* Four named accumulators, so that each stays in a register. */     \
      T a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];                          \
      const T *x1 = x + xs, *x2 = x + 2 * xs, *x3 = x + 3 * xs;              \
      T *y1 = y + ys, *y2 = y + 2 * ys, *y3 = y + 3 * ys;                    \
      for (i = 3 * LAG; i < n; i++) {                                        \
        y[i * sy] = a0 = F(a0, x[i * s]);                                    \
        y1[(i - LAG) * sy] = a1 = F(a1, x1[(i - LAG) * s]);                  \
        y2[(i - 2 * LAG) * sy] = a2 = F(a2, x2[(i - 2 * LAG) * s]);          \
        y3[(i - 3 * LAG) * sy] = a3 = F(a3, x3[(i - 3 * LAG) * s]);          \
      }                                                                      \
      a[0] = a0;                                                             \
      a[1] = a1;                                                             \
      a[2] = a2;                                                             \
      a[3] = a3;                                                             \
      for (int j = 1; j < 4; j++)                                            \
        for (int64_t m = n - j * LAG; m < n; m++)                            \
          y[j * ys + m * sy] = a[j] = F(a[j], x[j * xs + m * s]);            \
    } else                                                                   \
      for (int j = 0; j < k; j++)                                            \
        for (i = 0; i < n; i++)                                              \
          y[j * ys + i * sy] = a[j] = F(a[j], x[j * xs + i * s]);            \
    memcpy(a_, a, k * sizeof(T));                                            \
  }                                                                          \
  /* Four blocks' lanes at once, so that more additions are in flight. */ \
  static inline void four_##NAME(const T *x, T e, T *v) {             \
    T l[4][16];                                                              \
    for (int j = 0; j < 4; j++)                                              \
      for (int k = 0; k < 16; k++) l[j][k] = e;                              \
    for (int i = 0; i < BLOCK; i += 16)                                      \
      for (int j = 0; j < 4; j++)                                            \
        for (int k = 0; k < 16; k++)                                         \
          l[j][k] = F(l[j][k], x[j * BLOCK + i + k]);                        \
    for (int j = 0; j < 4; j++) {                                            \
      for (int h = 8; h >= 1; h /= 2)                                        \
        for (int k = 0; k < h; k++) l[j][k] = F(l[j][k], l[j][k + h]);      \
      v[j] = l[j][0];                                                        \
    }                                                                        \
  }                                                                          \
  static inline void one_##NAME(const T *x, T e, T *v) {                 \
    T l[16];                                                                 \
    for (int k = 0; k < 16; k++) l[k] = e;                                   \
    for (int i = 0; i < BLOCK; i += 16)                                      \
      for (int k = 0; k < 16; k++) l[k] = F(l[k], x[i + k]);                 \
    for (int h = 8; h >= 1; h /= 2)                                          \
      for (int k = 0; k < h; k++) l[k] = F(l[k], l[k + h]);                  \
    *v = l[0];                                                               \
  }                                                                          \
  static void blocks_##NAME(const uint8_t *x_, int64_t n, const uint8_t *e_, \
                            uint8_t *v_) {                                   \
    const T *x = (const T *)x_;                                              \
    T *v = (T *)v_, e;                                                       \
    memcpy(&e, e_, sizeof e);                                                \
    int64_t b = 0;                                                           \
    for (; b + 4 <= n; b += 4) four_##NAME(x + b * BLOCK, e, v + b);         \
    for (; b < n; b++) one_##NAME(x + b * BLOCK, e, v + b);                  \
  }                                                                          \
  static void column_##NAME(uint8_t *a_, const uint8_t *x_, int64_t st,      \
                            int64_t n, int64_t s, int64_t w) {               \
    T *a = (T *)a_;                                                          \
    const T *x = (const T *)x_;                                              \
    int64_t j = 0;                                                           \
    if (s == 1)                                                              \
      for (; j + 32 <= w; j += 32) {                                         \
        T v[32];                                                             \
        memcpy(v, a + j, sizeof v);                                          \
        for (int64_t i = 0; i < n; i++)                                      \
          for (int k = 0; k < 32; k++) v[k] = F(v[k], x[i * st + j + k]);    \
        memcpy(a + j, v, sizeof v);                                          \
      }                                                                      \
    for (; j < w; j++) {                                                     \
      T v = a[j];                                                            \
      for (int64_t i = 0; i < n; i++) v = F(v, x[i * st + j * s]);           \
      a[j] = v;                                                              \
    }                                                                        \
  }                                                                          \
  /* Sixteen lanes' values [a] combined with [b]'s. */                      \
  static inline void pair_##NAME(T *restrict a, const T *restrict b) {      \
    for (int i = 0; i < 16; i++) a[i] = F(a[i], b[i]);                       \
  }                                                                          \
  static void few_##NAME(const uint8_t *x_, const int64_t *off, int n,       \
                         int64_t s, int64_t w, int64_t s2, int64_t h,        \
                         const uint8_t *e_, uint8_t *y_, int64_t sy,         \
                         int64_t sy2) {                                      \
    const T *x = (const T *)x_;                                              \
    T *y = (T *)y_, e, l[16][STRIP(T)];                                      \
    memcpy(&e, e_, sizeof e);                                                \
    int64_t cap = STRIP(T);                                                  \
    /* Each term's values of a strip of outputs are copied side by side,   \
       as many whole rows as fit or a piece of one, and padded with the    \
       identity to whole vectors: the lanes and their tree then run over   \
       vectors of outputs however short the rows. Loops of whole vectors,  \
       since gcc at -O2 leaves a loop of any other count scalar. */         \
    for (int64_t r = 0; r < h;) {                                            \
      int64_t rows = w <= cap ? (h - r < cap / w ? h - r : cap / w) : 1;     \
      for (int64_t c = 0; c < w; c += cap) {                                 \
        int64_t m = w - c < cap ? w - c : cap, k = rows * m;                 \
        int64_t k16 = (k + 15) & ~(int64_t)15;                               \
        for (int t = 0; t < n; t++) {                                        \
          T *d = l[t];                                                       \
          for (int64_t q = 0; q < rows; q++, d += m) {                       \
            const T *a = x + (r + q) * s2 + c * s + off[t];                  \
            if (s == 1) memcpy(d, a, m * sizeof(T));                         \
            else                                                             \
              for (int64_t j = 0; j < m; j++) d[j] = a[j * s];               \
          }                                                                  \
          for (int64_t i = k; i < k16; i++) l[t][i] = e;                     \
          for (int64_t i = 0; i < k16; i += 16)                              \
            for (int j = 0; j < 16; j++) l[t][i + j] = F(e, l[t][i + j]);    \
        }                                                                    \
        for (int g = 8; g >= 1; g /= 2)                                      \
          for (int t = 0; t < g && t + g < n; t++)                           \
            for (int64_t i = 0; i < k16; i += 16)                            \
              pair_##NAME(l[t] + i, l[t + g] + i);                           \
        for (int64_t q = 0; q < rows; q++) {                                 \
          T *d = y + (r + q) * sy2 + c * sy;                                 \
          if (sy == 1) memcpy(d, l[0] + q * m, m * sizeof(T));               \
          else                                                               \
            for (int64_t j = 0; j < m; j++) d[j * sy] = l[0][q * m + j];     \
        }                                                                    \
      }                                                                      \
      r += rows;                                                             \
    }                                                                        \
  }

#define ADD(a, b) ((a) + (b))
#define MUL(a, b) ((a) * (b))
#define ANY(a, b) ((uint8_t)((a) | ((b) != 0)))
#define ALL(a, b) ((uint8_t)((a) & ((b) != 0)))

/* The four monoids of an integer dtype of storage [T], unsigned [U], at the
   compute suffix [S] for its extremes, [C] its compute type and [M] the
   unsigned type of [C]'s width. Sums and products run in [M], where they
   wrap: 8- and 16-bit operands would promote to int, whose product of two
   0xFFFF overflows. */
#define INTS(D, T, U, C, M, S)                                               \
  static inline T add_##D(T a, T b) { return (T)(U)((M)(U)a + (M)(U)b); }    \
  static inline T mul_##D(T a, T b) { return (T)(U)((M)(U)a * (M)(U)b); }    \
  static inline T max_##D(T a, T b) { return (T)nx_maximum_##S((C)a, (C)b); } \
  static inline T min_##D(T a, T b) { return (T)nx_minimum_##S((C)a, (C)b); } \
  MONOID(sum_##D, T, add_##D)                                                  \
  MONOID(prod_##D, T, mul_##D)                                                 \
  MONOID(max_##D, T, max_##D)                                                  \
  MONOID(min_##D, T, min_##D)

INTS(i8, int8_t, uint8_t, int32_t, uint32_t, i32)
INTS(u8, uint8_t, uint8_t, uint32_t, uint32_t, u32)
INTS(i16, int16_t, uint16_t, int32_t, uint32_t, i32)
INTS(u16, uint16_t, uint16_t, uint32_t, uint32_t, u32)
INTS(i32, int32_t, uint32_t, int32_t, uint32_t, i32)
INTS(u32, uint32_t, uint32_t, uint32_t, uint32_t, u32)
INTS(i64, int64_t, uint64_t, int64_t, uint64_t, i64)
INTS(u64, uint64_t, uint64_t, uint64_t, uint64_t, u64)

MONOID(sum_f32, float, ADD)
MONOID(prod_f32, float, MUL)
MONOID(max_f32, float, nx_maximum_f32)
MONOID(min_f32, float, nx_minimum_f32)
MONOID(sum_f64, double, ADD)
MONOID(prod_f64, double, MUL)
MONOID(max_f64, double, nx_maximum_f64)
MONOID(min_f64, double, nx_minimum_f64)
MONOID(max_b, uint8_t, ANY)
MONOID(min_b, uint8_t, ALL)

/* Moments scans. A unit scans [w] slices side by side by the network of
   nx_kernel.mli, a state rows of [w] lanes: its mu, S and n, and in the
   exact pass z, whether one of its terms is a NaN, 0 or 1.

   The terms enter in order, term t at index t of level 0. A state at an
   odd index joins the one before it, and the pair enters the next level at
   half its index. Where its index is even, the level's result is the state
   itself at index 0, else the next level's last result joined with it; a
   pair's result is the next level's. So a level holds the even state that
   awaits its partner and its last result, and the scratch three states per
   level: the waiting one, the arriving one and the joined one.

     t        0    1    2        3     4
     level 0  s0   s1   s2       s3    s4
     level 1       p0            p1
     level 2                     q0
     y        s0   p0   p0.s2    q0    q0.s4

   where p0 is s0.s1, p1 s2.s3, q0 p0.p1 and a.b the join of a and b.

   The fast pass computes by plain arithmetic, which gives the kinds' bits
   wherever no value is a NaN or an infinity. Where one of its results is
   not finite, the exact pass computes the unit again with the kinds' ops,
   one kind per loop: gcc leaves scalar a loop that chains two of their
   NaN selects. */

/* Whether [v] is finite: an infinity or a NaN less itself is a NaN. */
#define FINITE(v) ((v) - (v) == 0)

#define LANES_OF(D, E)                                                        \
  for (int64_t l = 0; l < w; l++) (D)[l] = E

/* A pass's arithmetic. */
typedef enum { FAST, EXACT } arith;

#define MOMENTS(T, S)                                                         \
  /* State [b] after state [a], into [r], each [w] elements a row. */       \
  static void fast_##S(int64_t w, const T *restrict a, const T *restrict b,  \
                       T *restrict r) {                                       \
    for (int64_t l = 0; l < w; l++) {                                         \
      T ma = a[l], va = a[w + l], na = a[2 * w + l];                          \
      T mb = b[l], vb = b[w + l], nb = b[2 * w + l];                          \
      T k = na + nb, d = mb - ma, f = nb / k;                                 \
      r[l] = ma + d * f;                                                      \
      r[w + l] = (va + vb) + (d * d) * (na * f);                              \
      r[2 * w + l] = k;                                                       \
    }                                                                        \
  }                                                                          \
  static void exact_##S(int64_t w, const T *restrict a, const T *restrict b, \
                        T *restrict r) {                                      \
    const T *ma = a, *va = a + w, *na = a + 2 * w, *za = a + 3 * w;           \
    const T *mb = b, *vb = b + w, *nb = b + 2 * w, *zb = b + 3 * w;           \
    T *k = r + 2 * w, d[NX_CPU_MOMENTS_LANES], f[NX_CPU_MOMENTS_LANES],       \
      t[NX_CPU_MOMENTS_LANES], sum[NX_CPU_MOMENTS_LANES];                     \
    LANES_OF(k, nx_add_##S(na[l], nb[l]));                                    \
    LANES_OF(d, nx_sub_##S(mb[l], ma[l]));                                    \
    LANES_OF(f, nx_fdiv_##S(nb[l], k[l]));                                    \
    LANES_OF(t, nx_mul_##S(d[l], f[l]));                                      \
    LANES_OF(t, nx_add_##S(ma[l], t[l]));                                     \
    LANES_OF(sum, nx_add_##S(ma[l], mb[l]));                                  \
    LANES_OF(r, (FINITE(ma[l]) & FINITE(mb[l]))                               \
                    ? t[l]                                                    \
                    : za[l] != 0 ? ma[l] : zb[l] != 0 ? mb[l] : sum[l]);      \
    LANES_OF(f, nx_mul_##S(na[l], f[l]));                                     \
    LANES_OF(d, nx_mul_##S(d[l], d[l]));                                      \
    LANES_OF(d, nx_mul_##S(d[l], f[l]));                                      \
    LANES_OF(t, nx_add_##S(va[l], vb[l]));                                    \
    LANES_OF(r + w, nx_add_##S(t[l], d[l]));                                  \
    LANES_OF(r + 3 * w, (T)((za[l] != 0) | (zb[l] != 0)));                    \
  }                                                                          \
  /* The states of term [x] of each lane, [xl] apart, shifted by [c]. */    \
  static void leaf_##S(arith a, int64_t w, const T *restrict x, int64_t xl,   \
                       const T *restrict c, T *restrict r) {                  \
    if (a == EXACT) LANES_OF(r, nx_sub_##S(x[l * xl], c[l]));                 \
    else LANES_OF(r, x[l * xl] - c[l]);                                       \
    LANES_OF(r + w, 0);                                                       \
    LANES_OF(r + 2 * w, 1);                                                   \
    if (a == EXACT) LANES_OF(r + 3 * w, (T)(x[l * xl] != x[l * xl]));         \
  }                                                                          \
  /* One pass: 0, or 1 where the fast pass met a value not finite. */       \
  static int pass_##S(arith a, const T *x, int64_t xs, int64_t xl, T *ym,     \
                      int64_t ms, int64_t ml, T *yv, int64_t vs, int64_t vl,  \
                      int64_t n, int64_t w, T *scratch) {                     \
    void (*join)(int64_t, const T *, const T *, T *) =                        \
        a == EXACT ? exact_##S : fast_##S;                                    \
    int64_t size = 4 * w;                                                     \
    int bad = 0, levels = nx_cpu_moments_levels(n);                           \
    T c[NX_CPU_MOMENTS_LANES];                                                \
    LANES_OF(c, FINITE(x[l * xl]) ? x[l * xl] : 0);                           \
    /* Level k's waiting, arriving and joined states, and its result. */    \
    T *held[64], *in[64], *joined[64];                                        \
    const T *got[64];                                                         \
    for (int k = 0; k < levels; k++) {                                        \
      held[k] = scratch + 3 * k * size;                                       \
      in[k] = held[k] + size;                                                 \
      joined[k] = in[k] + size;                                               \
    }                                                                        \
    for (int64_t t = 0; t < n; t++) {                                         \
      leaf_##S(a, w, x + t * xs, xl, c, in[0]);                               \
      int k = 0;                                                              \
      int64_t j = t;                                                          \
      for (; j & 1; k++, j >>= 1) join(w, held[k], in[k], in[k + 1]);         \
      if (j == 0) got[k] = in[k];                                             \
      else {                                                                  \
        join(w, got[k + 1], in[k], joined[k]);                                \
        got[k] = joined[k];                                                   \
      }                                                                      \
      T *e = held[k];                                                         \
      held[k] = in[k];                                                        \
      in[k] = e;                                                              \
      for (int i = 0; i < k; i++) got[i] = got[k];                            \
      /* The mean c + mu, the variance S over the count, or mu less        \
         itself where mu is not finite. */                                  \
      const T *r = got[0];                                                    \
      T m = (T)t + 1, *dm = ym + t * ms, *dv = yv + t * vs;                   \
      if (a == EXACT) {                                                       \
        T mean[NX_CPU_MOMENTS_LANES], var[NX_CPU_MOMENTS_LANES],              \
            nan[NX_CPU_MOMENTS_LANES];                                        \
        LANES_OF(mean, nx_add_##S(c[l], r[l]));                               \
        LANES_OF(var, nx_fdiv_##S(r[w + l], m));                              \
        LANES_OF(nan, nx_sub_##S(r[l], r[l]));                                \
        for (int64_t l = 0; l < w; l++) {                                     \
          dm[l * ml] = mean[l];                                               \
          dv[l * vl] = FINITE(r[l]) ? var[l] : nan[l];                        \
        }                                                                    \
        continue;                                                             \
      }                                                                      \
      for (int64_t l = 0; l < w; l++) {                                       \
        T mean = c[l] + r[l], var = r[w + l] / m;                             \
        bad |= !(FINITE(mean) & FINITE(var));                                 \
        dm[l * ml] = mean;                                                    \
        dv[l * vl] = var;                                                     \
      }                                                                      \
    }                                                                        \
    return bad;                                                               \
  }                                                                          \
  static void moments_##S(const uint8_t *x, int64_t xs, int64_t xl,          \
                          uint8_t *mean, int64_t ms, int64_t ml,              \
                          uint8_t *var, int64_t vs, int64_t vl, int64_t n,    \
                          int64_t w, uint8_t *scratch) {                      \
    if (pass_##S(FAST, (const T *)x, xs, xl, (T *)mean, ms, ml, (T *)var, vs, \
                 vl, n, w, (T *)scratch))                                     \
      pass_##S(EXACT, (const T *)x, xs, xl, (T *)mean, ms, ml, (T *)var, vs,  \
               vl, n, w, (T *)scratch);                                       \
  }

MOMENTS(float, f32)
MOMENTS(double, f64)

/* Arg, Logsumexp and Moments reductions: a block's state, two merged, an
   output's results.
   - Arg: the block's extreme by lanes, then its first term of the
     extreme's bits, or its first NaN where it has one. The later of two
     states takes over where the extreme of their two is not the
     earlier's, so a tie keeps the earlier position.
   - Logsumexp: first the terms' maximum m by lanes, merged by the
     maximum, and the center c, m where finite, else 0; then exp (x - c)
     summed in lane order, merged by addition, and c + log of the sum.
   - Moments: first the terms summed in lane order, merged by addition,
     and the center, their mean c; then x - c and (x - c)² summed so, and
     the variance, (Σ (x - c)² - (Σ (x - c))² / n) / n, +0 where that is
     negative, or Σ (x - c)² where that is +inf: Nx's expansion's
     arithmetic.
   A NaN result is replaced by the driver: by the first NaN term, or by
   nx_kinds.h's NaN where none is. Every function is inlined into each
   kernel, whose loops of 16 lanes the compiler then vectorises. */

#define KERNEL static __attribute__((flatten)) void

/* Arg by the extreme [F] over terms of [T], [U] the unsigned integer of
   its width: [BEAT] is [>] or [<], [F] without its order of NaNs and
   zeros, which the plain compare vectorises; [NANQ] tests a term for a
   NaN. */
#define ARG(NAME, T, U, F, BEAT, NANQ)                                       \
  /* The extreme by BEAT of the [n] terms at [x], or a NaN where one is:     \
     a lane takes each NaN it meets and keeps the last. */                   \
  static inline T top_##NAME(const T *x, int64_t n) {                        \
    T l[16];                                                                 \
    for (int k = 0; k < 16; k++) l[k] = x[0];                                \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16)                                             \
      for (int k = 0; k < 16; k++) {                                         \
        T v = x[i + k];                                                      \
        l[k] = v BEAT l[k] || NANQ(v) ? v : l[k];                            \
      }                                                                      \
    for (; i < n; i++) l[0] = x[i] BEAT l[0] || NANQ(x[i]) ? x[i] : l[0];    \
    for (int k = 1; k < 16; k++)                                             \
      l[0] = l[k] BEAT l[0] || NANQ(l[k]) ? l[k] : l[0];                     \
    return l[0];                                                             \
  }                                                                          \
  /* The first of the [n] terms at [x] of the bits of [e], or [n]: their     \
     bits compare as integers, 16 at a time until one is equal. */           \
  static inline U bits_##NAME(const T *x) {                                  \
    U u;                                                                     \
    memcpy(&u, x, sizeof u);                                                 \
    return u;                                                                \
  }                                                                          \
  static inline int64_t find_##NAME(const T *x, int64_t n, T e) {            \
    U b = bits_##NAME(&e);                                                   \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) {                                           \
      U hit = 0;                                                             \
      for (int k = 0; k < 16; k++) hit |= bits_##NAME(&x[i + k]) == b;       \
      if (hit) break;                                                        \
    }                                                                        \
    while (i < n && bits_##NAME(&x[i]) != b) i++;                            \
    return i;                                                                \
  }                                                                          \
  /* The first NaN, or the first term of the extreme's bits: of F's zero     \
     where the extreme is a zero and one is. */                              \
  KERNEL arg_##NAME(const uint8_t *x_, int64_t n, int64_t t0, double c,      \
                    nx_cpu_state *st) {                                      \
    (void)c;                                                                 \
    const T *x = (const T *)x_;                                              \
    T e = top_##NAME(x, n), z = F((T)0, (T)(-(T)0));                         \
    int64_t i = 0;                                                           \
    if (NANQ(e))                                                             \
      while (!NANQ(x[i])) i++;                                               \
    else if (e != 0)                                                         \
      i = find_##NAME(x, n, e);                                              \
    else if ((i = find_##NAME(x, n, z)) == n)                                \
      i = find_##NAME(x, n, (T)(-z));                                        \
    *st = (nx_cpu_state){.at = t0 + i};                                      \
    memcpy(&st->v, &x[i], sizeof(T));                                        \
  }                                                                          \
  static void arg_merge_##NAME(nx_cpu_state *a, const nx_cpu_state *b) {     \
    T va, vb;                                                                \
    memcpy(&va, &a->v, sizeof va);                                           \
    memcpy(&vb, &b->v, sizeof vb);                                           \
    T e = F(va, vb);                                                         \
    if (memcmp(&e, &va, sizeof e) != 0) *a = *b;                             \
  }                                                                          \
  static void arg_finish_##NAME(const nx_cpu_state *st, double c,            \
                                uint8_t *d0, uint8_t *d1) {                  \
    (void)c;                                                                 \
    memcpy(d0, &st->v, sizeof(T));                                           \
    memcpy(d1, &st->at, sizeof st->at);                                      \
  }

#define IS_NAN(v) ((v) != (v))
#define NO_NAN(v) 0

#define ARGS(D, T, U, F_MAX, F_MIN, NANQ)                                    \
  ARG(max_##D, T, U, F_MAX, >, NANQ)                                         \
  ARG(min_##D, T, U, F_MIN, <, NANQ)

ARGS(i8, int8_t, uint8_t, max_i8, min_i8, NO_NAN)
ARGS(u8, uint8_t, uint8_t, max_u8, min_u8, NO_NAN)
ARGS(i16, int16_t, uint16_t, max_i16, min_i16, NO_NAN)
ARGS(u16, uint16_t, uint16_t, max_u16, min_u16, NO_NAN)
ARGS(i32, int32_t, uint32_t, max_i32, min_i32, NO_NAN)
ARGS(u32, uint32_t, uint32_t, max_u32, min_u32, NO_NAN)
ARGS(i64, int64_t, uint64_t, max_i64, min_i64, NO_NAN)
ARGS(u64, uint64_t, uint64_t, max_u64, min_u64, NO_NAN)
ARGS(f32, float, uint32_t, nx_maximum_f32, nx_minimum_f32, IS_NAN)
ARGS(f64, double, uint64_t, nx_maximum_f64, nx_minimum_f64, IS_NAN)

/* Logsumexp and Moments at the float [T] of suffix [S], [MADE] the NaN
   nx_kinds.h makes from numbers, [CUT] the least integer whose exp is a
   normal [T]. */
#define STATES(T, S, MADE, CUT)                                              \
  /* exp [y], or 0 below CUT, computed at CUT: no lane makes a subnormal,    \
     which costs x86 a hundred cycles. */                                    \
  static inline T ex_##S(T y) {                                              \
    T e = nx_exp_##S(y < CUT ? CUT : y);                                     \
    return y < CUT ? 0 : e;                                                  \
  }                                                                          \
  KERNEL top_##S(const uint8_t *x, int64_t n, int64_t t0, double c,          \
                 nx_cpu_state *st) {                                         \
    (void)t0;                                                                \
    (void)c;                                                                 \
    *st = (nx_cpu_state){.s = top_max_##S((const T *)x, n)};                 \
  }                                                                          \
  static void top_merge_##S(nx_cpu_state *a, const nx_cpu_state *b) {        \
    T ma = (T)a->s, mb = (T)b->s;                                            \
    a->s = ma != ma ? ma : mb != mb ? mb : ma < mb ? mb : ma;                \
  }                                                                          \
  static void top_finish_##S(const nx_cpu_state *st, double c, uint8_t *d0,  \
                             uint8_t *d1) {                                  \
    (void)c;                                                                 \
    (void)d1;                                                                \
    double m = FINITE(st->s) ? st->s : 0;                                    \
    memcpy(d0, &m, sizeof m);                                                \
  }                                                                          \
  KERNEL lse_##S(const uint8_t *x_, int64_t n, int64_t t0, double c_,        \
                 nx_cpu_state *st) {                                         \
    (void)t0;                                                                \
    const T *x = (const T *)x_;                                              \
    T c = (T)c_, l[16] = {0};                                                \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16)                                             \
      for (int k = 0; k < 16; k++) l[k] += ex_##S(x[i + k] - c);             \
    for (; i < n; i++) l[i & 15] += ex_##S(x[i] - c);                        \
    for (int h = 8; h >= 1; h /= 2)                                          \
      for (int k = 0; k < h; k++) l[k] += l[k + h];                          \
    *st = (nx_cpu_state){.s = l[0]};                                         \
  }                                                                          \
  static void lse_merge_##S(nx_cpu_state *a, const nx_cpu_state *b) {        \
    a->s = (T)a->s + (T)b->s;                                                \
  }                                                                          \
  static void lse_finish_##S(const nx_cpu_state *st, double c, uint8_t *d0,  \
                             uint8_t *d1) {                                  \
    (void)d1;                                                                \
    T r = (T)c + nx_log_##S((T)st->s);                                       \
    memcpy(d0, &r, sizeof r);                                                \
  }                                                                          \
  /* The lanes' tree over lanes 0 to 7 [a] and 8 to 15 [b]: two arrays,      \
     which the compiler holds in registers, where one of 16 doubles it       \
     keeps in memory. */                                                     \
  static inline T halves_##S(T *a, const T *b) {                             \
    for (int k = 0; k < 8; k++) a[k] += b[k];                                \
    for (int h = 4; h >= 1; h /= 2)                                          \
      for (int k = 0; k < h; k++) a[k] += a[k + h];                          \
    return a[0];                                                             \
  }                                                                          \
  KERNEL mean_##S(const uint8_t *x_, int64_t n, int64_t t0, double c,        \
                  nx_cpu_state *st) {                                        \
    (void)t0;                                                                \
    (void)c;                                                                 \
    const T *x = (const T *)x_;                                              \
    T a[8] = {0}, b[8] = {0};                                                \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16)                                             \
      for (int k = 0; k < 8; k++) {                                          \
        a[k] += x[i + k];                                                    \
        b[k] += x[i + 8 + k];                                                \
      }                                                                      \
    for (; i < n; i++) (i & 8 ? b : a)[i & 7] += x[i];                       \
    *st = (nx_cpu_state){.n = n, .s = halves_##S(a, b)};                     \
  }                                                                          \
  static void sum_merge_##S(nx_cpu_state *a, const nx_cpu_state *b) {        \
    a->n += b->n;                                                            \
    a->s = (T)a->s + (T)b->s;                                                \
  }                                                                          \
  /* The count as the dtype rounds it, as Nx's expansion divides by it. */   \
  static void mean_finish_##S(const nx_cpu_state *st, double c,              \
                              uint8_t *d0, uint8_t *d1) {                    \
    (void)c;                                                                 \
    (void)d1;                                                                \
    double m = (T)st->s / (T)st->n;                                          \
    memcpy(d0, &m, sizeof m);                                                \
  }                                                                          \
  KERNEL var_##S(const uint8_t *x_, int64_t n, int64_t t0, double c_,        \
                 nx_cpu_state *st) {                                         \
    (void)t0;                                                                \
    const T *x = (const T *)x_;                                              \
    T c = (T)c_, a[8] = {0}, b[8] = {0}, p[8] = {0}, q[8] = {0};             \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16)                                             \
      for (int k = 0; k < 8; k++) {                                          \
        T d = x[i + k] - c, e = x[i + 8 + k] - c;                            \
        a[k] += d * d;                                                       \
        b[k] += e * e;                                                       \
        p[k] += d;                                                           \
        q[k] += e;                                                           \
      }                                                                      \
    for (; i < n; i++) {                                                     \
      T d = x[i] - c;                                                        \
      (i & 8 ? b : a)[i & 7] += d * d;                                       \
      (i & 8 ? q : p)[i & 7] += d;                                           \
    }                                                                        \
    *st = (nx_cpu_state){.n = n, .s = halves_##S(a, b),                      \
                         .d = halves_##S(p, q)};                             \
  }                                                                          \
  static void var_merge_##S(nx_cpu_state *a, const nx_cpu_state *b) {        \
    sum_merge_##S(a, b);                                                     \
    a->d = (T)a->d + (T)b->d;                                                \
  }                                                                          \
  /* A mean that is a NaN is the variance too. Squares summed to +inf are  \
     the variance, where d * d may overflow too and v be a NaN. */          \
  static void var_finish_##S(const nx_cpu_state *st, double c, uint8_t *d0,  \
                             uint8_t *d1) {                                  \
    T mean = MADE, var = MADE;                                               \
    if (st->n > 0) {                                                         \
      T n = (T)st->n, d = (T)st->d, s = (T)st->s;                            \
      T v = (s - d * d / n) / n;                                             \
      mean = (T)c;                                                           \
      var = c != c ? mean : s == (T)INFINITY ? s : v < 0 ? 0 : v;            \
    }                                                                        \
    memcpy(d0, &mean, sizeof mean);                                          \
    memcpy(d1, &var, sizeof var);                                            \
  }

STATES(float, f32, NX_NAN_F32, -87)
STATES(double, f64, NX_NAN_F64, -708)

/* The table's monoids, by monoid then dtype: NULL where declined. */
#define SET(M, DT, F)                                                        \
  t->monoid[M][DT] =                                                           \
      (nx_cpu_monoid){lanes_##F, combine_##F, scan_##F, blocks_##F, few_##F,  \
                    column_##F}

#define ALL4(DT, D)                                                          \
  SET(NX_MONOID_SUM, DT, sum_##D);                                            \
  SET(NX_MONOID_PROD, DT, prod_##D);                                          \
  SET(NX_MONOID_MAX, DT, max_##D);                                            \
  SET(NX_MONOID_MIN, DT, min_##D)

/* The table's stateful reductions, by reduction then dtype: a pass is the
   kernels K, M and F at the suffix D. */
#define PASS(K, M, F, D) ((nx_cpu_pass){K##_##D, M##_##D, F##_##D})
#define NONE ((nx_cpu_pass){NULL, NULL, NULL})

#define STATE(R, DT, CENTER, VALUE)                                          \
  t->stateful[R - NX_CPU_MONOIDS][DT] = (nx_cpu_stateful){CENTER, VALUE}

#define ARG2(DT, D)                                                          \
  STATE(NX_CPU_ARG_MAX, DT, NONE, PASS(arg, arg_merge, arg_finish, max_##D)); \
  STATE(NX_CPU_ARG_MIN, DT, NONE, PASS(arg, arg_merge, arg_finish, min_##D))

#define FLOATS(DT, D)                                                        \
  STATE(NX_CPU_LOGSUMEXP, DT, PASS(top, top_merge, top_finish, D),           \
        PASS(lse, lse_merge, lse_finish, D));                                \
  STATE(NX_CPU_MOMENTS, DT, PASS(mean, sum_merge, mean_finish, D),           \
        PASS(var, var_merge, var_finish, D))

static void set(nx_cpu_target *t) {
  ARG2(NX_FLOAT32, f32);
  ARG2(NX_FLOAT64, f64);
  ARG2(NX_INT8, i8);
  ARG2(NX_UINT8, u8);
  ARG2(NX_INT16, i16);
  ARG2(NX_UINT16, u16);
  ARG2(NX_INT32, i32);
  ARG2(NX_UINT32, u32);
  ARG2(NX_INT64, i64);
  ARG2(NX_UINT64, u64);
  FLOATS(NX_FLOAT32, f32);
  FLOATS(NX_FLOAT64, f64);
  ALL4(NX_FLOAT32, f32);
  ALL4(NX_FLOAT64, f64);
  ALL4(NX_INT8, i8);
  ALL4(NX_UINT8, u8);
  ALL4(NX_INT16, i16);
  ALL4(NX_UINT16, u16);
  ALL4(NX_INT32, i32);
  ALL4(NX_UINT32, u32);
  ALL4(NX_INT64, i64);
  ALL4(NX_UINT64, u64);
  SET(NX_MONOID_MAX, NX_BOOL, max_b);
  SET(NX_MONOID_MIN, NX_BOOL, min_b);
  t->moments[NX_FLOAT32] = moments_f32;
  t->moments[NX_FLOAT64] = moments_f64;
}

#if defined(NX_CPU_V3)
void nx_cpu_set_reduce_v3(nx_cpu_target *t) { set(t); }
#else
void nx_cpu_set_reduce_base(nx_cpu_target *t) { set(t); }
#endif

#endif

#if !defined(NX_CPU_V3)

#include <caml/fail.h>
#include <caml/mlvalues.h>


#define LANES NX_CPU_LANES
#define SCAN_CHUNK 4096

/* A streaming unit's row of outputs: 512 bytes, 128 float32. Byte buffers
   hold elements of the dtype, read and written as such, so they are aligned
   as the widest. */
#define ROW 512

/* The depth of the blocks' tree: 2^48 blocks exceed any array. */
#define LEVELS 48

/* Fewer terms per output take the streaming path. */
#define FEW 16

/* A scan's slices a unit runs side by side where its axis steps least. */
#define SLICES 4

/* Full blocks whose values the table computes in one call. */
#define RUN 64

/* Units the groups of blocks aim for where outputs are few. */
#define UNITS 64

/* A reduction of a lane of terms or fewer an output copies each term into its
   strip, then runs the identity and the lanes' tree there: about ten times
   memcpy's time for its operand (kimchi, one core: 180 us for the 1.35 MB
   of 2x2 windows, which memcpy moves in 19 us). Its jobs are priced so. */
#define FEW_COST 10

/* A term's time against memcpy's, by stateful reduction: Logsumexp's
   exp, Arg's and Moments' two walks of a block. */
static const double STATE_COST[NX_CPU_STATEFUL] = {8, 2, 2, 3};

/* What a call computes: a reduction, a scan, or a Moments scan. */
typedef enum { REDUCE, SCAN, MOMENTS } task;

/* An output's value is an element of the dtype under a monoid, a state
   under a stateful reduction. */
typedef struct {
  const nx_cpu_monoid *f;   /* a monoid's kernels, or NULL */
  const nx_cpu_pass *r;     /* a stateful reduction's pass, or NULL */
  double *centers;          /* its outputs' centers, a row of them a unit */
  int centering;            /* the pass stores the centers */
  int dtype, w, nan;        /* nan: NaN results take their first NaN term */
  int64_t size;             /* the bytes of an output's value */
  _Alignas(8) uint8_t id[8];
  /* The destinations, the second the first again for one, and the operand
     as loaded. */
  const nx_array *dst, *dst1, *src;
  /* The reduced axes in C order: extents and the operand's steps. */
  int nr;
  int64_t re[NX_MAX_RANK], rs[NX_MAX_RANK];
  /* The kept axes, the row axis last: extents, the operand's and the
     destinations' steps. */
  int nk;
  int64_t ke[NX_MAX_RANK], ks[NX_MAX_RANK], kd[NX_MAX_RANK], kd1[NX_MAX_RANK];
  int64_t x0, d0, d1;
  int64_t rd; /* a scan's destination step along its axis */
  int slices; /* a unit's outputs are slices scanned side by side */
  int64_t terms, blocks;
  int64_t row;    /* outputs per unit along the row axis; 1 per output */
  int64_t chunks; /* units along the row axis */
  int64_t band;   /* indices per unit of the kept axis before the row axis */
  int64_t bands;  /* units along that axis */
  int64_t outer;  /* units along the other kept axes */
  int64_t group;  /* blocks per group, a power of two */
  int64_t groups;
  uint8_t *scratch; /* each unit's group's values, where groups > 1 */
  int64_t off[LANES]; /* a lane or fewer terms: their positions from the
                         output's first */
} plan;

static void identity(int monoid, int dt, uint8_t *id) {
  memset(id, 0, 8);
  int w = nx_cpu_width(dt);
  switch (monoid) {
    case NX_MONOID_SUM: return;
    case NX_MONOID_PROD:
      if (dt == NX_FLOAT32) *(float *)id = 1.f;
      else if (dt == NX_FLOAT64) *(double *)id = 1.;
      else id[0] = 1; /* little-endian hosts: the low byte */
      return;
    case NX_MONOID_MAX:
      if (dt == NX_FLOAT32) *(float *)id = -INFINITY;
      else if (dt == NX_FLOAT64) *(double *)id = -INFINITY;
      else if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x80;
      return;
    default: /* NX_MONOID_MIN */
      if (dt == NX_FLOAT32) *(float *)id = INFINITY;
      else if (dt == NX_FLOAT64) *(double *)id = INFINITY;
      else if (dt == NX_BOOL) id[0] = 1;
      else {
        memset(id, 0xFF, w);
        if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x7F;
      }
  }
}

/* [n] identities at [d], copied in doubling runs. */
static void fill(const plan *q, uint8_t *d, int64_t n) {
  if (n == 0) return;
  memcpy(d, q->id, q->w);
  for (int64_t k = 1; k < n; k *= 2)
    memcpy(d + k * q->w, d, (k < n - k ? k : n - k) * q->w);
}

static int is_nan(const plan *q, const uint8_t *p) {
  if (q->dtype == NX_FLOAT32) {
    float v;
    memcpy(&v, p, 4);
    return v != v;
  }
  double v;
  memcpy(&v, p, 8);
  return v != v;
}

/* Whether one of the [n] values at [v] is NaN, in one pass the compiler
   vectorises. */
static int any_nan(const plan *q, const uint8_t *v, int64_t n) {
  int bad = 0;
  if (q->dtype == NX_FLOAT32) {
    const float *f = (const float *)v;
    for (int64_t j = 0; j < n; j++) bad |= f[j] != f[j];
  } else {
    const double *f = (const double *)v;
    for (int64_t j = 0; j < n; j++) bad |= f[j] != f[j];
  }
  return bad;
}

static const uint8_t *at(const plan *q, int64_t p) {
  return q->src->base + p * q->w;
}

/* Writes the NaN nx_kinds.h makes from numbers at [d]. */
static void made_nan(const plan *q, uint8_t *d) {
  if (q->dtype == NX_FLOAT32) {
    float v = NX_NAN_F32;
    memcpy(d, &v, 4);
    return;
  }
  double v = NX_NAN_F64;
  memcpy(d, &v, 8);
}

/* The operand's position of term [t] of the output at [p]. */
static int64_t term(const plan *q, int64_t p, int64_t t) {
  for (int i = q->nr - 1; i >= 0; i--) {
    p += t % q->re[i] * q->rs[i];
    t /= q->re[i];
  }
  return p;
}

/* The terms' positions step through the reduced axes as an odometer: the
   position of term [t] of the output at [p], and its indices into [idx]. */
static int64_t odometer(const plan *q, int64_t p, int64_t t, int64_t *idx) {
  for (int i = q->nr - 1; i >= 0; i--) {
    idx[i] = t % q->re[i];
    p += idx[i] * q->rs[i];
    t /= q->re[i];
  }
  return p;
}

/* The position of the term after the one at [pos], its indices [idx]
   stepped. */
static int64_t next(const plan *q, int64_t *idx, int64_t pos) {
  int i = q->nr - 1;
  pos += q->rs[i];
  while (++idx[i] == q->re[i] && i > 0) {
    pos -= q->re[i] * q->rs[i];
    idx[i--] = 0;
    pos += q->rs[i];
  }
  return pos;
}

/* Replaces the NaN result at [e] of the output at [p] by its first NaN
   term, or by nx_kinds.h's NaN where none is. */
static void first_nan(const plan *q, int64_t p, uint8_t *e) {
  int64_t t = 0;
  while (t < q->terms && !is_nan(q, at(q, term(q, p, t)))) t++;
  if (t < q->terms) memcpy(e, at(q, term(q, p, t)), q->w);
  else made_nan(q, e);
}

/* Blocks */

/* Adds terms [t0, t0 + n) of the output at [p] into the lanes [l], run by
   run along the innermost reduced axis. */
static void runs(const plan *q, int64_t p, int64_t t0, int64_t n, uint8_t *l) {
  int r = q->nr - 1;
  int64_t inner = q->re[r], s = q->rs[r];
  for (int64_t t = t0; t < t0 + n;) {
    int64_t i = t % inner, len = inner - i;
    if (len > t0 + n - t) len = t0 + n - t;
    q->f->lanes(at(q, term(q, p, t)), s, len, l, (int)(t & (LANES - 1)));
    t += len;
  }
}

/* The lanes' tree over [W] outputs: 16 rows of [W] values, [used] of them
   holding terms; the others are the identity, which changes no value. */
static void lane_tree(const plan *q, uint8_t *l, int64_t W, int used) {
  int64_t row = W * q->w;
  for (int h = LANES / 2; h >= 1; h /= 2)
    for (int i = 0; i < h && i + h < used; i++)
      q->f->combine(l + i * row, l + (i + h) * row, 1, W);
}

/* Block [b]'s states of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, their centers [c] or 0 where NULL,
   into [v]: each output's terms where they lie contiguous, else copied into
   a block. */
static void states(const plan *q, int64_t p, int64_t s, int64_t W,
                   const double *c, int64_t b, nx_cpu_state *v) {
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  _Alignas(16) uint8_t buf[BLOCK * 8];
  int contiguous = q->nr == 1 && q->rs[0] == 1;
  for (int64_t j = 0; j < W; j++) {
    int64_t o = p + j * s, idx[NX_MAX_RANK];
    const uint8_t *x = at(q, o + t0);
    if (!contiguous) {
      int64_t pos = odometer(q, o, t0, idx);
      for (int64_t t = 0; t < n; t++, pos = next(q, idx, pos))
        memcpy(buf + t * q->w, at(q, pos), q->w);
      x = buf;
    }
    q->r->block(x, n, t0, c != NULL ? c[j] : 0, &v[j]);
  }
}

/* Combines the values [b] of [W] outputs into [a]'s. */
static void join(const plan *q, uint8_t *a, const uint8_t *b, int64_t W) {
  if (q->f != NULL) {
    q->f->combine(a, b, 1, W);
    return;
  }
  for (int64_t j = 0; j < W; j++)
    q->r->merge((nx_cpu_state *)a + j, (const nx_cpu_state *)b + j);
}

/* Block [b]'s values of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, of centers [c], into [v]. */
static void block(const plan *q, int64_t p, int64_t s, int64_t W,
                  const double *c, int64_t b, uint8_t *v) {
  if (q->r != NULL) {
    states(q, p, s, W, c, b, (nx_cpu_state *)v);
    return;
  }
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  int used = n < LANES ? (int)n : LANES;
  if (W == 1) {
    _Alignas(16) uint8_t l[LANES * 8];
    fill(q, l, LANES);
    runs(q, p, t0, n, l);
    lane_tree(q, l, 1, used);
    memcpy(v, l, q->w);
    return;
  }
  _Alignas(16) uint8_t l[LANES * ROW];
  int64_t row = W * q->w;
  for (int i = 0; i < used; i++) fill(q, l + i * row, W);
  /* Along one reduced axis, lane i's terms lie LANES apart: the table adds
     each lane's in order, its accumulators in registers. */
  if (q->nr == 1) {
    int64_t st = q->rs[0];
    for (int i = 0; i < used; i++)
      q->f->column(l + i * row, at(q, p + (t0 + i) * st), LANES * st,
                   (n - i + LANES - 1) / LANES, s, W);
    lane_tree(q, l, W, used);
    memcpy(v, l, row);
    return;
  }
  int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t0, idx);
  for (int64_t t = t0; t < t0 + n; t++, pos = next(q, idx, pos))
    q->f->combine(l + (t & (LANES - 1)) * row, at(q, pos), s, W);
  lane_tree(q, l, W, used);
  memcpy(v, l, row);
}

/* The value of blocks [b0, b1) of the unit's outputs, of centers [c], by
   the counter, into [v]: the identity for no block. */
static void blocks(const plan *q, int64_t p, int64_t s, int64_t W,
                   const double *c, int64_t b0, int64_t b1, uint8_t *v) {
  _Alignas(16) uint8_t stack[LEVELS][ROW];
  _Alignas(16) uint8_t run[RUN * 8];
  int top = 0;
  int64_t row = W * q->size;
  /* A monoid's output's full blocks of contiguous terms take the table's
     blocks, RUN at a time. */
  int64_t full = q->f != NULL && W == 1 && q->nr == 1 && q->rs[0] == 1
                     ? q->terms / BLOCK
                     : 0;
  for (int64_t b = b0; b < b1;) {
    int64_t n = (b1 < full ? b1 : full) - b;
    if (n > RUN) n = RUN;
    if (n > 0)
      q->f->blocks(at(q, p + b * BLOCK), n, q->id, run);
    else
      n = 1;
    for (int64_t i = 0; i < n; i++, b++) {
      if (b < full) memcpy(stack[top++], run + i * q->w, q->w);
      else block(q, p, s, W, c, b, stack[top++]);
      for (int64_t k = b - b0 + 1; (k & 1) == 0; k >>= 1, top--)
        join(q, stack[top - 2], stack[top - 1], W);
    }
  }
  for (; top > 1; top--) join(q, stack[top - 2], stack[top - 1], W);
  if (top == 0) fill(q, v, W);
  else memcpy(v, stack[0], row);
}

/* Units */

/* Unit [u]'s outputs: [H] rows of [W], the first one's positions in the
   operand and the destinations, and the operand's and destinations' steps
   from one output to the next in a row, and from one row to the next. Units
   run outer index, then band, then chunk, then group. */
typedef struct {
  int64_t p, d, d1, W, s, sd, sd1, H, s2, sd2, g;
} unit;

static unit unit_of(const plan *q, int64_t u) {
  unit x = {.p = q->x0, .d = q->d0, .d1 = q->d1, .W = 1, .H = 1,
            .g = u % q->groups};
  u /= q->groups;
  int last = q->nk - 1;
  if (q->row > 1) {
    int64_t c = u % q->chunks;
    u /= q->chunks;
    x.s = q->ks[last];
    x.sd = q->kd[last];
    x.sd1 = q->kd1[last];
    x.W = q->ke[last] - c * q->row < q->row ? q->ke[last] - c * q->row : q->row;
    x.p += c * q->row * x.s;
    x.d += c * q->row * x.sd;
    x.d1 += c * q->row * x.sd1;
    last--;
  }
  if (q->band > 1) {
    int64_t c = u % q->bands;
    u /= q->bands;
    x.s2 = q->ks[last];
    x.sd2 = q->kd[last];
    x.H = q->ke[last] - c * q->band < q->band ? q->ke[last] - c * q->band
                                              : q->band;
    x.p += c * q->band * x.s2;
    x.d += c * q->band * x.sd2;
    x.d1 += c * q->band * q->kd1[last];
    last--;
  }
  for (int i = last; i >= 0; i--) {
    x.p += u % q->ke[i] * q->ks[i];
    x.d += u % q->ke[i] * q->kd[i];
    x.d1 += u % q->ke[i] * q->kd1[i];
    u /= q->ke[i];
  }
  return x;
}

/* Stores the results of the unit's states [v], of centers [c], each NaN
   replaced as store's are; or, where the pass finds them, the centers. */
static void store_states(const plan *q, const unit *x, double *c,
                         const nx_cpu_state *v) {
  int w1 = nx_cpu_width(q->dst1->dtype);
  for (int64_t j = 0; j < x->W; j++) {
    if (q->centering) {
      q->r->finish(&v[j], 0, (uint8_t *)&c[j], NULL);
      continue;
    }
    int64_t p = x->p + j * x->s;
    uint8_t *d0 = q->dst->base + (x->d + j * x->sd) * q->w;
    uint8_t *d1 = q->dst1->base + (x->d1 + j * x->sd1) * w1;
    q->r->finish(&v[j], c != NULL ? c[j] : 0, d0, d1);
    if (!q->nan) continue;
    if (is_nan(q, d0)) first_nan(q, p, d0);
    if (q->dst1 != q->dst && is_nan(q, d1)) first_nan(q, p, d1);
  }
}

/* Stores the unit's values [v], of centers [c], each NaN replaced by its
   first NaN term, or by nx_kinds.h's where none is. */
static void store(const plan *q, const unit *x, double *c, const uint8_t *v) {
  if (q->r != NULL) {
    store_states(q, x, c, (const nx_cpu_state *)v);
    return;
  }
  uint8_t *d = q->dst->base + x->d * q->w;
  if (x->sd == 1) memcpy(d, v, x->W * q->w);
  else
    for (int64_t j = 0; j < x->W; j++)
      memcpy(d + j * x->sd * q->w, v + j * q->w, q->w);
  if (!q->nan || !any_nan(q, v, x->W)) return;
  for (int64_t j = 0; j < x->W; j++)
    if (is_nan(q, v + j * q->w))
      first_nan(q, x->p + j * x->s, d + j * x->sd * q->w);
}

/* Outputs of a lane or fewer each, one block: their terms' positions from
   an output's first are the same for every output, and the table reduces
   the unit's rows at once, its lanes side by side. */
static void few(const plan *q, const unit *x) {
  q->f->few(at(q, x->p), q->off, (int)q->terms, x->s, x->W, x->s2, x->H, q->id,
            q->dst->base + x->d * q->w, x->sd, x->sd2);
  if (!q->nan) return;
  for (int64_t h = 0; h < x->H; h++) {
    int64_t p = x->p + h * x->s2;
    uint8_t *d = q->dst->base + (x->d + h * x->sd2) * q->w;
    if (x->sd == 1 && !any_nan(q, d, x->W)) continue;
    for (int64_t j = 0; j < x->W; j++)
      if (is_nan(q, d + j * x->sd * q->w))
        first_nan(q, p + j * x->s, d + j * x->sd * q->w);
  }
}

static void reduce_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan *q = ctx;
  _Alignas(16) uint8_t v[ROW];
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (q->f != NULL && q->terms <= LANES) {
      few(q, &x);
      continue;
    }
    int64_t b0 = x.g * q->group, b1 = b0 + q->group;
    if (b1 > q->blocks) b1 = q->blocks;
    double *c = q->centers ? q->centers + u / q->groups * q->row : NULL;
    if (q->groups == 1) {
      blocks(q, x.p, x.s, x.W, c, b0, b1, v);
      store(q, &x, c, v);
    } else
      blocks(q, x.p, x.s, x.W, c, b0, b1, q->scratch + u * q->row * q->size);
  }
}

/* Combines each unit's groups by the counter and stores them. */
static void finish_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan *q = ctx;
  int64_t row = q->row * q->size;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    _Alignas(16) uint8_t stack[LEVELS][ROW];
    int top = 0;
    for (int64_t g = 0; g < q->groups; g++) {
      memcpy(stack[top++], q->scratch + (u * q->groups + g) * row, row);
      for (int64_t k = g + 1; (k & 1) == 0; k >>= 1, top--)
        join(q, stack[top - 2], stack[top - 1], x.W);
    }
    for (; top > 1; top--) join(q, stack[top - 2], stack[top - 1], x.W);
    store(q, &x, q->centers ? q->centers + u * q->row : NULL, stack[0]);
  }
}

static int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }

/* Runs [body] over [total] units on at most [threads] threads, touching
   [bytes] in the time memcpy moves [cost]. */
static void job(int64_t total, int64_t bytes, double cost, int threads,
                rig_pool_body body, void *ctx) {
  if (threads == 1) body(0, total, 0, ctx);
  else nx_cpu_job(total, bytes, cost, body, ctx);
}

/* Adds the axis of extent [e] and steps [s] (operand), [d] and [d1]
   (destinations) to the kept axes, if it has more than one element. */
static void keep(plan *q, int64_t e, int64_t s, int64_t d, int64_t d1) {
  if (e <= 1) return;
  q->ke[q->nk] = e;
  q->ks[q->nk] = s;
  q->kd[q->nk] = d;
  q->kd1[q->nk++] = d1;
}

/* Picks the path for [q]'s axes and lays out its units. The kept axis that
   steps least through the operand becomes the row axis, last, where it
   steps less than every reduced axis or the outputs have few terms. A
   scan's slices otherwise run SLICES side by side along that axis, so that
   as many running sums are in flight. Outputs of a lane of terms or fewer
   whose rows are short take bands of rows of the kept axis before the row
   axis, so that a unit holds about a row's worth of outputs. */
static void choose(plan *q, task op) {
  int64_t outputs = 1;
  for (int i = 0; i < q->nk; i++) outputs *= q->ke[i];
  int least = -1;
  int64_t kept = INT64_MAX, red = INT64_MAX;
  for (int i = 0; i < q->nk; i++)
    if (magnitude(q->ks[i]) < kept) {
      least = i;
      kept = magnitude(q->ks[i]);
    }
  for (int i = 0; i < q->nr; i++)
    if (q->re[i] > 1 && magnitude(q->rs[i]) < red) red = magnitude(q->rs[i]);
  q->row = 1;
  q->chunks = 1;
  q->band = 1;
  q->bands = 1;
  q->outer = outputs;
  q->blocks = (q->terms + BLOCK - 1) / BLOCK;
  int rows = least >= 0 && (kept < red || q->terms < FEW);
  if (least < 0 || (!rows && op != SCAN)) return;
  int64_t e = q->ke[least], s = q->ks[least], d = q->kd[least],
          d1 = q->kd1[least];
  for (int i = least; i < q->nk - 1; i++) {
    q->ke[i] = q->ke[i + 1];
    q->ks[i] = q->ks[i + 1];
    q->kd[i] = q->kd[i + 1];
    q->kd1[i] = q->kd1[i + 1];
  }
  q->ke[q->nk - 1] = e;
  q->ks[q->nk - 1] = s;
  q->kd[q->nk - 1] = d;
  q->kd1[q->nk - 1] = d1;
  q->slices = !rows;
  int64_t most = q->slices ? SLICES : ROW / q->size;
  q->row = most < e ? most : e;
  q->chunks = (e + q->row - 1) / q->row;
  q->outer = outputs / e;
  if (op == SCAN || q->f == NULL || q->terms > LANES || q->nk < 2 ||
      q->chunks > 1)
    return;
  int64_t next = q->ke[q->nk - 2], want = (most + q->row - 1) / q->row;
  q->bands = (next + want - 1) / want;
  q->band = (next + q->bands - 1) / q->bands;
  q->outer /= next;
}

/* Lays out a reduction from the coalesced loop [l] over the destinations
   and the operand, the destinations stepping 0 along the reduced axes.
   Where units are few and blocks many, each unit takes a group of blocks,
   its size from the shape alone. */
static void plan_reduce(plan *q, const nx_loop *l) {
  q->nr = q->nk = 0;
  for (int i = 0; i < l->rank; i++)
    if (l->step[0][i] != 0)
      keep(q, l->extent[i], l->step[2][i], l->step[0][i], l->step[1][i]);
    else if (l->extent[i] > 1) {
      q->re[q->nr] = l->extent[i];
      q->rs[q->nr++] = l->step[2][i];
    }
  if (q->nr == 0) {
    q->re[0] = 1;
    q->rs[0] = 0;
    q->nr = 1;
  }
  q->x0 = l->first[2];
  q->d0 = l->first[0];
  q->d1 = l->first[1];
  choose(q, REDUCE);
  for (int64_t t = 0; t < q->terms && t < LANES; t++) q->off[t] = term(q, 0, t);
  int64_t units = q->outer * q->bands * q->chunks;
  q->group = q->blocks > 0 ? q->blocks : 1;
  if (units < UNITS && q->blocks > 1) {
    int64_t want = (UNITS + units - 1) / units;
    int64_t per = (q->blocks + want - 1) / want;
    for (q->group = 1; q->group < per;) q->group *= 2;
  }
  q->groups = q->blocks > 0 ? (q->blocks + q->group - 1) / q->group : 1;
}

/* The elements of [d], each of [w] bytes, all set to the bits [e]: a
   reduction of no term. */
static void fill_dst(const nx_array *d, const uint8_t *e, int w) {
  int64_t extent[NX_MAX_RANK], step[1][NX_MAX_RANK];
  for (int i = 0; i < d->rank; i++) {
    extent[i] = d->dim[i];
    step[0][i] = d->dim[d->rank + i];
  }
  int r = nx_coalesce_dims(1, d->rank, extent, step);
  int64_t n = 1;
  for (int i = 0; i < r; i++) n *= extent[i];
  for (int64_t k = 0; k < n; k++) {
    int64_t p = d->offset, rest = k;
    for (int i = r - 1; i >= 0; i--) {
      p += rest % extent[i] * step[0][i];
      rest /= extent[i];
    }
    memcpy(d->base + p * w, e, w);
  }
}

/* Scans */

/* Chunk [c]'s total of the unit's slices, into [v]: slice by slice where
   they run side by side. */
static void total(const plan *q, const unit *x, int64_t c, uint8_t *v) {
  int64_t b0 = c * (SCAN_CHUNK / BLOCK), b1 = b0 + SCAN_CHUNK / BLOCK;
  if (b1 > q->blocks) b1 = q->blocks;
  if (!q->slices) {
    blocks(q, x->p, x->s, x->W, NULL, b0, b1, v);
    return;
  }
  for (int64_t j = 0; j < x->W; j++)
    blocks(q, x->p + j * x->s, 0, 1, NULL, b0, b1, v + j * q->w);
}

/* Chunk [c]'s results of the unit's slices, from the carry [a], which it
   leaves holding the chunk's last results: side by side, or a row of
   outputs at a time. */
static void rescan(const plan *q, const unit *x, int64_t c, uint8_t *a) {
  int64_t t0 = c * SCAN_CHUNK;
  int64_t n = q->terms - t0 < SCAN_CHUNK ? q->terms - t0 : SCAN_CHUNK;
  uint8_t *y = q->dst->base + x->d * q->w;
  if (q->slices || x->W == 1) {
    q->f->scan(a, at(q, x->p + t0 * q->rs[0]), q->rs[0], x->s,
               y + t0 * q->rd * q->w, q->rd, x->sd, n, (int)x->W);
    return;
  }
  for (int64_t t = t0; t < t0 + n; t++) {
    uint8_t *row = y + t * q->rd * q->w;
    q->f->combine(a, at(q, x->p + t * q->rs[0]), x->s, x->W);
    if (x->sd == 1) memcpy(row, a, x->W * q->w);
    else
      for (int64_t j = 0; j < x->W; j++)
        memcpy(row + j * x->sd * q->w, a + j * q->w, q->w);
  }
}

/* Each of the unit's slices whose last result is NaN takes its first NaN
   term from that term on, and nx_kinds.h's NaN for each NaN result before
   it. */
static void settle(const plan *q, const unit *x) {
  if (!q->nan || q->terms == 0) return;
  uint8_t *y = q->dst->base;
  for (int64_t j = 0; j < x->W; j++) {
    int64_t p = x->p + j * x->s, d = x->d + j * x->sd;
    if (!is_nan(q, y + (d + (q->terms - 1) * q->rd) * q->w)) continue;
    int64_t k = 0;
    while (k < q->terms && !is_nan(q, at(q, p + k * q->rs[0]))) k++;
    for (int64_t t = 0; t < k; t++) {
      uint8_t *e = y + (d + t * q->rd) * q->w;
      if (is_nan(q, e)) made_nan(q, e);
    }
    for (int64_t t = k; t < q->terms; t++)
      memcpy(y + (d + t * q->rd) * q->w, at(q, p + k * q->rs[0]), q->w);
  }
}

/* One pass per unit: each chunk's results from the carry, which then takes
   the chunk's total. The last chunk's total carries nowhere. */
static void scan_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan *q = ctx;
  _Alignas(16) uint8_t carry[ROW], a[ROW], v[ROW];
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    fill(q, carry, x.W);
    for (int64_t c = 0; c * SCAN_CHUNK < q->terms; c++) {
      memcpy(a, carry, x.W * q->w);
      rescan(q, &x, c, a);
      if ((c + 1) * SCAN_CHUNK >= q->terms) break;
      total(q, &x, c, v);
      q->f->combine(carry, v, 1, x.W);
    }
    settle(q, &x);
  }
}

/* Few slices, long axes: every chunk's total in one job, each unit's
   carries in order, every chunk's results in a second job. A unit's group
   is its chunk. */
static void total_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan *q = ctx;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (x.g < q->groups - 1) total(q, &x, x.g, q->scratch + u * q->row * q->w);
  }
}

static void rescan_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan *q = ctx;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    rescan(q, &x, x.g, q->scratch + u * q->row * q->w);
  }
}

/* One slice's chunks four at a time, side by side, where a unit is one
   slice: their carries lie side by side in the scratch. A short last chunk
   runs alone. */
static void rescan_quads(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan *q = ctx;
  int64_t quads = (q->groups + 3) / 4;
  uint8_t *y = q->dst->base;
  for (int64_t v = lo; v < hi; v++) {
    int64_t u = v / quads, c = v % quads * 4;
    unit x = unit_of(q, u * q->groups);
    uint8_t *a = q->scratch + (u * q->groups + c) * q->w;
    int k = q->groups - c < 4 ? (int)(q->groups - c) : 4;
    int full = (c + k) * SCAN_CHUNK <= q->terms ? k : k - 1;
    int64_t t0 = c * SCAN_CHUNK;
    if (full > 0)
      q->f->scan(a, at(q, x.p + t0 * q->rs[0]), q->rs[0], SCAN_CHUNK * q->rs[0],
                 y + (x.d + t0 * q->rd) * q->w, q->rd, SCAN_CHUNK * q->rd,
                 SCAN_CHUNK, full);
    if (full < k) rescan(q, &x, c + full, a + full * q->w);
  }
}

static void carries(const plan *q, int64_t units) {
  int64_t row = q->row * q->w;
  _Alignas(16) uint8_t carry[ROW], v[ROW];
  for (int64_t u = 0; u < units; u++) {
    unit x = unit_of(q, u * q->groups);
    fill(q, carry, x.W);
    for (int64_t c = 0; c < q->groups; c++) {
      uint8_t *s = q->scratch + (u * q->groups + c) * row;
      memcpy(v, s, row);
      memcpy(s, carry, row);
      if (c < q->groups - 1) q->f->combine(carry, v, 1, x.W);
    }
  }
}

/* Lays out a scan along the axis [axis] of the destination [d] and the
   operand [x]: the axes before it and those after it coalesce apart. */
static void plan_scan(plan *q, const nx_array *d, const nx_array *x, int axis) {
  int r = x->rank;
  q->nr = 1;
  q->re[0] = x->dim[axis];
  q->rs[0] = x->dim[r + axis];
  q->rd = d->dim[r + axis];
  q->x0 = x->offset;
  q->d0 = d->offset;
  q->nk = 0;
  int parts[2][2] = {{0, axis}, {axis + 1, r}};
  for (int k = 0; k < 2; k++) {
    int64_t extent[NX_MAX_RANK], step[2][NX_MAX_RANK];
    int n = parts[k][1] - parts[k][0];
    for (int i = 0; i < n; i++) {
      int a = parts[k][0] + i;
      extent[i] = x->dim[a];
      step[0][i] = d->dim[r + a];
      step[1][i] = x->dim[r + a];
    }
    n = nx_coalesce_dims(2, n, extent, step);
    for (int i = 0; i < n; i++)
      keep(q, extent[i], step[1][i], step[0][i], step[0][i]);
  }
  choose(q, SCAN);
  int64_t chunks = (q->terms + SCAN_CHUNK - 1) / SCAN_CHUNK;
  q->groups = q->outer * q->chunks < UNITS && chunks > 1 ? chunks : 1;
}

/* Whether the destination's shape [y] of rank [yr] is the result's for an
   operand of shape [x] of rank [xr]: [x] without [axes] for a reduction,
   [x] for a scan along its axis. Sets the terms per output and the
   outputs. */
static int fits(task op, const int64_t *x, int xr, const int64_t *y, int yr,
                const int *axes, int naxes, int64_t *terms, int64_t *outputs) {
  int j = 0, k = 0;
  *terms = *outputs = 1;
  for (int i = 0; i < xr; i++) {
    if (j < naxes && axes[j] == i) {
      *terms *= x[i];
      j++;
      if (op == REDUCE) continue;
    } else
      *outputs *= x[i];
    if (k >= yr || y[k++] != x[i]) return 0;
  }
  return j == naxes && k == yr;
}

/* A reduction's units: each unit's blocks, or each unit's groups of them
   then the groups combined. */
static void reduce_jobs(plan *q, int64_t units, int64_t bytes, double cost,
                        int threads) {
  if (q->groups == 1) {
    job(units, bytes, cost, threads, reduce_units, q);
    return;
  }
  int64_t values = units * q->groups * q->row * q->size;
  job(units * q->groups, bytes, cost, threads, reduce_units, q);
  job(units, values, values, threads, finish_units, q);
}

/* Reduces or scans, as [op] says, [x] into [y] and [y1] by [r]: a monoid
   (reduced or scanned) into [y], [y1] being [y]; or reduced, Logsumexp
   into [y], [y1] being [y], Arg into the extreme [y] and its int64
   position [y1], or Moments into the mean [y] and the variance [y1].
   NX_OK, NX_SHAPE for shapes that do not fit, an extreme of no term
   included, or -1 if host memory runs out. */
static int reduce_into(task op, int r, int dt, const int *axes, int naxes,
                       const nx_array *y, const nx_array *y1,
                       const nx_array *x, int threads) {
  int64_t terms, outputs;
  if (!fits(op, x->dim, x->rank, y->dim, y->rank, axes, naxes, &terms,
            &outputs) ||
      !fits(op, x->dim, x->rank, y1->dim, y1->rank, axes, naxes, &terms,
            &outputs))
    return NX_SHAPE;
  int extreme = r == NX_MONOID_MAX || r == NX_MONOID_MIN ||
                r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  if (op == REDUCE && terms == 0 && outputs > 0 && extreme) return NX_SHAPE;
  int stateful = r >= NX_CPU_MONOIDS, w = nx_cpu_width(dt);
  int arg = r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  const nx_cpu_stateful *st =
      stateful ? &nx_cpu_table->stateful[r - NX_CPU_MONOIDS][dt] : NULL;
  plan q = {.f = stateful ? NULL : &nx_cpu_table->monoid[r][dt],
            .r = stateful ? &st->value : NULL,
            .dtype = dt,
            .w = w,
            .nan = (dt == NX_FLOAT32 || dt == NX_FLOAT64) && !arg,
            .size = stateful ? (int64_t)sizeof(nx_cpu_state) : w,
            .dst = y,
            .dst1 = y1,
            .src = x,
            .terms = terms};
  if (outputs > 0 && terms == 0 && op == REDUCE) {
    /* The results of no term: the identity, or a state of none's. */
    _Alignas(8) uint8_t e0[8], e1[8];
    if (stateful) q.r->finish(&(nx_cpu_state){0}, 0, e0, e1);
    else identity(r, dt, e0);
    fill_dst(y, e0, w);
    if (y1 != y) fill_dst(y1, e1, nx_cpu_width(y1->dtype));
  }
  if (outputs == 0 || terms == 0) return NX_OK;
  if (!stateful) identity(r, dt, q.id);
  int rank = x->rank;
  if (op == SCAN) plan_scan(&q, y, x, axes[0]);
  else {
    /* The loop: the destinations step 0 along the reduced axes. */
    nx_loop l = {.rank = rank, .first = {y->offset, y1->offset, x->offset}};
    for (int i = 0, k = 0, j = 0; i < rank; i++) {
      l.extent[i] = x->dim[i];
      l.step[2][i] = x->dim[rank + i];
      int reduced = j < naxes && axes[j] == i;
      j += reduced;
      l.step[0][i] = reduced ? 0 : y->dim[y->rank + k];
      l.step[1][i] = reduced ? 0 : y1->dim[y1->rank + k];
      k += !reduced;
    }
    l.rank = nx_coalesce_dims(3, rank, l.extent, l.step);
    plan_reduce(&q, &l);
  }
  int64_t units = q.outer * q.bands * q.chunks;
  int64_t bytes = terms * outputs * w;
  double cost = stateful ? STATE_COST[r - NX_CPU_MONOIDS] * bytes
                : op == REDUCE && terms <= LANES ? FEW_COST * bytes
                                                 : bytes;
  if (q.groups > 1)
    q.scratch = malloc((size_t)(units * q.groups * q.row * q.size));
  int center = stateful && st->center.block != NULL;
  if (center) q.centers = malloc((size_t)(units * q.row) * sizeof(double));
  if ((q.groups > 1 && q.scratch == NULL) || (center && q.centers == NULL)) {
    free(q.scratch);
    free(q.centers);
    return -1;
  }
  if (op == REDUCE && center) {
    plan c = q;
    c.r = &st->center;
    c.centering = 1;
    reduce_jobs(&c, units, bytes, cost, threads);
  }
  if (op == REDUCE) reduce_jobs(&q, units, bytes, cost, threads);
  else if (q.groups == 1) job(units, bytes, cost, threads, scan_units, &q);
  else {
    job(units * q.groups, bytes, bytes, threads, total_units, &q);
    carries(&q, units);
    if (q.row == 1)
      job(units * ((q.groups + 3) / 4), bytes, bytes, threads, rescan_quads,
          &q);
    else
      job(units * q.groups, bytes, bytes, threads, rescan_units, &q);
    for (int64_t u = 0; u < units; u++) {
      unit x = unit_of(&q, u * q.groups);
      settle(&q, &x);
    }
  }
  free(q.scratch);
  free(q.centers);
  return NX_OK;
}

/* Moments scans */

/* A Moments scan: the outer axes, those but the scanned one, coalesced,
   with the steps of the mean (0), the variance (1) and the operand (2);
   the scanned axis's extent and steps. [failed] is set where a unit's
   scratch could not be allocated. A unit runs NX_CPU_MOMENTS_LANES slices
   side by side, along the outer axis that steps least. */
typedef struct {
  const nx_array *a[3];
  nx_loop outer;
  int64_t len, along[3];
  int64_t rows, chunks; /* outer positions but the last axis; units along it */
  _Atomic int failed;
} moments;

static void moments_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  moments *q = ctx;
  const nx_loop *l = &q->outer;
  int last = l->rank - 1, w8 = nx_cpu_width(q->a[2]->dtype);
  int64_t inner = l->extent[last], most = NX_CPU_MOMENTS_LANES;
  int64_t lanes = inner < most ? inner : most;
  int64_t levels = nx_cpu_moments_levels(q->len);
  uint8_t *scratch = malloc((size_t)(12 * levels * lanes * w8));
  if (scratch == NULL) {
    q->failed = 1;
    return;
  }
  nx_cpu_moments run = nx_cpu_table->moments[q->a[2]->dtype];
  for (int64_t u = lo; u < hi; u++) {
    int64_t row = u / q->chunks, from = (u % q->chunks) * most;
    int64_t at[3] = {l->first[0], l->first[1], l->first[2]};
    for (int i = last - 1; i >= 0; i--) {
      int64_t k = row % l->extent[i];
      row /= l->extent[i];
      for (int o = 0; o < 3; o++) at[o] += k * l->step[o][i];
    }
    for (int o = 0; o < 3; o++) at[o] += from * l->step[o][last];
    int64_t w = inner - from < most ? inner - from : most;
    run(q->a[2]->base + at[2] * w8, q->along[2], l->step[2][last],
        q->a[0]->base + at[0] * w8, q->along[0], l->step[0][last],
        q->a[1]->base + at[1] * w8, q->along[1], l->step[1][last], q->len, w,
        scratch);
  }
  free(scratch);
}

/* Scans [x] along [axis] by Moments into the mean [m] and the variance [v],
   float32 or float64 of [x]'s dtype: NX_OK, NX_SHAPE for shapes that do
   not fit, or -1 if host memory runs out. */
static int moments_into(int axis, const nx_array *m, const nx_array *v,
                        const nx_array *x, int threads) {
  int64_t terms, outputs;
  if (!fits(MOMENTS, x->dim, x->rank, m->dim, m->rank, &axis, 1, &terms,
            &outputs) ||
      !fits(MOMENTS, x->dim, x->rank, v->dim, v->rank, &axis, 1, &terms,
            &outputs))
    return NX_SHAPE;
  if (outputs == 0 || terms == 0) return NX_OK;
  moments q = {.a = {m, v, x}, .len = terms};
  nx_loop *l = &q.outer;
  int r = x->rank, k = 0;
  for (int o = 0; o < 3; o++) {
    l->first[o] = q.a[o]->offset;
    q.along[o] = q.a[o]->dim[r + axis];
  }
  for (int i = 0; i < r; i++) {
    if (i == axis) continue;
    l->extent[k] = x->dim[i];
    for (int o = 0; o < 3; o++) l->step[o][k] = q.a[o]->dim[r + i];
    k++;
  }
  /* A slice alone keeps one outer axis, of extent 1. */
  if (k == 0) {
    l->extent[0] = 1;
    for (int o = 0; o < 3; o++) l->step[o][0] = 0;
    k = 1;
  }
  l->rank = nx_coalesce_dims(3, k, l->extent, l->step);
  int64_t inner = l->extent[l->rank - 1];
  q.rows = outputs / inner;
  q.chunks = (inner + NX_CPU_MOMENTS_LANES - 1) / NX_CPU_MOMENTS_LANES;
  int64_t bytes = 3 * terms * outputs * nx_cpu_width(x->dtype);
  job(q.rows * q.chunks, bytes, 4 * bytes, threads, moments_units, &q);
  return q.failed ? -1 : NX_OK;
}

/* Entries */

/* Reads [x], one destination [d0] or two, [d0] and [d1], and, where
   [vpad] is Some (pad, fill), [x] copied padded with the bits [fill] into
   C-heap scratch (assemble.c), and computes [c] of them: [r]'s reduction
   along the [naxes] axes [axes], a monoid's scan along [axes[0]], or a
   Moments scan along [axes[0]] into the two destinations, as [op] says.
   The destinations are of [x]'s dtype, but for Arg's int64 positions. The
   pad is trusted: one that fits no operand of [x]'s shape is refused
   before anything reads [x]. */
typedef struct {
  task op;
  int r, naxes, threads;
  int axes[NX_MAX_RANK];
} call;

static value run(const call *c, value d0, value d1, value vx, value vpad) {
  CAMLparam4(d0, d1, vx, vpad);
  int dt = nx_array_dtype(vx), nd = d1 == Val_unit ? 1 : 2;
  int arg =
      c->op == REDUCE && (c->r == NX_CPU_ARG_MAX || c->r == NX_CPU_ARG_MIN);
  /* The pad and the fill, copied before anything that can move them. */
  int padded = Is_block(vpad);
  nx_cpu_pad pad;
  uint8_t bits[16] = {0};
  if (padded) {
    int64_t xs[2 * NX_MAX_RANK], ys[NX_MAX_RANK], off;
    int rank = nx_array_layout(vx, xs, &off);
    value fill = Field(Field(vpad, 0), 1);
    size_t n = caml_string_length(fill);
    memcpy(bits, String_val(fill), n < sizeof bits ? n : sizeof bits);
    if (nx_cpu_pad_of(Field(Field(vpad, 0), 0), &pad) ||
        nx_cpu_padded_shape(&pad, rank, xs, ys))
      CAMLreturn(Val_int(NX_SHAPE));
  }
  nx_operand in[3] = {{d0, dt, 1}, {d1, arg ? NX_INT64 : dt, 1}};
  in[nd] = (nx_operand){vx, dt, 0};
  nx_array a[3], loaded, *x = &a[nd];
  int e = nx_read(nd + 1, in, a);
  if (e) CAMLreturn(Val_int(e));
  if (padded) {
    if (nx_cpu_unpad(x, &pad, bits, &loaded)) {
      nx_done(nd + 1, a);
      caml_raise_out_of_memory();
    }
    x = &loaded;
  }
  int threads = c->threads;
  e = c->op == MOMENTS
          ? moments_into(c->axes[0], &a[0], &a[1], x, threads)
          : reduce_into(c->op, c->r, dt, c->axes, c->naxes, &a[0],
                        &a[nd - 1], x, threads);
  if (padded) free(loaded.base);
  nx_done(nd + 1, a);
  if (e < 0) caml_raise_out_of_memory();
  CAMLreturn(Val_int(e));
}

/* Copies the axis [v] into [a]: 0, or 1 for an axis no array has. */
static int axis_of(value v, int *a) {
  intnat x = Long_val(v);
  if (x < 0 || x >= NX_MAX_RANK) return 1;
  *a = (int)x;
  return 0;
}

/* [dsts] holds Arg's and Moments' two destinations, the others' one. */
value nx_cpu_reduce(value r, value axes, value dsts, value x, value pad,
                    value threads) {
  call c = {.op = REDUCE,
            .r = Int_val(r),
            .naxes = (int)Wosize_val(axes),
            .threads = Int_val(threads)};
  int two = c.r == NX_CPU_ARG_MAX || c.r == NX_CPU_ARG_MIN ||
            c.r == NX_CPU_MOMENTS;
  if (Wosize_val(dsts) != (mlsize_t)(1 + two)) return Val_int(NX_ARITY);
  if (c.naxes > NX_MAX_RANK) return Val_int(NX_SHAPE);
  for (int i = 0; i < c.naxes; i++)
    if (axis_of(Field(axes, i), &c.axes[i])) return Val_int(NX_SHAPE);
  return run(&c, Field(dsts, 0), two ? Field(dsts, 1) : Val_unit, x, pad);
}

value nx_cpu_reduce_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_reduce(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

value nx_cpu_scan(value monoid, value axis, value dst, value x, value pad,
                  value threads) {
  call c = {.op = SCAN,
            .r = Int_val(monoid),
            .naxes = 1,
            .threads = Int_val(threads)};
  if (axis_of(axis, &c.axes[0])) return Val_int(NX_SHAPE);
  return run(&c, dst, Val_unit, x, pad);
}

value nx_cpu_scan_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_scan(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

value nx_cpu_moments_scan(value axis, value mean, value var, value x,
                          value pad, value threads) {
  call c = {.op = MOMENTS, .naxes = 1, .threads = Int_val(threads)};
  if (axis_of(axis, &c.axes[0])) return Val_int(NX_SHAPE);
  return run(&c, mean, var, x, pad);
}

value nx_cpu_moments_scan_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_moments_scan(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5]);
}

#elif !defined(__x86_64__)

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_v3;

#endif