/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reductions and scans of one operand by Sum, Prod, Max, Min, Logsumexp
   and Arg, and by Moments.

   Association. An output's terms are numbered t = 0, 1, … in C order of the
   reduced axes' indices. They fall into blocks of NX_CPU_LANE_BLOCK
   consecutive terms and, within a block, into NX_CPU_LANES lanes from the
   identity, term t into lane t mod 16. Lane i takes lane i + 8 for i < 8,
   then lane i + 4, i + 2 and i + 1, but not a lane holding no term: lane 0
   is the block's value. A complex product's lanes start from their first
   term: 1 + 0i times a term would change its signed zeros and make NaN
   parts from infinite ones, where a real 1 changes nothing. The blocks
   combine by the binary tree whose left part holds the largest power of
   two of blocks below their count. A binary counter computes it in one
   pass: it combines the two newest values once per trailing zero of the
   number of blocks done, then what remains, newest first.

   A scan cuts each slice along its axis into chunks of SCAN_CHUNK terms from
   its start. A chunk's total is its terms reduced in the order above. The
   carry into chunk 0 is the slice's given carry, else the identity, but a
   complex product's chunk 0 given none has none and starts from its first
   term; the carry into chunk c + 1 is the
   carry into c combined with c's total, and result i of chunk c is the
   carry combined with the chunk's terms up to i, left to right.

   Only float sums and float and complex products depend on this order:
   integers wrap and extremes are exact. Neither the layout, the path nor
   the thread count changes it. A thread takes a group of blocks, a power
   of two of them from a multiple of that power, and such a group's value
   is a subtree of the blocks' tree: the groups' values combine by the same
   counter.

   NaN. A float result that is NaN, or a complex one with a NaN part, is
   replaced by its first such term in index order, quieted, found by a
   second walk taken only then, or by nx_kinds.h's NaN, in each part of a
   complex, where no term is NaN. Quieting sets each NaN part's quiet bit,
   the top of its fraction, and keeps its sign and the rest of its payload,
   as IEEE 754 quiets a signalling NaN an operation returns. A complex
   sum's parts are float sums, each replaced so apart. A scan's results
   stay NaN once they are, since adding, multiplying and the extremes keep
   a NaN: where a slice's last result is NaN, every result from its first
   NaN term on is that term, quieted, and every NaN result before it
   nx_kinds.h's. A Max's or a Min's carry is a term before the slice's
   first, so a NaN one is every result, quieted; a sum's or a product's
   counts as made from numbers.

   Paths. The operand and the destination, read at the operand's shape with
   step 0 along the reduced axes, coalesce: reduced and kept axes never
   merge, and the axes keep their order. A unit of work reduces W outputs:
   - one output, its runs of terms along the innermost reduced axis into
     lanes, where a reduced axis steps least through the operand;
   - W outputs along the kept axis that steps least, each term a row of W
     elements combined into one of 16 accumulator rows, elsewhere and where
     outputs have few terms. A 2x2 window's maximum takes this path.

   Stage. A monoid's terms reduce in their carrier (cpu.h): a narrow
   float's in float32, int4's and uint4's in int8 and uint8, bit's in bool.
   A narrow operand's terms are converted into it as the paths read them, a
   block of one output's or 16 terms of a row's outputs at a time, each a
   few KiB, so that they stay in L1; a network scan's operand is converted
   whole first. Arg, Logsumexp and Moments reduce in the carrier too.
   Results go into the destination where it is of the carrier, else into a
   carrier copy of it, rounded into it once at the end, as a cast does. A
   NaN keeps its sign and payload through the carrier, quieted
   (nx_dtype.h), so a NaN result rounded into the operand's dtype is its
   first NaN term quieted, as the rule above gives it.

   States. Logsumexp, Arg and Moments reduce by the same blocks, tree and
   paths, an output's value a state of its terms in place of a monoid's
   element: a kernel makes a block's state from its terms, contiguous in
   the operand or copied so, and merges two states as the tree combines
   values. A float result that is NaN is replaced as a monoid's is, but
   Arg's, whose state holds its first NaN term, quieted as it is stored.
   Arg reads a staged operand in the carrier too; its extreme into the
   operand's dtype is the term at its position, copied and quieted.

   Arg's scans. An Arg scan cuts the same chunks and carries a state per
   slice: a chunk's total is its terms' state from none, the carry into
   chunk c + 1 the carry into c merged with c's total, and result i of
   chunk c the carry taken on through the chunk's terms up to i. A unit's
   sequences, slices side by side or one slice's chunks, are the kernel's
   lanes, read and written in place, but a staged operand's, which go
   through rows of the carrier. Arg is exact, so its chunk totals are its
   reduction's.

   Programs. A reduction of a program's output reads its terms through
   the program: each place that reads terms asks map.c's evaluator for
   them (nx_cpu_prog), in the output's held dtype, converted into the
   carrier as a narrow operand's are, so the paths that stage a narrow
   operand read a program's output too. Each term is the output there as
   a map computes it, and the order is a function of the shapes alone, so
   the reduction has the bits of the output mapped into an array, then
   reduced.

   Networks. A Logsumexp or Moments scan computes nx_kernel.mli's network
   on each slice, op by op with the kinds' arithmetic, as Nx's expansion
   does, so that the two give the same bits; a slice whose results a NaN
   term reaches has them quieted after. It holds O(log n) states per
   slice.

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
   whose ties order -0 below +0. Contiguous terms take the lanes as
   vectors. */

#define BLOCK NX_CPU_LANE_BLOCK

/* Unrolls a loop under gcc. Arg's search for its extreme's bits and
   Moments' lanes take it; Logsumexp's lanes run faster unrolled whole.
   Each loop and dtype takes the count that is faster or no slower than
   none at -O2 and -O3, on arm64 and x86, and none where no count is. clang
   unrolls the loops itself and takes the pragma as its own, so it applies
   under gcc alone. */
#define PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#define UNROLL(n)
#else
#define UNROLL(n) PRAGMA(GCC unroll n)
#endif

/* The outputs whose lanes a few-term reduction holds side by side: a lane's
   share of its scratch. */
#define STRIP(T) (NX_CPU_FEW_BYTES / NX_CPU_LANES / (int64_t)sizeof(T))

/* How far a scan's sequences run behind one another, in elements. */
#define LAG 16

/* Lanes as vectors. A monoid's 16 lanes D_l, at the suffix D, come with
   splat_D, load_D and store_D, and its steps on_ (16 terms into the
   lanes), pair_ and start_ (lanes in memory, as few holds them), run_ (a
   run of terms into the lanes) and cols_ (columns into accumulators), from
   which MONOID builds its kernels; VMONOID provides them over vectors,
   XMONOID over whole vectors, AMONOID over arrays, and SMONOID over an
   array of its own. A real dtype's lanes are a struct of vectors of the
   target's width, NX_VECTOR_BYTES, or of one vector of all 16 where they
   are narrower: FIELDS(W) fields a, b, … for a dtype of W bytes. gcc keeps
   named fields in registers, where it keeps an array of lanes, or a single
   vector wider than the target's, in memory. EACH(n, M, …) applies
   M(field, its index, …) to each of n fields.

   An exact monoid, an integer's or a boolean's, gives the same result by
   any association. Its runs also go through XFIELDS(W) whole vectors, at
   least four, at the suffix Dx: their lanes fill the vectors, and four
   chains are in flight, which an emulated product of bytes, several
   cycles long, needs. Their lane p takes the terms of lane p mod 16 from
   the identity, and is then combined into it. Where four whole vectors
   hold no more than the 16 lanes (8-byte dtypes, and 4-byte ones on
   16-byte targets), runs keep the 16 lanes. */
#if NX_VECTOR_BYTES == 32
#define FIELDS_1 1
#define FIELDS_2 1
#define FIELDS_4 2
#define FIELDS_8 4
#define XFIELDS_1 4
#define XFIELDS_2 4
#define XFIELDS_4 4
#define XFIELDS_8 4
#elif NX_VECTOR_BYTES == 16
#define FIELDS_1 1
#define FIELDS_2 2
#define FIELDS_4 4
#define FIELDS_8 8
#define XFIELDS_1 4
#define XFIELDS_2 4
#define XFIELDS_4 4
#define XFIELDS_8 8
#else
#error "NX_VECTOR_BYTES is neither 16 nor 32"
#endif
#define FIELDS(W) FIELDS_##W
#define XFIELDS(W) XFIELDS_##W
#define EACH(n, M, ...) EACH_(n, M, __VA_ARGS__)
#define EACH_(n, M, ...) EACH##n(M, __VA_ARGS__)
#define EACH1(M, ...) M(a, 0, __VA_ARGS__)
#define EACH2(M, ...) EACH1(M, __VA_ARGS__) M(b, 1, __VA_ARGS__)
#define EACH4(M, ...)                                                        \
  EACH2(M, __VA_ARGS__) M(c, 2, __VA_ARGS__) M(d, 3, __VA_ARGS__)
#define EACH8(M, ...)                                                        \
  EACH4(M, __VA_ARGS__) M(e, 4, __VA_ARGS__) M(f, 5, __VA_ARGS__)            \
  M(g, 6, __VA_ARGS__) M(h, 7, __VA_ARGS__)

#define FIELD_DECL(f, i, V) V f;
#define FIELD_SET(f, i, l, v) l.f = v;
#define FIELD_LOAD(f, i, l, x, M) l.f = ((const M *)(x))[i];
#define FIELD_STORE(f, i, l, x, M) ((M *)(x))[i] = l.f;
#define FIELD_ON(f, i, l, r, VF) l->f = VF(l->f, r.f);

/* [N] vectors of [B] bytes of [T] at the suffix [D]: the vectors D_v, of
   [U] D_u, the struct D_l, all of one value, loaded from its terms, stored
   to them. A load or store goes through D_m, a D_v at any address and of
   any type's memory: gcc keeps in memory lanes that memcpy fills. */
#define VECTORS(D, T, U, B, N)                                               \
  typedef T D##_v __attribute__((vector_size(B)));                           \
  typedef U D##_u __attribute__((vector_size(B)));                           \
  typedef T D##_m __attribute__((vector_size(B), aligned(1), may_alias));    \
  typedef struct {                                                           \
    EACH(N, FIELD_DECL, D##_v)                                               \
  } D##_l;                                                                   \
  static inline D##_l splat_##D(T e) {                                       \
    /* e less a vector of zeros, which is e in each element */               \
    D##_v v = e - (D##_v){0};                                                \
    D##_l l;                                                                 \
    EACH(N, FIELD_SET, l, v)                                                 \
    return l;                                                                \
  }                                                                          \
  static inline D##_l load_##D(const T *x) {                                 \
    D##_l l;                                                                 \
    EACH(N, FIELD_LOAD, l, x, D##_m)                                         \
    return l;                                                                \
  }                                                                          \
  static inline void store_##D(T *x, D##_l l) {                              \
    EACH(N, FIELD_STORE, l, x, D##_m)                                        \
  }

/* The 16 lanes of a real dtype of [T], [W] bytes, at the suffix [D]. */
#define LANE_VECTORS(D, T, U, W)                                             \
  VECTORS(D, T, U, 16 * W / FIELDS(W), FIELDS(W))

/* The 16 lanes of an exact monoid's dtype, and its whole vectors at Dx. */
#define WHOLE_VECTORS(D, T, U, W)                                            \
  LANE_VECTORS(D, T, U, W)                                                   \
  VECTORS(D##x, T, U, NX_VECTOR_BYTES, XFIELDS(W))

/* [NAME], the [N] vectors [l] of the suffix [D] combined with as many
   terms of [T] at [x] by the vector op [VF]. */
#define ON(NAME, D, T, N, VF)                                                \
  static inline void NAME(D##_l *l, const T *x) {                            \
    D##_l r = load_##D(x);                                                   \
    EACH(N, FIELD_ON, l, r, VF)                                              \
  }

/* [x] where the mask [m] is set, else [y], vectors [V] of the unsigned
   vectors [U]. */
#define SEL(U, V, m, x, y)                                                   \
  ((V)(((U)(m) & (U)(x)) | (~(U)(m) & (U)(y))))

/* Vector ops over vectors [a] and [b] of the suffix [D]. Integer sums and
   products wrap in its unsigned vectors, as add_ and mul_ do; float
   extremes are nx_kinds.h's Maximum and Minimum, their selects and mask
   over vectors; booleans' are any and all, a lane 0 or 1. */
#define V_ADD(D, a, b) ((a) + (b))
#define V_MUL(D, a, b) ((a) * (b))
#define V_WRAP_ADD(D, a, b) ((D##_v)((D##_u)(a) + (D##_u)(b)))
#define V_WRAP_MUL(D, a, b) ((D##_v)((D##_u)(a) * (D##_u)(b)))
#define V_MAX(D, a, b) SEL(D##_u, D##_v, (a) > (b), a, b)
#define V_MIN(D, a, b) SEL(D##_u, D##_v, (a) < (b), a, b)
#define V_MAXIMUM(D, a, b)                                                   \
  ((D##_v)((D##_u)SEL(D##_u, D##_v, (a) != (a), a, V_MAX(D, a, b)) &         \
           ((D##_u)(a) | (D##_u)((a) != (b)))))
#define V_MINIMUM(D, a, b)                                                   \
  ((D##_v)((D##_u)SEL(D##_u, D##_v, (a) != (a), a, V_MIN(D, a, b)) |         \
           ((D##_u)(a) & ~(D##_u)((a) != (b)))))
#define V_ANY(D, a, b) ((a) | (D##_v)(((b) != 0) & 1))
#define V_ALL(D, a, b) ((a) & (D##_v)(((b) != 0) & 1))

/* 16 lanes of [T] as an array, of the suffix [D], for the monoids no
   vector op computes: complex products and extremes, and 64-bit integer
   ones the target lacks. ON_EACH's [NAME] combines the lanes with 16 terms
   by [F], lane by lane. */
#define ARRAYS(D, T)                                                         \
  typedef struct {                                                           \
    T k[16];                                                                 \
  } D##_l;                                                                   \
  static inline D##_l splat_##D(T e) {                                       \
    D##_l l;                                                                 \
    for (int k = 0; k < 16; k++) l.k[k] = e;                                 \
    return l;                                                                \
  }                                                                          \
  static inline D##_l load_##D(const T *x) {                                 \
    D##_l l;                                                                 \
    memcpy(l.k, x, sizeof l.k);                                              \
    return l;                                                                \
  }                                                                          \
  static inline void store_##D(T *x, D##_l l) { memcpy(x, l.k, sizeof l.k); }

#define ON_EACH(NAME, D, T, F)                                               \
  static inline void NAME(D##_l *l, const T *x) {                            \
    for (int k = 0; k < 16; k++) l->k[k] = F(l->k[k], x[k]);                 \
  }

/* The blocks a step of blocks_ takes at once at the lanes' suffix [D]:
   four, or two where the lanes span more than four vectors, so that the
   lanes stay in the vector registers. */
#define GROUP(D) (sizeof(D##_l) > 4 * NX_VECTOR_BYTES ? 2 : 4)

/* [FN] adds the [n] rows of [x], a row [st] further than the one before,
   into the [w] accumulators at [a], 32 columns at once as two lanes'
   worth, and returns the columns it took. */
#define PAIRS(FN, NAME, D, T)                                                \
  static inline int64_t FN(T *a, const T *x, int64_t st, int64_t n,          \
                           int64_t w) {                                      \
    int64_t j = 0;                                                           \
    for (; j + 32 <= w; j += 32) {                                           \
      D##_l v0 = load_##D(a + j), v1 = load_##D(a + j + 16);                 \
      for (int64_t i = 0; i < n; i++) {                                      \
        on_##NAME(&v0, x + i * st + j);                                      \
        on_##NAME(&v1, x + i * st + j + 16);                                 \
      }                                                                      \
      store_##D(a + j, v0);                                                  \
      store_##D(a + j + 16, v1);                                             \
    }                                                                        \
    return j;                                                                \
  }

/* A monoid's runs and columns 16 lanes at a time. run_ adds the [n] terms
   of [T] at [x] into the lanes [v] of the suffix [D], 16 at a time, and
   returns how many it took; it takes the identity [e] for WHOLE_RUNS's
   sake. cols_ is PAIRS's. */
#define RUNS(NAME, D, T)                                                     \
  static inline int64_t run_##NAME(D##_l *v, const T *x, int64_t n, T e) {   \
    (void)e;                                                                 \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) on_##NAME(v, x + i);                        \
    return i;                                                                \
  }                                                                          \
  PAIRS(cols_##NAME, NAME, D, T)

/* The terms an exact monoid's whole vectors hold, a multiple of 16. */
#define XN(D, T) ((int64_t)(sizeof(D##x_l) / sizeof(T)))

/* An exact monoid's run_ and cols_: RUNS's, with whole vectors Dx where
   they hold more than 16 lanes. run_ starts the whole vectors' lanes from
   [e] and folds them into the 16 lanes at the end of the run. Where a
   vector holds 16 columns or more, cols_ takes two vectors' columns at a
   time, then one vector's, then 16 columns, so that two chains are in
   flight; elsewhere it is PAIRS's. */
#define WHOLE_RUNS(NAME, D, T)                                               \
  static inline int64_t run_##NAME(D##_l *v, const T *x, int64_t n, T e) {   \
    int64_t i = 0;                                                           \
    if (XN(D, T) > 16 && n >= XN(D, T)) {                                    \
      D##x_l l = splat_##D##x(e);                                            \
      for (; i + XN(D, T) <= n; i += XN(D, T)) onx_##NAME(&l, x + i);        \
      T t[XN(D, T)];                                                         \
      int64_t m = XN(D, T);                                                  \
      /* Where a vector holds 16 lanes or more, its lane k is lane k mod     \
         16 in every vector: the four combine in pairs in registers, and     \
         one vector folds. */                                                \
      if (NX_VECTOR_BYTES / (int64_t)sizeof(T) >= 16) {                      \
        *(D##x_m *)t = opx_##NAME(opx_##NAME(l.a, l.b), opx_##NAME(l.c, l.d)); \
        m = NX_VECTOR_BYTES / (int64_t)sizeof(T);                            \
      } else                                                                 \
        store_##D##x(t, l);                                                  \
      for (int64_t k = 0; k < m; k += 16) on_##NAME(v, t + k);               \
    }                                                                        \
    for (; i + 16 <= n; i += 16) on_##NAME(v, x + i);                        \
    return i;                                                                \
  }                                                                          \
  PAIRS(pairs_##NAME, NAME, D, T)                                            \
  static inline int64_t cols_##NAME(T *a, const T *x, int64_t st, int64_t n, \
                                    int64_t w) {                             \
    int64_t j = 0, m = NX_VECTOR_BYTES / (int64_t)sizeof(T);                 \
    if (m < 16) return pairs_##NAME(a, x, st, n, w);                         \
    for (; j + 2 * m <= w; j += 2 * m) {                                     \
      D##x_v v0 = *(const D##x_m *)(a + j), v1 = *(const D##x_m *)(a + j + m); \
      for (int64_t i = 0; i < n; i++) {                                      \
        v0 = opx_##NAME(v0, *(const D##x_m *)(x + i * st + j));              \
        v1 = opx_##NAME(v1, *(const D##x_m *)(x + i * st + j + m));          \
      }                                                                      \
      *(D##x_m *)(a + j) = v0;                                               \
      *(D##x_m *)(a + j + m) = v1;                                           \
    }                                                                        \
    for (; j + m <= w; j += m) {                                             \
      D##x_v v = *(const D##x_m *)(a + j);                                   \
      for (int64_t i = 0; i < n; i++)                                        \
        v = opx_##NAME(v, *(const D##x_m *)(x + i * st + j));                \
      *(D##x_m *)(a + j) = v;                                                \
    }                                                                        \
    for (; j + 16 <= w; j += 16) {                                           \
      D##_l v = load_##D(a + j);                                             \
      for (int64_t i = 0; i < n; i++) on_##NAME(&v, x + i * st + j);         \
      store_##D(a + j, v);                                                   \
    }                                                                        \
    return j;                                                                \
  }

/* A lane step and a combine of [F] over elements of [T], whose lanes at
   the suffix [D] on_NAME combines. */
#define MONOID(NAME, D, T, F)                                                \
  static void lanes_##NAME(const uint8_t *x_, int64_t s, int64_t n,          \
                           const uint8_t *e_, uint8_t *l_, int first) {     \
    const T *x = (const T *)x_;                                              \
    T *l = (T *)l_, e;                                                       \
    memcpy(&e, e_, sizeof e);                                                \
    int64_t i = 0;                                                           \
    for (; i < n && ((first + i) & 15) != 0; i++)                            \
      l[(first + i) & 15] = F(l[(first + i) & 15], x[i * s]);                \
    if (s == 1 && n - i >= 16) {                                             \
      D##_l v = load_##D(l);                                                 \
      i += run_##NAME(&v, x + i, n - i, e);                                  \
      store_##D(l, v);                                                       \
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
  /* The lanes' tree: lane i takes lane i + 8, then i + 4, i + 2, i + 1. */  \
  static inline T fold_##NAME(D##_l v) {                                     \
    T l[16];                                                                 \
    store_##D(l, v);                                                         \
    for (int h = 8; h >= 1; h /= 2)                                          \
      for (int k = 0; k < h; k++) l[k] = F(l[k], l[k + h]);                  \
    return l[0];                                                             \
  }                                                                          \
  /* GROUP(D) blocks' lanes at once, so that more additions are in           \
     flight. */                                                              \
  static inline void group_##NAME(const T *x, T e, T *v) {                   \
    D##_l l0 = splat_##D(e), l1 = l0, l2 = l0, l3 = l0;                      \
    for (int i = 0; i < BLOCK; i += 16) {                                    \
      on_##NAME(&l0, x + i);                                                 \
      on_##NAME(&l1, x + BLOCK + i);                                         \
      if (GROUP(D) == 2) continue;                                           \
      on_##NAME(&l2, x + 2 * BLOCK + i);                                     \
      on_##NAME(&l3, x + 3 * BLOCK + i);                                     \
    }                                                                        \
    v[0] = fold_##NAME(l0);                                                  \
    v[1] = fold_##NAME(l1);                                                  \
    if (GROUP(D) == 2) return;                                               \
    v[2] = fold_##NAME(l2);                                                  \
    v[3] = fold_##NAME(l3);                                                  \
  }                                                                          \
  static inline void one_##NAME(const T *x, T e, T *v) {                     \
    D##_l l = splat_##D(e);                                                  \
    run_##NAME(&l, x, BLOCK, e);                                             \
    *v = fold_##NAME(l);                                                     \
  }                                                                          \
  static void blocks_##NAME(const uint8_t *x_, int64_t n, const uint8_t *e_, \
                            uint8_t *v_) {                                   \
    const T *x = (const T *)x_;                                              \
    T *v = (T *)v_, e;                                                       \
    memcpy(&e, e_, sizeof e);                                                \
    int64_t b = 0;                                                           \
    for (; b + GROUP(D) <= n; b += GROUP(D))                                 \
      group_##NAME(x + b * BLOCK, e, v + b);                                 \
    for (; b < n; b++) one_##NAME(x + b * BLOCK, e, v + b);                  \
  }                                                                          \
  static void column_##NAME(uint8_t *a_, const uint8_t *x_, int64_t st,      \
                            int64_t n, int64_t s, int64_t w) {               \
    T *a = (T *)a_;                                                          \
    const T *x = (const T *)x_;                                              \
    int64_t j = s == 1 ? cols_##NAME(a, x, st, n, w) : 0;                    \
    for (; j < w; j++) {                                                     \
      T v = a[j];                                                            \
      for (int64_t i = 0; i < n; i++) v = F(v, x[i * st + j * s]);           \
      a[j] = v;                                                              \
    }                                                                        \
  }                                                                          \
  static void few_##NAME(const uint8_t *x_, const int64_t *off, int n,       \
                         int64_t s, int64_t w, int64_t s2, int64_t h,        \
                         const uint8_t *e_, uint8_t *y_, int64_t sy,         \
                         int64_t sy2, uint8_t *l_) {                         \
    const T *x = (const T *)x_;                                              \
    T *y = (T *)y_, e, (*l)[STRIP(T)] = (T(*)[STRIP(T)])l_;                  \
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
          for (int64_t i = 0; i < k16; i += 16) start_##NAME(l[t] + i, e);   \
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

/* Sixteen lanes in memory, as few holds them: [pair] combines a's with
   b's, [start] takes x's terms into lanes from [e]. IN_LANES's go through
   the lanes of the suffix [D], IN_ARRAY's by [F], lane by lane, in
   place. */
#define IN_LANES(NAME, D, T)                                                 \
  static inline void pair_##NAME(T *restrict a, const T *restrict b) {       \
    D##_l v = load_##D(a);                                                   \
    on_##NAME(&v, b);                                                        \
    store_##D(a, v);                                                         \
  }                                                                          \
  static inline void start_##NAME(T *x, T e) {                               \
    D##_l v = splat_##D(e);                                                  \
    on_##NAME(&v, x);                                                        \
    store_##D(x, v);                                                         \
  }

#define IN_ARRAY(NAME, T, F)                                                 \
  static inline void pair_##NAME(T *restrict a, const T *restrict b) {       \
    for (int k = 0; k < 16; k++) a[k] = F(a[k], b[k]);                       \
  }                                                                          \
  static inline void start_##NAME(T *x, T e) {                               \
    for (int k = 0; k < 16; k++) x[k] = F(e, x[k]);                          \
  }

/* A monoid of a real dtype, its lanes combined by the vector op [OP]. */
#define VMONOID(NAME, D, T, W, F, OP)                                        \
  static inline D##_v op_##NAME(D##_v a, D##_v b) { return OP(D, a, b); }    \
  ON(on_##NAME, D, T, FIELDS(W), op_##NAME)                                  \
  IN_LANES(NAME, D, T)                                                       \
  RUNS(NAME, D, T)                                                           \
  MONOID(NAME, D, T, F)

/* An exact monoid of a real dtype: VMONOID's, its runs and columns over
   whole vectors too. */
#define XMONOID(NAME, D, T, W, F, OP)                                        \
  static inline D##_v op_##NAME(D##_v a, D##_v b) { return OP(D, a, b); }    \
  static inline D##x_v opx_##NAME(D##x_v a, D##x_v b) {                      \
    return OP(D##x, a, b);                                                   \
  }                                                                          \
  ON(on_##NAME, D, T, FIELDS(W), op_##NAME)                                  \
  ON(onx_##NAME, D##x, T, XFIELDS(W), opx_##NAME)                            \
  IN_LANES(NAME, D, T)                                                       \
  WHOLE_RUNS(NAME, D, T)                                                     \
  MONOID(NAME, D, T, F)

/* A monoid whose lanes are an array, combined lane by lane. */
#define AMONOID(NAME, D, T, F)                                               \
  ON_EACH(on_##NAME, D, T, F)                                                \
  IN_ARRAY(NAME, T, F)                                                       \
  RUNS(NAME, D, T)                                                           \
  MONOID(NAME, D, T, F)

/* A monoid of a real dtype whose vector op the target lacks: VMONOID's
   arguments, its lanes an array of their own. */
#define SMONOID(NAME, D, T, W, F, OP)                                        \
  ARRAYS(NAME, T)                                                            \
  AMONOID(NAME, NAME, T, F)

/* 64-bit integers' products multiply lane by lane: no target here has a
   vector multiply of 64-bit integers, which the compilers emulate at a
   third of the speed of scalar ones. Their extremes and Arg's lanes too
   where the target has no compare of 64-bit integers, x86-64 below
   SSE4.2. */
#if defined(__x86_64__) && !defined(__SSE4_2__)
#define EXTREMES_64 SMONOID
#define ARGS_64 ARGS_A
#else
#define EXTREMES_64 XMONOID
#define ARGS_64 ARGS
#endif

#define ADD(a, b) ((a) + (b))
#define MUL(a, b) ((a) * (b))
#define ANY(a, b) ((uint8_t)((a) | ((b) != 0)))
#define ALL(a, b) ((uint8_t)((a) & ((b) != 0)))

/* The four monoids of an integer dtype of storage [T], of [W] bytes,
   unsigned [U], at the compute suffix [S] for its extremes, [C] its compute
   type and [M] the unsigned type of [C]'s width, [PROD] and [EXT] the forms
   of its product and extremes. Sums and products run in [M], where they
   wrap: 8- and 16-bit operands would promote to int, whose product of two
   0xFFFF overflows. */
#define INTS(D, T, U, C, M, S, W, PROD, EXT)                                 \
  static inline T add_##D(T a, T b) { return (T)(U)((M)(U)a + (M)(U)b); }    \
  static inline T mul_##D(T a, T b) { return (T)(U)((M)(U)a * (M)(U)b); }    \
  static inline T max_##D(T a, T b) { return (T)nx_maximum_##S((C)a, (C)b); } \
  static inline T min_##D(T a, T b) { return (T)nx_minimum_##S((C)a, (C)b); } \
  WHOLE_VECTORS(D, T, U, W)                                                  \
  XMONOID(sum_##D, D, T, W, add_##D, V_WRAP_ADD)                             \
  PROD(prod_##D, D, T, W, mul_##D, V_WRAP_MUL)                               \
  EXT(max_##D, D, T, W, max_##D, V_MAX)                                      \
  EXT(min_##D, D, T, W, min_##D, V_MIN)

INTS(i8, int8_t, uint8_t, int32_t, uint32_t, i32, 1, XMONOID, XMONOID)
INTS(u8, uint8_t, uint8_t, uint32_t, uint32_t, u32, 1, XMONOID, XMONOID)
INTS(i16, int16_t, uint16_t, int32_t, uint32_t, i32, 2, XMONOID, XMONOID)
INTS(u16, uint16_t, uint16_t, uint32_t, uint32_t, u32, 2, XMONOID, XMONOID)
INTS(i32, int32_t, uint32_t, int32_t, uint32_t, i32, 4, XMONOID, XMONOID)
INTS(u32, uint32_t, uint32_t, uint32_t, uint32_t, u32, 4, XMONOID, XMONOID)
INTS(i64, int64_t, uint64_t, int64_t, uint64_t, i64, 8, SMONOID,
     EXTREMES_64)
INTS(u64, uint64_t, uint64_t, uint64_t, uint64_t, u64, 8, SMONOID,
     EXTREMES_64)

/* The four monoids of a float dtype of [T], [W] bytes, unsigned [U]. */
#define FLOATS4(D, T, U, W)                                                  \
  LANE_VECTORS(D, T, U, W)                                                   \
  VMONOID(sum_##D, D, T, W, ADD, V_ADD)                                      \
  VMONOID(prod_##D, D, T, W, MUL, V_MUL)                                     \
  VMONOID(max_##D, D, T, W, nx_maximum_##D, V_MAXIMUM)                       \
  VMONOID(min_##D, D, T, W, nx_minimum_##D, V_MINIMUM)

FLOATS4(f32, float, uint32_t, 4)
FLOATS4(f64, double, uint64_t, 8)

/* Booleans' Max and Min: a lane is 0 or 1 from its identity on. */
WHOLE_VECTORS(b, uint8_t, uint8_t, 1)
XMONOID(max_b, b, uint8_t, 1, ANY, V_ANY)
XMONOID(min_b, b, uint8_t, 1, ALL, V_ALL)

/* Complex sums, products and extremes. A sum adds each part as a float
   sum does; a product is nx_kinds.h's Mul, whose lanes start from their
   first term (plan's [first]); the extremes are its Maximum and Minimum,
   which order the real parts, then the imaginary ones. A sum's 16 lanes,
   of the suffix C##p, are its parts' sums' 32 lanes, of the parts' suffix
   [P], type [E] and [W] bytes, lanes 0 to 7 in [lo], 8 to 15 in [hi]. */
#define COMPLEX(C, P, E, W)                                                  \
  static inline nx_##C cadd_##C(nx_##C a, nx_##C b) {                        \
    return nx_##C##_of(a.re + b.re, a.im + b.im);                            \
  }                                                                          \
  typedef struct {                                                           \
    P##_l lo, hi;                                                            \
  } C##p_l;                                                                  \
  static inline C##p_l load_##C##p(const nx_##C *x) {                        \
    return (C##p_l){load_##P((const E *)x), load_##P((const E *)x + 16)};    \
  }                                                                          \
  static inline void store_##C##p(nx_##C *x, C##p_l l) {                     \
    store_##P((E *)x, l.lo);                                                 \
    store_##P((E *)x + 16, l.hi);                                            \
  }                                                                          \
  static inline C##p_l splat_##C##p(nx_##C e) {                              \
    /* the real part in the even elements, the imaginary in the odd */       \
    P##_v v;                                                                 \
    for (int k = 0; k < (int)(sizeof v / sizeof(E)); k++)                    \
      v[k] = k & 1 ? e.im : e.re;                                            \
    P##_l l;                                                                 \
    EACH(FIELDS(W), FIELD_SET, l, v)                                         \
    return (C##p_l){l, l};                                                   \
  }                                                                          \
  static inline void on_sum_##C(C##p_l *l, const nx_##C *x) {                \
    on_sum_##P(&l->lo, (const E *)x);                                        \
    on_sum_##P(&l->hi, (const E *)x + 16);                                   \
  }                                                                          \
  IN_LANES(sum_##C, C##p, nx_##C)                                            \
  RUNS(sum_##C, C##p, nx_##C)                                                \
  MONOID(sum_##C, C##p, nx_##C, cadd_##C)                                    \
  ARRAYS(C, nx_##C)                                                          \
  AMONOID(prod_##C, C, nx_##C, nx_mul_##C)                                   \
  AMONOID(max_##C, C, nx_##C, nx_maximum_##C)                                \
  AMONOID(min_##C, C, nx_##C, nx_minimum_##C)

COMPLEX(c64, f32, float, 4)
COMPLEX(c128, f64, double, 8)

/* Moments scans. A unit scans [w] slices side by side by the network of
   nx_kernel.mli, a state rows of [w] lanes: its mu, S and n, and in the
   exact pass z, whether one of its terms is a NaN, 0 or 1. Every lane's
   state at a place in the network counts the same terms, so n is held once,
   in its row's first element, and a join's weight nb / n is one division
   for all its lanes.

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

/* Prefetches, for a write where [rw] is 1, the row of [w] lanes at [y].
   Where a scan's lanes are contiguous, its terms' rows lie a row of the
   array apart. x86's stride prefetchers follow steps of at most 2 KiB and
   stay within a 4 KiB page (Intel's optimization manual), so rows further
   apart are not fetched ahead; rows closer, as few slices make, are, and
   their prefetches are redundant. */
#define PREFETCH_ROW(y, w, rw)                                                \
  for (int64_t b = 0; b < (w) * (int64_t)sizeof *(y); b += 64)                \
    __builtin_prefetch((const char *)(y) + b, rw)

/* The terms ahead whose rows a network scan prefetches: on fudge,
   scan-moments-f32-1024x4096-axis0 took 1.41 ms without, 0.89 at 8 terms,
   0.95 at 16. */
#define NETWORK_AHEAD 8

/* A pass's arithmetic. */
typedef enum { FAST, EXACT } arith;

/* The terms a network scan's tile holds: 2^NETWORK_P. A float32 Moments
   scan of 1M terms took 4.2 ms on one M1 core at 5, 3.5 at 6, 3.1 at 7,
   3.0 at 8, 2.9 at 9 and 10. */
#define NETWORK_P 8
#define PT (1 << NETWORK_P)

/* The streaming network's levels over a slice's tiles: fewer than 2^63
   terms make fewer than 2^(63 - NETWORK_P) tiles. */
#define TILE_LEVELS (64 - NETWORK_P)

/* The scratch elements a slice's tiles take: its levels' states, packed,
   their results, packed with one more a level, and the streaming
   network's sets; Moments', Logsumexp's being fewer. */
#define NETWORK_ONE                                                         \
  (4 * PT + 3 * (2 * PT + NETWORK_P + 1) + 12 * TILE_LEVELS)

/* Where a tile's level k starts among its states, packed level by level,
   level k's PT >> k of them, and among its results, one more a level. */
static int64_t states_at(int k) { return 2 * PT - (2 * PT >> k); }
static int64_t results_at(int k) { return states_at(k) + k; }

/* A Moments scan's fast pass over one slice by tiles. The network's results
   at a tile of PT terms aligned from the slice's start depend only on its
   terms and on the results before it at each level. Bottom-up, its terms'
   states join in pairs level by level, up to the tile's state at level
   NETWORK_P, which the streaming network over the tiles' states takes on
   to its result; top-down, each level's results follow from the next
   one's: an odd index's is the next level's result at its half, an even
   one's that level's result before joined with its own state. These are
   pass_'s joins of the same states in the same order, by fast_'s
   arithmetic, so the bits are pass_'s; each level's joins run over its
   positions at once. Level k's states hold 2^k terms each, so their joins'
   weights are the same for the whole level. Results hold position 0 for
   the result before the tile. */
#define TILES(T, S)                                                          \
  /* Level k's [m] states from level k - 1's pairs, their weights [f] and    \
     [g]: arguments, whose restrict lets the compiler vectorise. */          \
  static void up_##S(int64_t m, T f, T g, const T *restrict am,              \
                     const T *restrict as, T *restrict um, T *restrict us) { \
    for (int64_t i = 0; i < m; i++) {                                        \
      T u = am[2 * i], d = am[2 * i + 1] - u;                                \
      um[i] = u + d * f;                                                     \
      us[i] = (as[2 * i] + as[2 * i + 1]) + (d * d) * g;                     \
    }                                                                        \
  }                                                                          \
  /* The means and variances of [n] terms from term [t0], their results      \
     [rm], [rs] centred on [c], into [ym], [yv], stepping [ms], [vs]: 1      \
     where one is not finite. Apart, whose restrict lets the compiler        \
     vectorise the divisions. */                                             \
  static int results_##S(int64_t n, T c, int64_t t0, const T *restrict rm,   \
                         const T *restrict rs, T *restrict ym, int64_t ms,   \
                         T *restrict yv, int64_t vs) {                       \
    int bad = 0;                                                             \
    for (int64_t i = 0; i < n; i++) {                                        \
      T mean = c + rm[i], var = rs[i] / ((T)(t0 + i) + 1);                   \
      bad |= !(FINITE(mean) & FINITE(var));                                  \
      ym[i * ms] = mean;                                                     \
      yv[i * vs] = var;                                                      \
    }                                                                        \
    return bad;                                                              \
  }                                                                          \
  /* Level k's results at the even positions from [i0] on, of [e] of         \
     them: the next level's results before them [um], [us], [un], joined     \
     with their states [bm], [bs] of [nb] terms. */                          \
  static void down_##S(int64_t i0, int64_t e, T nb, const T *restrict um,    \
                       const T *restrict us, const T *restrict un,           \
                       const T *restrict bm, const T *restrict bs,           \
                       T *restrict om, T *restrict os, T *restrict on) {     \
    for (int64_t i = i0; i < e; i++) {                                       \
      T na = un[i], w = na + nb, f = nb / w, g = na * f;                     \
      T u = um[i], d = bm[2 * i] - u;                                        \
      om[2 * i + 1] = u + d * f;                                             \
      os[2 * i + 1] = (us[i] + bs[2 * i]) + (d * d) * g;                     \
      on[2 * i + 1] = w;                                                     \
    }                                                                        \
  }                                                                          \
  static int tiles_##S(const T *x, int64_t xs, T *ym, int64_t ms, T *yv,     \
                       int64_t vs, int64_t n, const nx_cpu_terms *src,       \
                       T *scratch) {                                         \
    /* The levels packed in [scratch]: states' level k at states_at(k),      \
       results' at results_at(k); then the streaming network over the        \
       tiles' states, one lane's. */                                         \
    /* The regions are apart: restrict lets the compiler vectorise. */       \
    T *restrict sm = scratch, *restrict ss = sm + 2 * PT;                    \
    T *restrict rm = ss + 2 * PT, *restrict rs = rm + 2 * PT + NETWORK_P + 1;\
    T *restrict rn = rs + 2 * PT + NETWORK_P + 1;                            \
    T *sets = rn + 2 * PT + NETWORK_P + 1;                                   \
    T *held[TILE_LEVELS], *in[TILE_LEVELS], *joined[TILE_LEVELS];            \
    const T *got[TILE_LEVELS];                                               \
    for (int k = 0; k < TILE_LEVELS; k++) {                                  \
      held[k] = sets + 12 * k;                                               \
      in[k] = held[k] + 4;                                                   \
      joined[k] = in[k] + 4;                                                 \
    }                                                                        \
    T c = 0;                                                                 \
    int bad = 0;                                                             \
    for (int64_t t0 = 0, q = 0; t0 < n; t0 += PT, q++) {                     \
      int64_t L = n - t0 < PT ? n - t0 : PT;                                 \
      /* The terms into level 0, a program's a tile of its own at a          \
         time, less the center, the first term where finite. */              \
      if (src != NULL)                                                       \
        for (int64_t i = 0, most = NX_CPU_NETWORK_TILE; i < L; i += most)    \
          src->fill(src->ctx, t0 + i, L - i < most ? L - i : most,           \
                    (uint8_t *)(sm + i));                                    \
      else                                                                   \
        for (int64_t i = 0; i < L; i++) sm[i] = x[(t0 + i) * xs];            \
      if (t0 == 0) c = FINITE(sm[0]) ? sm[0] : 0;                            \
      for (int64_t i = 0; i < L; i++) {                                      \
        sm[i] -= c;                                                          \
        ss[i] = 0;                                                           \
      }                                                                      \
      for (int k = 1; k <= NETWORK_P; k++) {                                 \
        T half = (T)(1 << (k - 1)), w = half + half, f = half / w;           \
        T g = half * f;                                                      \
        int64_t a = states_at(k - 1), u = states_at(k);                      \
        up_##S(L >> k, f, g, sm + a, ss + a, sm + u, ss + u);                \
      }                                                                      \
      if (L == PT) {                                                         \
        in[0][0] = sm[states_at(NETWORK_P)];                                 \
        in[0][1] = ss[states_at(NETWORK_P)];                                 \
        in[0][2] = (T)PT;                                                    \
        stream_##S(fast_##S, 1, q, held, in, joined, got);                   \
        int64_t r = results_at(NETWORK_P) + 1;                               \
        rm[r] = got[0][0];                                                   \
        rs[r] = got[0][1];                                                   \
        rn[r] = got[0][2];                                                   \
      }                                                                      \
      for (int k = NETWORK_P - 1; k >= 0; k--) {                             \
        int64_t m = L >> k, u = results_at(k + 1), o = results_at(k);        \
        T nb = (T)(1 << k);                                                  \
        const T *restrict um = rm + u, *restrict us = rs + u;                \
        const T *restrict un = rn + u, *restrict bm = sm + states_at(k);     \
        const T *restrict bs = ss + states_at(k);                            \
        T *restrict om = rm + o, *restrict os = rs + o;                      \
        T *restrict on = rn + o;                                             \
        for (int64_t i = 0; i < m / 2; i++) {                                \
          om[2 * i + 2] = um[i + 1];                                         \
          os[2 * i + 2] = us[i + 1];                                         \
          on[2 * i + 2] = un[i + 1];                                         \
        }                                                                    \
        int64_t i0 = 0;                                                      \
        if (t0 == 0 && m > 0) {                                              \
          om[1] = bm[0];                                                     \
          os[1] = bs[0];                                                     \
          on[1] = nb;                                                        \
          i0 = 1;                                                            \
        }                                                                    \
        down_##S(i0, (m + 1) / 2, nb, um, us, un, bm, bs, om, os, on);       \
      }                                                                      \
      bad |= results_##S(L, c, t0, rm + 1, rs + 1, ym + t0 * ms, ms,         \
                         yv + t0 * vs, vs);                                  \
      /* Each level's last result, the one before the next tile. */          \
      for (int k = 0; k <= NETWORK_P; k++) {                                 \
        int64_t o = results_at(k);                                           \
        rm[o] = rm[o + (L >> k)];                                            \
        rs[o] = rs[o + (L >> k)];                                            \
        rn[o] = rn[o + (L >> k)];                                            \
      }                                                                      \
    }                                                                        \
    return bad;                                                              \
  }

#define MOMENTS(T, S)                                                         \
  /* State [b] after state [a], into [r], each [w] elements a row. */       \
  static void fast_##S(int64_t w, const T *restrict a, const T *restrict b,   \
                       T *restrict r) {                                       \
    T na = a[2 * w], nb = b[2 * w], k = na + nb, f = nb / k, g = na * f;      \
    for (int64_t l = 0; l < w; l++) {                                         \
      T ma = a[l], d = b[l] - ma;                                             \
      r[l] = ma + d * f;                                                      \
      r[w + l] = (a[w + l] + b[w + l]) + (d * d) * g;                         \
    }                                                                         \
    r[2 * w] = k;                                                             \
  }                                                                           \
  static void exact_##S(int64_t w, const T *restrict a, const T *restrict b,  \
                        T *restrict r) {                                      \
    const T *ma = a, *va = a + w, *za = a + 3 * w;                            \
    const T *mb = b, *vb = b + w, *zb = b + 3 * w;                            \
    T na = a[2 * w], nb = b[2 * w], k = nx_add_##S(na, nb);                   \
    T f = nx_fdiv_##S(nb, k), g = nx_mul_##S(na, f);                          \
    T d[NX_CPU_NETWORK_LANES], t[NX_CPU_NETWORK_LANES],                       \
        sum[NX_CPU_NETWORK_LANES];                                            \
    LANES_OF(d, nx_sub_##S(mb[l], ma[l]));                                    \
    LANES_OF(t, nx_mul_##S(d[l], f));                                         \
    LANES_OF(t, nx_add_##S(ma[l], t[l]));                                     \
    LANES_OF(sum, nx_add_##S(ma[l], mb[l]));                                  \
    LANES_OF(r, (FINITE(ma[l]) & FINITE(mb[l]))                               \
                    ? t[l]                                                    \
                    : za[l] != 0 ? ma[l] : zb[l] != 0 ? mb[l] : sum[l]);      \
    LANES_OF(d, nx_mul_##S(d[l], d[l]));                                      \
    LANES_OF(d, nx_mul_##S(d[l], g));                                         \
    LANES_OF(t, nx_add_##S(va[l], vb[l]));                                    \
    LANES_OF(r + w, nx_add_##S(t[l], d[l]));                                  \
    r[2 * w] = k;                                                             \
    LANES_OF(r + 3 * w, (T)((za[l] != 0) | (zb[l] != 0)));                    \
  }                                                                           \
  /* The states of the terms [x] of each lane, shifted by [c]. */             \
  static void leaf_##S(arith a, int64_t w, const T *restrict x,               \
                       const T *restrict c, T *restrict r) {                  \
    if (a == EXACT) LANES_OF(r, nx_sub_##S(x[l], c[l]));                      \
    else LANES_OF(r, x[l] - c[l]);                                            \
    LANES_OF(r + w, 0);                                                       \
    r[2 * w] = 1;                                                             \
    if (a == EXACT) LANES_OF(r + 3 * w, (T)(x[l] != x[l]));                   \
  }                                                                           \
  /* [tn] rows of [w] elements from [y] into [r], row i lane l from           \
     y[i ys + l yl]. Element by element: nx_copy_box's transposed tiles       \
     took scan-moments-f32-4096x1024-axis1 1.04 times as long on fudge. */    \
  static void rows_##S(int64_t tn, int64_t w, const T *restrict y,            \
                       int64_t ys, int64_t yl, T *restrict r) {               \
    for (int64_t l = 0; l < w; l++)                                           \
      for (int64_t i = 0; i < tn; i++) r[i * w + l] = y[i * ys + l * yl];     \
  }                                                                           \
  /* [tn] rows of [w] elements from [r] into [y], as rows_ reads them. */     \
  static void unrows_##S(int64_t tn, int64_t w, const T *restrict r, T *y,    \
                         int64_t ys, int64_t yl) {                            \
    for (int64_t l = 0; l < w; l++)                                           \
      for (int64_t i = 0; i < tn; i++) y[i * ys + l * yl] = r[i * w + l];     \
  }                                                                           \
  /* Takes the [t]th state of a streaming network, at [in[0]], on through     \
     its levels, each level's [w] lanes' states 4 w elements: joins it with   \
     the states [held] there, and leaves the result at [t] at [got[0]]. */    \
  static void stream_##S(void (*join)(int64_t, const T *, const T *, T *),    \
                         int64_t w, int64_t t, T **held, T **in, T **joined,  \
                         const T **got) {                                     \
    int k = 0;                                                                \
    int64_t j = t;                                                            \
    for (; j & 1; k++, j >>= 1) join(w, held[k], in[k], in[k + 1]);           \
    if (j == 0) got[k] = in[k];                                               \
    else {                                                                    \
      join(w, got[k + 1], in[k], joined[k]);                                  \
      got[k] = joined[k];                                                     \
    }                                                                         \
    T *e = held[k];                                                           \
    held[k] = in[k];                                                          \
    in[k] = e;                                                                \
    for (int h = 0; h < k; h++) got[h] = got[k];                              \
  }                                                                           \
  /* One pass: 0, or 1 where the fast pass met a value not finite. Terms      \
     go in tiles of NX_CPU_NETWORK_TILE, and lanes not contiguous in [x],     \
     or in a result, through rows of the scratch, so that every loop over     \
     the lanes reads and writes contiguous elements. Terms that [src]         \
     computes go into those rows too, the first tile's first. */              \
  static int pass_##S(arith a, const T *x, int64_t xs, int64_t xl, T *ym,     \
                      int64_t ms, int64_t ml, T *yv, int64_t vs, int64_t vl,  \
                      int64_t n, int64_t w, T *scratch,                       \
                      const nx_cpu_terms *src) {                              \
    void (*join)(int64_t, const T *, const T *, T *) =                        \
        a == EXACT ? exact_##S : fast_##S;                                    \
    int64_t size = 4 * w, tile = NX_CPU_NETWORK_TILE;                         \
    int bad = 0, levels = nx_cpu_network_levels(n);                           \
    T c[NX_CPU_NETWORK_LANES];                                                \
    T *tx = scratch + 3 * levels * size, *tm = tx + tile * w,                 \
      *tv = tm + tile * w;                                                    \
    /* Computed terms are read from the tile's rows: xl is 0 then. */         \
    if (src != NULL) {                                                        \
      src->fill(src->ctx, 0, n < tile ? n : tile, (uint8_t *)tx);             \
      LANES_OF(c, FINITE(tx[l]) ? tx[l] : 0);                                 \
      xl = 0;                                                                 \
    } else                                                                    \
      LANES_OF(c, FINITE(x[l * xl]) ? x[l * xl] : 0);                         \
    /* Level k's waiting, arriving and joined states, and its result. */    \
    T *held[64], *in[64], *joined[64];                                        \
    const T *got[64];                                                         \
    for (int k = 0; k < levels; k++) {                                        \
      held[k] = scratch + 3 * k * size;                                       \
      in[k] = held[k] + size;                                                 \
      joined[k] = in[k] + size;                                               \
    }                                                                        \
    for (int64_t t0 = 0; t0 < n; t0 += tile) {                                \
      int64_t tn = n - t0 < tile ? n - t0 : tile;                             \
      if (src != NULL) {                                                      \
        if (t0 > 0) src->fill(src->ctx, t0, tn, (uint8_t *)tx);               \
      } else if (xl != 1)                                                     \
        rows_##S(tn, w, x + t0 * xs, xs, xl, tx);                             \
      for (int64_t i = 0; i < tn; i++) {                                      \
        int64_t t = t0 + i, p = t + NETWORK_AHEAD;                            \
        if (p < n && xl == 1) PREFETCH_ROW(x + p * xs, w, 0);                 \
        if (p < n && ml == 1) PREFETCH_ROW(ym + p * ms, w, 1);                \
        if (p < n && vl == 1) PREFETCH_ROW(yv + p * vs, w, 1);                \
        leaf_##S(a, w, xl == 1 ? x + t * xs : tx + i * w, c, in[0]);          \
        stream_##S(join, w, t, held, in, joined, got);                        \
        /* The mean c + mu, the variance S over the count, or mu less         \
           itself where mu is not finite. */                                  \
        const T *r = got[0];                                                  \
        T m = (T)t + 1;                                                       \
        T *dm = ml == 1 ? ym + t * ms : tm + i * w;                           \
        T *dv = vl == 1 ? yv + t * vs : tv + i * w;                           \
        if (a == EXACT) {                                                     \
          T nan[NX_CPU_NETWORK_LANES];                                        \
          LANES_OF(dm, nx_add_##S(c[l], r[l]));                               \
          LANES_OF(dv, nx_fdiv_##S(r[w + l], m));                             \
          LANES_OF(nan, nx_sub_##S(r[l], r[l]));                              \
          LANES_OF(dv, FINITE(r[l]) ? dv[l] : nan[l]);                        \
          continue;                                                           \
        }                                                                     \
        for (int64_t l = 0; l < w; l++) {                                     \
          T mean = c[l] + r[l], var = r[w + l] / m;                           \
          bad |= !(FINITE(mean) & FINITE(var));                               \
          dm[l] = mean;                                                       \
          dv[l] = var;                                                        \
        }                                                                     \
      }                                                                       \
      if (ml != 1) unrows_##S(tn, w, tm, ym + t0 * ms, ms, ml);               \
      if (vl != 1) unrows_##S(tn, w, tv, yv + t0 * vs, vs, vl);               \
    }                                                                         \
    return bad;                                                               \
  }                                                                           \
  TILES(T, S)                                                                 \
  static void moments_##S(const uint8_t *x, int64_t xs, int64_t xl,           \
                          uint8_t *mean, int64_t ms, int64_t ml,              \
                          uint8_t *var, int64_t vs, int64_t vl, int64_t n,    \
                          int64_t w, uint8_t *scratch,                        \
                          const nx_cpu_terms *src) {                          \
    /* A slice alone runs its fast pass by tiles, and the exact one, if      \
       needed, as pass_'s. */                                               \
    if (w == 1) {                                                            \
      if (tiles_##S((const T *)x, xs, (T *)mean, ms, (T *)var, vs, n, src,   \
                    (T *)scratch))                                           \
        pass_##S(EXACT, (const T *)x, xs, xl, (T *)mean, ms, ml, (T *)var,   \
                 vs, vl, n, w, (T *)scratch, src);                           \
      return;                                                                \
    }                                                                        \
    if (pass_##S(FAST, (const T *)x, xs, xl, (T *)mean, ms, ml, (T *)var, vs, \
                 vl, n, w, (T *)scratch, src))                                \
      pass_##S(EXACT, (const T *)x, xs, xl, (T *)mean, ms, ml, (T *)var, vs,  \
               vl, n, w, (T *)scratch, src);                                  \
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
     the variance, (Σ (x - c)² - Σ (x - c) (Σ (x - c) / n)) / n, +0 where
     that is negative, or Σ (x - c)² where that is +inf: Nx's expansion's
     arithmetic.
   A NaN result is replaced by the driver: by the first NaN term, or by
   nx_kinds.h's NaN where none is. Every function is inlined into each
   kernel, whose loops of 16 lanes the compiler then vectorises. */

#define KERNEL static __attribute__((flatten)) void

/* Arg's step of its lanes [L], beats_, by the compare over vectors
   (VBEATS) or lane by lane over an array (ABEATS): a lane takes the term
   where it beats the lane by [BEAT] or is a NaN by [NANQ]. */
#define VBEATS(NAME, L, T, W, BEAT, NANQ)                                    \
  static inline L##_v beat_##NAME(L##_v l, L##_v v) {                        \
    return SEL(L##_u, L##_v, (v BEAT l) | NANQ(v), v, l);                    \
  }                                                                          \
  ON(beats_##NAME, L, T, FIELDS(W), beat_##NAME)

#define ABEATS(NAME, L, T, W, BEAT, NANQ)                                    \
  static inline T beat_##NAME(T l, T v) {                                   \
    return v BEAT l || NANQ(v) ? v : l;                                      \
  }                                                                          \
  ON_EACH(beats_##NAME, L, T, beat_##NAME)

/* Arg by the extreme [F] over terms of [T] of [W] bytes, its lanes at the
   suffix [L] stepped by [BEATS], [U] the unsigned integer of its width:
   [BEAT] is [>] or [<], [F] without its order of NaNs and zeros, which the
   plain compare vectorises; [NANQ] tests a term, or a vector of them, for
   a NaN; [FIND] the pragma, if any, of the loop over 16 terms' bits. */
#define ARG(NAME, L, T, U, W, F, BEAT, NANQ, FIND, BEATS)                    \
  BEATS(NAME, L, T, W, BEAT, NANQ)                                           \
  /* The extreme by BEAT of the [n] terms at [x], or a NaN where one is:     \
     a lane takes each NaN it meets and keeps the last. */                   \
  static inline T top_##NAME(const T *x, int64_t n) {                        \
    L##_l v = splat_##L(x[0]);                                               \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16) beats_##NAME(&v, x + i);                    \
    T l[16];                                                                 \
    store_##L(l, v);                                                         \
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
      FIND for (int k = 0; k < 16; k++)                                      \
        hit |= bits_##NAME(&x[i + k]) == b;                                  \
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
    memcpy(st->v, &x[i], sizeof(T));                                         \
  }                                                                          \
  ARG_STATE(NAME, T, F, same_##U)

/* Arg's merge, results and scan over terms of [T] by the extreme [F],
   [SAME] whether the terms at two addresses have the same bits. */
#define ARG_STATE(NAME, T, F, SAME)                                          \
  static void arg_merge_##NAME(nx_cpu_state *a, const nx_cpu_state *b) {     \
    T va, vb;                                                                \
    memcpy(&va, a->v, sizeof va);                                            \
    memcpy(&vb, b->v, sizeof vb);                                            \
    T e = F(va, vb);                                                         \
    if (memcmp(&e, &va, sizeof e) != 0) *a = *b;                             \
  }                                                                          \
  static void arg_finish_##NAME(const nx_cpu_state *st, double c,            \
                                uint8_t *d0, uint8_t *d1) {                  \
    (void)c;                                                                 \
    memcpy(d0, st->v, sizeof(T));                                            \
    memcpy(d1, &st->at, sizeof st->at);                                      \
  }                                                                          \
  ARG_SCAN(NAME, T, F, SAME)

/* Arg's scan over terms of [T] by the extreme [F], [SAME] whether two of
   them have the same bits: a lane's extreme takes over where F of it and
   the term is not its bits, as arg_merge's does, so ties keep the earlier
   position and the first NaN holds; a lane of no term takes its first, as
   F of it and itself, which for booleans is 0 or 1. F returns one of its
   operands, or the same bits, so the lane keeps F's result. Lanes run side
   by side, row by row, so that the compiler vectorises their selects. */
#define ARG_SCAN(NAME, T, F, SAME)                                           \
  static void argscan_##NAME(nx_cpu_state *restrict a, const uint8_t *x_,    \
                             int64_t xs, int64_t xl, int64_t m, int64_t w,   \
                             uint8_t *y_, int64_t ys, int64_t yl,            \
                             int64_t *p, int64_t ps, int64_t pl) {           \
    const T *x = (const T *)x_;                                              \
    T *y = (T *)y_, e[16];                                                   \
    int64_t at[16], n[16];                                                   \
    for (int64_t l = 0; l < w; l++) {                                        \
      memcpy(&e[l], a[l].v, sizeof(T));                                      \
      at[l] = a[l].at;                                                       \
      n[l] = a[l].n;                                                         \
      if (at[l] < 0 && m > 0) {                                              \
        e[l] = F(x[l * xl], x[l * xl]);                                      \
        at[l] = n[l];                                                        \
      }                                                                      \
    }                                                                        \
    /* At the widths units take, a constant one, so that the lanes' states \
       stay in registers. */                                                \
    if (w == 1) {                                                            \
      ARG_ROWS(T, F, SAME, 1)                                                \
    } else if (w == 4) {                                                     \
      ARG_ROWS(T, F, SAME, 4)                                                \
    } else if (w == 16) {                                                    \
      ARG_ROWS(T, F, SAME, 16)                                               \
    } else {                                                                 \
      ARG_ROWS(T, F, SAME, w)                                                \
    }                                                                        \
    for (int64_t l = 0; l < w; l++) {                                        \
      memcpy(a[l].v, &e[l], sizeof(T));                                      \
      a[l].at = at[l];                                                       \
      a[l].n = n[l] + m;                                                     \
    }                                                                        \
  }                                                                          \
  static void argscan_merge_##NAME(nx_cpu_state *a, const nx_cpu_state *b) { \
    int64_t n = b->n;                                                        \
    if (a->at < 0) *a = *b;                                                  \
    else arg_merge_##NAME(a, b);                                             \
    a->n = n;                                                                \
  }

/* argscan_'s rows over [W] lanes. */
#define ARG_ROWS(T, F, SAME, W)                                              \
  for (int64_t i = 0; i < m; i++)                                            \
    for (int64_t l = 0; l < (W); l++) {                                      \
      T f = F(e[l], x[i * xs + l * xl]);                                     \
      at[l] = SAME(&f, &e[l]) ? at[l] : n[l] + i;                            \
      e[l] = f;                                                              \
      y[i * ys + l * yl] = f;                                                \
      p[i * ps + l * pl] = at[l];                                            \
    }

/* Whether the terms at [a] and [b] have the same bits: a real's as its
   unsigned integer [U], a complex number's as its bytes. */
#define SAME_AS(U)                                                           \
  static inline int same_##U(const void *a, const void *b) {                 \
    U x, y;                                                                  \
    memcpy(&x, a, sizeof x);                                                 \
    memcpy(&y, b, sizeof y);                                                 \
    return x == y;                                                           \
  }

SAME_AS(uint8_t)
SAME_AS(uint16_t)
SAME_AS(uint32_t)
SAME_AS(uint64_t)

static inline int same_c64(const void *a, const void *b) {
  return memcmp(a, b, sizeof(nx_c64)) == 0;
}

static inline int same_c128(const void *a, const void *b) {
  return memcmp(a, b, sizeof(nx_c128)) == 0;
}

#define IS_NAN(v) ((v) != (v))
#define NO_NAN(v) 0

/* Arg's search for the extreme's bits unrolls by 8, which keeps gcc at
   -O3 from unrolling it whole and leaving it scalar, but not for 64-bit
   integers, which at every count run slower at -O3 on arm64. */
#define ARGS(D, T, U, W, F_MAX, F_MIN, NANQ, FIND)                           \
  ARG(max_##D, D, T, U, W, F_MAX, >, NANQ, FIND, VBEATS)                     \
  ARG(min_##D, D, T, U, W, F_MIN, <, NANQ, FIND, VBEATS)

/* ARGS whose lanes step lane by lane, over the arrays of lanes that
   SMONOID made for the dtype's extremes (EXTREMES_64). */
#define ARGS_A(D, T, U, W, F_MAX, F_MIN, NANQ, FIND)                         \
  ARG(max_##D, max_##D, T, U, W, F_MAX, >, NANQ, FIND, ABEATS)               \
  ARG(min_##D, min_##D, T, U, W, F_MIN, <, NANQ, FIND, ABEATS)

ARGS(i8, int8_t, uint8_t, 1, max_i8, min_i8, NO_NAN, UNROLL(8))
ARGS(u8, uint8_t, uint8_t, 1, max_u8, min_u8, NO_NAN, UNROLL(8))
ARGS(i16, int16_t, uint16_t, 2, max_i16, min_i16, NO_NAN, UNROLL(8))
ARGS(u16, uint16_t, uint16_t, 2, max_u16, min_u16, NO_NAN, UNROLL(8))
ARGS(i32, int32_t, uint32_t, 4, max_i32, min_i32, NO_NAN, UNROLL(8))
ARGS(u32, uint32_t, uint32_t, 4, max_u32, min_u32, NO_NAN, UNROLL(8))
ARGS_64(i64, int64_t, uint64_t, 8, max_i64, min_i64, NO_NAN, )
ARGS_64(u64, uint64_t, uint64_t, 8, max_u64, min_u64, NO_NAN, )
ARGS(f32, float, uint32_t, 4, nx_maximum_f32, nx_minimum_f32,
     IS_NAN, UNROLL(8))
ARGS(f64, double, uint64_t, 8, nx_maximum_f64, nx_minimum_f64,
     IS_NAN, UNROLL(8))

/* Arg over complex terms of [C] by [F]: the lanes' extreme by F, a NaN
   where a term is one, then its first term of the extreme's bits, or the
   first NaN. Without NaNs the order is total, so the extreme's bits are one
   number's. */
#define ARG_COMPLEX(NAME, C, F)                                              \
  KERNEL arg_##NAME(const uint8_t *x_, int64_t n, int64_t t0, double c,      \
                    nx_cpu_state *st) {                                      \
    (void)c;                                                                 \
    const nx_##C *x = (const nx_##C *)x_;                                    \
    nx_##C l[16];                                                            \
    for (int k = 0; k < 16; k++) l[k] = x[0];                                \
    int64_t i = 0;                                                           \
    for (; i + 16 <= n; i += 16)                                             \
      for (int k = 0; k < 16; k++) l[k] = F(l[k], x[i + k]);                 \
    for (; i < n; i++) l[0] = F(l[0], x[i]);                                 \
    for (int k = 1; k < 16; k++) l[0] = F(l[0], l[k]);                       \
    i = 0;                                                                   \
    if (nx_nan_##C(l[0]))                                                    \
      while (!nx_nan_##C(x[i])) i++;                                         \
    else                                                                     \
      while (memcmp(&x[i], &l[0], sizeof l[0]) != 0) i++;                    \
    *st = (nx_cpu_state){.at = t0 + i};                                      \
    memcpy(st->v, &x[i], sizeof x[i]);                                       \
  }                                                                          \
  ARG_STATE(NAME, nx_##C, F, same_##C)

/* Arg over booleans, a byte true where it is not 0: Max's first true term,
   else the first, Min's first false one, else the first; the extreme's
   bits are 1 or 0, which TRUE_OR and TRUE_AND give of any bytes. */
#define TRUE_OR(a, b) ((uint8_t)(((a) != 0) | ((b) != 0)))
#define TRUE_AND(a, b) ((uint8_t)(((a) != 0) & ((b) != 0)))

#define ARG_BOOL(NAME, F, WANT)                                              \
  KERNEL arg_##NAME(const uint8_t *x, int64_t n, int64_t t0, double c,       \
                    nx_cpu_state *st) {                                      \
    (void)c;                                                                 \
    int64_t i = 0;                                                           \
    while (i < n && (x[i] != 0) != WANT) i++;                                \
    if (i == n) i = 0;                                                       \
    *st = (nx_cpu_state){.at = t0 + i};                                      \
    st->v[0] = x[i] != 0;                                                    \
  }                                                                          \
  ARG_STATE(NAME, uint8_t, F, same_uint8_t)

ARG_BOOL(max_b, TRUE_OR, 1)
ARG_BOOL(min_b, TRUE_AND, 0)

ARG_COMPLEX(max_c64, c64, nx_maximum_c64)
ARG_COMPLEX(min_c64, c64, nx_minimum_c64)
ARG_COMPLEX(max_c128, c128, nx_maximum_c128)
ARG_COMPLEX(min_c128, c128, nx_minimum_c128)

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
      UNROLL(16) for (int k = 0; k < 16; k++)                                \
        l[k] += ex_##S(x[i + k] - c);                                        \
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
  /* The lanes' tree over lanes 0 to 7 [a] and 8 to 15 [b]. */               \
  static inline T halves_##S(T *a, const T *b) {                             \
    for (int k = 0; k < 8; k++) a[k] += b[k];                                \
    for (int h = 4; h >= 1; h /= 2)                                          \
      for (int k = 0; k < h; k++) a[k] += a[k + h];                          \
    return a[0];                                                             \
  }                                                                          \
  /* Moments' loops over the lanes unroll by 8, under a bound, i <= n - 16,  \
     that cannot overflow. Without the pragma gcc keeps the lanes in         \
     memory at -O2. At -O3 it unrolls Var's loop over four arrays of 8       \
     lanes whole and vectorises the loop over i instead, with permutes;      \
     over two arrays of 16 lanes it does not. Under OCaml's -fwrapv it       \
     cannot count the steps of i + 16 <= n, which may wrap, and at -O3 on    \
     arm64 leaves the lanes in memory. */                                    \
  KERNEL mean_##S(const uint8_t *x_, int64_t n, int64_t t0, double c,        \
                  nx_cpu_state *st) {                                        \
    (void)t0;                                                                \
    (void)c;                                                                 \
    const T *x = (const T *)x_;                                              \
    T a[8] = {0}, b[8] = {0};                                                \
    int64_t i = 0;                                                           \
    for (; i <= n - 16; i += 16)                                             \
      UNROLL(8) for (int k = 0; k < 8; k++) {                                \
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
    T c = (T)c_, s[16] = {0}, p[16] = {0};                                   \
    int64_t i = 0;                                                           \
    for (; i <= n - 16; i += 16)                                             \
      UNROLL(8) for (int k = 0; k < 16; k++) {                               \
        T d = x[i + k] - c;                                                  \
        s[k] += d * d;                                                       \
        p[k] += d;                                                           \
      }                                                                      \
    for (; i < n; i++) {                                                     \
      T d = x[i] - c;                                                        \
      s[i & 15] += d * d;                                                    \
      p[i & 15] += d;                                                        \
    }                                                                        \
    *st = (nx_cpu_state){.n = n, .s = halves_##S(s, s + 8),                  \
                         .d = halves_##S(p, p + 8)};                         \
  }                                                                          \
  static void var_merge_##S(nx_cpu_state *a, const nx_cpu_state *b) {        \
    sum_merge_##S(a, b);                                                     \
    a->d = (T)a->d + (T)b->d;                                                \
  }                                                                          \
  /* A mean that is a NaN is the variance too. Squares summed to +inf are  \
     the variance, where d * (d / n) may overflow too and v be a NaN. */    \
  static void var_finish_##S(const nx_cpu_state *st, double c, uint8_t *d0,  \
                             uint8_t *d1) {                                  \
    T mean = MADE, var = MADE;                                               \
    if (st->n > 0) {                                                         \
      T n = (T)st->n, d = (T)st->d, s = (T)st->s;                            \
      T v = (s - d * (d / n)) / n;                                           \
      mean = (T)c;                                                           \
      var = c != c ? mean : s == (T)INFINITY ? s : v < 0 ? 0 : v;            \
    }                                                                        \
    memcpy(d0, &mean, sizeof mean);                                          \
    memcpy(d1, &var, sizeof var);                                            \
  }                                                                          \
  /* Logsumexp's network (nx_kernel.mli). The sum of state [a], of        \
     maximum [ma] and sum [sa], then state [b], [m] the two's maximum:       \
     where m is finite, the sum of the side whose maximum is m as it is,     \
     its exp 0 being 1, plus the other's rescaled; 0 where m is -∞; m        \
     itself elsewhere, +∞ as the network makes it, or a NaN no result       \
     reads. Branch free, so that its loops vectorise. */                     \
  static inline T lse_sum_##S(T ma, T sa, T mb, T sb, T m) {                 \
    int hi = ma == m;                                                        \
    T lo = hi ? mb : ma, sl = hi ? sb : sa, sh = hi ? sa : sb;               \
    T s = sh + sl * ex_##S(lo - m);                                          \
    return FINITE(m) ? s : m == -(T)INFINITY ? 0 : m;                        \
  }                                                                          \
  /* A term's sum: exp 0 where finite, 0 for -∞, else the term. */           \
  static inline T lse_leaf_##S(T x) {                                        \
    return FINITE(x) ? 1 : x == -(T)INFINITY ? 0 : x;                        \
  }                                                                          \
  static inline T lse_result_##S(T m, T s) {                                 \
    T r = m + nx_log_##S(s);                                                 \
    return FINITE(m) ? r : m;                                                \
  }                                                                          \
  /* State [b] after state [a], into [r], each [w] maxima then [w] sums. */  \
  static void lse_join_##S(int64_t w, const T *restrict a,                   \
                           const T *restrict b, T *restrict r) {             \
    for (int64_t l = 0; l < w; l++) {                                        \
      T m = nx_maximum_##S(a[l], b[l]);                                      \
      r[w + l] = lse_sum_##S(a[l], a[w + l], b[l], b[w + l], m);             \
      r[l] = m;                                                              \
    }                                                                        \
  }                                                                          \
  /* Pairs of level k - 1's [2m] states into level k's [m]. */               \
  static void lse_up_##S(int64_t m, const T *restrict am,                    \
                         const T *restrict as, T *restrict um,               \
                         T *restrict us) {                                   \
    for (int64_t i = 0; i < m; i++) {                                        \
      T ma = am[2 * i], mb = am[2 * i + 1], g = nx_maximum_##S(ma, mb);      \
      us[i] = lse_sum_##S(ma, as[2 * i], mb, as[2 * i + 1], g);              \
      um[i] = g;                                                             \
    }                                                                        \
  }                                                                          \
  /* Level k's results at the even positions from [i0] on, of [e] of        \
     them: the next level's results before them [um], [us], joined with     \
     their states [bm], [bs]. */                                            \
  static void lse_down_##S(int64_t i0, int64_t e, const T *restrict um,      \
                           const T *restrict us, const T *restrict bm,       \
                           const T *restrict bs, T *restrict om,             \
                           T *restrict os) {                                 \
    for (int64_t i = i0; i < e; i++) {                                       \
      T ma = um[i], mb = bm[2 * i], g = nx_maximum_##S(ma, mb);              \
      os[2 * i + 1] = lse_sum_##S(ma, us[i], mb, bs[2 * i], g);              \
      om[2 * i + 1] = g;                                                     \
    }                                                                        \
  }                                                                          \
  /* One slice by tiles, as tiles_ scans Moments: the same joins in the     \
     same order as lse_pass_'s, each level's at once. */                     \
  static void lse_tiles_##S(const T *x, int64_t xs, T *y, int64_t ys,        \
                            int64_t n, const nx_cpu_terms *src,              \
                            T *scratch) {                                    \
    T *restrict sm = scratch, *restrict ss = sm + 2 * PT;                    \
    T *restrict rm = ss + 2 * PT, *restrict rs = rm + 2 * PT + NETWORK_P + 1;\
    T *sets = rs + 2 * PT + NETWORK_P + 1;                                   \
    T *held[TILE_LEVELS], *in[TILE_LEVELS], *joined[TILE_LEVELS];            \
    const T *got[TILE_LEVELS];                                               \
    for (int k = 0; k < TILE_LEVELS; k++) {                                  \
      held[k] = sets + 6 * k;                                                \
      in[k] = held[k] + 2;                                                   \
      joined[k] = in[k] + 2;                                                 \
    }                                                                        \
    for (int64_t t0 = 0, q = 0; t0 < n; t0 += PT, q++) {                     \
      int64_t L = n - t0 < PT ? n - t0 : PT;                                 \
      if (src != NULL)                                                       \
        for (int64_t i = 0, most = NX_CPU_NETWORK_TILE; i < L; i += most)    \
          src->fill(src->ctx, t0 + i, L - i < most ? L - i : most,           \
                    (uint8_t *)(sm + i));                                    \
      else                                                                   \
        for (int64_t i = 0; i < L; i++) sm[i] = x[(t0 + i) * xs];            \
      for (int64_t i = 0; i < L; i++) ss[i] = lse_leaf_##S(sm[i]);           \
      for (int k = 1; k <= NETWORK_P; k++) {                                 \
        int64_t a = states_at(k - 1), u = states_at(k);                      \
        lse_up_##S(L >> k, sm + a, ss + a, sm + u, ss + u);                  \
      }                                                                      \
      if (L == PT) {                                                         \
        in[0][0] = sm[states_at(NETWORK_P)];                                 \
        in[0][1] = ss[states_at(NETWORK_P)];                                 \
        stream_##S(lse_join_##S, 1, q, held, in, joined, got);               \
        int64_t r = results_at(NETWORK_P) + 1;                               \
        rm[r] = got[0][0];                                                   \
        rs[r] = got[0][1];                                                   \
      }                                                                      \
      for (int k = NETWORK_P - 1; k >= 0; k--) {                             \
        int64_t m = L >> k, u = results_at(k + 1), o = results_at(k);        \
        const T *restrict um = rm + u, *restrict us = rs + u;                \
        const T *restrict bm = sm + states_at(k);                            \
        const T *restrict bs = ss + states_at(k);                            \
        T *restrict om = rm + o, *restrict os = rs + o;                      \
        for (int64_t i = 0; i < m / 2; i++) {                                \
          om[2 * i + 2] = um[i + 1];                                         \
          os[2 * i + 2] = us[i + 1];                                         \
        }                                                                    \
        int64_t i0 = 0;                                                      \
        if (t0 == 0 && m > 0) {                                              \
          om[1] = bm[0];                                                     \
          os[1] = bs[0];                                                     \
          i0 = 1;                                                            \
        }                                                                    \
        lse_down_##S(i0, (m + 1) / 2, um, us, bm, bs, om, os);               \
      }                                                                      \
      for (int64_t i = 0; i < L; i++)                                        \
        y[(t0 + i) * ys] = lse_result_##S(rm[1 + i], rs[1 + i]);             \
      for (int k = 0; k <= NETWORK_P; k++) {                                 \
        int64_t o = results_at(k);                                           \
        rm[o] = rm[o + (L >> k)];                                            \
        rs[o] = rs[o + (L >> k)];                                            \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  /* [w] slices side by side, as pass_ scans Moments: a state of rows of   \
     [w] maxima then [w] sums, terms and results through rows of the        \
     scratch where their lanes are not contiguous. */                       \
  static void lse_pass_##S(const T *x, int64_t xs, int64_t xl, T *y,         \
                           int64_t ys, int64_t yl, int64_t n, int64_t w,     \
                           T *scratch, const nx_cpu_terms *src) {            \
    int64_t size = 2 * w, tile = NX_CPU_NETWORK_TILE;                        \
    int levels = nx_cpu_network_levels(n);                                   \
    T *tx = scratch + 3 * levels * size, *ty = tx + tile * w;                \
    /* Computed terms are read from the tile's rows. */                      \
    if (src != NULL) xl = 0;                                                 \
    T *held[64], *in[64], *joined[64];                                       \
    const T *got[64];                                                        \
    for (int k = 0; k < levels; k++) {                                       \
      held[k] = scratch + 3 * k * size;                                      \
      in[k] = held[k] + size;                                                \
      joined[k] = in[k] + size;                                              \
    }                                                                        \
    for (int64_t t0 = 0; t0 < n; t0 += tile) {                               \
      int64_t tn = n - t0 < tile ? n - t0 : tile;                            \
      if (src != NULL) src->fill(src->ctx, t0, tn, (uint8_t *)tx);           \
      else if (xl != 1) rows_##S(tn, w, x + t0 * xs, xs, xl, tx);            \
      for (int64_t i = 0; i < tn; i++) {                                     \
        int64_t t = t0 + i, p = t + NETWORK_AHEAD;                           \
        if (p < n && xl == 1) PREFETCH_ROW(x + p * xs, w, 0);                \
        if (p < n && yl == 1) PREFETCH_ROW(y + p * ys, w, 1);                \
        const T *xi = xl == 1 ? x + t * xs : tx + i * w;                     \
        for (int64_t l = 0; l < w; l++) {                                    \
          in[0][l] = xi[l];                                                  \
          in[0][w + l] = lse_leaf_##S(xi[l]);                                \
        }                                                                    \
        stream_##S(lse_join_##S, w, t, held, in, joined, got);               \
        const T *r = got[0];                                                 \
        T *dy = yl == 1 ? y + t * ys : ty + i * w;                           \
        for (int64_t l = 0; l < w; l++)                                      \
          dy[l] = lse_result_##S(r[l], r[w + l]);                            \
      }                                                                      \
      if (yl != 1) unrows_##S(tn, w, ty, y + t0 * ys, ys, yl);               \
    }                                                                        \
  }                                                                          \
  static void logsumexp_##S(const uint8_t *x, int64_t xs, int64_t xl,        \
                            uint8_t *y, int64_t ys, int64_t yl, uint8_t *y1, \
                            int64_t s1, int64_t l1, int64_t n, int64_t w,    \
                            uint8_t *scratch, const nx_cpu_terms *src) {     \
    (void)y1;                                                                \
    (void)s1;                                                                \
    (void)l1;                                                                \
    if (w == 1)                                                              \
      lse_tiles_##S((const T *)x, xs, (T *)y, ys, n, src, (T *)scratch);     \
    else                                                                     \
      lse_pass_##S((const T *)x, xs, xl, (T *)y, ys, yl, n, w, (T *)scratch, \
                   src);                                                     \
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

/* A scan's kernels R and M at the suffix D. */
#define SCAN(R, M, D) ((nx_cpu_scan_pass){R##_##D, M##_##D})
#define NO_SCAN ((nx_cpu_scan_pass){NULL, NULL})

#define STATE(R, DT, CENTER, VALUE, SCANS)                                   \
  t->stateful[R - NX_CPU_MONOIDS][DT] =                                      \
      (nx_cpu_stateful){CENTER, VALUE, SCANS}

#define ARG2(DT, D)                                                          \
  STATE(NX_CPU_ARG_MAX, DT, NONE, PASS(arg, arg_merge, arg_finish, max_##D), \
        SCAN(argscan, argscan_merge, max_##D));                              \
  STATE(NX_CPU_ARG_MIN, DT, NONE, PASS(arg, arg_merge, arg_finish, min_##D), \
        SCAN(argscan, argscan_merge, min_##D))

#define FLOATS(DT, D)                                                        \
  STATE(NX_CPU_LOGSUMEXP, DT, PASS(top, top_merge, top_finish, D),           \
        PASS(lse, lse_merge, lse_finish, D),                                 \
        NO_SCAN);                                                            \
  STATE(NX_CPU_MOMENTS, DT, PASS(mean, sum_merge, mean_finish, D),           \
        PASS(var, var_merge, var_finish, D), NO_SCAN)

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
  ARG2(NX_BOOL, b);
  ARG2(NX_COMPLEX64, c64);
  ARG2(NX_COMPLEX128, c128);
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
  ALL4(NX_COMPLEX64, c64);
  ALL4(NX_COMPLEX128, c128);
  t->logsumexp[NX_FLOAT32] = logsumexp_f32;
  t->logsumexp[NX_FLOAT64] = logsumexp_f64;
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

#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>


#define LANES NX_CPU_LANES
#define SCAN_CHUNK 4096

/* A streaming unit's row of outputs: 512 bytes, 128 float32. Byte buffers
   hold elements of the dtype, read and written as such, so they are aligned
   as the widest. */
#define ROW 512

/* A monoid scan's row of outputs on one thread, where its outputs lie side
   by side: 16 KiB, so that a unit sweeps rows of terms as long runs. A
   scan of 1024x4096 float32 down its columns took 3.75 ms on one M1 core
   in units of ROW, whose 512-byte runs lie 16 KiB apart, and 0.56 ms in
   these; on ten threads, units of ROW took 0.64 ms and of 410 outputs
   0.80. */
#define SCAN_ROW 16384

/* The rows ahead whose terms a scan of a unit's row of outputs
   prefetches, and on ARM its results: its runs of at most 512 bytes lie a
   row of the operand apart, which the hardware fetches late. On ten M1
   threads, scan-sum-f32-1024x4096-axis0 took about 545 us without, about
   370 with both at 2, 4 or 8 rows ahead, 460 at 16, and the terms' alone
   gained nothing. On fudge's six threads, the terms' alone took it from
   308 to 296 us, and with the results' to 353. */
#define SCAN_AHEAD 8

/* The bytes of the widest element, a complex128. */
#define WIDEST 16

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

/* What a call computes: a reduction, a scan, or a scan by nx_kernel.mli's
   network, Logsumexp's or Moments'. */
typedef enum { REDUCE, SCAN, NETWORK } task;

/* A program's output as the terms (Programs): output [k] of [prog], bound
   to its [nloads] loads [loads], read at virtual positions, a term's place
   in C order of the loop's shape. The loop of the virtual operand and the
   loads coalesced has [rank] axes of extents [ext]; [vs] are the virtual
   positions' steps along them, C-contiguous, [ls] each load's and [l0]
   each load's first position. [index] says whether the program reads the
   index, the virtual position. */
typedef struct {
  const nx_cpu_prog *prog;
  const nx_array *loads;
  int k, nloads, index, rank;
  int64_t ext[NX_MAX_RANK], vs[NX_MAX_RANK];
  int64_t ls[NX_MAX_OPERANDS][NX_MAX_RANK], l0[NX_MAX_OPERANDS];
} fused;

/* An output's value is an element of the dtype under a monoid, a state
   under a stateful reduction. */
typedef struct {
  const nx_cpu_monoid *f;   /* a monoid's kernels, or NULL */
  const nx_cpu_pass *r;     /* a stateful reduction's pass, or NULL */
  const nx_cpu_scan_pass *z; /* an Arg scan's kernels, or NULL */
  double *centers;          /* its outputs' centers, a row of them a unit */
  int centering;            /* the pass stores the centers */
  int w, nan;               /* nan: NaN results take their first NaN term */
  int part, parts;          /* the dtype of a float or its parts, 1 or 2 */
  int first;                /* lanes, rows and slices start from their
                               first term: a complex product's */
  int extreme;              /* a Max or a Min, whose scan's carry is a
                               term before its slice */
  int slots;                /* 2 for a complex sum's parts, NaN apart, or 1 */
  int carrier, staged;      /* the dtype terms reduce in; src is not in it */
  int own;                  /* Arg's extreme is of src's dtype, not the
                               carrier's: its term, copied */
  int arg;                  /* an Arg, whose chunk totals are reductions */
  int64_t size;             /* the bytes of an output's value */
  _Alignas(16) uint8_t id[WIDEST];
  /* The destinations, the second the first again for one, and the operand
     as loaded, or a program's virtual one, of no buffer, read through
     [z]. */
  const nx_array *dst, *dst1, *src;
  const fused *fused;
  /* A scan's carries, of the carrier, or NULL: the destination's shape but
     one along its axis. */
  const nx_array *cy;
  /* The reduced axes in C order: extents and the operand's steps. */
  int nr;
  int64_t re[NX_MAX_RANK], rs[NX_MAX_RANK];
  /* The kept axes, the row axis last: extents, the operand's and the
     destinations' steps. */
  int nk;
  int64_t ke[NX_MAX_RANK], ks[NX_MAX_RANK], kd[NX_MAX_RANK], kd1[NX_MAX_RANK];
  int64_t kc[NX_MAX_RANK]; /* the carries' steps */
  int64_t x0, d0, d1, c0;
  int64_t rd, rd1; /* a scan's destinations' steps along its axis */
  int slices; /* a unit's outputs are slices scanned side by side */
  int64_t terms, blocks;
  int threads;    /* the most threads the call's jobs run on */
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
  _Atomic int *failed; /* set where a thread's work could not be taken */
} plan;

/* A thread's scratch for its units: every buffer the paths under a unit
   hold, in one of the thread's kept buffers (nx_cpu_take), so that its
   stack holds frames alone. A thread OCaml makes may have 128 KiB of
   stack (musl's default), and a unit runs on it under the door's and the
   plan's frames: the deepest chain from an entry of this file, through the
   other files of nx.cpu, took at most 31 KiB, and 40 KiB from a program's
   reduction, 12 KiB of it the door's descriptors (gcc's -O3 call graphs,
   arm64 and x86-64). Each buffer is named for the paths that use it: no
   path writes a buffer while a caller's data in it is live. */
typedef struct {
  /* The counter's values: the blocks' (blocks) or the groups'
     (finish_units), and a run of the table's blocks. The buffers every
     reduction takes come first, so that a short one touches the work's
     first pages alone. */
  _Alignas(64) uint8_t run[RUN * WIDEST];
  _Alignas(64) uint8_t tree[LEVELS][ROW];
  /* A block's lanes (block, staged_block), or a few-term reduction's
     (few). */
  _Alignas(64) uint8_t lanes[NX_CPU_FEW_BYTES];
  /* Terms in the carrier: a block's (states, staged_block), a unit's
     (few), a piece's (run_states, quad), or one output's (nan_term). */
  _Alignas(64) uint8_t terms[2 * BLOCK * WIDEST];
  /* Arg's extremes in the carrier, before their terms (run_states). */
  _Alignas(64) uint8_t extremes[LANES * ROW];
  /* A program's terms in their held dtype (eval_in). */
  _Alignas(64) uint8_t held[2 * LANES * ROW];
  /* A scan's carry, results and chunk total (scan_units, carries). */
  _Alignas(64) uint8_t carry[SCAN_ROW], a[SCAN_ROW], v[SCAN_ROW];
  /* A staged operand's or a program's terms, a piece of a chunk (rescan). */
  _Alignas(64) uint8_t pieces[2 * SCAN_ROW];
} work;

_Static_assert(LANES * ROW <= NX_CPU_FEW_BYTES, "a block's lanes fit few's");

/* The float dtype of [dt]'s parts: a complex's, or [dt] itself. */
static int part_of(int dt) {
  return dt == NX_COMPLEX64    ? NX_FLOAT32
         : dt == NX_COMPLEX128 ? NX_FLOAT64
                               : dt;
}

/* [n] parts of the float dtype [p], each [v] rounded, at [d]. */
static void parts_at(uint8_t *d, int p, double v, int n) {
  float f = (float)v;
  int w = nx_cpu_width(p);
  int64_t c = p == NX_FLOAT32 || p == NX_FLOAT64 ? 0 : nx_double_to_bits(p, v);
  for (int i = 0; i < n; i++)
    if (p == NX_FLOAT32) memcpy(d + 4 * i, &f, 4);
    else if (p == NX_FLOAT64) memcpy(d + 8 * i, &v, 8);
    else memcpy(d + w * i, &c, w); /* little-endian hosts: the low bytes */
}

/* A complex identity is 1 + 0i, or both parts the float's infinity. */
static void identity(int monoid, int dt, uint8_t *id) {
  memset(id, 0, WIDEST);
  int w = nx_cpu_width(dt), p = part_of(dt), n = p == dt ? 1 : 2;
  int fl = p == NX_FLOAT32 || p == NX_FLOAT64;
  switch (monoid) {
    case NX_MONOID_SUM: return;
    case NX_MONOID_PROD:
      if (fl) parts_at(id, p, 1., 1);
      else id[0] = 1; /* little-endian hosts: the low byte */
      return;
    case NX_MONOID_MAX:
      if (fl) parts_at(id, p, -INFINITY, n);
      else if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x80;
      return;
    default: /* NX_MONOID_MIN */
      if (fl) parts_at(id, p, INFINITY, n);
      else if (dt == NX_BOOL) id[0] = 1;
      else {
        memset(id, 0xFF, w);
        if (nx_dtype_row_of(dt).kind == NX_KIND_SIGNED) id[w - 1] = 0x7F;
      }
  }
}

/* An Arg scan's states of no term at [v], [n] of them, their first
   term [t0]. */
static void none(nx_cpu_state *v, int64_t n, int64_t t0) {
  for (int64_t j = 0; j < n; j++)
    v[j] = (nx_cpu_state){.n = t0, .at = -1};
}

/* [n] identities of the plan's monoid at [d], copied in doubling runs. */
static void ids(const plan *q, uint8_t *d, int64_t n) {
  if (n == 0) return;
  memcpy(d, q->id, q->w);
  for (int64_t k = 1; k < n; k *= 2)
    memcpy(d + k * q->w, d, (k < n - k ? k : n - k) * q->w);
}

/* [n] values of no term at [d]: identities, or an Arg scan's states
   of no term. */
static void fill(const plan *q, uint8_t *d, int64_t n) {
  if (q->z != NULL) {
    none((nx_cpu_state *)d, n, 0);
    return;
  }
  ids(q, d, n);
}

/* Whether one of the [n] floats of dtype [part] at [v] is NaN: float32's
   and float64's in one pass the compiler vectorises, a narrow float's of
   whole bytes by its code. */
static int floats_nan(int part, const uint8_t *v, int64_t n) {
  int bad = 0;
  if (part == NX_FLOAT32) {
    const float *f = (const float *)v;
    for (int64_t j = 0; j < n; j++) bad |= f[j] != f[j];
  } else if (part == NX_FLOAT64) {
    const double *f = (const double *)v;
    for (int64_t j = 0; j < n; j++) bad |= f[j] != f[j];
  } else {
    const uint16_t *h = (const uint16_t *)v;
    int two = nx_cpu_width(part) == 2;
    for (int64_t j = 0; j < n; j++) {
      float f = nx_bits_to_float(part, two ? h[j] : v[j]);
      bad |= f != f;
    }
  }
  return bad;
}

/* Whether one of the [n] values at [v] is NaN, a complex one where a part
   is. */
static int any_nan(const plan *q, const uint8_t *v, int64_t n) {
  return floats_nan(q->part, v, n * q->parts);
}

static int is_nan(const plan *q, const uint8_t *p) { return any_nan(q, p, 1); }

/* The quiet bit of the float dtype [p]'s NaNs, the top bit of its
   fraction; 0 for a dtype whose NaNs all have it, float8_e4m3fn's one NaN
   of each sign, or that has none. */
static uint64_t quiet_bit(int p) {
  switch (p) {
    case NX_FLOAT64: return (uint64_t)1 << 51;
    case NX_FLOAT32: return (uint64_t)1 << 22;
    case NX_FLOAT16: return 0x200;
    case NX_BFLOAT16: return 0x40;
    case NX_FLOAT8_E5M2: return 0x2;
    default: return 0;
  }
}

/* Quiets each NaN part of the element of [dt] at [d]: sets its quiet bit,
   keeping its sign and the rest of its payload, as IEEE 754 quiets a
   signalling NaN an operation returns. */
static void quiet(int dt, uint8_t *d) {
  int p = part_of(dt), w = nx_cpu_width(p), n = p == dt ? 1 : 2;
  uint64_t bit = quiet_bit(p);
  if (bit == 0) return;
  for (int i = 0; i < n; i++) {
    uint8_t *e = d + i * w;
    if (!floats_nan(p, e, 1)) continue;
    uint64_t c = 0;
    memcpy(&c, e, w); /* little-endian hosts: the low bytes */
    c |= bit;
    memcpy(e, &c, w);
  }
}

/* A value's slots are NaN or not as a whole: the value, or each part of a
   complex sum, whose parts are float sums. Whether the slot at [s] is. */
static int slot_nan(const plan *q, const uint8_t *s) {
  return floats_nan(q->part, s, q->parts / q->slots);
}

static const uint8_t *at(const plan *q, int64_t p) {
  return q->src->base + p * q->w;
}

/* Programs */

/* The loads' positions at the virtual position [p], the index's after
   them, into [at]. An axis of one element, as a loop of one element has,
   steps 0. */
static void loads_at(const fused *z, int64_t p, int64_t *at) {
  for (int k = 0; k < z->nloads; k++) at[k] = z->l0[k];
  for (int d = 0; d < z->rank; d++) {
    if (z->ext[d] <= 1) continue;
    int64_t i = p / z->vs[d] % z->ext[d];
    for (int k = 0; k < z->nloads; k++) at[k] += i * z->ls[k][d];
  }
  if (z->index) at[z->nloads] = p;
}

/* The loads' steps, the index's after them, along the virtual step [s],
   into [st]: 0 for no step. [s] moves along one of the coalesced loop's
   axes, [s / vs[d]] places along axis d, the innermost whose step divides
   it into fewer places than its extent. */
static void loads_along(const fused *z, int64_t s, int64_t *st) {
  int d = z->rank - 1;
  while (d >= 0 && (s == 0 || z->ext[d] <= 1 || s % z->vs[d] != 0 ||
                    s / z->vs[d] >= z->ext[d]))
    d--;
  for (int k = 0; k < z->nloads; k++)
    st[k] = d >= 0 ? s / z->vs[d] * z->ls[k][d] : 0;
  if (z->index) st[z->nloads] = s;
}

/* [z] for the program bound to its loads [loads] over the virtual
   operand [x], C-contiguous over the loop's shape: the loop of [x] and the
   loads coalesced. Apart, so that its loop's frame is off the stack the
   jobs run on. */
static __attribute__((noinline)) void fuse(fused *z, const nx_array *x,
                                           const nx_array *loads) {
  int r = x->rank;
  int64_t ext[NX_MAX_RANK], st[NX_MAX_OPERANDS][NX_MAX_RANK];
  for (int i = 0; i < r; i++) {
    ext[i] = x->dim[i];
    st[0][i] = x->dim[r + i];
    for (int k = 0; k < z->nloads; k++) st[1 + k][i] = loads[k].dim[r + i];
  }
  z->loads = loads;
  z->rank = nx_coalesce_dims(1 + z->nloads, r, ext, st);
  for (int i = 0; i < z->rank; i++) {
    z->ext[i] = ext[i];
    z->vs[i] = st[0][i];
    for (int k = 0; k < z->nloads; k++) z->ls[k][i] = st[1 + k][i];
  }
  for (int k = 0; k < z->nloads; k++) z->l0[k] = loads[k].offset;
}

/* The program's output over [n1] rows of [n0] terms from the virtual
   position [p], stepping [s0] along a row and [s1] across rows, each a
   step of one axis, into [out] in its held dtype: row r's term i at
   out + (r·n0 + i)·w. */
static void eval(const fused *z, int64_t p, int64_t s0, int64_t n0,
                 int64_t s1, int64_t n1, uint8_t *out) {
  nx_cpu_block b = {.n0 = n0, .n1 = n1, .n2 = 1};
  loads_at(z, p, b.at);
  loads_along(z, s0, b.s0);
  loads_along(z, s1, b.s1);
  uint8_t *to[NX_MAX_OPERANDS] = {NULL};
  to[z->k] = out;
  nx_cpu_prog_terms(z->prog, &b, to);
}

/* As eval, into [out] in the dtype [c]: the held values, through [held],
   converted into it where they differ. [held] holds the held values. */
static void eval_in(const fused *z, int c, int64_t p, int64_t s0, int64_t n0,
                    int64_t s1, int64_t n1, uint8_t *out, uint8_t *held) {
  int h = nx_cpu_prog_held(z->prog, z->k);
  if (h == c) {
    eval(z, p, s0, n0, s1, n1, out);
    return;
  }
  eval(z, p, s0, n0, s1, n1, held);
  nx_cpu_table->convert[h][c](held, out, n0 * n1);
}

/* The operand's element at [pos] in the carrier: in place, or staged into
   [tmp], or a program's computed into it. */
static const uint8_t *element(const plan *q, int64_t pos, uint8_t *tmp) {
  if (q->fused != NULL) {
    _Alignas(16) uint8_t held[WIDEST];
    eval_in(q->fused, q->carrier, pos, 0, 1, 0, 1, tmp, held);
    return tmp;
  }
  if (!q->staged) return at(q, pos);
  nx_cpu_stage_as(q->src, pos, 1, 0, 1, 1, q->carrier, tmp, 1);
  return tmp;
}

/* Whether the paths read terms through stage_terms and element: a narrow
   operand's or a program's. */
static int sourced(const plan *q) { return q->staged || q->fused != NULL; }

/* Writes the NaN nx_kinds.h makes from numbers into the slot at [d], in
   each of its parts: a narrow float's rounded from float32's. */
static void made_nan(const plan *q, uint8_t *d) {
  parts_at(d, q->part, q->part == NX_FLOAT64 ? NX_NAN_F64 : NX_NAN_F32,
           q->parts / q->slots);
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

/* The stage */

/* The program's terms as stage_terms gives the operand's, in the carrier.
   Its rows take at most 2·LANES·ROW bytes, as every caller's buffer, their
   held values through [held]. */
static void fused_terms(const plan *q, int64_t p, int64_t s, int64_t W,
                        int64_t t, int64_t m, uint8_t *buf, uint8_t *held) {
  int r = q->nr - 1, c = q->carrier;
  if (W == 1)
    for (int64_t i = 0; i < m;) {
      int64_t len = q->re[r] - (t + i) % q->re[r];
      if (len > m - i) len = m - i;
      eval_in(q->fused, c, term(q, p, t + i), q->rs[r], len, 0, 1,
              buf + i * q->w, held);
      i += len;
    }
  else if (q->nr == 1)
    eval_in(q->fused, c, p + t * q->rs[0], s, W, q->rs[0], m, buf, held);
  else {
    int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t, idx);
    for (int64_t i = 0; i < m; i++, pos = next(q, idx, pos))
      eval_in(q->fused, c, pos, s, W, 0, 1, buf + i * W * q->w, held);
  }
}

/* Converts into the carrier at [buf] the [W] elements, stepping [s], of
   each of the [m] terms from [t] of the outputs from [p]: term t + i's row
   at buf + i·W·w. One output's terms stage run by run along the innermost
   reduced axis; several outputs' along one reduced axis, at once. A
   program's held values go through [wk]. */
static void stage_terms(const plan *q, work *wk, int64_t p, int64_t s,
                        int64_t W, int64_t t, int64_t m, uint8_t *buf) {
  int r = q->nr - 1;
  if (q->fused != NULL) {
    fused_terms(q, p, s, W, t, m, buf, wk->held);
    return;
  }
  if (W == 1) {
    for (int64_t i = 0; i < m;) {
      int64_t len = q->re[r] - (t + i) % q->re[r];
      if (len > m - i) len = m - i;
      nx_cpu_stage_as(q->src, term(q, p, t + i), q->rs[r], 0, len, 1,
                      q->carrier, buf + i * q->w, len);
      i += len;
    }
    return;
  }
  if (q->nr == 1) {
    nx_cpu_stage_as(q->src, p + t * q->rs[0], s, q->rs[0], W, m, q->carrier,
                    buf, W);
    return;
  }
  int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t, idx);
  for (int64_t i = 0; i < m; i++, pos = next(q, idx, pos))
    nx_cpu_stage_as(q->src, pos, s, 0, W, 1, q->carrier, buf + i * W * q->w,
                    W);
}

/* The index of the first term of the output at [p] whose slot [k] is
   NaN, the term copied into [e], quieted; or q->terms where none is. A
   staged operand's or a program's terms come a buffer at a time. */
static int64_t nan_term(const plan *q, work *wk, int64_t p, int k,
                        uint8_t *e) {
  int64_t sw = q->w / q->slots;
  if (!sourced(q)) {
    for (int64_t t = 0; t < q->terms; t++) {
      const uint8_t *x = at(q, term(q, p, t));
      if (!slot_nan(q, x + k * sw)) continue;
      memcpy(e, x, q->w);
      quiet(q->carrier, e);
      return t;
    }
    return q->terms;
  }
  uint8_t *buf = wk->terms;
  int64_t most = LANES * ROW / q->w;
  for (int64_t t = 0; t < q->terms; t += most) {
    int64_t m = q->terms - t < most ? q->terms - t : most;
    stage_terms(q, wk, p, 0, 1, t, m, buf);
    for (int64_t i = 0; i < m; i++) {
      if (!slot_nan(q, buf + i * q->w + k * sw)) continue;
      memcpy(e, buf + i * q->w, q->w);
      quiet(q->carrier, e);
      return t + i;
    }
  }
  return q->terms;
}

/* Replaces each NaN slot of the result at [e] of the output at [p] by its
   first term's slot that is NaN, quieted, or by nx_kinds.h's NaN where
   none is. */
static void first_nan(const plan *q, work *wk, int64_t p, uint8_t *e) {
  int64_t sw = q->w / q->slots;
  _Alignas(16) uint8_t first[WIDEST];
  for (int k = 0; k < q->slots; k++) {
    uint8_t *s = e + k * sw;
    if (!slot_nan(q, s)) continue;
    if (nan_term(q, wk, p, k, first) < q->terms)
      memcpy(s, first + k * sw, sw);
    else made_nan(q, s);
  }
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
    q->f->lanes(at(q, term(q, p, t)), s, len, q->id, l,
                (int)(t & (LANES - 1)));
    t += len;
  }
}

/* Copies term [t0 + i] of the [W] outputs, the first at [p] and each next
   [s] further along the operand, into row i of [W] values of the lanes
   [l], for i below [used]: the lanes start from their first terms. */
static void seed(const plan *q, int64_t p, int64_t s, int64_t W, int64_t t0,
                 int used, uint8_t *l) {
  for (int i = 0; i < used; i++)
    for (int64_t j = 0; j < W; j++)
      memcpy(l + (i * W + j) * q->w, at(q, term(q, p + j * s, t0 + i)), q->w);
}

/* The lanes' tree over [W] outputs: 16 rows of [W] values, [used] of them
   holding terms, which alone take part. */
static void lane_tree(const plan *q, uint8_t *l, int64_t W, int used) {
  int64_t row = W * q->w;
  for (int h = LANES / 2; h >= 1; h /= 2)
    for (int i = 0; i < h && i + h < used; i++)
      q->f->combine(l + i * row, l + (i + h) * row, 1, W);
}

/* Block [b]'s states of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, their centers [c] or 0 where NULL,
   into [v]: each output's terms where they lie contiguous, else copied into
   a block. Along one reduced axis, the blocks of as many outputs as 32 KiB
   holds are copied at once: a row of the operand then serves them all,
   where one output at a time would read each row once per output. */
static void states(const plan *q, work *wk, int64_t p, int64_t s, int64_t W,
                   const double *c, int64_t b, nx_cpu_state *v) {
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  uint8_t *buf = wk->terms;
  int contiguous = q->nr == 1 && q->rs[0] == 1;
  if (!contiguous && q->nr == 1 && W > 1) {
    int64_t most = (int64_t)sizeof wk->terms / (n * q->w), rs = q->rs[0];
    for (int64_t j = 0; j < W; j += most) {
      int64_t k = W - j < most ? W - j : most, o = p + j * s + t0 * rs;
      if (q->fused != NULL)
        eval_in(q->fused, q->carrier, o, rs, n, s, k, buf, wk->held);
      else nx_cpu_stage_as(q->src, o, rs, s, n, k, q->carrier, buf, n);
      for (int64_t i = 0; i < k; i++)
        q->r->block(buf + i * n * q->w, n, t0, c != NULL ? c[j + i] : 0,
                    &v[j + i]);
    }
    return;
  }
  for (int64_t j = 0; j < W; j++) {
    int64_t o = p + j * s;
    const uint8_t *x = buf;
    if (sourced(q) || !contiguous) stage_terms(q, wk, o, 0, 1, t0, n, buf);
    else x = at(q, o + t0);
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

/* Block [b]'s values of the unit's [W] outputs from a staged operand or a
   program, as block's: its terms in the carrier, a buffer's of one
   output's at a time, else 16 terms at a time, each a row of the outputs'.
   The lanes start from their first terms where the plan's do. */
static void staged_block(const plan *q, work *wk, int64_t p, int64_t s,
                         int64_t W, int64_t b, uint8_t *v) {
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  int used = n < LANES ? (int)n : LANES;
  int64_t row = W * q->w, step = W == 1 ? LANES * ROW / q->w : LANES;
  uint8_t *l = wk->lanes, *buf = wk->terms;
  if (!q->first) ids(q, l, used * W);
  for (int64_t t = 0; t < n; t += step) {
    int64_t m = n - t < step ? n - t : step;
    stage_terms(q, wk, p, s, W, t0 + t, m, buf);
    /* The rows the lanes start from: the block's first ones. */
    int64_t k = t == 0 && q->first ? used : 0;
    if (k > 0) memcpy(l, buf, (size_t)(k * row));
    if (W == 1)
      q->f->lanes(buf + k * q->w, 1, m - k, q->id, l,
                  (int)((t + k) & (LANES - 1)));
    else
      for (int64_t i = k; i < m; i++)
        q->f->combine(l + i * row, buf + i * row, 1, W);
  }
  lane_tree(q, l, W, used);
  memcpy(v, l, row);
}

/* Block [b]'s values of the unit's [W] outputs, the first at [p] and each
   next [s] further along the operand, of centers [c], into [v]. */
static void block(const plan *q, work *wk, int64_t p, int64_t s, int64_t W,
                  const double *c, int64_t b, uint8_t *v) {
  if (q->r != NULL) {
    states(q, wk, p, s, W, c, b, (nx_cpu_state *)v);
    return;
  }
  if (sourced(q)) {
    staged_block(q, wk, p, s, W, b, v);
    return;
  }
  int64_t t0 = b * BLOCK, n = q->terms - t0 < BLOCK ? q->terms - t0 : BLOCK;
  int used = n < LANES ? (int)n : LANES;
  /* The terms the lanes start from, where they start from their first. */
  int k = q->first ? used : 0;
  if (W == 1) {
    _Alignas(16) uint8_t l[LANES * WIDEST];
    if (q->first) seed(q, p, s, 1, t0, used, l);
    else ids(q, l, LANES);
    runs(q, p, t0 + k, n - k, l);
    lane_tree(q, l, 1, used);
    memcpy(v, l, q->w);
    return;
  }
  uint8_t *l = wk->lanes;
  int64_t row = W * q->w;
  if (q->first) seed(q, p, s, W, t0, used, l);
  else
    for (int i = 0; i < used; i++) ids(q, l + i * row, W);
  /* Along one reduced axis, lane i's terms lie LANES apart: the table adds
     each lane's in order, its accumulators in registers. */
  if (q->nr == 1) {
    int64_t st = q->rs[0], from = q->first ? LANES : 0;
    for (int i = 0; i < used; i++)
      q->f->column(l + i * row, at(q, p + (t0 + i + from) * st), LANES * st,
                   (n - i - from + LANES - 1) / LANES, s, W);
    lane_tree(q, l, W, used);
    memcpy(v, l, row);
    return;
  }
  int64_t idx[NX_MAX_RANK], pos = odometer(q, p, t0 + k, idx);
  for (int64_t t = t0 + k; t < t0 + n; t++, pos = next(q, idx, pos))
    q->f->combine(l + (t & (LANES - 1)) * row, at(q, pos), s, W);
  lane_tree(q, l, W, used);
  memcpy(v, l, row);
}

/* The value of blocks [b0, b1) of the unit's outputs, of centers [c], by
   the counter, into [v]: the identity for no block. */
static void blocks(const plan *q, work *wk, int64_t p, int64_t s, int64_t W,
                   const double *c, int64_t b0, int64_t b1, uint8_t *v) {
  uint8_t(*stack)[ROW] = wk->tree, *run = wk->run;
  int top = 0;
  int64_t row = W * q->size;
  /* A monoid's output's full blocks of contiguous terms take the table's
     blocks, RUN at a time, whose lanes start from the identity. */
  int contiguous = W == 1 && q->nr == 1 && q->rs[0] == 1;
  int64_t full = q->f != NULL && !sourced(q) && !q->first && contiguous
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
      else block(q, wk, p, s, W, c, b, stack[top++]);
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
   operand, the destinations and a scan's carries, and the operand's,
   destinations' and carries' steps from one output to the next in a row,
   and from one row to the next. Units run outer index, then band, then
   chunk, then group. */
typedef struct {
  int64_t p, d, d1, c, W, s, sd, sd1, sc, H, s2, sd2, g;
} unit;

static unit unit_of(const plan *q, int64_t u) {
  unit x = {.p = q->x0, .d = q->d0, .d1 = q->d1, .c = q->c0, .W = 1, .H = 1,
            .g = u % q->groups};
  u /= q->groups;
  int last = q->nk - 1;
  if (q->row > 1) {
    int64_t c = u % q->chunks;
    u /= q->chunks;
    x.s = q->ks[last];
    x.sd = q->kd[last];
    x.sd1 = q->kd1[last];
    x.sc = q->kc[last];
    x.W = q->ke[last] - c * q->row < q->row ? q->ke[last] - c * q->row : q->row;
    x.p += c * q->row * x.s;
    x.d += c * q->row * x.sd;
    x.d1 += c * q->row * x.sd1;
    x.c += c * q->row * x.sc;
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
    x.c += c * q->band * q->kc[last];
    last--;
  }
  for (int i = last; i >= 0; i--) {
    x.p += u % q->ke[i] * q->ks[i];
    x.d += u % q->ke[i] * q->kd[i];
    x.d1 += u % q->ke[i] * q->kd1[i];
    x.c += u % q->ke[i] * q->kc[i];
    u /= q->ke[i];
  }
  return x;
}

/* Copies element [s] of [a] into element [d] of [b], of [a]'s dtype, bits
   for bits. */
static void copy_element(const nx_array *b, int64_t d, const nx_array *a,
                         int64_t s) {
  int bits = nx_dtype_row_of(a->dtype).bits;
  if (bits < 8) nx_sub_store(b->base, bits, d, nx_sub_load(a->base, bits, s));
  else memcpy(b->base + d * (bits / 8), a->base + s * (bits / 8), bits / 8);
}

/* Arg's extreme of the operand's dtype into element [d] of the
   destination: the operand's term at [pos], or a program's computed
   there, quieted. */
static void own_term(const plan *q, int64_t d, int64_t pos) {
  int64_t w = nx_cpu_width(q->dst->dtype);
  if (q->fused == NULL) copy_element(q->dst, d, q->src, pos);
  else eval(q->fused, pos, 0, 1, 0, 1, q->dst->base + d * w);
  quiet(q->dst->dtype, q->dst->base + d * w);
}

/* Stores the results of the unit's states [v], of centers [c], each NaN
   replaced as store's are, Arg's extreme quieted, of the operand's dtype
   copied from its term; or, where the pass finds them, the centers. */
static void store_states(const plan *q, work *wk, const unit *x, double *c,
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
    if (q->own) {
      _Alignas(16) uint8_t e[WIDEST];
      q->r->finish(&v[j], 0, e, d1);
      own_term(q, x->d + j * x->sd, term(q, p, v[j].at));
      continue;
    }
    q->r->finish(&v[j], c != NULL ? c[j] : 0, d0, d1);
    if (q->arg) quiet(q->dst->dtype, d0);
    if (!q->nan) continue;
    if (is_nan(q, d0)) first_nan(q, wk, p, d0);
    if (q->dst1 != q->dst && is_nan(q, d1)) first_nan(q, wk, p, d1);
  }
}

/* Stores the unit's values [v], of centers [c], each NaN replaced by its
   first NaN term, or by nx_kinds.h's where none is. */
static void store(const plan *q, work *wk, const unit *x, double *c,
                  const uint8_t *v) {
  if (q->r != NULL) {
    store_states(q, wk, x, c, (const nx_cpu_state *)v);
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
      first_nan(q, wk, x->p + j * x->s, d + j * x->sd * q->w);
}

/* Outputs of a lane or fewer each, one block: their terms' positions from
   an output's first are the same for every output, and the table reduces
   the unit's rows at once, its lanes side by side. */
static void few(const plan *q, work *wk, const unit *x) {
  int64_t n = q->terms;
  if (!sourced(q)) {
    q->f->few(at(q, x->p), q->off, (int)n, x->s, x->W, x->s2, x->H, q->id,
              q->dst->base + x->d * q->w, x->sd, x->sd2, wk->lanes);
  } else {
    /* The terms in the carrier, term t of row h a row of the W outputs'
       at buf + (h n + t)·W·w: within a unit's H·W < 2·ROW / w outputs of
       at most 16 terms each. */
    uint8_t *buf = wk->terms;
    int64_t off[LANES];
    for (int64_t h = 0; h < x->H; h++)
      stage_terms(q, wk, x->p + h * x->s2, x->s, x->W, 0, n,
                  buf + h * n * x->W * q->w);
    for (int64_t t = 0; t < n; t++) off[t] = t * x->W;
    q->f->few(buf, off, (int)n, 1, x->W, n * x->W, x->H, q->id,
              q->dst->base + x->d * q->w, x->sd, x->sd2, wk->lanes);
  }
  if (!q->nan) return;
  for (int64_t h = 0; h < x->H; h++) {
    int64_t p = x->p + h * x->s2;
    uint8_t *d = q->dst->base + (x->d + h * x->sd2) * q->w;
    if (x->sd == 1 && !any_nan(q, d, x->W)) continue;
    for (int64_t j = 0; j < x->W; j++)
      if (is_nan(q, d + j * x->sd * q->w))
        first_nan(q, wk, p + j * x->s, d + j * x->sd * q->w);
  }
}

static void reduce_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  _Alignas(16) uint8_t v[ROW];
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (q->f != NULL && !q->first && q->terms <= LANES) {
      few(q, wk, &x);
      continue;
    }
    int64_t b0 = x.g * q->group, b1 = b0 + q->group;
    if (b1 > q->blocks) b1 = q->blocks;
    double *c = q->centers ? q->centers + u / q->groups * q->row : NULL;
    if (q->groups == 1) {
      blocks(q, wk, x.p, x.s, x.W, c, b0, b1, v);
      store(q, wk, &x, c, v);
    } else
      blocks(q, wk, x.p, x.s, x.W, c, b0, b1,
             q->scratch + u * q->row * q->size);
  }
}

/* Combines each unit's groups by the counter and stores them. */
static void finish_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  int64_t row = q->row * q->size;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    uint8_t(*stack)[ROW] = wk->tree;
    int top = 0;
    for (int64_t g = 0; g < q->groups; g++) {
      memcpy(stack[top++], q->scratch + (u * q->groups + g) * row, row);
      for (int64_t k = g + 1; (k & 1) == 0; k >>= 1, top--)
        join(q, stack[top - 2], stack[top - 1], x.W);
    }
    for (; top > 1; top--) join(q, stack[top - 2], stack[top - 1], x.W);
    store(q, wk, &x, q->centers ? q->centers + u * q->row : NULL, stack[0]);
  }
}

static int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }

/* A plan's units [lo, hi) on one thread, with the thread's work. */
typedef void (*units_body)(const plan *q, work *wk, int64_t lo, int64_t hi);

typedef struct {
  units_body body;
  const plan *q;
} plan_job;

/* A plan's job's range on one thread: its work taken for the range, or the
   call failed where it cannot be. */
static void with_work(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const plan_job *j = ctx;
  work *wk = nx_cpu_take(sizeof *wk);
  if (wk == NULL) {
    *j->q->failed = 1;
    return;
  }
  j->body(j->q, wk, lo, hi);
  nx_cpu_give(wk);
}

/* nx_cpu_job of the plan [q]'s units each by [body]. Once a range's work
   could not be taken, no later job runs: it would read what that range
   left unwritten. */
static void units_job(int64_t total, int64_t bytes, double cost,
                      units_body body, const plan *q) {
  if (*q->failed) return;
  plan_job j = {body, q};
  nx_cpu_job(total, bytes, cost, with_work, &j);
}

/* As units_job, on the calling thread alone: each unit reads the results
   of the units before it. */
static void units_serial(int64_t total, units_body body, const plan *q) {
  if (*q->failed) return;
  plan_job j = {body, q};
  with_work(0, total, 0, &j);
}

/* Adds the axis of extent [e] and steps [s] (operand), [d] and [d1]
   (destinations) and [c] (carries) to the kept axes, if it has more than
   one element. */
static void keep(plan *q, int64_t e, int64_t s, int64_t d, int64_t d1,
                 int64_t c) {
  if (e <= 1) return;
  q->ke[q->nk] = e;
  q->ks[q->nk] = s;
  q->kd[q->nk] = d;
  q->kc[q->nk] = c;
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
          d1 = q->kd1[least], c = q->kc[least];
  for (int i = least; i < q->nk - 1; i++) {
    q->ke[i] = q->ke[i + 1];
    q->ks[i] = q->ks[i + 1];
    q->kd[i] = q->kd[i + 1];
    q->kd1[i] = q->kd1[i + 1];
    q->kc[i] = q->kc[i + 1];
  }
  q->ke[q->nk - 1] = e;
  q->ks[q->nk - 1] = s;
  q->kd[q->nk - 1] = d;
  q->kd1[q->nk - 1] = d1;
  q->kc[q->nk - 1] = c;
  q->slices = !rows;
  int64_t most = q->slices ? SLICES : ROW / q->size;
  /* A monoid scan on one thread sweeps long rows; on several, its units
     of ROW share the work more evenly, which gains more. Its bits do not
     depend on the rows. */
  if (!q->slices && op == SCAN && q->f != NULL && q->threads == 1)
    most = SCAN_ROW / q->size;
  q->row = most < e ? most : e;
  q->chunks = (e + q->row - 1) / q->row;
  q->outer = outputs / e;
  if (op == SCAN || q->f == NULL || q->first || q->terms > LANES ||
      q->nk < 2 || q->chunks > 1)
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
      keep(q, l->extent[i], l->step[2][i], l->step[0][i], l->step[1][i], 0);
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

/* [d]'s axes coalesced (nx_coalesce_dims) into [extent] and [step]: its
   rank, the last axis its rows. */
static int rows_of(const nx_array *d, int64_t *extent,
                   int64_t (*step)[NX_MAX_RANK]) {
  for (int i = 0; i < d->rank; i++) {
    extent[i] = d->dim[i];
    step[0][i] = d->dim[d->rank + i];
  }
  return nx_coalesce_dims(1, d->rank, extent, step);
}

/* The elements of [d], each of [w] bytes, all set to the bits [e]: a
   reduction of no term. */
static void fill_dst(const nx_array *d, const uint8_t *e, int w) {
  int64_t extent[NX_MAX_RANK], step[1][NX_MAX_RANK];
  int r = rows_of(d, extent, step);
  int64_t n = extent[r - 1], s = step[0][r - 1], rows = 1;
  for (int i = 0; i < r - 1; i++) rows *= extent[i];
  nx_cpu_row0 row = nx_cpu_table->fill[__builtin_ctz((unsigned)w)];
  int64_t at[NX_MAX_RANK] = {0}, p = d->offset;
  for (int64_t k = 0; k < rows && n > 0; k++) {
    row(n, d->base + p * w, s, e);
    /* The next row: the odometer's last outer axis steps, carrying. */
    int i = r - 2;
    for (; i >= 0 && ++at[i] == extent[i]; i--) {
      at[i] = 0;
      p -= (extent[i] - 1) * step[0][i];
    }
    if (i >= 0) p += step[0][i];
  }
}

/* Scans */

/* Converts rows [t, t + m) of the unit's [W] sequences into the carrier
   at [buf], row i's sequence l at buf + (i W + l)·w, or computes a
   program's there: sequence l is the unit's slice l, or, where [lt] is not
   0, the unit's slice from term l·lt on. */
static void stage_lanes(const plan *q, work *wk, const unit *x, int64_t W,
                        int64_t lt, int64_t t, int64_t m, uint8_t *buf) {
  int64_t rs = q->rs[0], s = (lt ? 0 : x->s) + lt * rs;
  if (q->fused != NULL) {
    eval_in(q->fused, q->carrier, x->p + t * rs, s, W, rs, m, buf, wk->held);
    return;
  }
  nx_cpu_stage_as(q->src, x->p + t * rs, s, rs, W, m, q->carrier, buf, W);
}

/* Quiets the NaNs among Arg's [m] by [W] extremes at [y], a scan's
   results, [q->rd] and [dl] elements apart. */
static void quiet_rows(const plan *q, uint8_t *y, int64_t m, int64_t W,
                       int64_t dl) {
  if (quiet_bit(q->part) == 0) return;
  for (int64_t i = 0; i < m; i++)
    for (int64_t l = 0; l < W; l++)
      quiet(q->dst->dtype, y + (i * q->rd + l * dl) * q->w);
}

/* Terms [t0, t0 + n) of an Arg scan's [W] sequences into their states
   [a], and each result into the destinations. Sequence l is
   the unit's slice l, or, where [lt] is not 0, the unit's slice from term
   l·lt on. The kernel reads the operand and writes the destinations in
   place, a staged operand's terms converted into the carrier a piece at a
   time; an extreme of the operand's dtype goes through a piece of scratch
   and is then copied from the term at its position. */
static void run_states(const plan *q, work *wk, const unit *x, nx_cpu_state *a,
                       int64_t W, int64_t lt, int64_t t0, int64_t n) {
  int64_t rs = q->rs[0], ls = lt ? 0 : x->s, xl = ls + lt * rs;
  int64_t dl = lt ? lt * q->rd : x->sd, d1l = lt ? lt * q->rd1 : x->sd1;
  /* A staged operand's or a program's terms come side by side. */
  int staged = sourced(q);
  int64_t piece = staged ? LANES * ROW / (W * WIDEST) : n;
  uint8_t *buf = wk->terms, *e = wk->extremes;
  for (int64_t t = t0; t < t0 + n; t += piece) {
    int64_t m = t0 + n - t < piece ? t0 + n - t : piece;
    const uint8_t *src = staged ? buf : at(q, x->p + t * rs);
    if (staged) stage_lanes(q, wk, x, W, lt, t, m, buf);
    int64_t xs = staged ? W : rs, xli = staged ? 1 : xl;
    uint8_t *y = q->own ? e : q->dst->base + (x->d + t * q->rd) * q->w;
    int64_t *p = (int64_t *)q->dst1->base + x->d1 + t * q->rd1;
    q->z->rows(a, src, xs, xli, m, W, y, q->own ? W : q->rd, q->own ? 1 : dl,
               p, q->rd1, d1l);
    if (!q->own) {
      quiet_rows(q, y, m, W, dl);
      continue;
    }
    for (int64_t i = 0; i < m; i++)
      for (int64_t l = 0; l < W; l++)
        own_term(q, x->d + (t + i) * q->rd + l * dl,
                 x->p + l * ls + p[i * q->rd1 + l * d1l] * rs);
  }
}

/* Chunk [c]'s total of the unit's slices, into [v]: slice by slice where
   they run side by side. An Arg scan's, being exact, is its reduction's,
   numbered on to the chunk's end. */
static void total(const plan *q, work *wk, const unit *x, int64_t c,
                  uint8_t *v) {
  int64_t t0 = c * SCAN_CHUNK;
  int64_t n = q->terms - t0 < SCAN_CHUNK ? q->terms - t0 : SCAN_CHUNK;
  int64_t b0 = c * (SCAN_CHUNK / BLOCK), b1 = b0 + SCAN_CHUNK / BLOCK;
  if (b1 > q->blocks) b1 = q->blocks;
  if (!q->slices)
    /* A row of a scan's outputs, in pieces the blocks' path holds. */
    for (int64_t j = 0, most = ROW / q->size; j < x->W; j += most)
      blocks(q, wk, x->p + j * x->s, x->s, x->W - j < most ? x->W - j : most,
             NULL, b0, b1, v + j * q->size);
  else
    for (int64_t j = 0; j < x->W; j++)
      blocks(q, wk, x->p + j * x->s, 0, 1, NULL, b0, b1, v + j * q->size);
  if (q->z != NULL)
    for (int64_t j = 0; j < x->W; j++) ((nx_cpu_state *)v)[j].n = t0 + n;
}

/* Results [t0, t0 + n) of the unit's slices, from the carry [a], which it
   leaves holding the last ones, term t of slice j at
   x + ((t - t0)·st + j·sj)·w: side by side, or a row of outputs at a
   time. */
static void scan_run(const plan *q, const unit *x, uint8_t *a,
                     const uint8_t *x0, int64_t st, int64_t sj, int64_t t0,
                     int64_t n) {
  uint8_t *y = q->dst->base + (x->d + t0 * q->rd) * q->w;
  if (q->slices || x->W == 1) {
    q->f->scan(a, x0, st, sj, y, q->rd, x->sd, n, (int)x->W);
    return;
  }
  for (int64_t t = 0; t < n; t++) {
    uint8_t *row = y + t * q->rd * q->w;
    if (t + SCAN_AHEAD < n && x->W * q->w <= ROW) {
      /* A staged operand's terms are in a buffer the stage just wrote. */
      if (sj == 1 && !sourced(q))
        PREFETCH_ROW(x0 + (t + SCAN_AHEAD) * st * q->w, x->W * q->w, 0);
#if defined(__aarch64__)
      if (x->sd == 1)
        PREFETCH_ROW(row + SCAN_AHEAD * q->rd * q->w, x->W * q->w, 1);
#endif
    }
    q->f->combine(a, x0 + t * st * q->w, sj, x->W);
    if (x->sd == 1) memcpy(row, a, x->W * q->w);
    else
      for (int64_t j = 0; j < x->W; j++)
        memcpy(row + j * x->sd * q->w, a + j * q->w, q->w);
  }
}

/* Whether a scan's chunk 0 has no carry and starts from its first terms:
   a complex product's, given none. */
static int bare(const plan *q) { return q->first && q->cy == NULL; }

/* The carries of the unit's slices into chunk 0, at [a]: the given ones,
   else values of no term. */
static void start(const plan *q, const unit *x, uint8_t *a) {
  if (q->cy == NULL) {
    fill(q, a, x->W);
    return;
  }
  for (int64_t j = 0; j < x->W; j++)
    memcpy(a + j * q->w, q->cy->base + (x->c + j * x->sc) * q->w, q->w);
}

/* Chunk [c]'s results of the unit's slices, from the carry [a], which it
   leaves holding the chunk's last results. A chunk 0 that has no carry
   starts from its first terms. A staged operand's or a program's terms
   come into the carrier a piece at a time. */
static void rescan(const plan *q, work *wk, const unit *x, int64_t c,
                   uint8_t *a) {
  int64_t t0 = c * SCAN_CHUNK;
  int64_t n = q->terms - t0 < SCAN_CHUNK ? q->terms - t0 : SCAN_CHUNK;
  if (q->z != NULL) {
    run_states(q, wk, x, (nx_cpu_state *)a, x->W, 0, t0, n);
    return;
  }
  if (c == 0 && bare(q)) {
    uint8_t *y = q->dst->base + x->d * q->w;
    _Alignas(16) uint8_t tmp[WIDEST];
    for (int64_t j = 0; j < x->W; j++) {
      memcpy(a + j * q->w, element(q, x->p + j * x->s, tmp), q->w);
      memcpy(y + j * x->sd * q->w, a + j * q->w, q->w);
    }
    t0 = 1;
    n--;
  }
  if (!sourced(q)) {
    scan_run(q, x, a, at(q, x->p + t0 * q->rs[0]), q->rs[0], x->s, t0, n);
    return;
  }
  /* Pieces of LANES * ROW bytes, or of two rows where a row is longer
     than ROW. */
  uint8_t *buf = wk->pieces;
  int64_t row = x->W * q->w;
  int64_t piece = (row <= ROW ? LANES * ROW : 2 * SCAN_ROW) / row;
  for (int64_t t = t0; t < t0 + n; t += piece) {
    int64_t m = t0 + n - t < piece ? t0 + n - t : piece;
    stage_terms(q, wk, x->p, x->s, x->W, t, m, buf);
    scan_run(q, x, a, buf, x->W, 1, t, m);
  }
}

/* Slot [k] of the slice at [p] whose results start at [d], from the carry
   at [cy] or none: where its last result is NaN, it takes its first NaN
   term's from that term on, and nx_kinds.h's NaN for each NaN result
   before it; but where an extreme's carry is NaN, every result is it,
   quieted. */
static void settle_slot(const plan *q, work *wk, int64_t p, int64_t d,
                        const uint8_t *cy, int k) {
  int64_t sw = q->w / q->slots;
  uint8_t *y = q->dst->base + k * sw;
  _Alignas(16) uint8_t first[WIDEST];
  if (!slot_nan(q, y + (d + (q->terms - 1) * q->rd) * q->w)) return;
  int64_t i = 0;
  if (q->extreme && cy != NULL && slot_nan(q, cy + k * sw)) {
    memcpy(first, cy, q->w);
    quiet(q->carrier, first);
  } else i = nan_term(q, wk, p, k, first);
  for (int64_t t = 0; t < i; t++) {
    uint8_t *e = y + (d + t * q->rd) * q->w;
    if (slot_nan(q, e)) made_nan(q, e);
  }
  if (i == q->terms) return;
  for (int64_t t = i; t < q->terms; t++)
    memcpy(y + (d + t * q->rd) * q->w, first + k * sw, sw);
}

/* Settles each slot of each of the unit's slices. */
static void settle(const plan *q, work *wk, const unit *x) {
  if (!q->nan || q->terms == 0) return;
  for (int64_t j = 0; j < x->W; j++) {
    const uint8_t *cy =
        q->cy != NULL ? q->cy->base + (x->c + j * x->sc) * q->w : NULL;
    for (int k = 0; k < q->slots; k++)
      settle_slot(q, wk, x->p + j * x->s, x->d + j * x->sd, cy, k);
  }
}

static void settle_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    settle(q, wk, &x);
  }
}

/* The carry [a] of [W] slices into the chunk after [c], from [a] into [c]
   and [c]'s total [v]: the total alone where chunk 0 has no carry, as an
   Arg scan's has none. */
static void carry_on(const plan *q, uint8_t *a, const uint8_t *v, int64_t c,
                     int64_t W) {
  if (c == 0 && (bare(q) || q->z != NULL)) memcpy(a, v, W * q->size);
  else if (q->z == NULL) q->f->combine(a, v, 1, W);
  else
    for (int64_t j = 0; j < W; j++)
      q->z->merge((nx_cpu_state *)a + j, (const nx_cpu_state *)v + j);
}

/* One pass per unit: each chunk's results from the carry, which then takes
   the chunk's total. The last chunk's total carries nowhere. */
static void scan_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  uint8_t *carry = wk->carry, *a = wk->a, *v = wk->v;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    start(q, &x, carry);
    for (int64_t c = 0; c * SCAN_CHUNK < q->terms; c++) {
      memcpy(a, carry, x.W * q->size);
      rescan(q, wk, &x, c, a);
      if ((c + 1) * SCAN_CHUNK >= q->terms) break;
      total(q, wk, &x, c, v);
      carry_on(q, carry, v, c, x.W);
    }
    settle(q, wk, &x);
  }
}

/* Few slices, long axes: every chunk's total in one job, each unit's
   carries in order, every chunk's results in a second job. A unit's group
   is its chunk. */
static void total_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    if (x.g < q->groups - 1)
      total(q, wk, &x, x.g, q->scratch + u * q->row * q->size);
  }
}

/* The unit of span [v] of a job over each unit's first [chunks] chunks,
   SLICES of them a span, and the span's first chunk [c] and count [k]. */
static int64_t span(int64_t v, int64_t chunks, int64_t *c, int *k) {
  int64_t spans = (chunks + SLICES - 1) / SLICES;
  *c = v % spans * SLICES;
  *k = (int)(chunks - *c < SLICES ? chunks - *c : SLICES);
  return v / spans;
}

static void rescan_units(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u);
    rescan(q, wk, &x, x.g, q->scratch + u * q->row * q->size);
  }
}

/* [k] whole chunks from [t0] of the slice [x] from their carries [a], side
   by side: in place, or a staged operand's or a program's a piece of each
   at a time, in the carrier. */
static void quad(const plan *q, work *wk, const unit *x, uint8_t *a,
                 int64_t t0, int k) {
  if (q->z != NULL) {
    run_states(q, wk, x, (nx_cpu_state *)a, k, SCAN_CHUNK, t0, SCAN_CHUNK);
    return;
  }
  int64_t rs = q->rs[0];
  uint8_t *y = q->dst->base + (x->d + t0 * q->rd) * q->w;
  if (!sourced(q)) {
    q->f->scan(a, at(q, x->p + t0 * rs), rs, SCAN_CHUNK * rs, y, q->rd,
               SCAN_CHUNK * q->rd, SCAN_CHUNK, k);
    return;
  }
  uint8_t *buf = wk->terms;
  int64_t piece = LANES * ROW / (4 * q->w);
  for (int64_t i = 0; i < SCAN_CHUNK; i += piece) {
    stage_lanes(q, wk, x, k, SCAN_CHUNK, t0 + i, piece, buf);
    q->f->scan(a, buf, k, 1, y + i * q->rd * q->w, q->rd,
               SCAN_CHUNK * q->rd, piece, k);
  }
}

/* One slice's chunks SLICES at a time, side by side, where a unit is one
   slice: their carries lie side by side in the scratch. A short last chunk
   runs alone, and so does a chunk 0 that has no carry. */
static void rescan_quads(const plan *q, work *wk, int64_t lo, int64_t hi) {
  for (int64_t v = lo; v < hi; v++) {
    int64_t c;
    int k;
    int64_t u = span(v, q->groups, &c, &k);
    unit x = unit_of(q, u * q->groups);
    uint8_t *a = q->scratch + (u * q->groups + c) * q->size;
    if (c == 0 && bare(q)) {
      rescan(q, wk, &x, 0, a);
      c = 1;
      a += q->size;
      k--;
    }
    int full = (c + k) * SCAN_CHUNK <= q->terms ? k : k - 1;
    int64_t t0 = c * SCAN_CHUNK;
    if (full > 0) quad(q, wk, &x, a, t0, full);
    if (full < k) rescan(q, wk, &x, c + full, a + full * q->size);
  }
}

static void carries(const plan *q, work *wk, int64_t lo, int64_t hi) {
  int64_t row = q->row * q->size;
  uint8_t *carry = wk->carry, *v = wk->v;
  for (int64_t u = lo; u < hi; u++) {
    unit x = unit_of(q, u * q->groups);
    start(q, &x, carry);
    for (int64_t c = 0; c < q->groups; c++) {
      uint8_t *s = q->scratch + (u * q->groups + c) * row;
      memcpy(v, s, row);
      memcpy(s, carry, row);
      if (c < q->groups - 1) carry_on(q, carry, v, c, x.W);
    }
  }
}

/* Lays out a scan along the axis [axis] of the destinations [d] and [d1]
   and the operand [x]: the axes before it and those after it coalesce
   apart, and with a program's loads, so that each kept axis steps every
   load evenly. */
static void plan_scan(plan *q, const nx_array *d, const nx_array *d1,
                      const nx_array *x, int axis) {
  int nl = q->fused != NULL ? q->fused->nloads : 0;
  int r = x->rank;
  q->nr = 1;
  q->re[0] = x->dim[axis];
  q->rs[0] = x->dim[r + axis];
  q->rd = d->dim[r + axis];
  q->rd1 = d1->dim[r + axis];
  q->x0 = x->offset;
  q->d0 = d->offset;
  q->d1 = d1->offset;
  q->c0 = q->cy != NULL ? q->cy->offset : 0;
  q->nk = 0;
  /* The carries, where given, coalesce with the rest, after the loads. */
  int nc = q->cy != NULL;
  int parts[2][2] = {{0, axis}, {axis + 1, r}};
  for (int k = 0; k < 2; k++) {
    int64_t extent[NX_MAX_RANK], step[NX_MAX_OPERANDS + 1][NX_MAX_RANK];
    int n = parts[k][1] - parts[k][0];
    for (int i = 0; i < n; i++) {
      int a = parts[k][0] + i;
      extent[i] = x->dim[a];
      step[0][i] = d->dim[r + a];
      step[1][i] = d1->dim[r + a];
      step[2][i] = x->dim[r + a];
      for (int o = 0; o < nl; o++)
        step[3 + o][i] = q->fused->loads[o].dim[r + a];
      if (nc) step[3 + nl][i] = q->cy->dim[r + a];
    }
    n = nx_coalesce_dims(3 + nl + nc, n, extent, step);
    for (int i = 0; i < n; i++)
      keep(q, extent[i], step[2][i], step[0][i], step[1][i],
           nc ? step[3 + nl][i] : 0);
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

/* Lays out [q]'s units for [op] of [x] along [axes] into [y] and [y1]. */
static void lay_out(plan *q, task op, const int *axes, int naxes,
                    const nx_array *y, const nx_array *y1, const nx_array *x) {
  if (op == SCAN) {
    plan_scan(q, y, y1, x, axes[0]);
    return;
  }
  /* The loop: the destinations step 0 along the reduced axes; a program's
     loads follow the operand, so that each of the plan's axes steps every
     load evenly. */
  int rank = x->rank, nl = q->fused != NULL ? q->fused->nloads : 0;
  nx_loop l = {.rank = rank, .first = {y->offset, y1->offset, x->offset}};
  for (int i = 0, k = 0, j = 0; i < rank; i++) {
    l.extent[i] = x->dim[i];
    l.step[2][i] = x->dim[rank + i];
    for (int o = 0; o < nl; o++)
      l.step[3 + o][i] = q->fused->loads[o].dim[rank + i];
    int reduced = j < naxes && axes[j] == i;
    j += reduced;
    l.step[0][i] = reduced ? 0 : y->dim[y->rank + k];
    l.step[1][i] = reduced ? 0 : y1->dim[y1->rank + k];
    k += !reduced;
  }
  l.rank = nx_coalesce_dims(3 + nl, rank, l.extent, l.step);
  plan_reduce(q, &l);
}

/* A reduction's units: each unit's blocks, or each unit's groups of them
   then the groups combined. */
static void reduce_jobs(plan *q, int64_t units, int64_t bytes, double cost) {
  if (q->groups == 1) {
    units_job(units, bytes, cost, reduce_units, q);
    return;
  }
  int64_t values = units * q->groups * q->row * q->size;
  units_job(units * q->groups, bytes, cost, reduce_units, q);
  units_job(units, values, values, finish_units, q);
}

/* Whether [op] by [r] of [x] along the [naxes] axes [axes] fits the
   destinations [y] and [y1]: NX_OK, setting the terms per output and the
   outputs, or NX_SHAPE, an extreme of no term included. */
static int checked(task op, int r, const int *axes, int naxes,
                   const nx_array *y, const nx_array *y1, const nx_array *x,
                   int64_t *terms, int64_t *outputs) {
  if (!fits(op, x->dim, x->rank, y->dim, y->rank, axes, naxes, terms,
            outputs) ||
      !fits(op, x->dim, x->rank, y1->dim, y1->rank, axes, naxes, terms,
            outputs))
    return NX_SHAPE;
  int extreme = r == NX_MONOID_MAX || r == NX_MONOID_MIN ||
                r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  if (op == REDUCE && *terms == 0 && *outputs > 0 && extreme) return NX_SHAPE;
  return NX_OK;
}

/* Reduces or scans, as [op] says, [x], of [dt] or staged into it, into
   [y] and [y1] by [r] in the dtype [dt]: a monoid into [y], [y1] being
   [y]; Logsumexp into [y], [y1] being [y], or Arg into the extreme [y]
   and its int64 position [y1]; or reduced, Moments into the mean [y] and
   the variance [y1]. The destinations are of [dt],
   but Arg's positions, and its extreme where that is of [x]'s dtype, its
   term copied. A monoid's scan starts from the carries [cy] where they are
   not NULL. NX_OK, NX_SHAPE for shapes that do not fit, an extreme of no
   term included, or -1 if host memory runs out. Apart, so that the casts
   reduce_into runs around it take none of its plan's frame. */
static __attribute__((noinline)) int reduce_in(
    task op, int r, int dt, const int *axes, int naxes, const nx_array *y,
    const nx_array *y1, const nx_array *x, const nx_array *cy,
    const fused *z) {
  int64_t terms, outputs;
  int e = checked(op, r, axes, naxes, y, y1, x, &terms, &outputs);
  if (e) return e;
  _Atomic int failed = 0;
  int stateful = r >= NX_CPU_MONOIDS, w = nx_cpu_width(dt), p = part_of(dt);
  int arg = r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  const nx_cpu_stateful *st =
      stateful ? &nx_cpu_table->stateful[r - NX_CPU_MONOIDS][dt] : NULL;
  plan q = {.f = stateful ? NULL : &nx_cpu_table->monoid[r][dt],
            .r = stateful ? &st->value : NULL,
            .z = stateful && op == SCAN ? &st->scan : NULL,
            .w = w,
            .nan = (p == NX_FLOAT32 || p == NX_FLOAT64) && !arg,
            .part = p,
            .parts = p == dt ? 1 : 2,
            .first = r == NX_MONOID_PROD && p != dt,
            .extreme = r == NX_MONOID_MAX || r == NX_MONOID_MIN,
            .slots = p != dt && r == NX_MONOID_SUM ? 2 : 1,
            .carrier = dt,
            .staged = x->dtype != dt,
            .own = arg && x->dtype != dt && y->dtype == x->dtype,
            .arg = arg,
            .size = stateful ? (int64_t)sizeof(nx_cpu_state) : w,
            .dst = y,
            .dst1 = y1,
            .src = x,
            .fused = z,
            .cy = cy,
            .terms = terms,
            .failed = &failed};
  if (outputs > 0 && terms == 0 && op == REDUCE) {
    /* The results of no term: the identity, or a state of none's. */
    _Alignas(16) uint8_t e0[WIDEST], e1[WIDEST];
    if (stateful) q.r->finish(&(nx_cpu_state){0}, 0, e0, e1);
    else identity(r, dt, e0);
    fill_dst(y, e0, w);
    if (y1 != y) fill_dst(y1, e1, nx_cpu_width(y1->dtype));
  }
  if (outputs == 0 || terms == 0) return NX_OK;
  if (!stateful) identity(r, dt, q.id);
  int64_t bytes = terms * outputs * w;
  double cost = stateful ? STATE_COST[r - NX_CPU_MONOIDS] * bytes
                : op == REDUCE && terms <= LANES ? FEW_COST * bytes
                                                 : bytes;
  /* A program's kinds take about memcpy's time each, per term. */
  if (z != NULL) cost += (double)nx_cpu_prog_kinds(z->prog) * bytes;
  q.threads = nx_cpu_threads(bytes, cost);
  lay_out(&q, op, axes, naxes, y, y1, x);
  int64_t units = q.outer * q.bands * q.chunks;
  if (q.groups > 1)
    q.scratch = malloc((size_t)(units * q.groups * q.row * q.size));
  int center = op == REDUCE && stateful && st->center.block != NULL;
  if (center) q.centers = malloc((size_t)(units * q.row) * sizeof(double));
  if ((q.groups > 1 && q.scratch == NULL) || (center && q.centers == NULL)) {
    free(q.scratch);
    free(q.centers);
    return -1;
  }
  if (center) {
    q.r = &st->center;
    q.centering = 1;
    reduce_jobs(&q, units, bytes, cost);
    q.r = &st->value;
    q.centering = 0;
  }
  if (op == REDUCE) reduce_jobs(&q, units, bytes, cost);
  else if (q.groups == 1) units_job(units, bytes, cost, scan_units, &q);
  else {
    units_job(units * q.groups, bytes, cost, total_units, &q);
    units_serial(units, carries, &q);
    if (q.row == 1)
      units_job(units * ((q.groups + SLICES - 1) / SLICES), bytes, cost,
                rescan_quads, &q);
    else units_job(units * q.groups, bytes, cost, rescan_units, &q);
    units_serial(units, settle_units, &q);
  }
  free(q.scratch);
  free(q.centers);
  return failed ? -1 : NX_OK;
}

/* A C-contiguous array of [y]'s shape in the dtype [c], in scratch at
   [t->base] that the caller gives back: 0, or 1 if host memory runs out. */
static int carrier_copy(const nx_array *y, int c, nx_array *t) {
  int r = y->rank, w = nx_cpu_width(c);
  *t = (nx_array){.dtype = c, .bits = 8 * w, .rank = r};
  int64_t n = 1;
  for (int i = r - 1; i >= 0; i--) {
    t->dim[i] = y->dim[i];
    t->dim[r + i] = n;
    n *= y->dim[i];
  }
  t->base = nx_cpu_take((n > 0 ? n : 1) * w);
  return t->base == NULL;
}

/* Where a copy lives: the thread's kept scratch, which the caller gives
   back, or the heap, which it frees. */
typedef enum { KEPT_SCRATCH, HEAP } home;

/* A C-contiguous copy of [x] at [t->base], in [h]: 0, or 1 if host memory
   runs out. */
static int own_copy(const nx_array *x, nx_array *t, home h) {
  int r = x->rank;
  *t = (nx_array){.dtype = x->dtype, .bits = x->bits, .rank = r};
  int64_t n = 1;
  for (int i = r - 1; i >= 0; i--) {
    t->dim[i] = x->dim[i];
    t->dim[r + i] = n;
    n *= x->dim[i];
  }
  /* An operand of no element has no base, as nx_read gives it. */
  if (n == 0) return 0;
  int64_t bytes = (n * x->bits + 7) / 8;
  t->base = h == KEPT_SCRATCH ? nx_cpu_take(bytes) : malloc((size_t)bytes);
  if (t->base == NULL) return 1;
  nx_array pair[2] = {*t, *x};
  nx_loop l;
  nx_coalesce(2, pair, &l);
  nx_cpu_copy_loop(t->base, x->base, x->bits, &l);
  return 0;
}

/* Casts [a] into [b], of one shape, as a cast does. Apart, so that its
   loop's 5 KiB are off the frame its caller's jobs run under. */
static __attribute__((noinline)) void cast(const nx_array *b,
                                           const nx_array *a) {
  nx_array pair[2] = {*b, *a};
  nx_loop l;
  nx_coalesce(2, pair, &l);
  nx_cpu_cast_loop(b, a, &l);
}

/* As reduce_in, of [x] of any dtype in its carrier, into destinations of
   any dtype but Arg's int64 positions: where theirs is not the carrier,
   into carrier_copy's arrays, then rounded once into them, but Arg's
   extreme of [x]'s dtype, which reduce_in copies from [x]. A monoid's scan
   starts from the carries [cy], of the carrier, where they are not
   NULL. */
static int reduce_into(task op, int r, const int *axes, int naxes,
                       const nx_array *y, const nx_array *y1,
                       const nx_array *x, const nx_array *cy,
                       const fused *z) {
  int c = nx_cpu_carrier(x->dtype);
  int arg = r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN;
  const nx_array *y2[2] = {y, y1}, *d[2] = {y, y1};
  nx_array t[2];
  int copied[2] = {0, 0}, e = NX_OK;
  for (int k = 0; k < 2 && e == NX_OK; k++) {
    int kept = arg && (k == 1 || y->dtype == x->dtype);
    if (k == 1 && y1 == y) d[1] = d[0];
    else if (!kept && y2[k]->dtype != c) {
      if (carrier_copy(y2[k], c, &t[k])) e = -1;
      else {
        d[k] = &t[k];
        copied[k] = 1;
      }
    }
  }
  if (e == NX_OK)
    e = reduce_in(op, r, c, axes, naxes, d[0], d[1], x, cy, z);
  for (int k = 0; k < 2; k++) {
    if (!copied[k]) continue;
    if (e == NX_OK) cast(y2[k], &t[k]);
    nx_cpu_give(t[k].base);
  }
  return e;
}

/* Network scans */

/* A scan by Logsumexp or Moments [r]: the outer axes, those but the
   scanned one, coalesced, with the steps of the result (0), Moments'
   variance (1), the result again for Logsumexp, and the operand (2); the
   scanned axis's extent and steps. [failed] is set where a unit's scratch
   could not be allocated. A unit runs NX_CPU_NETWORK_LANES slices side by
   side, along the outer axis that steps least. */
typedef struct {
  int r;
  const nx_array *a[3];
  nx_loop outer;
  int64_t len, along[3];
  int64_t rows, chunks; /* outer positions but the last axis; units along it */
  const fused *fused;   /* a program's terms, or NULL */
  _Atomic int failed;
} network;

/* A unit's terms of a program: [w] slices from the virtual position [p],
   [lane] apart, each stepping [along], in the carrier [c], their held
   values through [held]. */
typedef struct {
  const fused *fused;
  int64_t p, lane, along, w;
  int c;
  uint8_t *held;
} slices;

/* Terms [t0, t0 + tn) of the slices, at most a tile of them, a narrow
   float's of two bytes or fewer held. */
static void fill_slices(const void *ctx, int64_t t0, int64_t tn,
                        uint8_t *out) {
  const slices *c = ctx;
  eval_in(c->fused, c->c, c->p + t0 * c->along, c->lane, c->w, c->along, tn,
          out, c->held);
}

/* Quiets the NaN results of the [w] slices whose results start at the
   positions [at]. A slice that a NaN term reaches ends in a NaN, and its
   results from that term on carry its bits through the kinds'
   arithmetic: such a slice alone is walked again. */
static void quiet_slices(const network *q, const int64_t *at, int64_t w) {
  const nx_loop *l = &q->outer;
  int last = l->rank - 1, dt = q->a[0]->dtype, w8 = nx_cpu_width(dt);
  int two = q->r == NX_CPU_MOMENTS;
  for (int64_t j = 0; j < w; j++) {
    uint8_t *m = q->a[0]->base + (at[0] + j * l->step[0][last]) * w8;
    uint8_t *v = q->a[1]->base + (at[1] + j * l->step[1][last]) * w8;
    if (!floats_nan(dt, m + (q->len - 1) * q->along[0] * w8, 1)) continue;
    for (int64_t i = 0; i < q->len; i++) {
      quiet(dt, m + i * q->along[0] * w8);
      if (two) quiet(dt, v + i * q->along[1] * w8);
    }
  }
}

static void network_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  network *q = ctx;
  const nx_loop *l = &q->outer;
  int last = l->rank - 1, w8 = nx_cpu_width(q->a[2]->dtype);
  int64_t inner = l->extent[last], most = NX_CPU_NETWORK_LANES;
  int64_t lanes = inner < most ? inner : most;
  int64_t levels = nx_cpu_network_levels(q->len);
  int64_t states = (12 * levels + 3 * NX_CPU_NETWORK_TILE) * lanes;
  if (states < NETWORK_ONE) states = NETWORK_ONE;
  /* The pass's states, then a tile of a program's held values, which are
     no wider than the carrier. */
  int64_t held = NX_CPU_NETWORK_TILE * lanes;
  uint8_t *scratch = nx_cpu_take((states + held) * w8);
  if (scratch == NULL) {
    q->failed = 1;
    return;
  }
  int dt = q->a[2]->dtype;
  nx_cpu_network run = q->r == NX_CPU_MOMENTS ? nx_cpu_table->moments[dt]
                                              : nx_cpu_table->logsumexp[dt];
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
    slices c = {q->fused,       at[2], l->step[2][last], q->along[2], w,
                q->a[2]->dtype, scratch + states * w8};
    nx_cpu_terms src = {fill_slices, &c};
    /* A program's operand is virtual, of no buffer. */
    const uint8_t *x = q->fused != NULL ? NULL : q->a[2]->base + at[2] * w8;
    run(x, q->along[2], l->step[2][last],
        q->a[0]->base + at[0] * w8, q->along[0], l->step[0][last],
        q->a[1]->base + at[1] * w8, q->along[1], l->step[1][last], q->len, w,
        scratch, q->fused != NULL ? &src : NULL);
    quiet_slices(q, at, w);
  }
  nx_cpu_give(scratch);
}

/* Scans [x] along [axis], [outputs] slices of [terms] terms, by
   Logsumexp or Moments [r] into [y] and Moments' variance [y1], [y1]
   being [y] for Logsumexp, float32 or float64 of [x]'s dtype, all of
   shapes that fit: NX_OK, or -1 if host memory runs out. [x] is a
   program's virtual operand in its carrier where [z] is not NULL, the
   outer axes coalesced with its loads. Apart, so that the casts around
   it run on a small frame. */
static __attribute__((noinline)) int network_in(
    int r, int axis, int64_t terms, int64_t outputs, const nx_array *y,
    const nx_array *y1, const nx_array *x, const fused *z) {
  network q = {.r = r, .a = {y, y1, x}, .len = terms, .fused = z};
  nx_loop *l = &q.outer;
  int rank = x->rank, k = 0, nl = z != NULL ? z->nloads : 0;
  for (int o = 0; o < 3; o++) {
    l->first[o] = q.a[o]->offset;
    q.along[o] = q.a[o]->dim[rank + axis];
  }
  for (int i = 0; i < rank; i++) {
    if (i == axis) continue;
    l->extent[k] = x->dim[i];
    for (int o = 0; o < 3; o++) l->step[o][k] = q.a[o]->dim[rank + i];
    for (int o = 0; o < nl; o++) l->step[3 + o][k] = z->loads[o].dim[rank + i];
    k++;
  }
  /* A slice alone keeps one outer axis, of extent 1. */
  if (k == 0) {
    l->extent[0] = 1;
    for (int o = 0; o < 3 + nl; o++) l->step[o][0] = 0;
    k = 1;
  }
  l->rank = nx_coalesce_dims(3 + nl, k, l->extent, l->step);
  int64_t inner = l->extent[l->rank - 1];
  q.rows = outputs / inner;
  q.chunks = (inner + NX_CPU_NETWORK_LANES - 1) / NX_CPU_NETWORK_LANES;
  int64_t bytes = 3 * terms * outputs * nx_cpu_width(x->dtype);
  double cost = 4.0 * bytes;
  if (z != NULL) cost += (double)nx_cpu_prog_kinds(z->prog) * bytes / 3;
  nx_cpu_job(q.rows * q.chunks, bytes, cost, network_units, &q);
  return q.failed ? -1 : NX_OK;
}

/* As network_in, of [x] of any float dtype in its carrier, into
   destinations of any dtype: a narrow [x] cast whole into carrier_copy's
   array first, or a program's terms converted tile by tile, and
   destinations not of the carrier computed into such arrays, then rounded
   once into them. NX_OK, NX_SHAPE for shapes that do not fit, or -1 if host
   memory runs out. */
static int network_into(int r, int axis, const nx_array *y,
                        const nx_array *y1, const nx_array *x,
                        const fused *z) {
  int64_t terms, outputs;
  if (!fits(NETWORK, x->dim, x->rank, y->dim, y->rank, &axis, 1, &terms,
            &outputs) ||
      !fits(NETWORK, x->dim, x->rank, y1->dim, y1->rank, &axis, 1, &terms,
            &outputs))
    return NX_SHAPE;
  if (outputs == 0 || terms == 0) return NX_OK;
  int c = nx_cpu_carrier(x->dtype), e = NX_OK;
  int nd = y1 == y ? 1 : 2;
  const nx_array *a[3] = {y, y1, x}, *d[3] = {y, y1, x};
  nx_array t[3];
  int copied[3] = {0, 0, 0};
  for (int k = 0; k < 3 && e == NX_OK; k++) {
    if (k == 1 && nd == 1) {
      d[1] = d[0];
      continue;
    }
    if (a[k]->dtype == c) continue;
    if (k == 2 && z != NULL) {
      t[2] = *x;
      t[2].dtype = c;
      t[2].bits = 8 * nx_cpu_width(c);
      d[2] = &t[2];
      continue;
    }
    if (carrier_copy(a[k], c, &t[k])) {
      e = -1;
      continue;
    }
    if (k == 2) cast(&t[k], x);
    d[k] = &t[k];
    copied[k] = 1;
  }
  if (e == NX_OK)
    e = network_in(r, axis, terms, outputs, d[0], d[1], d[2], z);
  for (int k = 0; k < 3; k++) {
    if (!copied[k]) continue;
    if (e == NX_OK && k < 2) cast(a[k], &t[k]);
    nx_cpu_give(t[k].base);
  }
  return e;
}

/* Entries */

/* Reads [x], one destination [d0] or two, [d0] and [d1], and, where
   [vpad] is Some (pad, fill), [x] copied padded with the bits [fill] into
   C-heap scratch (assemble.c), and computes [c] of them: [r]'s reduction
   along the [naxes] axes [axes], its scan along [axes[0]], or a Moments
   scan along [axes[0]] into the two destinations, as [op] says.
   The destinations are of any dtype, but Arg's int64 positions. The pad
   is trusted: one that fits no operand of [x]'s shape is refused before
   anything reads [x]. */
typedef struct {
  task op;
  int r, naxes;
  int axes[NX_MAX_RANK];
} call;

/* What a term of a call takes (cpu.h, Calls, scalar): a reduction's or a
   scan's by sums and products, extremes, Logsumexp, Arg and Moments,
   along adjacent terms or strided ones whose neighbours along a kept axis
   share their lines. Worst float16, float32 or float64 terms: reduced,
   29 (M1), 57 (fudge), 236 (M1), 27 and 70 (M1); scanned, 101, 194, 350,
   504 and 189 (M1). A reduction of few terms to an output (a lane or
   fewer) takes up to 57 a term of a monoid (FEW_TERM_COST, M1 alone). A
   term no neighbour shares its line with adds a line (NX_CPU_LINE); so
   does a stateful reduction's strided or few one. */
enum { BY_SUMS, BY_EXTREMES, BY_LOGSUMEXP, BY_ARG, BY_MOMENTS };
static const double TERM_COST[2][5] = {{64, 128, 512, 64, 128},
                                       {256, 512, 1024, 1024, 512}};
#define FEW_TERM_COST 128
#define DEAREST_TERM 1024 /* TERM_COST's largest */

/* The row of TERM_COST a term of [c] takes. */
static int term_of(const call *c) {
  if (c->r == NX_CPU_MOMENTS) return BY_MOMENTS;
  if (c->r == NX_CPU_LOGSUMEXP) return BY_LOGSUMEXP;
  if (c->r == NX_CPU_ARG_MAX || c->r == NX_CPU_ARG_MIN) return BY_ARG;
  return c->r <= NX_MONOID_PROD ? BY_SUMS : BY_EXTREMES;
}

/* Whether the loaded operand [x], of the loaded shape [s] and rank [r],
   steps one element along axis [i]: a [copied] one, padded or aliased, is
   read from a C-contiguous copy. */
static int unit_step(const nx_array *x, const int64_t *s, int r, int copied,
                     int i) {
  if (copied) {
    for (int j = i + 1; j < r; j++)
      if (s[j] != 1) return 0;
    return 1;
  }
  return x->dim[r + i] == 1 || x->dim[r + i] == -1;
}

/* The price (cpu.h, Calls) of a reduction of no term's fill of [d]
   (fill_dst): streaming its bytes, and ROW_COST for each of its coalesced
   rows, 102 measured for uint8 rows of 2, 64 bytes apart (M1 alone); or a
   line for each element where a row's elements lie apart. */
#define ROW_COST 256

static double fill_cost(const nx_array *d) {
  int64_t extent[NX_MAX_RANK], step[1][NX_MAX_RANK];
  int r = rows_of(d, extent, step);
  double rows = 1, n = (double)extent[r - 1];
  for (int i = 0; i < r - 1; i++) rows *= (double)extent[i];
  int apart = step[0][r - 1] != 1 && step[0][r - 1] != -1;
  return nx_cpu_bytes(d) + rows * (apart ? n * NX_CPU_LINE : ROW_COST);
}

/* The price (cpu.h, Calls) of the call [c] over the operand [x] of the
   loaded shape [s] and rank [r], [copied] or not (unit_step), into the
   [nd] destinations [d]: its terms, each [extra] more, or, without a
   term, the destinations' fills. A call too small to cost
   NX_CPU_RELEASE_BYTES at the dearest term is priced so. */
static double call_cost(const call *c, const nx_array *x, const int64_t *s,
                        int r, int copied, double extra, const nx_array *d,
                        int nd) {
  double numel = 1, cost = 0, pad = extra;
  for (int i = 0; i < r; i++) numel *= (double)s[i];
  if (numel == 0) {
    for (int k = 0; k < nd; k++) cost += fill_cost(&d[k]);
    return cost;
  }
  /* The most a term takes: a line and a scanned term's dearest. */
  double most = NX_CPU_LINE + DEAREST_TERM + pad;
  if (numel * most < NX_CPU_RELEASE_BYTES) return numel * most;
  double terms = 1;
  int reduced[NX_MAX_RANK] = {0}, shared = 0;
  for (int i = 0; i < c->naxes; i++)
    if (c->axes[i] < r) {
      terms *= (double)s[c->axes[i]];
      reduced[c->axes[i]] = 1;
    }
  for (int i = 0; i < r && !shared; i++)
    shared = !reduced[i] && s[i] >= LANES && unit_step(x, s, r, copied, i);
  int last = c->naxes > 0 ? c->axes[c->naxes - 1] : r;
  int adjacent = last >= r || unit_step(x, s, r, copied, last);
  int few = c->op == REDUCE && terms <= LANES;
  int t = term_of(c), stateful = t != BY_SUMS && t != BY_EXTREMES;
  double term = TERM_COST[c->op != REDUCE][t];
  if (few && !stateful) term = FEW_TERM_COST;
  if ((!adjacent && !shared) || (stateful && (!adjacent || few)))
    term += NX_CPU_LINE;
  return numel * (term + pad);
}

/* Whether the carries [cy], where not NULL, fit [c], a monoid's scan into
   [y]: NX_OK; NX_ARITY for another call's; NX_SHAPE unless they have [y]'s
   shape but one along the scan's axis. */
static int carried(const call *c, const nx_array *cy, const nx_array *y) {
  if (cy == NULL) return NX_OK;
  if (c->op != SCAN || c->r > NX_MONOID_MIN) return NX_ARITY;
  if (cy->rank != y->rank) return NX_SHAPE;
  for (int i = 0; i < y->rank; i++)
    if (cy->dim[i] != (i == c->axes[0] ? 1 : y->dim[i])) return NX_SHAPE;
  return NX_OK;
}

/* reduce.c's code of the Nx_kernel.reduction [r]: a Monoid's monoid,
   NX_CPU_LOGSUMEXP for Logsumexp, NX_CPU_ARG_MAX or NX_CPU_ARG_MIN for an
   Arg, NX_CPU_MOMENTS for Moments. */
static int code_of(value r) {
  if (Is_long(r))
    return Long_val(r) == NX_REDUCTION_LOGSUMEXP ? NX_CPU_LOGSUMEXP
                                                 : NX_CPU_MOMENTS;
  int k = Int_val(Field(r, 0));
  return Tag_val(r) == 0 ? k : NX_CPU_ARG_MAX + k;
}

/* The dtype of the program [vp]'s one load where [vp] is that load's
   identity (Prog.is_identity), else -1. Reads [vp] alone. */
static int identity_of(value vp) {
  value ins = Field(vp, NX_PROG_INS), outs = Field(vp, NX_PROG_OUTS);
  if (Wosize_val(ins) != 1 || Wosize_val(outs) != 1) return -1;
  value node = Field(Field(vp, NX_PROG_NODES), Int_val(Field(outs, 0)));
  return Tag_val(node) == NX_NODE_IN ? Int_val(Field(ins, 0)) : -1;
}

/* Computes [c] of [vx], a program's one load that is its identity, of
   the program's dtype [dt], read in place, into the destinations [d0] and
   [d1] (Val_unit where [c] has one), from the carries [vcarry] where it is
   Some; [vpad] holds the load's pad and fill where it is padded. Nothing
   is copied before the door: a call of a few elements allocates nothing.
   A padded load is read from its padded copy; one a destination aliases,
   from a copy too, since a scan reads its terms again after it writes
   their results: for its chunks' totals, and in a Moments scan's exact
   pass. */
static value run(const call *c, int dt, value d0, value d1, value vx,
                 value vpad, value vcarry) {
  CAMLparam5(d0, d1, vx, vpad, vcarry);
  int nd = d1 == Val_unit ? 1 : 2;
  int arg = c->r == NX_CPU_ARG_MAX || c->r == NX_CPU_ARG_MIN;
  /* The pad and the fill, copied before anything that can move them. */
  int padded = Is_block(vpad);
  nx_cpu_pad pad;
  uint8_t bits[16] = {0};
  if (padded && nx_cpu_padding_of(Field(vpad, 0), vx, &pad, bits))
    CAMLreturn(Val_int(NX_SHAPE));
  int dt1 = nd == 1 ? dt : arg ? NX_INT64 : nx_array_dtype(d1);
  int nc = Is_block(vcarry), n = nd + 1 + nc;
  nx_operand in[4] = {{d0, nx_array_dtype(d0), 1}, {d1, dt1, 1}};
  in[nd] = (nx_operand){vx, dt, 0};
  if (nc) in[nd + 1] = (nx_operand){Field(vcarry, 0), nx_cpu_carrier(dt), 0};
  nx_array a[4], loaded, *x = &a[nd];
  int e = nx_read(n, in, a);
  if (e) CAMLreturn(Val_int(e));
  const nx_array *cy = nc ? &a[nd + 1] : NULL;
  int copy = padded || x->alias;
  int64_t ys[NX_MAX_RANK], terms, outputs;
  if (padded) nx_cpu_padded_shape(&pad, x->rank, x->dim, ys);
  int r = padded ? pad.rank + pad.nwindows : x->rank;
  /* The destinations fit the operand, padded, before it is copied. */
  nx_array loop = {.rank = r};
  memcpy(loop.dim, padded ? ys : x->dim, sizeof(int64_t) * (size_t)r);
  if ((e = checked(c->op, c->r, c->axes, c->naxes, &a[0], &a[nd - 1], &loop,
                   &terms, &outputs)) ||
      (e = carried(c, cy, &a[0]))) {
    nx_done(n, a);
    CAMLreturn(Val_int(e));
  }
  double extra = copy ? NX_CPU_STRIDED_COST * nx_cpu_width(dt) : 0;
  int released = nx_cpu_release(
      call_cost(c, x, padded ? ys : x->dim, r, copy, extra, a, nd));
  /* A carry a destination aliases is read from a copy too: a Max's or a
     Min's scan reads it again after it writes the results (settle). */
  nx_array kept_carry = {0};
  int failed = cy != NULL && cy->alias && own_copy(cy, &kept_carry, HEAP);
  if (!failed)
    failed = padded     ? nx_cpu_unpad(x, &pad, bits, &loaded)
             : x->alias ? own_copy(x, &loaded, KEPT_SCRATCH)
                        : 0;
  if (failed) {
    free(kept_carry.base);
    nx_cpu_reacquire(released);
    nx_done(n, a);
    caml_raise_out_of_memory();
  }
  if (copy) x = &loaded;
  if (kept_carry.base != NULL) cy = &kept_carry;
  e = c->op == NETWORK
          ? network_into(c->r, c->axes[0], &a[0], &a[nd - 1], x, NULL)
          : reduce_into(c->op, c->r, c->axes, c->naxes, &a[0], &a[nd - 1],
                        x, cy, NULL);
  if (padded) free(loaded.base);
  else if (copy) nx_cpu_give(loaded.base);
  free(kept_carry.base);
  nx_cpu_reacquire(released);
  nx_done(n, a);
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

/* Whether [r] has two destinations: Arg's and Moments'. */
static int two(int r) {
  return r == NX_CPU_ARG_MAX || r == NX_CPU_ARG_MIN || r == NX_CPU_MOMENTS;
}

/* What run_fused holds on the C heap: the program's copy, past
   NX_MAX_OPERANDS entries its tables, and its padded loads' pads. A custom
   block owns them, so that an exception nx_read raises, Out_of_memory or
   Rig.Lost, leaves them to its finalizer. */
typedef struct {
  nx_cpu_prog *g;
  void *tables;
  nx_cpu_pad *pads;
} fused_held;

static void release(value owner) {
  fused_held *h = Data_custom_val(owner);
  if (h->g != NULL) nx_cpu_prog_free(h->g);
  free(h->tables);
  free(h->pads);
  h->g = NULL;
  h->tables = NULL;
  h->pads = NULL;
}

/* release, what a count sized freed with the domain lock released where
   [many] (nx_free_released). */
static void drop(value owner, int many) {
  if (!many) {
    release(owner);
    return;
  }
  fused_held h = *(fused_held *)Data_custom_val(owner);
  *(fused_held *)Data_custom_val(owner) = (fused_held){NULL, NULL, NULL};
  caml_enter_blocking_section_no_pending();
  nx_cpu_prog_free(h.g);
  free(h.tables);
  free(h.pads);
  caml_leave_blocking_section();
}

static struct custom_operations fused_ops = {
    "nx.cpu.reduce",            release,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* The operands of a fused call as the door reads them, from the roots
   [vdsts], [vloads] and [vcarry]: the destinations, each of its own dtype
   or the one [dts] names, then the loads, as [g] loads them, then the
   carry, in the carrier of output [k]'s held dtype. */
typedef struct {
  const value *vdsts, *vloads, *vcarry;
  const int *dts;
  const nx_cpu_prog *g;
  int nd, nl, k;
} fused_operands;

static int fused_operand(void *ctx, int k, nx_operand *in) {
  const fused_operands *o = ctx;
  if (k < o->nd) {
    value y = Field(*o->vdsts, k);
    int dt = o->dts[k] < 0 ? nx_array_dtype(y) : o->dts[k];
    *in = (nx_operand){y, dt, 1};
  } else if (k < o->nd + o->nl) {
    int i = k - o->nd;
    *in = (nx_operand){Field(Field(*o->vloads, i), 0), nx_cpu_prog_in(o->g, i),
                       0};
  } else
    *in = (nx_operand){Field(*o->vcarry, 0),
                       nx_cpu_carrier(nx_cpu_prog_held(o->g, o->k)), 0};
  return NX_OK;
}

/* Computes [c] of the program [vp] over the loop of [rank] axes of extents
   [shape], or of its first destination's shape where [rank] is -1, its
   loads [vloads], each copied padded where [vpads], empty where none is
   padded, holds Some (pad, fill), by the reductions [vrs], an array of
   pairs of an Nx_kernel.reduction and an output, or a scan's one pair
   where [single], in turn into the destinations [vdsts], each reduction's
   one or two as run takes them. The loop's terms are virtual, read
   through the program (Programs), in the output's held dtype. Every
   destination is checked before any is written, so that a refusal writes
   nothing. The program and the pads are
   copied before the door. Each reduction's loop holds its one or two
   destinations, the virtual operand and the loads, whatever the count of
   reductions, which nothing bounds: their tables, and the door's of every
   destination and load, live on the stack up to NX_MAX_OPERANDS entries
   and past it on the heap, which a custom block owns with the program's
   copy. */
static value run_fused(const call *c, value vp, value vrs, int single,
                       int rank, const int64_t *shape, value vdsts,
                       value vloads, value vpads, value vcarry) {
  CAMLparam5(vp, vrs, vdsts, vloads, vpads);
  CAMLxparam1(vcarry);
  CAMLlocal1(owner);
  owner = caml_alloc_custom(&fused_ops, sizeof(fused_held), 0, 1);
  *(fused_held *)Data_custom_val(owner) = (fused_held){NULL, NULL, NULL};
  nx_cpu_prog *g = nx_cpu_prog_copy(&vp);
  if (g == NULL) caml_raise_out_of_memory();
  ((fused_held *)Data_custom_val(owner))->g = g;
  int nl = nx_cpu_prog_ins(g), nd = (int)Wosize_val(vdsts), e = NX_OK;
  int nc = Is_block(vcarry);
  int nr = single ? 1 : (int)Wosize_val(vrs), want = 0;
  /* The tables, on the stack up to NX_MAX_OPERANDS entries, else in one
     block of the heap: the door's descriptors and each destination's
     dtype, -1 for its own, then each reduction's code, output and first
     destination. */
  int nt = nr > NX_MAX_OPERANDS ? nr : NX_MAX_OPERANDS;
  int no = nd + nl + nc > NX_MAX_OPERANDS ? nd + nl + nc : NX_MAX_OPERANDS;
  nx_array a_stack[NX_MAX_OPERANDS], *a = a_stack;
  int tables[4 * NX_MAX_OPERANDS], *dts = tables;
  if (nt > NX_MAX_OPERANDS || no > NX_MAX_OPERANDS) {
    uint8_t *b = malloc((size_t)no * (sizeof *a + sizeof *dts) +
                        3 * (size_t)nt * sizeof *dts);
    if (b == NULL) {
      release(owner);
      caml_raise_out_of_memory();
    }
    ((fused_held *)Data_custom_val(owner))->tables = b;
    a = (nx_array *)b;
    dts = (int *)(b + (size_t)no * sizeof *a);
  }
  int *rs = dts + no, *ks = rs + nt, *first = ks + nt;
  /* Tables or a program sized by a count, freed with the lock released. */
  int many = nt > NX_MAX_OPERANDS || no > NX_MAX_OPERANDS ||
             nx_cpu_prog_nodes(g) > NX_CPU_ITEMS;
  if (nr < 1 || (c->op != REDUCE && nr != 1) || (int)Wosize_val(vloads) != nl)
    e = NX_ARITY;
  /* The reductions are read from the root, letting other domains collect
     every NX_CPU_ITEMS of them. */
  for (int i = 0, next = NX_CPU_ITEMS; !e && i < nr; i++) {
    if (i == next) {
      nx_let_collect();
      next += NX_CPU_ITEMS;
    }
    value pair = single ? vrs : Field(vrs, i);
    rs[i] = code_of(Field(pair, 0));
    ks[i] = Int_val(Field(pair, 1));
    first[i] = want;
    if (want < nd) dts[want] = -1;
    if (two(rs[i]) && want + 1 < nd)
      dts[want + 1] =
          rs[i] == NX_CPU_ARG_MAX || rs[i] == NX_CPU_ARG_MIN ? NX_INT64 : -1;
    want += 1 + two(rs[i]);
    if (ks[i] < 0 || ks[i] >= nx_cpu_prog_outs(g)) e = NX_ARITY;
  }
  if (!e && want != nd) e = NX_ARITY;
  /* A reduction's loop holds its destinations, the virtual operand and the
     loads: a program's loads leave room for them (Prog.max_operands). */
  if (!e && 3 + nl > NX_MAX_OPERANDS) e = NX_ARITY;
  /* The pads on the heap, where a load is padded, as its copy will be. */
  int padded[NX_MAX_OPERANDS] = {0};
  nx_cpu_pad *pad = NULL;
  uint8_t fill[NX_MAX_OPERANDS][16] = {{0}};
  for (int i = 0; !e && i < nl && Wosize_val(vpads) > 0; i++) {
    value v = Field(vpads, i);
    padded[i] = Is_block(v);
    if (!padded[i]) continue;
    if (pad == NULL) {
      pad = malloc((size_t)nl * sizeof *pad);
      if (pad == NULL) {
        release(owner);
        caml_raise_out_of_memory();
      }
      ((fused_held *)Data_custom_val(owner))->pads = pad;
    }
    if (nx_cpu_padding_of(Field(v, 0), Field(Field(vloads, i), 0), &pad[i],
                          fill[i]))
      e = NX_SHAPE;
  }
  if (e) {
    drop(owner, many);
    CAMLreturn(Val_int(e));
  }
  /* The door: past NX_MAX_OPERANDS operands in runs, else as one. */
  fused_operands o = {&vdsts, &vloads, &vcarry, dts, g, nd, nl, ks[0]};
  int n = nd + nl + nc;
  if (n > NX_MAX_OPERANDS) e = nx_read_each(n, fused_operand, &o, a);
  else {
    nx_operand in[NX_MAX_OPERANDS];
    for (int k = 0; k < n; k++) fused_operand(&o, k, &in[k]);
    e = nx_read(n, in, a);
  }
  if (e) {
    drop(owner, many);
    CAMLreturn(Val_int(e));
  }
  /* A call whose reductions and nodes cost a release's worth to check and
     price does both with the lock released, whatever its work costs. */
  int early = nr + nx_cpu_prog_nodes(g) > NX_CPU_ITEMS, released = 0;
  if (early) released = nx_cpu_release(NX_CPU_RELEASE_BYTES);
  if (rank < 0) {
    rank = a[0].rank;
    shape = a[0].dim;
  }
  const nx_array *cy = nc ? &a[nd + nl] : NULL;
  /* Every reduction's destinations fit the loop, before any load is
     copied; nx_cpu.ml checked that the loads have its shape. */
  nx_array loop = {.rank = rank};
  memcpy(loop.dim, shape, sizeof(int64_t) * (size_t)rank);
  for (int i = 0; !e && i < nr; i++) {
    int64_t terms, outputs;
    const nx_array *y = &a[first[i]];
    e = checked(c->op, rs[i], c->axes, c->naxes, y,
                two(rs[i]) ? y + 1 : y, &loop, &terms, &outputs);
  }
  if (!e && nc) {
    call ci = *c;
    ci.r = rs[0];
    e = carried(&ci, cy, &a[first[0]]);
  }
  if (e) {
    nx_cpu_reacquire(released);
    nx_done(n, a);
    drop(owner, many);
    CAMLreturn(Val_int(e));
  }
  /* The call's price (cpu.h, Calls): each reduction's, as run prices
     one over the loop's C-contiguous virtual operand (call_cost), each term
     also running the program's kinds (nx_cpu_prog_cost), and the copies of
     padded and aliased loads. */
  double per = 0, cost = 0;
  for (int i = 0; i < nl; i++)
    if (padded[i] || a[nd + i].alias)
      per += NX_CPU_STRIDED_COST * nx_cpu_width(a[nd + i].dtype);
  per += nx_cpu_prog_cost(g);
  for (int i = 0; i < nr; i++) {
    call ci = *c;
    ci.r = rs[i];
    cost += call_cost(&ci, &a[nd], shape, rank, 1, per, &a[first[i]],
                      1 + two(rs[i]));
  }
  if (!early) released = nx_cpu_release(cost);
  /* The loop of the program and its loads as it reads them, in the
     thread's kept scratch, taken after the door and given back before its
     end, since nothing raises in between: with a reduction's two carrier
     copies and its work, four kept buffers at most. */
  uint8_t *held = nx_cpu_take((int64_t)sizeof(fused) + nl * sizeof(nx_array));
  int copied = 0, oom = held == NULL;
  fused *z = (fused *)held;
  nx_array *loads = oom ? NULL : (nx_array *)(held + sizeof(fused));
  /* A padded load is read from its padded copy; one a destination
     aliases, from a copy too, as run reads such an operand: on the heap,
     as many as the loads, so that the thread's kept buffers hold no more
     than the reductions' copies and work. */
  for (; copied < nl && !oom; copied++) {
    nx_array *l = &a[nd + copied];
    loads[copied] = *l;
    if (padded[copied])
      oom = nx_cpu_unpad(l, &pad[copied], fill[copied], &loads[copied]);
    else if (l->alias) oom = own_copy(l, &loads[copied], HEAP);
  }
  /* A carry a destination aliases, from a copy, as run reads it. */
  nx_array kept_carry = {0};
  if (!oom && cy != NULL && cy->alias) oom = own_copy(cy, &kept_carry, HEAP);
  if (kept_carry.base != NULL) cy = &kept_carry;
  /* The virtual operand is C-contiguous over the loop's shape, the
     loads'. */
  int index = 0;
  if (!oom) e = nx_cpu_prog_bind(g, loads, rank, shape, &index);
  nx_array x = {.rank = rank};
  if (!oom) *z = (fused){.prog = g, .nloads = nl, .index = index};
  if (!oom && !e) {
    int64_t past = 1;
    for (int d = rank - 1; d >= 0; d--) {
      x.dim[d] = shape[d];
      x.dim[rank + d] = past;
      past *= x.dim[d];
    }
    fuse(z, &x, loads);
  }
  for (int i = 0; !oom && !e && i < nr; i++) {
    int dt = nx_cpu_prog_held(g, ks[i]);
    const nx_array *y = &a[first[i]], *y1 = two(rs[i]) ? y + 1 : y;
    x.dtype = dt;
    x.bits = nx_dtype_row_of(dt).bits;
    z->k = ks[i];
    e = c->op == NETWORK
            ? network_into(rs[i], c->axes[0], y, y1, &x, z)
            : reduce_into(c->op, rs[i], c->axes, c->naxes, y, y1, &x, cy, z);
  }
  /* A program that could not have its arenas computed nothing. */
  if (!oom && !e && nx_cpu_prog_failed(g)) e = -1;
  /* A copy that failed left its base NULL: free skips it. */
  for (int i = 0; i < copied; i++)
    if (padded[i] || a[nd + i].alias) free(loads[i].base);
  free(kept_carry.base);
  nx_cpu_give(held);
  nx_cpu_reacquire(released);
  nx_done(n, a);
  drop(owner, many);
  if (oom || e < 0) caml_raise_out_of_memory();
  CAMLreturn(Val_int(e));
}

/* Computes [c] by the one pair [vr] of a reduction and an output of [vp]
   into [vdsts], where [vp] is its one load's identity: NX_ARITY unless
   the destinations are the reduction's. Sets [*done] to 0, and computes
   nothing, where [vp] is not. */
static value in_place(call *c, value vp, value vr, value vdsts,
                      value vloads, value vpads, value vcarry, int *done) {
  int dt = identity_of(vp);
  *done = dt >= 0 && Int_val(Field(vr, 1)) == 0 && Wosize_val(vloads) == 1;
  if (!*done) return Val_unit;
  c->r = code_of(Field(vr, 0));
  int nd = 1 + two(c->r);
  if (Wosize_val(vdsts) != (mlsize_t)nd) return Val_int(NX_ARITY);
  value pad = Wosize_val(vpads) > 0 ? Field(vpads, 0) : Val_none;
  return run(c, dt, Field(vdsts, 0), nd == 2 ? Field(vdsts, 1) : Val_unit,
             Field(Field(vloads, 0), 0), pad, vcarry);
}

/* Reduces the program [vp] over the loop of shape [vshape] along [vaxes],
   its [loads] and their paddings [pads], by the reductions [vrs], pairs
   of an Nx_kernel.reduction and an output, in turn into [vdsts]. A
   program that is its one load's identity reads the load in place. */
value nx_cpu_reduce(value vp, value vrs, value vshape, value vaxes,
                    value vdsts, value vloads, value vpads) {
  call c = {.op = REDUCE, .naxes = (int)Wosize_val(vaxes)};
  int rank = (int)Wosize_val(vshape);
  if (c.naxes > NX_MAX_RANK || rank > NX_MAX_RANK) return Val_int(NX_SHAPE);
  int64_t shape[NX_MAX_RANK];
  for (int d = 0; d < rank; d++)
    if ((shape[d] = Long_val(Field(vshape, d))) < 0) return Val_int(NX_SHAPE);
  for (int i = 0; i < c.naxes; i++)
    if (axis_of(Field(vaxes, i), &c.axes[i])) return Val_int(NX_SHAPE);
  int done = 0;
  if (Wosize_val(vrs) == 1) {
    value r = in_place(&c, vp, Field(vrs, 0), vdsts, vloads, vpads, Val_none,
                       &done);
    if (done) return r;
  }
  return run_fused(&c, vp, vrs, 0, rank, shape, vdsts, vloads, vpads,
                   Val_none);
}

value nx_cpu_reduce_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_reduce(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                       argv[6]);
}

/* Scans the program [vp] along [vaxis] by the pair [vr] of a reduction and
   an output, Logsumexp and Moments by the network, its [loads] and their
   paddings [pads], into [vdsts], over their first one's shape, from the
   carries [vcarry] where it is Some. */
value nx_cpu_scan(value vp, value vr, value vaxis, value vdsts, value vloads,
                  value vpads, value vcarry) {
  call c = {.op = Is_long(Field(vr, 0)) ? NETWORK : SCAN, .naxes = 1};
  if (axis_of(vaxis, &c.axes[0])) return Val_int(NX_SHAPE);
  int done = 0;
  value r = in_place(&c, vp, vr, vdsts, vloads, vpads, vcarry, &done);
  if (done) return r;
  return run_fused(&c, vp, vr, 1, -1, NULL, vdsts, vloads, vpads, vcarry);
}

value nx_cpu_scan_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_scan(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                     argv[6]);
}

#elif !defined(__x86_64__)

/* ISO C forbids an empty unit. */
typedef int nx_cpu_no_v3;

#endif