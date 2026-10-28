/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* reduce.c's kernels: each monoid's, state's and network's at each
   dtype, and the table that holds them. Included once, after cpu.h and
   nx_kinds.h. */

/* The scratch a few-term reduction's lanes take: 1 KiB of outputs in each
   lane. */
#define NX_CPU_FEW_BYTES (NX_CPU_LANES * 1024)

/* The monoids nx.cpu reduces by: NX_MONOID_SUM to NX_MONOID_MIN
   (nx_kernel.h). */
enum { NX_CPU_MONOIDS = NX_MONOID_MIN + 1 };

/* A monoid's kernels at a dtype. [lanes] adds the [n] terms at
   [x], stepping [s] elements, into the NX_CPU_LANES lanes at [l]: term i
   into lane (first + i) mod NX_CPU_LANES; [e] is the identity. [combine]
   adds the [n] elements at [x], stepping [s], into the [n] contiguous
   accumulators at [a], element by element. [scan] runs [k] sequences, at
   most 4, side by side: sequence j adds the [n] elements at [x + j·xs],
   stepping [s], into the accumulator [a]'s element j in order, storing each
   sum at [y + j·ys], stepping [sy]. [blocks] stores into [v] the values of
   the [n] blocks of NX_CPU_LANE_BLOCK contiguous terms from [x], each its
   lanes from the identity [e] and the lanes' tree, as [lanes] and [combine]
   give them. [few] stores the values of [h] rows of [w] outputs of [n] terms
   each, n at most NX_CPU_LANES, output j of row i at [y + i·sy2 + j·sy]: its
   term t at [x + i·s2 + j·s + off[t]], in lane t from the identity [e], then
   the lanes' tree, as one block of [lanes] gives them; [l] is
   NX_CPU_FEW_BYTES of scratch on 16 bytes for the lanes. [column] adds into
   each of the [w] contiguous accumulators at [a] in turn the [n] rows of
   [x], row i at [x + i·st] and its element j [s] further than j - 1:
   accumulator j takes row 0's element j, then row 1's, and so on. */
typedef struct {
  void (*lanes)(const uint8_t *x, int64_t s, int64_t n, const uint8_t *e,
                uint8_t *l, int first);
  void (*combine)(uint8_t *a, const uint8_t *x, int64_t s, int64_t n);
  void (*scan)(uint8_t *a, const uint8_t *x, int64_t s, int64_t xs,
               uint8_t *y, int64_t sy, int64_t ys, int64_t n, int k);
  void (*blocks)(const uint8_t *x, int64_t n, const uint8_t *e, uint8_t *v);
  void (*few)(const uint8_t *x, const int64_t *off, int n, int64_t s,
              int64_t w, int64_t s2, int64_t h, const uint8_t *e, uint8_t *y,
              int64_t sy, int64_t sy2, uint8_t *l);
  void (*column)(uint8_t *a, const uint8_t *x, int64_t st, int64_t n,
                 int64_t s, int64_t w);
} nx_cpu_monoid;

/* A sum of few products at a float dtype stores the [h] rows of
   [w] sums of [n] products each, n from 1 to NX_CPU_LANES, row i's output
   j at [y + i·sy2 + j·sy]: its product t
   is the element [i·sa + oa[t] + j] of [a] times the element
   [i·sb + ob[t] + j] of [b], each a term in lane t from +0, then the
   lanes' tree, as one block of a monoid's [lanes] gives them. */
typedef void (*nx_cpu_products)(int n, int64_t w, int64_t h,
                                const uint8_t *a, const int64_t *oa,
                                int64_t sa, const uint8_t *b,
                                const int64_t *ob, int64_t sb, uint8_t *y,
                                int64_t sy, int64_t sy2);

/* The reductions nx.cpu computes by a state of each block of terms,
   numbered on from the monoids: Logsumexp, Arg by Max and by Min, and
   Moments. */
enum {
  NX_CPU_LOGSUMEXP = NX_CPU_MONOIDS,
  NX_CPU_ARG_MAX,
  NX_CPU_ARG_MIN,
  NX_CPU_MOMENTS,
  NX_CPU_REDUCTIONS
};

#define NX_CPU_STATEFUL (NX_CPU_REDUCTIONS - NX_CPU_MONOIDS)

/* The state of a block of an output's terms, or of several merged: Arg's
   extreme's bits [v], a complex128's at most, and its position [at], or a
   value [s] of the terms and their count [n]: Logsumexp's maximum, or its
   sum, and Moments' sums, [d] the deviations' with [s] their squares'. An
   Arg scan's state numbers its next term [n] and holds its extreme and
   position, none where [at] is -1. A float is its dtype's value, which a
   double holds exactly. */
typedef struct {
  int64_t n, at;
  union {
    uint8_t v[16];
    struct {
      double s, d;
    };
  };
} nx_cpu_state;

/* A pass of a stateful reduction at a dtype. [block] stores
   into [st] the state of the [n] contiguous terms at [x], 1 to
   NX_CPU_LANE_BLOCK of them, the first numbered [t0], of an output whose
   center is [c]. [merge] makes [a] the state of [a]'s terms then [b]'s.
   [finish] stores the results of an output's state [st], center [c], at
   [d0] and, but for Logsumexp, [d1]. */
typedef struct {
  void (*block)(const uint8_t *x, int64_t n, int64_t t0, double c,
                nx_cpu_state *st);
  void (*merge)(nx_cpu_state *a, const nx_cpu_state *b);
  void (*finish)(const nx_cpu_state *st, double c, uint8_t *d0, uint8_t *d1);
} nx_cpu_pass;

/* An Arg scan at a dtype. [rows] runs [w] lanes, at most
   16, of [m] terms each: lane l's state a[l] takes the terms at
   x + (i xs + l xl)·size in order of i, the first numbered a[l].n, and
   stores its extreme after each at y + (i ys + l yl) elements and its
   position at p + i ps + l pl, its chunks' totals being its reduction's.
   [merge] makes [a] the state of [a]'s terms then [b]'s, [b]'s n. A state
   of no term has n its first term's number and at -1. */
typedef struct {
  void (*rows)(nx_cpu_state *a, const uint8_t *x, int64_t xs, int64_t xl,
               int64_t m, int64_t w, uint8_t *y, int64_t ys, int64_t yl,
               int64_t *p, int64_t ps, int64_t pl);
  void (*merge)(nx_cpu_state *a, const nx_cpu_state *b);
} nx_cpu_scan_pass;

/* A stateful reduction at a dtype: where [center.block] is not NULL, a
   first pass whose [finish] stores each output's center at [d0], a double,
   then the [value] pass, which takes it. Logsumexp's center is the terms'
   maximum where finite, else 0; Moments', their mean. [scan] scans by an
   Arg, rows NULL where the table declines. */
typedef struct {
  nx_cpu_pass center, value;
  nx_cpu_scan_pass scan;
} nx_cpu_stateful;

/* Terms a kernel asks for where they are computed, not read: [fill] stores
   terms [t0, t0 + tn) of [w] slices into [out], term t's of slice l at
   element (t - t0)·w + l, [w] the kernel's. */
typedef struct {
  void (*fill)(const void *ctx, int64_t t0, int64_t tn, uint8_t *out);
  const void *ctx;
} nx_cpu_terms;

/* A scan by nx_kernel.mli's network, Logsumexp's or Moments', of [w]
   slices of [n] terms side by side, w at most
   NX_CPU_NETWORK_LANES: slice l's term t at x + t xs + l xl, or from [src]
   where it is not NULL, its result at y + t ys + l yl and Moments'
   variance at y1 + t s1 + l l1, in elements; [scratch] holds
   (12 nx_cpu_network_levels(n) + 3 NX_CPU_NETWORK_TILE) w elements, and
   for one slice at least the elements its tiles take. With a
   [span], it runs one step of the scan by tiles; without, the whole
   scan. */
#define NX_CPU_NETWORK_LANES 64

/* A step of a network scan by tiles of PT terms from its slices' start,
   [states] holding a state of the [w] slices for each
   full tile: TOTALS stores tiles [q0, q1)'s states there, CARRIES takes
   the first [q1] states to the network's results over them, in their
   place, RESULTS computes tiles [q0, q1)'s results from those before
   each tile and at its last term, setting [bad] where Moments' fast pass
   met a value not finite, and EXACT, after a RESULTS that set it, runs
   Moments' exact pass over the whole slices. */
typedef enum {
  NX_CPU_TOTALS,
  NX_CPU_CARRIES,
  NX_CPU_RESULTS,
  NX_CPU_EXACT
} nx_cpu_step;

typedef struct {
  nx_cpu_step step;
  int64_t q0, q1;
  uint8_t *states;
  int bad;
} nx_cpu_span;

/* A network scan's terms per tile. On fudge,
   scan-moments-f32-4096x1024-axis1 took 1.18 ms at 16, 1.04 at 32, 1.12 at
   64. */
#define NX_CPU_NETWORK_TILE 32

/* The levels of a network scan over [n] terms, n >= 1: one per bit of
   n. */
static inline int nx_cpu_network_levels(int64_t n) {
  int k = 1;
  for (; n > 1; n >>= 1) k++;
  return k;
}

typedef void (*nx_cpu_network)(const uint8_t *x, int64_t xs, int64_t xl,
                               uint8_t *y, int64_t ys, int64_t yl,
                               uint8_t *y1, int64_t s1, int64_t l1,
                               int64_t n, int64_t w, uint8_t *scratch,
                               const nx_cpu_terms *src, nx_cpu_span *span);

/* A monoid at a dtype is its identity, a lane step that adds terms
   into NX_CPU_LANES lanes by their index, and a combine that adds a row of
   terms into a row of accumulators, element by element. Integers compute in
   their own width, unsigned, so they wrap as a wider sum stored back would;
   booleans reduce as any and all of their bytes' being non-zero. A float sum
   or product uses the instruction. Maxima and minima are nx_kinds.h's,
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
  /* The lanes' tree. */                                                     \
  static inline T fold_##NAME(D##_l v) {                                     \
    T l[16];                                                                 \
    store_##D(l, v);                                                         \
    NX_CPU_LANE_TREE(i, j, l[i] = F(l[i], l[j]));                            \
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
   arguments, its lanes an array of their own. 64-bit integers' products
   take it: no target here has a vector multiply of 64-bit integers, which
   the compilers emulate at a third of the speed of scalar ones. */
#define SMONOID(NAME, D, T, W, F, OP)                                        \
  ARRAYS(NAME, T)                                                            \
  AMONOID(NAME, NAME, T, F)

#define ADD(a, b) ((a) + (b))
#define MUL(a, b) ((a) * (b))
#define ANY(a, b) ((uint8_t)((a) | ((b) != 0)))
#define ALL(a, b) ((uint8_t)((a) & ((b) != 0)))

/* The four monoids of an integer dtype of storage [T], of [W] bytes,
   unsigned [U], at the compute suffix [S] for its extremes, [C] its compute
   type and [M] the unsigned type of [C]'s width, [PROD] the form of its
   product. Sums and products run in [M], where they wrap: 8- and 16-bit
   operands would promote to int, whose product of two 0xFFFF overflows. */
#define INTS(D, T, U, C, M, S, W, PROD)                                      \
  static inline T add_##D(T a, T b) { return (T)(U)((M)(U)a + (M)(U)b); }    \
  static inline T mul_##D(T a, T b) { return (T)(U)((M)(U)a * (M)(U)b); }    \
  static inline T max_##D(T a, T b) { return (T)nx_maximum_##S((C)a, (C)b); } \
  static inline T min_##D(T a, T b) { return (T)nx_minimum_##S((C)a, (C)b); } \
  WHOLE_VECTORS(D, T, U, W)                                                  \
  XMONOID(sum_##D, D, T, W, add_##D, V_WRAP_ADD)                             \
  PROD(prod_##D, D, T, W, mul_##D, V_WRAP_MUL)                               \
  XMONOID(max_##D, D, T, W, max_##D, V_MAX)                                  \
  XMONOID(min_##D, D, T, W, min_##D, V_MIN)

INTS(i8, int8_t, uint8_t, int32_t, uint32_t, i32, 1, XMONOID)
INTS(u8, uint8_t, uint8_t, uint32_t, uint32_t, u32, 1, XMONOID)
INTS(i16, int16_t, uint16_t, int32_t, uint32_t, i32, 2, XMONOID)
INTS(u16, uint16_t, uint16_t, uint32_t, uint32_t, u32, 2, XMONOID)
INTS(i32, int32_t, uint32_t, int32_t, uint32_t, i32, 4, XMONOID)
INTS(u32, uint32_t, uint32_t, uint32_t, uint32_t, u32, 4, XMONOID)
INTS(i64, int64_t, uint64_t, int64_t, uint64_t, i64, 8, SMONOID)
INTS(u64, uint64_t, uint64_t, uint64_t, uint64_t, u64, 8, SMONOID)

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

/* The fewest slices side by side a network scan's unit runs where units
   are few: on fudge, scan/host-logsumexp-f32-1024x1024-cols took 0.36 ms
   in 16 units of 64, 0.31 in 64 of 16, 0.41 to 0.47 in 128 of 8. */
#define NETWORK_LEAST 16

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
  (4 * PT + 3 * (2 * PT + NETWORK_P + 1) + 9 * TILE_LEVELS)

/* Where a tile's level k starts among its states, packed level by level,
   level k's PT >> k of them, and among its results, one more a level. */
static int64_t states_at(int k) { return 2 * PT - (2 * PT >> k); }
static int64_t results_at(int k) { return states_at(k) + k; }

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
     that is negative, or Σ (x - c)² where that is +inf, as
     Nx_kernel.reduction's Moments defines it.
   Every function is inlined into each kernel, whose loops of 16 lanes the
   compiler then vectorises. */

#define KERNEL static __attribute__((flatten)) void

/* Arg by the extreme [F] over terms of [T] of [W] bytes, its lanes at the
   suffix [L], [U] the unsigned integer of its width: [BEAT] is [>] or [<],
   [F] without its order of NaNs and zeros, which the plain compare
   vectorises; [NANQ] tests a term, or a vector of them, for a NaN; [FIND]
   the pragma, if any, of the loop over 16 terms' bits. */
#define ARG(NAME, L, T, U, W, F, BEAT, NANQ, FIND)                           \
  /* beats_ steps the lanes: a lane takes the term where it beats the        \
     lane by BEAT or is a NaN. */                                            \
  static inline L##_v beat_##NAME(L##_v l, L##_v v) {                        \
    return SEL(L##_u, L##_v, (v BEAT l) | NANQ(v), v, l);                    \
  }                                                                          \
  ON(beats_##NAME, L, T, FIELDS(W), beat_##NAME)                             \
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
  ARG(max_##D, D, T, U, W, F_MAX, >, NANQ, FIND)                             \
  ARG(min_##D, D, T, U, W, F_MIN, <, NANQ, FIND)

ARGS(i8, int8_t, uint8_t, 1, max_i8, min_i8, NO_NAN, UNROLL(8))
ARGS(u8, uint8_t, uint8_t, 1, max_u8, min_u8, NO_NAN, UNROLL(8))
ARGS(i16, int16_t, uint16_t, 2, max_i16, min_i16, NO_NAN, UNROLL(8))
ARGS(u16, uint16_t, uint16_t, 2, max_u16, min_u16, NO_NAN, UNROLL(8))
ARGS(i32, int32_t, uint32_t, 4, max_i32, min_i32, NO_NAN, UNROLL(8))
ARGS(u32, uint32_t, uint32_t, 4, max_u32, min_u32, NO_NAN, UNROLL(8))
ARGS(i64, int64_t, uint64_t, 8, max_i64, min_i64, NO_NAN, )
ARGS(u64, uint64_t, uint64_t, 8, max_u64, min_u64, NO_NAN, )
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

/* The float dtypes' sums of few products, Logsumexp and Moments
   (reduce_float.h), once per type. */
#define T float
#define S(name) name##_f32
#define MADE nx_bits_float(0x7FC00000u)
#define CUT -87
#include "reduce_float.h"
#undef T
#undef S
#undef MADE
#undef CUT

#define T double
#define S(name) name##_f64
#define MADE nx_bits_double(0x7FF8000000000000u)
#define CUT -708
#include "reduce_float.h"
#undef T
#undef S
#undef MADE
#undef CUT

/* The table's monoids, by monoid then dtype: NULL where declined. */
#define SET(M, DT, F)                                                        \
  .monoid[M][DT] = {lanes_##F, combine_##F, scan_##F, blocks_##F, few_##F,   \
                    column_##F},

#define ALL4(DT, D)                                                          \
  SET(NX_MONOID_SUM, DT, sum_##D)                                            \
  SET(NX_MONOID_PROD, DT, prod_##D)                                          \
  SET(NX_MONOID_MAX, DT, max_##D)                                            \
  SET(NX_MONOID_MIN, DT, min_##D)

/* The table's stateful reductions, by reduction then dtype: a pass is the
   kernels K, M and F at the suffix D. */
#define PASS(K, M, F, D) {K##_##D, M##_##D, F##_##D}
#define NONE {NULL, NULL, NULL}

/* A scan's kernels R and M at the suffix D. */
#define SCAN(R, M, D) {R##_##D, M##_##D}
#define NO_SCAN {NULL, NULL}

#define STATE(R, DT, CENTER, VALUE, SCANS)                                   \
  .stateful[R - NX_CPU_MONOIDS][DT] = {CENTER, VALUE, SCANS},

#define ARG2(DT, D)                                                          \
  STATE(NX_CPU_ARG_MAX, DT, NONE, PASS(arg, arg_merge, arg_finish, max_##D), \
        SCAN(argscan, argscan_merge, max_##D))                               \
  STATE(NX_CPU_ARG_MIN, DT, NONE, PASS(arg, arg_merge, arg_finish, min_##D), \
        SCAN(argscan, argscan_merge, min_##D))

#define FLOATS(DT, D)                                                        \
  STATE(NX_CPU_LOGSUMEXP, DT, PASS(top, top_merge, top_finish, D),           \
        PASS(lse, lse_merge, lse_finish, D), NO_SCAN)                        \
  STATE(NX_CPU_MOMENTS, DT, PASS(mean, sum_merge, mean_finish, D),           \
        PASS(var, var_merge, var_finish, D), NO_SCAN)

/* The kernels by reduction and dtype. monoid[m][dt] reduces by the monoid
   m (NX_MONOID_SUM to NX_MONOID_MIN) at dt: lanes NULL where the table
   declines. products[dt] sums few products at float32 and float64: NULL
   where the table declines. logsumexp[dt] and moments[dt] scan by
   Logsumexp and Moments at float32 and float64.
   stateful[r - NX_CPU_MONOIDS][dt] reduces by r (NX_CPU_LOGSUMEXP to
   NX_CPU_MOMENTS) at dt: block NULL where the table declines. */
static const struct {
  nx_cpu_monoid monoid[NX_CPU_MONOIDS][NX_DTYPE_COUNT];
  nx_cpu_products products[NX_DTYPE_COUNT];
  nx_cpu_network logsumexp[NX_DTYPE_COUNT], moments[NX_DTYPE_COUNT];
  nx_cpu_stateful stateful[NX_CPU_STATEFUL][NX_DTYPE_COUNT];
} table = {
    ARG2(NX_FLOAT32, f32)
    ARG2(NX_FLOAT64, f64)
    ARG2(NX_INT8, i8)
    ARG2(NX_UINT8, u8)
    ARG2(NX_INT16, i16)
    ARG2(NX_UINT16, u16)
    ARG2(NX_INT32, i32)
    ARG2(NX_UINT32, u32)
    ARG2(NX_INT64, i64)
    ARG2(NX_UINT64, u64)
    ARG2(NX_BOOL, b)
    ARG2(NX_COMPLEX64, c64)
    ARG2(NX_COMPLEX128, c128)
    FLOATS(NX_FLOAT32, f32)
    FLOATS(NX_FLOAT64, f64)
    ALL4(NX_FLOAT32, f32)
    ALL4(NX_FLOAT64, f64)
    ALL4(NX_INT8, i8)
    ALL4(NX_UINT8, u8)
    ALL4(NX_INT16, i16)
    ALL4(NX_UINT16, u16)
    ALL4(NX_INT32, i32)
    ALL4(NX_UINT32, u32)
    ALL4(NX_INT64, i64)
    ALL4(NX_UINT64, u64)
    SET(NX_MONOID_MAX, NX_BOOL, max_b)
    SET(NX_MONOID_MIN, NX_BOOL, min_b)
    ALL4(NX_COMPLEX64, c64)
    ALL4(NX_COMPLEX128, c128)
    .logsumexp = {[NX_FLOAT32] = lse_network_f32,
                  [NX_FLOAT64] = lse_network_f64},
    .moments = {[NX_FLOAT32] = moments_network_f32,
                [NX_FLOAT64] = moments_network_f64},
    .products = {[NX_FLOAT32] = products_f32, [NX_FLOAT64] = products_f64},
};
