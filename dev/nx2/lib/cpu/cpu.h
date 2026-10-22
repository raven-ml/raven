/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cpu's private interface: targets, jobs, walks and the stage.

   A kernel reads its operands through the door (nx_array.h) and coalesces
   them into a loop. The walk cuts the loop into 2-D blocks and hands them
   to a job, which runs them on the pool's threads. Per block the kernel
   reads each operand in place where the target's runs can, and otherwise
   stages it into a buffer in L1; it writes its result through the unstage,
   which converts as a cast does.

     kernel ─► walk ─► job ─► rig.pool threads ─► block
                                                   │
                     operand ─► stage ─► buffer ─► unstage ─► result
                                     (target's runs)
*/

#ifndef NX_CPU_H
#define NX_CPU_H

#include <stdint.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__x86_64__)
#include <emmintrin.h>
#endif

#include <caml/signals.h>

#include "nx_array.h"
#include "nx_contract_view.h"
#include "nx_contract_view_fill.h"
#include "nx_kernel.h"
#include "nx_prog.h"
#include "rig_pool.h"

/* Targets

   A target is the instruction set the kernels' runs are compiled for: one
   table per target, chosen once at initialisation. Every table computes the
   same bits. */

/* The target's vector, in bytes: 32 where the target has AVX (x86-64's
   v3 table, built for AVX2), else 16. */
#if defined(__AVX__)
#define NX_VECTOR_BYTES 32
#else
#define NX_VECTOR_BYTES 16
#endif

/* A run converts [n] contiguous elements at [src] into [dst]. */
typedef void (*nx_cpu_run)(const void *src, void *dst, int64_t n);

/* The lanes of a sum whose order is fixed by shape: term t of a block of
   NX_CPU_LANE_BLOCK consecutive terms adds into lane t modulo
   NX_CPU_LANES. */
#define NX_CPU_LANES 16
#define NX_CPU_LANE_BLOCK 1024

/* The scratch a few-term reduction's lanes take: 1 KiB of outputs in each
   lane. */
#define NX_CPU_FEW_BYTES (NX_CPU_LANES * 1024)

/* Where a microkernel's sums start: the tile's outputs, or +0, the tile
   then only written. */
typedef enum { NX_CPU_FROM_TILE, NX_CPU_FROM_ZERO } nx_cpu_from;

/* A contraction's microkernel adds to the [m] × [n] outputs at [c], n at
   most NR, output (i, j) at c + (i·ldc + j)·w, the products of [k] steps:
   step p's element of a's row i at a + (i·a_row + p·a_k)·w, of b's column
   j at b + (p·b_k + j)·w. It runs MR rows at a time, so that b's steps
   stay in the caches across them, and reads b as nx_cpu_b_read says. Each
   output adds its products in increasing p, each fused into its addition,
   to the start [from] names. */
typedef void (*nx_cpu_kernel)(int64_t m, int64_t n, int64_t k, const void *a,
                              int64_t a_row, int64_t a_k, const void *b,
                              int64_t b_k, void *c, int64_t ldc,
                              nx_cpu_from from);

/* How a microkernel reads a: an element at a time, so that a is read in
   place however its rows lie; or MR elements of a step at once, from a
   packed block, a_row 1, whose rows past m up to a multiple of MR are
   padding, any value (nx_cpu_pack). */
typedef enum { NX_CPU_A_ELEMENTS, NX_CPU_A_PACKED } nx_cpu_a_read;

/* How a microkernel reads b. NX_CPU_B_SLIVER: NR columns a step, so that
   a sliver of fewer columns is packed. NX_CPU_B_COLUMNS: the n columns it
   is given, so that such a sliver is read in place too. NX_CPU_B_HALVES:
   the n columns it is given where n is at least NR / 2, so that only a
   sliver of fewer is packed. */
typedef enum {
  NX_CPU_B_SLIVER,
  NX_CPU_B_COLUMNS,
  NX_CPU_B_HALVES
} nx_cpu_b_read;

/* A dot adds, for each of [r] contiguous rows of [a], row i at
   a + i·lda·w, r at most NX_CPU_DOT_ROWS, the [n] products with the
   contiguous [b] into row i's lanes at lanes + i·NX_CPU_LANES·w: term t
   into lane t modulo NX_CPU_LANES, each fused. */
#define NX_CPU_DOT_ROWS 4
typedef void (*nx_cpu_dot)(const void *a, int64_t lda, int r, const void *b,
                           int64_t n, void *lanes);

/* An axpy adds, for each of [r] rows, the element of [a] at a + i·lda·w
   times each of the [n] contiguous elements of [b] into row i's [n]
   contiguous accumulators at y + i·ldy·w, each fused. It prefetches the
   [n] elements at [next], the b of a later call. */
typedef void (*nx_cpu_axpy)(const void *a, int64_t lda, int r, const void *b,
                            int64_t n, void *y, int64_t ldy,
                            const void *next);

/* A lane tile sets y[j], for each of the [c] columns of b at [b], c at
   most NX_CPU_LANES, row t at b + t·b_k·w, to lane order's sum of one
   block of [k] products, k at most NX_CPU_LANE_BLOCK, of a's row, its
   element t at a + t·a_k·w, and the column: term t into lane t modulo
   NX_CPU_LANES, fused, the lanes summed by their tree (gemm.c). It reads
   no column past c. */
typedef void (*nx_cpu_lane_tile)(const void *a, int64_t a_k, const void *b,
                                 int64_t b_k, int64_t k, int64_t c,
                                 void *y);

/* A gemv adds into each of the [m] contiguous accumulators y[i] the [k]
   products of a's element at a + (i + p·lda)·w and x's [p]th, in
   increasing p, each fused: y += A x for the m × k A whose columns lie
   lda apart. */
typedef void (*nx_cpu_gemv)(int64_t m, int64_t k, const void *a, int64_t lda,
                            const void *x, void *y);

/* A quads dot sets y[i·ldy + c], for each of 2 rows i of a, at a + i·lda,
   and 4 columns c of b, at b + c·ldb, to the sum of the products of the
   [n] words of the row and of the column: each word four elements of
   quads[d]'s 8-bit dtype along k, adjacent (gemm.c, Quads). */
typedef void (*nx_cpu_quads_dot)(const uint32_t *a, int64_t lda,
                                 const uint32_t *b, int64_t ldb, int64_t n,
                                 uint32_t *y, int64_t ldy);

/* A quads axpy adds into y[i·ldy + j], for each of [rows] rows i of a,
   at a + i·lda, and [n] columns j, the sum over the [k] elements t of the
   row, adjacent, of a[t] · b[t·ldb + j], b's rows adjacent along their
   columns: elements of quads[d]'s 8-bit dtype, the sum in 32 bits (gemm.c,
   Quads). A row of a holds zeros past k to a multiple of 4, which the axpy
   may read. It may prefetch the [k] rows of [n] elements at [next], [ldb]
   apart, the b of a later call. */
typedef void (*nx_cpu_quads_axpy)(const uint8_t *a, int64_t lda,
                                  int64_t rows, const uint8_t *b, int64_t ldb,
                                  int64_t k, int64_t n, uint32_t *y,
                                  int64_t ldy, const uint8_t *next);

/* A pack moves [n1] rows of [n0] elements of acc from [s], stepping [s0]
   along a row and [s1] across rows, one of them 1, into [d]: element i of
   row j to d + (j·pitch + i)·w. [pitch] is the next multiple of the
   kernel's MR, for a's blocks, or of its NR, for b's slivers. A pack may
   leave any value past n0: a kernel's outputs from those elements lie past
   m or n and are never stored, and a kernel on hardware that slows on
   subnormal inputs masks its loads past n. It packs without the stage, whose
   moves serve every dtype and layout. */
typedef void (*nx_cpu_pack)(int64_t n0, int64_t n1, const void *s,
                            int64_t s0, int64_t s1, void *d, int64_t pitch);

/* A microkernel, its tile of MR × NR outputs, how it reads a and b, its
   speed: the flops it runs on one thread in the time memcpy moves a byte,
   which prices its jobs; and the cores that share the hardware it runs on:
   1 for a core's vector unit, a cluster's for the M1's matrix unit. */
typedef struct {
  nx_cpu_kernel f;
  int mr, nr;
  nx_cpu_a_read a;
  nx_cpu_b_read b;
  double flops_per_byte;
  int shared;
} nx_cpu_micro;

/* A contraction's kernels for chain order in one accumulator dtype. */
typedef struct nx_cpu_gemm {
  nx_cpu_micro kernel;
  int64_t mc, kc, nc;    /* the driver's blocks of rows, of k and of columns */
  int64_t few;           /* the rows at most which a product may stream b
                            past all of a (gemm.c's streams) */
  nx_cpu_pack pack;      /* or NULL: operands pack through the stage */
  const struct nx_cpu_gemm *other; /* or NULL: kernels that compute a
                                      product where they take less time */
  void (*enter)(void), (*leave)(void); /* or NULL: on a thread, its calls
                                          of kernel and pack run between
                                          enter and leave */
  double start; /* the cost, in bytes of memcpy, that each thread of a job
                   on these kernels adds to its start (nx_cpu_job_at); 0
                   where it adds none */
  double call;  /* the cost, in bytes of memcpy, that a product on these
                   kernels on one thread adds to its flops' (gemm.c's
                   fastest); 0 where it adds none */
} nx_cpu_gemm;

/* A row of a kind of one operand: n elements of [d] from [x], each stepping
   its own count of elements; of two, from [x] and [y]; of three, from [c],
   [x] and [y]. Each operand's elements and [d]'s are disjoint or identical:
   the same address and step. */
typedef void (*nx_cpu_row1)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *x, int64_t sx);
typedef void (*nx_cpu_row2)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *x, int64_t sx, const uint8_t *y,
                            int64_t sy);
typedef void (*nx_cpu_row3)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *c, int64_t sc, const uint8_t *x,
                            int64_t sx, const uint8_t *y, int64_t sy);
/* A fill's row: n elements of [d] set to the element whose bits are at
   [bits]. */
typedef void (*nx_cpu_row0)(int64_t n, uint8_t *d, int64_t sd,
                            const uint8_t *bits);

/* The monoids nx.cpu reduces by: NX_MONOID_SUM to NX_MONOID_MIN
   (nx_kernel.h). */
enum { NX_CPU_MONOIDS = NX_MONOID_MIN + 1 };

/* A monoid's kernels at a dtype (reduce.c). [lanes] adds the [n] terms at
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

/* The reductions nx.cpu computes by a state of each block of terms
   (reduce.c), numbered on from the monoids: Logsumexp, Arg by Max and by
   Min, and Moments. */
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

/* A pass of a stateful reduction at a dtype (reduce.c). [block] stores
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

/* An Arg scan at a dtype (reduce.c). [rows] runs [w] lanes, at most
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
   element (t - t0)·w + l, [w] the kernel's (reduce.c). */
typedef struct {
  void (*fill)(const void *ctx, int64_t t0, int64_t tn, uint8_t *out);
  const void *ctx;
} nx_cpu_terms;

/* A scan by nx_kernel.mli's network, Logsumexp's or Moments', of [w]
   slices of [n] terms side by side (reduce.c), w at most
   NX_CPU_NETWORK_LANES: slice l's term t at x + t xs + l xl, or from [src]
   where it is not NULL, its result at y + t ys + l yl and Moments'
   variance at y1 + t s1 + l l1, in elements; [scratch] holds
   (12 nx_cpu_network_levels(n) + 3 NX_CPU_NETWORK_TILE) w elements, and
   for one slice at least the elements its tiles take (reduce.c). */
#define NX_CPU_NETWORK_LANES 64

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
                               const nx_cpu_terms *src);

/* The bytes of the largest tile of any target's microkernel: 32 × 32
   float32 on the matrix unit. */
#define NX_CPU_TILE 4096

typedef struct {
  const char *name;
  /* The bytes of L2 cache of a core, as the host reports them when the
     program starts (target.c). */
  int64_t l2;
  /* The bytes of L1 data cache of the cores a job runs on (target.c). */
  int64_t l1;
  /* The bytes of the largest cache the host reports (target.c). */
  int64_t llc;
  /* Whether a kernel that reads operand rows several times at a step other
     than one copies them into unit steps first: on x86-64, where strided
     loads cost more than the copy. In cache, adding rows 9 elements apart
     took 0.26 ns an element on an Arrow Lake core and 0.15 on an M1 Max
     one, whose strided adds outrun the copy. */
  int copy_strided;
  /* convert[s][d] converts elements of the dtype s into the dtype d, as a
     cast does, where s is a carrier, or where s is a narrow float and d
     float32. A sub-byte dtype has one element per byte, in its low bits:
     int4 and uint4 their value modulo 16, float4 its code, bit 0 or 1. */
  nx_cpu_run convert[NX_DTYPE_COUNT][NX_DTYPE_COUNT];
  /* gemm[acc] contracts in acc in chain order; its kernel is NULL where
     the target has none. dot[acc], axpy[acc] and tile[acc] are lane
     order's; gemv[dt] serves the factorisations' leaves (linalg.c).
     quads[d] contracts operands of the 8-bit dtype d into 32 bits in
     chain order, each element of a and b a 32-bit word of four of d along
     k, each product of words the sum of their four products (gemm.c,
     Quads); its kernel is NULL where the target has none. quads_dot[d]
     and quads_axpy[d] are its dot and axpy for products of few rows, or
     NULL where the target has none; from quads_zip_rows rows (INT_MAX
     where it never does), quads[d]'s kernel over b's rows zipped into
     words outruns the axpy. */
  nx_cpu_gemm gemm[NX_DTYPE_COUNT];
  nx_cpu_gemm quads[NX_DTYPE_COUNT];
  nx_cpu_quads_dot quads_dot[NX_DTYPE_COUNT];
  nx_cpu_quads_axpy quads_axpy[NX_DTYPE_COUNT];
  int quads_zip_rows;
  nx_cpu_dot dot[NX_DTYPE_COUNT];
  nx_cpu_axpy axpy[NX_DTYPE_COUNT];
  nx_cpu_lane_tile tile[NX_DTYPE_COUNT];
  nx_cpu_gemv gemv[NX_DTYPE_COUNT];
  /* op1[k][dt] and op2[k][dt] compute the kind of one or two operands k
     (nx_prog.h's code) at dt, fma[dt] fma: NULL where the table declines
     (rows.c). */
  nx_cpu_row1 op1[NX_OP1_COUNT][NX_DTYPE_COUNT];
  nx_cpu_row2 op2[NX_OP2_COUNT][NX_DTYPE_COUNT];
  nx_cpu_row3 fma[NX_DTYPE_COUNT];
  /* where[i] and fill[i] move elements of 2^i bytes, i in 0..4. */
  nx_cpu_row3 where[5];
  nx_cpu_row0 fill[5];
  /* monoid[m][dt] reduces by the monoid m (NX_MONOID_SUM to
     NX_MONOID_MIN) at dt: lanes NULL where the table declines. */
  nx_cpu_monoid monoid[NX_CPU_MONOIDS][NX_DTYPE_COUNT];
  /* logsumexp[dt] and moments[dt] scan by Logsumexp and Moments at
     float32 and float64. */
  nx_cpu_network logsumexp[NX_DTYPE_COUNT], moments[NX_DTYPE_COUNT];
  /* stateful[r - NX_CPU_MONOIDS][dt] reduces by r (NX_CPU_LOGSUMEXP to
     NX_CPU_MOMENTS) at dt: block NULL where the table declines. */
  nx_cpu_stateful stateful[NX_CPU_STATEFUL][NX_DTYPE_COUNT];
  /* fft runs fft.c's transform t (its C2C, R2C or C2R; inverse for C2C's
     Inverse) of x into y along [axes]: 0 if memory runs out. */
  int (*fft)(int t, int inverse, const nx_array *x, const nx_array *y,
             const int *axes, int na);
  /* linalg runs units of linalg.c's call [ctx], a job's body. */
  rig_pool_body linalg;
} nx_cpu_target;

/* The tables, each set when the program starts on a host that runs it, by
   convert.c, gemm_generic.c, rows.c, reduce.c, fft.c and linalg.c
   compiled for its target, then by the kernels of its instructions:
   gemm_neon.c's on arm64, gemm_avx2.c's for v3, and gemm_vnni.c's quads
   where the host has AVX-VNNI-INT8. amx is base's table with gemm_amx.c's kernels for the
   matrix unit of Apple's M1. */
extern nx_cpu_target nx_cpu_base;
void nx_cpu_set_convert_base(nx_cpu_target *t);
void nx_cpu_set_gemm_base(nx_cpu_target *t);
void nx_cpu_set_rows_base(nx_cpu_target *t);
void nx_cpu_set_reduce_base(nx_cpu_target *t);
void nx_cpu_set_fft_base(nx_cpu_target *t);
void nx_cpu_set_linalg_base(nx_cpu_target *t);
#if defined(__aarch64__)
void nx_cpu_set_neon(nx_cpu_target *t);
#endif
#if defined(__APPLE__) && defined(__aarch64__)
extern nx_cpu_target nx_cpu_amx;
int nx_cpu_has_amx(void);
void nx_cpu_set_amx(nx_cpu_target *t);
#endif
#if defined(__x86_64__)
extern nx_cpu_target nx_cpu_v3;
void nx_cpu_set_convert_v3(nx_cpu_target *t);
void nx_cpu_set_gemm_v3(nx_cpu_target *t);
void nx_cpu_set_rows_v3(nx_cpu_target *t);
void nx_cpu_set_reduce_v3(nx_cpu_target *t);
void nx_cpu_set_fft_v3(nx_cpu_target *t);
void nx_cpu_set_linalg_v3(nx_cpu_target *t);
void nx_cpu_set_avx2(nx_cpu_target *t);
void nx_cpu_set_vnni(nx_cpu_target *t);
#endif

/* The table the kernels run: the best one whose instructions the host
   has. */
extern const nx_cpu_target *nx_cpu_table;

/* The dtype a block of [dt] is staged in: float32 for a narrow float, int8
   for int4, uint8 for uint4, bool for bit, and [dt] itself otherwise. */
static inline int nx_cpu_carrier(int dt) {
  switch (dt) {
    case NX_FLOAT16:
    case NX_BFLOAT16:
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2:
    case NX_FLOAT4_E2M1FN: return NX_FLOAT32;
    case NX_INT4: return NX_INT8;
    case NX_UINT4: return NX_UINT8;
    case NX_BIT: return NX_BOOL;
    default: return dt;
  }
}

/* The bytes a staged element of [dt] takes: a sub-byte one takes a byte. */
static inline int nx_cpu_width(int dt) {
  int bits = nx_dtype_row_of(dt).bits;
  return bits < 8 ? 1 : bits / 8;
}

/* The elements of [a]. */
static inline double nx_cpu_elements(const nx_array *a) {
  double n = 1;
  for (int i = 0; i < a->rank; i++) n *= (double)a->dim[i];
  return n;
}

/* The bytes of [a]'s elements, each nx_cpu_width wide. */
static inline double nx_cpu_bytes(const nx_array *a) {
  return nx_cpu_elements(a) * nx_cpu_width(a->dtype);
}

/* Calls

   A call keeps other domains waiting to collect for at most about one
   core's copy of NX_CPU_HOLD_BYTES, and never waits on another domain's
   job while it holds the domain lock. Between nx_read and nx_done an
   external calls nx_cpu_release with the price of its whole call, every
   pass, fill and copy it runs inline included, then runs its work, then
   nx_cpu_reacquire: the domain lock is released for the work where the
   price reaches NX_CPU_RELEASE_BYTES, and a held call runs its jobs on
   the calling thread alone (job.c). The work reads no OCaml value and
   raises nothing, held or released, so every call that passes
   nx_cpu_release reaches nx_cpu_reacquire. A call reads many operands
   through nx_read_each, and a long program or many reductions as
   NX_CPU_ITEM says, so that no count of them lengthens a held stretch.

   NX_CPU_HOLD_BYTES is 1 MiB: one core copies it in 12 to 15 us (M1 Max,
   fudge's Core Ultra 5 235), and a release and reacquisition cost 9 to
   150 ns, at most 1% of that, with or without systhreads and another
   domain collecting. NX_CPU_RELEASE_BYTES, below it, is the least cost
   at which a job takes a second thread: a call whose jobs could use the
   pool releases first, and a held one runs on one thread, priced under a
   quarter of NX_CPU_HOLD_BYTES but for the plans a transform makes on a
   cache's miss, which fft.c holds within NX_CPU_HOLD_BYTES.

   A price is in bytes one core's memcpy moves in the work's time, the
   reference nx.cpu's one-thread copy of a filled 128 KiB array. It is an
   upper bound in three classes:
   - streaming: work that moves its bytes in order, the bytes times its
     kernel's weight against memcpy (nx_cpu_walk_cost);
   - lines: an element or term that touches a cache line of its own,
     NX_CPU_LINE each;
   - scalar: work bound by arithmetic, its operations times their cost:
     an elementwise kind's per element (nx_cpu_kind_cost), a family's own
     for its routines.
   Each constant is 1.5 times the most one core took of its class's
   members, on the M1 Max and fudge under the timing lock, rounded up to a
   power of two, and names its worst member; a constant measured on the
   M1 alone says so. */

/* The most a held call may cost. */
#define NX_CPU_HOLD_BYTES (1024 * 1024)

/* The cost of a job for each of its threads (nx_cpu_threads). */
#define NX_CPU_THREAD_BYTES (128 * 1024)

/* The least price at which a call releases the domain lock: the least
   cost at which a job takes a second thread. */
#define NX_CPU_RELEASE_BYTES (2 * NX_CPU_THREAD_BYTES)

/* An item of a call's own reading and analysis, before its work: a
   program's node or a reduction, each copied, checked and priced. A node
   took about 20 ns (M1 alone, a 20000-node map under load), 1.6 KiB of
   memcpy. A call whose items cost NX_CPU_RELEASE_BYTES releases the domain
   lock before it analyses them, and a held loop that reads them from
   OCaml lets other domains collect (nx_let_collect) every NX_CPU_ITEMS
   items, a release's worth. */
#define NX_CPU_ITEM 4096
#define NX_CPU_ITEMS (NX_CPU_RELEASE_BYTES / NX_CPU_ITEM)

/* A cache line's fetch for an element or term that touches its own, on
   top of the term's own price: Logsumexp along strided float32 terms, its
   worst member, took 900 a term (M1 alone), which this and the term's 512
   (reduce.c) cover. */
#define NX_CPU_LINE 1024

/* The cost of computing the kind [k] of [arity] operands, nx_prog.h's
   code, at one element of [dt] (target.c). */
double nx_cpu_kind_cost(int arity, int k, int dt);

/* Whether the calling thread runs a held call's work, whose jobs then run
   on it alone (job.c): nx_cpu_release writes it at every call's entry and
   nx_cpu_reacquire clears it at its exit. */
extern _Thread_local int nx_cpu_held;

/* Whether the calling thread runs every job on itself alone, as a test of
   nx_cpu.ml's one_thread asks: the bits are those of any threads. */
extern _Thread_local int nx_cpu_one;

/* Releases the domain lock for a call's work that costs [price],
   NX_CPU_RELEASE_BYTES or more, else marks the call held; answers
   whether it released, for nx_cpu_reacquire. */
static inline int nx_cpu_release(double price) {
  nx_cpu_held = price < NX_CPU_RELEASE_BYTES;
  if (nx_cpu_held) return 0;
  /* The release that runs no pending signal handler: one that raised
     there would leave the door's claims held. */
  caml_enter_blocking_section_no_pending();
  return 1;
}

/* Reacquires the domain lock if nx_cpu_release released it, else ends
   the held call. Pending handlers run once the external returns. */
static inline void nx_cpu_reacquire(int released) {
  if (released) caml_leave_blocking_section();
  else nx_cpu_held = 0;
}

/* Jobs */

/* Chunks a job cuts its units into per thread: a thread that finishes its
   strip takes the chunks left in the others', so a thread that runs
   slower, an efficiency core's say, runs fewer. */
#define NX_CPU_CHUNKS 8

/* Runs [body] over the units [0, total) of work that touches [bytes] bytes
   and takes as long as memcpy of [cost] bytes, on as many of the pool's
   threads as the cost pays for: [body] reads no OCaml value, as a
   worker is no OCaml thread. The cost is an estimate, a double: one that
   multiplies extents of several operands may pass an int64_t. A body may
   begin a job of its own: it runs on the body's thread alone where the
   outer job runs on several (rig_pool.h, Scheduling). */
void nx_cpu_job(int64_t total, int64_t bytes, double cost, rig_pool_body body,
                void *ctx);

/* The threads nx_cpu_job gives such work, if it has as many units: at
   least 1, one per NX_CPU_THREAD_BYTES of cost, the bytes a core moves in
   the time a job takes to start, and 1 in a held call or under
   nx_cpu_one (job.c). */
int nx_cpu_threads(int64_t bytes, double cost);

/* nx_cpu_job and nx_cpu_threads with one thread per [thread_bytes] of
   cost: the bytes a core moves in the time a job of that work takes to
   start, where it starts slower than a job's NX_CPU_THREAD_BYTES. */
void nx_cpu_job_at(int64_t total, int64_t bytes, double cost,
                   double thread_bytes, rig_pool_body body, void *ctx);
int nx_cpu_threads_at(int64_t bytes, double cost, double thread_bytes);

/* Boxes (copy.c) */

/* A box: [extent[0]] planes of [extent[1]] rows of [extent[2]] elements, in
   C order, as nx_loop's axes. Element (p, i, j) of operand k, 0 the
   destination and 1 the source, lies at position
   first[k] + p·step[k][0] + i·step[k][1] + j·step[k][2], counted in
   elements from its base. Positions are at least 0; steps have either
   sign. */
typedef struct {
  int64_t extent[3];
  int64_t first[2];
  int64_t step[2][3]; /* elements, by operand then axis */
} nx_box;

/* Copies the box [b] of elements of [bits] bits from [src] to [dst], bits
   for bits. The destination's elements are distinct and share no byte with
   the source's. Bytes that hold only copied elements take plain stores; a
   byte shared with other elements takes a compare-and-swap, as
   nx_sub_store does. Each plane copies as one block: rows of adjacent
   elements are memcpy; where one operand steps one element across rows and
   the other one along them, whichever operand is transposed, it moves
   square blocks through registers, 8x8 of 4-byte elements, 32 bytes from
   each of eight columns into each of eight rows, and 4x4 of the others.
   Two adjacent runs of the source interleaved into adjacent elements, as a
   window two elements wide reads its rows, zip in registers. The case is
   chosen once per box. */
void nx_copy_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                 int bits);

/* nx_copy_box, its square blocks of 4- and 8-byte elements storing each
   row's 32 bytes past the caches (nx_stream32) where the destination's
   rows, and each plane's pieces of them, lie on whole 64-byte lines,
   drained before it returns: for a destination far larger than the
   caches. A line that plain and streamed stores share, or that a stream
   fills in parts far apart in time, is written in parts: on x86-64 a
   transposed 4000x4000 float32 copy in tiles 65 elements wide took twice
   as long streamed. */
void nx_stream_box(uint8_t *dst, const uint8_t *src, const nx_box *b,
                   int bits);

/* Stores past the caches

   A store into a line the cache lacks first reads the line. A result far
   larger than the caches, written in pieces that lie apart, as a
   transpose's rows are, reads each line it writes: stores that bypass the
   caches read none. NX_STREAMS is 1 where the target has them: x86-64's
   movntdq, arm64's stnp. Which results gain is each kernel's rule. */

#if defined(__x86_64__) || defined(__aarch64__)
#define NX_STREAMS 1
#else
#define NX_STREAMS 0
#endif

/* Stores the 32 bytes at [s] at [d], on 32 bytes, past the caches where
   NX_STREAMS: one AVX store or two SSE2 ones on x86-64, arm64's stnp in
   assembly, since clang lowers __builtin_nontemporal_store to plain
   stores there. Inline, a source in registers stays in them. */
static inline __attribute__((always_inline)) void nx_stream32(void *d,
                                                              const void *s) {
#if defined(__AVX__)
  _mm256_stream_si256((__m256i *)d, _mm256_loadu_si256((const __m256i *)s));
#elif defined(__x86_64__)
  _mm_stream_si128((__m128i *)d, _mm_loadu_si128((const __m128i *)s));
  _mm_stream_si128((__m128i *)d + 1,
                   _mm_loadu_si128((const __m128i *)s + 1));
#elif defined(__aarch64__)
  uint8x16_t lo = vld1q_u8((const uint8_t *)s);
  uint8x16_t hi = vld1q_u8((const uint8_t *)s + 16);
  __asm__ volatile("stnp %q0, %q1, [%2]" : : "w"(lo), "w"(hi), "r"(d)
                   : "memory");
#else
  __builtin_memcpy(d, s, 32);
#endif
}

/* Orders the stores past the caches before the stores that follow: x86's
   are weakly ordered and need a fence. arm64's stnp relaxes ordering only
   for a load's address dependency (the non-temporal hint), so like any
   store it is ordered by the release with which a job's worker leaves
   (rig_pool.c) and the acquire with which its caller sees the job end. */
static inline void nx_stream_drain(void) {
#if defined(__x86_64__)
  _mm_sfence();
#endif
}

/* Scratch */

/* [bytes] of scratch on 128 bytes, NULL if they cannot be allocated: one
   of the calling thread's kept buffers where one is free and [bytes] fit
   in it (job.c), else a new allocation. nx_cpu_give gives it back. */
void *nx_cpu_take(int64_t bytes);
void nx_cpu_give(void *p);

/* Matrices */

/* A contraction's order of summation, numbered as Nx_kernel.order's
   constructors are. */
enum { NX_CPU_CHAIN = 0, NX_CPU_LANE = 1 };

/* Stores into op[NX_VIEW_DST] the contraction in [acc], float32 or
   float64, and in [order], of op[NX_VIEW_A], op[NX_VIEW_B] and
   op[NX_VIEW_INIT] (or NULL), arrays of [acc] laid out as [v] (gemm.c).
   Answers 0 if an allocation failed. */
int nx_cpu_contract_on(int acc, int order, const nx_contract_view *v,
                       const nx_array *const *op);

/* The speed (nx_cpu_micro) of the core's vector kernel of products in
   [acc], float32 or float64: the table's, or its other kernels' where the
   table's runs on a unit that cores share, as the M1's matrix unit
   (gemm.c). It prices work whose units each run on one thread. */
double nx_cpu_core_speed(int acc);

/* Walks */

/* A block of a loop: [n2] planes of [n1] rows of [n0] elements. Operand k's
   first element is at position at[k] from its base, and it steps s0[k]
   elements along a row, s1[k] from a row to the next and s2[k] from a plane
   to the next. */
typedef struct {
  int64_t n0, n1, n2;
  int64_t at[NX_MAX_OPERANDS];
  int64_t s0[NX_MAX_OPERANDS];
  int64_t s1[NX_MAX_OPERANDS];
  int64_t s2[NX_MAX_OPERANDS];
} nx_cpu_block;

typedef void (*nx_cpu_block_fn)(const nx_cpu_block *b, void *ctx);

/* Calls [f] on blocks of at most [most] elements that cover the loop [l]
   over the [n] operands [a], operand 0 the written one, each element once,
   from a job, or on the calling thread for a loop that one block holds
   and one thread runs. [f] takes [weight] times as long as a memcpy of
   the operands' bytes, at least 1, and the job sizes its threads by
   nx_cpu_walk_cost. Order and threads decide no bit: [f] computes each element
   from operands' elements at its index alone. Inline, so that a kernel's
   call of [f] on its one block is direct (walk.c, rule 0). */
static inline void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l,
                               int64_t most, int64_t weight,
                               nx_cpu_block_fn f, void *ctx);

/* A walk that moves elements one by one, through tiles or along strides,
   costs as much time as this many times its bytes moved by memcpy: a
   transposed 512x512 float32 copy, tile by tile, takes 4.6 times memcpy's
   time on one kimchi core. */
#define NX_CPU_STRIDED_COST 4

/* The cost (job.c) of a walk over [bytes] whose block function takes
   [weight] times as long as memcpy, [strided] where an operand steps by
   more than one element along the rows. */
static inline int64_t nx_cpu_walk_cost(int64_t bytes, int64_t weight,
                                       int strided) {
  return weight * (strided ? NX_CPU_STRIDED_COST : 1) * bytes;
}

/* The price (cpu.h, Calls) of an elementwise call over the loop [l] of [n]
   operands, [per] an element where its operands step one element along
   its rows (a kind's, nx_cpu_kind_cost), NX_CPU_STRIDED_COST times that
   where one steps more or the loop keeps several axes. */
static inline double nx_cpu_loop_cost(int n, const nx_loop *l, double per) {
  double elements = 1;
  int strided = l->rank > 1;
  for (int i = 0; i < l->rank; i++) elements *= (double)l->extent[i];
  for (int k = 0; k < n; k++)
    strided |= l->step[k][l->rank - 1] > 1 || l->step[k][l->rank - 1] < -1;
  return elements * per * (strided ? NX_CPU_STRIDED_COST : 1);
}

/* nx_cpu_walk for a loop that rule 0 does not run (walk.c, rules 1 to
   3). */
void nx_cpu_cut(int n, const nx_array *a, const nx_loop *l, int64_t most,
                int64_t weight, nx_cpu_block_fn f, void *ctx);

static inline void nx_cpu_walk(int n, const nx_array *a, const nx_loop *l,
                               int64_t most, int64_t weight,
                               nx_cpu_block_fn f, void *ctx) {
  if (l->rank > 1 || l->extent[0] > most) {
    nx_cpu_cut(n, a, l, most, weight, f, ctx);
    return;
  }
  nx_cpu_block b = {.n0 = l->extent[0], .n1 = 1, .n2 = 1};
  int64_t bits = 0;
  int strided = 0;
  for (int k = 0; k < n; k++) {
    b.at[k] = l->first[k];
    b.s0[k] = l->step[k][0];
    bits += a[k].bits;
    strided |= b.s0[k] > 1 || b.s0[k] < -1;
  }
  /* Below two threads' cost, nx_cpu_threads gives one. */
  int64_t bytes = b.n0 * bits / 8;
  if (nx_cpu_walk_cost(bytes, weight, strided) >= 2 * NX_CPU_THREAD_BYTES) {
    nx_cpu_cut(n, a, l, most, weight, f, ctx);
    return;
  }
  if (b.n0 > 0) f(&b, ctx);
}

/* Copies */

/* Copies the elements of [bits] bits of the loop [l]'s operand 1 from [src]
   to its operand 0 at [dst], bits for bits, as a copy's walk does
   (copy.c). */
void nx_cpu_copy_loop(uint8_t *dst, const uint8_t *src, int bits,
                      const nx_loop *l);

/* Casts the loop [l]'s operand 1, [src], into its operand 0, [dst], as a
   cast's walk does (copy.c): each element rounds once to [dst]'s dtype. */
void nx_cpu_cast_loop(const nx_array *dst, const nx_array *src,
                      const nx_loop *l);

/* Padded loads */

/* A padding, Nx_kernel.pad, copied out of OCaml: [lo], [hi] and
   [interior] by axis of its [rank], then by window its axis, size, step
   and dilation. */
typedef struct {
  int rank, nwindows;
  int64_t lo[NX_MAX_RANK], hi[NX_MAX_RANK], interior[NX_MAX_RANK];
  int64_t window[NX_MAX_RANK][4];
} nx_cpu_pad;

/* Copies the Nx_kernel.pad [v] into [p] and answers NX_OK, or NX_SHAPE
   where [lo], [hi] and [interior] differ in length or the rank and the
   windows number more than NX_MAX_RANK. Reads [v] alone and allocates
   nothing (assemble.c). */
int nx_cpu_pad_of(value v, nx_cpu_pad *p);

/* Fills [y] with the shape an operand of shape [x] and rank [r] has once
   padded by [p], its windows' axes last, as Nx_kernel.Shape.padded
   computes it; nx_cpu.ml checks a load's shape by Shape.padded before a
   kernel copies it. Answers NX_OK, or NX_SHAPE for a pad it cannot
   apply: [r] other than [p]'s rank, an extent below zero or past int64,
   or a window whose size, step or dilation is below one. */
int nx_cpu_padded_shape(const nx_cpu_pad *p, int r, const int64_t *x,
                        int64_t *y);

/* Copies a load's padding [v], a (Nx_kernel.pad * string) of the pad and
   its fill's bits, into [p] and [fill], 16 bytes zero-padded, and answers
   NX_OK, or NX_SHAPE where the pad fits no operand of the shape of [x],
   the load's array. Reads [v] and [x] alone and allocates nothing, so a
   kernel copies it before the door, which can run OCaml code
   (assemble.c). */
int nx_cpu_padding_of(value v, value x, nx_cpu_pad *p, uint8_t *fill);

/* Fills [out] with the padded load [p] of the operand [a]: a plain
   descriptor over a C-contiguous copy of [a] padded with the element of
   bits [fill], laid out by [p]'s windows, which a kernel reads as any
   operand (assemble.c). [a]'s shape is one nx_cpu_padded_shape answers
   NX_OK for. The copy lives in C-heap memory at [out->base], which the
   caller frees; it is NULL for a load with no element. Answers 0, or 1 if
   host memory runs out, having allocated nothing. Reads no OCaml
   value. */
int nx_cpu_unpad(const nx_array *a, const nx_cpu_pad *p,
                 const uint8_t *fill, nx_array *out);

/* Programs

   A map's program as map.c runs it, block by block, for a kernel that
   reads a program's outputs as its terms, as reduce.c does. The program
   is copied out of OCaml before the door, which can run OCaml code, bound
   to the loads the door read, then run over blocks of them: each node
   computes as a map's does, so an output's elements have the map's bits.

   A block of a bound program numbers its operands as the program's loads,
   then, where it reads a coordinate, the index: the position in C order of
   the loop's shape, of no buffer. */
typedef struct nx_cpu_prog nx_cpu_prog;

/* The Prog.t at the root [vp] copied, or NULL if host memory runs out.
   Reads [vp] alone, letting other domains collect every NX_CPU_ITEMS
   nodes. */
nx_cpu_prog *nx_cpu_prog_copy(const value *vp);
void nx_cpu_prog_free(nx_cpu_prog *j);

/* [j]'s nodes, its loads and outputs, and their dtypes. */
int nx_cpu_prog_nodes(const nx_cpu_prog *j);
int nx_cpu_prog_ins(const nx_cpu_prog *j);
int nx_cpu_prog_in(const nx_cpu_prog *j, int k);
int nx_cpu_prog_outs(const nx_cpu_prog *j);
int nx_cpu_prog_out(const nx_cpu_prog *j, int o);

/* Binds [j] to its loads [a], read by the door, over a loop of [rank] axes
   of extents [shape]: NX_OK, NX_SHAPE for a coordinate past the rank, or
   NX_DECLINED for a program map.c declines. Sets [index] to whether the
   program reads the index. */
int nx_cpu_prog_bind(nx_cpu_prog *j, const nx_array *a, int rank,
                     const int64_t *shape, int *index);

/* The dtype output [o] of [j] is held in: its own, a narrow float's
   codes included, or its carrier for a sub-byte dtype. */
int nx_cpu_prog_held(const nx_cpu_prog *j, int o);

/* Runs the bound [j] over the block [b], one plane of n1 rows of n0
   elements, and stores each output [o] whose [to[o]] is not NULL there,
   C-contiguous in its held dtype: row r's element i at
   to[o] + (r·n0 + i)·w, w that dtype's width. */
void nx_cpu_prog_terms(const nx_cpu_prog *j, const nx_cpu_block *b,
                       uint8_t *const *to);

/* The kinds [j] runs an element: each takes about memcpy's time for its
   element's bytes. */
int64_t nx_cpu_prog_kinds(const nx_cpu_prog *j);

/* What an element of [j] costs (Calls): the kinds it runs, each at its
   cost (nx_cpu_kind_cost). */
double nx_cpu_prog_cost(const nx_cpu_prog *j);

/* Whether a run of [j] could not have an arena on the heap, as a program
   of many values needs: its outputs are then unspecified. */
int nx_cpu_prog_failed(const nx_cpu_prog *j);

/* The stage */

/* The bytes of a staged block: a walk for the stage takes blocks of at most
   NX_CPU_SLOT / w elements, w the widest form its elements take, so that a
   block's buffers stay in L1. */
#define NX_CPU_SLOT (16 * 1024)

/* Copies operand [k] of block [b], the array [a], into [dst] in its carrier:
   row j's element i at [dst + j·row + i·w], w the carrier's width. */
void nx_cpu_stage(const nx_array *a, const nx_cpu_block *b, int k, uint8_t *dst,
                  int64_t row);

/* Writes [src], elements of the dtype [c] with row j's element i at
   [src + j·row + i·w], w [c]'s width, into operand [k] of block [b], the
   array [a], converted as a cast does. [c] is a carrier. */
void nx_cpu_unstage(const nx_array *a, const nx_cpu_block *b, int k,
                    const uint8_t *src, int64_t row, int c);

/* Copies the [n1] rows of [n0] elements of [a] from position [at], each
   stepping [s0] along a row and [s1] across rows, into [dst] in the
   carrier [d], converted as a cast does: row j's element i at
   dst + (j·pitch + i)·w, w [d]'s width. An operand of the dtype [d] is
   one stage, whatever its size; another goes through a slot, in pieces
   whose every form fits it. */
void nx_cpu_stage_as(const nx_array *a, int64_t at, int64_t s0, int64_t s1,
                     int64_t n0, int64_t n1, int d, uint8_t *dst,
                     int64_t pitch);

#endif /* NX_CPU_H */
