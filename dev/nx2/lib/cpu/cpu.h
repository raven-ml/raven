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

/* A run converts [n] contiguous elements at [src] into [dst]. */
typedef void (*nx_cpu_run)(const void *src, void *dst, int64_t n);

/* The lanes of a sum whose order is fixed by shape: term t of a block of
   NX_CPU_LANE_BLOCK consecutive terms adds into lane t modulo
   NX_CPU_LANES. */
#define NX_CPU_LANES 16
#define NX_CPU_LANE_BLOCK 1024

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

/* A monoid's kernels at a dtype (reduce.c). [lanes] adds the [n] terms at [x],
   stepping [s] elements, into the NX_CPU_LANES lanes at [l]: term i into
   lane (first + i) mod NX_CPU_LANES. [combine] adds the [n] elements at
   [x], stepping [s], into the [n] contiguous accumulators at [a], element
   by element. [scan] runs [k] sequences, at most 4, side by side: sequence
   j adds the [n] elements at [x + j·xs], stepping [s], into the
   accumulator [a]'s element j in order, storing each sum at
   [y + j·ys], stepping [sy]. [blocks] stores into [v] the values of the [n] blocks of
   NX_CPU_LANE_BLOCK contiguous terms from [x], each its lanes from the
   identity [e] and the lanes' tree, as [lanes] and [combine] give them.
   [few] stores the values of [h] rows of [w] outputs of [n] terms each, n
   at most NX_CPU_LANES, output j of row i at [y + i·sy2 + j·sy]: its term
   t at [x + i·s2 + j·s + off[t]], in lane t from the identity [e], then
   the lanes' tree, as one block of [lanes] gives them. [column] adds
   into each of the [w] contiguous accumulators at [a] in turn the [n] rows
   of [x], row i at [x + i·st] and its element j [s] further than j - 1:
   accumulator j takes row 0's element j, then row 1's, and so on. */
typedef struct {
  void (*lanes)(const uint8_t *x, int64_t s, int64_t n, uint8_t *l,
                int first);
  void (*combine)(uint8_t *a, const uint8_t *x, int64_t s, int64_t n);
  void (*scan)(uint8_t *a, const uint8_t *x, int64_t s, int64_t xs,
               uint8_t *y, int64_t sy, int64_t ys, int64_t n, int k);
  void (*blocks)(const uint8_t *x, int64_t n, const uint8_t *e, uint8_t *v);
  void (*few)(const uint8_t *x, const int64_t *off, int n, int64_t s,
              int64_t w, int64_t s2, int64_t h, const uint8_t *e, uint8_t *y,
              int64_t sy, int64_t sy2);
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
   extreme's bits [v] and its position [at], or a value [s] of the terms
   and their count [n]: Logsumexp's maximum, or its sum, and Moments' sums,
   [d] the deviations' with [s] their squares'. A float is its dtype's
   value, which a double holds exactly. */
typedef struct {
  int64_t n, at;
  uint64_t v;
  double s, d;
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

/* A stateful reduction at a dtype: where [center.block] is not NULL, a
   first pass whose [finish] stores each output's center at [d0], a double,
   then the [value] pass, which takes it. Logsumexp's center is the terms'
   maximum where finite, else 0; Moments', their mean. */
typedef struct {
  nx_cpu_pass center, value;
} nx_cpu_stateful;

/* A Moments scan of [w] slices of [n] terms side by side (reduce.c), w at
   most NX_CPU_MOMENTS_LANES: slice l's term t at x + t xs + l xl, its mean
   and variance at mean + t ms + l ml and var + t vs + l vl, in elements;
   [scratch] holds (12 nx_cpu_moments_levels(n) + 3 NX_CPU_MOMENTS_TILE) w
   elements. */
#define NX_CPU_MOMENTS_LANES 64

/* A Moments scan's terms per tile. On fudge,
   scan-moments-f32-4096x1024-axis1 took 1.18 ms at 16, 1.04 at 32, 1.12 at
   64. */
#define NX_CPU_MOMENTS_TILE 32

/* The levels of a Moments scan's network over [n] terms, n >= 1: one per
   bit of n. */
static inline int nx_cpu_moments_levels(int64_t n) {
  int k = 1;
  for (; n > 1; n >>= 1) k++;
  return k;
}

typedef void (*nx_cpu_moments)(const uint8_t *x, int64_t xs, int64_t xl,
                               uint8_t *mean, int64_t ms, int64_t ml,
                               uint8_t *var, int64_t vs, int64_t vl,
                               int64_t n, int64_t w, uint8_t *scratch);

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
  /* moments[dt] scans by Moments at float32 and float64. */
  nx_cpu_moments moments[NX_DTYPE_COUNT];
  /* stateful[r - NX_CPU_MONOIDS][dt] reduces by r (NX_CPU_LOGSUMEXP to
     NX_CPU_MOMENTS) at dt: block NULL where the table declines. */
  nx_cpu_stateful stateful[NX_CPU_STATEFUL][NX_DTYPE_COUNT];
  /* fft runs fft.c's transform t (its C2C, R2C or C2R; inverse for C2C's
     Inverse) of x into y along [axes] on [threads] threads: 0 if memory
     runs out. */
  int (*fft)(int t, int inverse, const nx_array *x, const nx_array *y,
             const int *axes, int na, int threads);
} nx_cpu_target;

/* The tables, each set when the program starts on a host that runs it, by
   convert.c, gemm_generic.c, rows.c, reduce.c and fft.c compiled for its
   target, then by the kernels of its instructions: gemm_neon.c's on arm64,
   gemm_avx2.c's for v3, and gemm_vnni.c's quads where the host has
   AVX-VNNI-INT8. amx is base's table with gemm_amx.c's kernels for the
   matrix unit of Apple's M1. */
extern nx_cpu_target nx_cpu_base;
void nx_cpu_set_convert_base(nx_cpu_target *t);
void nx_cpu_set_gemm_base(nx_cpu_target *t);
void nx_cpu_set_rows_base(nx_cpu_target *t);
void nx_cpu_set_reduce_base(nx_cpu_target *t);
void nx_cpu_set_fft_base(nx_cpu_target *t);
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

/* Jobs */

/* Chunks a job cuts its units into per thread: a thread that finishes its
   strip takes the chunks left in the others', so a thread that runs
   slower, an efficiency core's say, runs fewer. */
#define NX_CPU_CHUNKS 8

/* Runs [body] over the units [0, total) of work that touches [bytes] bytes
   and takes as long as memcpy of [cost] bytes, on as many of the pool's
   threads as the cost pays for. A job of more than one thread, or a long
   one, runs with the runtime released: [body] reads no OCaml value. The
   cost is an estimate, a double: one that multiplies extents of several
   operands may pass an int64_t. A body may begin a job of its own: it runs
   with the runtime still released, on the body's thread alone where the
   outer job runs on several (rig_pool.h, Scheduling). */
void nx_cpu_job(int64_t total, int64_t bytes, double cost, rig_pool_body body,
                void *ctx);

/* The threads nx_cpu_job gives such work, if it has as many units: at
   least 1, one per NX_CPU_THREAD_BYTES of cost, the bytes a core moves in
   the time a job takes to start (job.c). */
int nx_cpu_threads(int64_t bytes, double cost);
#define NX_CPU_THREAD_BYTES (128 * 1024)

/* nx_cpu_job and nx_cpu_threads with one thread per [thread_bytes] of
   cost: the bytes a core moves in the time a job of that work takes to
   start, where it starts slower than a job's NX_CPU_THREAD_BYTES. */
void nx_cpu_job_at(int64_t total, int64_t bytes, double cost,
                   double thread_bytes, rig_pool_body body, void *ctx);
int nx_cpu_threads_at(int64_t bytes, double cost, double thread_bytes);

/* nx_cpu_job's work of one thread: [body] over [0, total) on the calling
   thread, with the runtime released where [cost] is as much as a held
   job may take (job.c), as for a job nx_cpu_threads gives one thread.
   For work that decides its threads itself, as a contraction's call
   does (gemm.c). */
void nx_cpu_one_thread(int64_t total, double cost, rig_pool_body body,
                       void *ctx);

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
   computes it. Answers NX_OK, or NX_SHAPE where Shape.padded answers
   [Error]: [r] other than [p]'s rank, an extent below zero or past int64,
   or a window that does not fit its padded axis. */
int nx_cpu_padded_shape(const nx_cpu_pad *p, int r, const int64_t *x,
                        int64_t *y);

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
