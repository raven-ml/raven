/*---------------------------------------------------------------------------
   Copyright (c) 1992-2025 The University of Tennessee and The University of
   Tennessee Research Foundation, The University of California Berkeley,
   The University of Colorado Denver (LAPACK). BSD-3-Clause-Open-MPI (see
   dev/nx2/LICENSE-lapack).
   Copyright (c) 2016 Elmar Peise (ReLAPACK). MIT (see
   dev/nx2/LICENSE-relapack).
   Copyright (c) 2026 The Raven authors. All rights reserved.

   SPDX-License-Identifier: BSD-3-Clause-Open-MPI AND MIT AND ISC
  ---------------------------------------------------------------------------*/

/* Factorisations, triangular solves and decompositions: Nx_kernel.S's
   cholesky, lu, qr, solve_triangular, eigh, svd and eig.

   A call computes a routine on each matrix of its operands' batch. Each
   matrix is copied once from its operand, whatever its layout, into its
   result, C-contiguous, or into scratch, and computed there in place
   (linalg_body.h, one instance per dtype). A unit of the job is a run of
   matrices: one matrix each where there are as many as the job's threads,
   else all of them, one after the other, their products and their
   triangular solves' units of columns on the job's threads. Which thread
   computes a matrix or a unit changes no bit: a matrix's arithmetic is a
   function of its shape alone.

     operand ──stage──► result or scratch ──routine──► result
                                    │
                                    ├──── products (gemm.c)
                                    └──── solves' units of columns

   The routines follow LAPACK's (BSD) and ReLAPACK's recursive ones (MIT),
   named at each.

   No C file of nx2 contracts a product into an fma it does not name
   (dev/nx2/dune), so every target's results have the same bits. */

#include <complex.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_kinds.h"

/* The most columns of a recursive routine's leaf; the columns of the
   blocks of reflectors eigh, svd and eig reduce and apply; the columns of
   eigh's update panels; the columns of Cholesky's blocks, and the order
   of its update's tiles; the least columns of a triangular solve's units
   on the call's threads, or rows of x ← x l⁻ᴴ's. */
#define LEAF 24
#define NB 32
#define HERK 128
#define CHOL_NB 64
#define SOLVE_COLUMNS 32

/* CHOL_NB: on the i9-9900K's 16 threads, blocks of 48, 96 and 128 took
   1.06, 1.07 and 1.35 of cholesky-f64-1024's time at 64, 1.11, 1.19 and
   1.35 of cholesky-f64-256's, and 0.95, 1.33 and 1.38 of
   cholesky-c128-256's (one run each). */

/* The columns of LU's blocks: LU_NB where a matrix holds LU_BLOCKS of
   them, else half as many, so that a smaller matrix keeps blocks enough
   for the lookahead to overlap; a tile of a step's update is at least a
   block wide. On the i9-9900K's 16 threads, ABBA over 8 rounds, against
   64 columns and tiles of at least 32: 32 columns took 0.77 of the time
   of lu-f64-256 and 0.79 of lu-c128-256; at order 1024, tiles of at
   least 64 took 0.90 (f64) and 0.84 (f32), where 32 columns took
   1.03-1.17, 48 1.04-1.12, and 96 or 128 1.10-1.26. */
#define LU_NB 64
#define LU_BLOCKS 16

/* The bytes of a vector of SSE2 and NEON: an LU leaf's scan for its pivot
   compares that many at once. */
#define VECTOR_BYTES 16

/* The elements each row of QR's factored matrix takes past its n, and
   each column of an LU leaf past its rows, so that for a power-of-two
   length they lie no power of two apart: a block's products read down
   QR's columns and across a leaf's, and lines that far apart share one
   set of the cache. */
#define PAD 8

/* The columns of QR's blocks. On one core of the i9-9900K, blocks of 32,
   48 and 64 columns took 1.71, 1.69 and 1.92 ms on qr-f64-256, 74.0,
   67.5 and 67.8 ms on qr-f64-1024, and 6.59, 6.76 and 7.72 ms on
   qr-c128-256; on 16 threads 48 beat 64 at every size. On the M1, 64
   took 2-3% less than 48 at 512 and 1024 float64 columns and 3-8% more
   at 256 and in complex128; on fudge's six cores 4-7% more at each. */
#define QR_NB 48

/* The elements QR's blocks' T take for [k] reflectors, QR_NB × QR_NB a
   block. */
static int64_t qr_ts(int64_t k) { return (k + QR_NB) * QR_NB; }

/* The elements of scratch QR of [m] × [n] takes besides its factored
   matrix and τ: its blocks' T, a block's V, a leaf's copy and row, and two
   products of a block (linalg_body.h). */
static int64_t qr_scratch(int64_t m, int64_t n) {
  int64_t k = m < n ? m : n, big = m > n ? m : n;
  return qr_ts(k) + m * QR_NB + (m + 1) * 2 * LEAF + 2 * QR_NB * big;
}

/* The chunks of svd's reduction's panel (labrd): its products of the
   trailing matrix with a vector (xGEMV 'N' and 'C') by rows and by
   columns, and the sums of the panel's products with u_i by BRD_COLUMNS
   columns, which each chunk of rows adds again.

   A chunk of rows has an MV_CHUNKS-th of the panel's, at least MV_ROWS:
   a phase waits for its last chunk, and a chunk's rows are read a
   column at a time. On fudge's 6 P-cores (load 2.4-5.7, 4 rounds,
   minima) the one-stage reduction of svd-f64-1024 took 94.8, 74.8, 66.9
   and 57.8 ms in chunks of 16, 32, 64 and 128 rows, of svd-c128-512
   29.2, 26.6, 25.3 and 28.2, of svd-f64-256 2.14, 1.97, 2.07 and 2.17:
   an eighth of the rows, or 32, was the fastest at each. On the M1's 8
   P-cores (load 6-9, 2 rounds) the three took 53.4, 50.9 and 51.6 ms,
   29.6, 23.7 and 22.4, and 2.96, 2.64 and 2.54 at 128, 64 and 32 rows.

   MV_COLUMNS is four of mvh's groups of four columns, a multiple of 4 so
   that each chunk's columns group as the whole's do. BRD_COLUMNS groups
   the sums of Yᴴ u_i and U u_i, so it sets their bits; time hardly moves
   with it: on the M1 (timing lock, load 10-13, 2 rounds) the reduction
   of svd-f64-1024 took 52.8, 51.3, 51.2 and 51.3 ms at 16, 32, 64 and
   128 columns, of svd-c128-512 20.4, 20.4, 20.4 and 21.0, of
   svd-f64-256 2.27, 2.25, 2.31 and 2.41. */
#define MV_CHUNKS 8
#define MV_ROWS 32
#define MV_COLUMNS 16
#define BRD_COLUMNS 64

/* The roots of a divide and conquer merge in a unit of its work on the
   threads (xLAED3's loops over the roots). */
#define MERGE_ROOTS 32

/* Node [j] of depth [t] of lasd0's tree over [n] values and [sqre]: its
   first row into [c], its values into [size] and its sqre. The first
   half of a node has n / 2 values and one more column; the last the rest
   but one, and its parent's sqre. */
static void lasd_node(int64_t n, int sqre, int t, int64_t j, int64_t *c,
                      int64_t *size, int *sq) {
  *c = 0;
  *size = n;
  *sq = sqre;
  for (int b = t - 1; b >= 0; b--) {
    int64_t nl = *size / 2;
    if ((j >> b) & 1) {
      *c += nl + 1;
      *size -= nl + 1;
    } else {
      *size = nl;
      *sq = 1;
    }
  }
}

/* The reals lasd1 at a node of [size] values takes (or lasdq at a
   leaf). */
static int64_t lasd_work(int64_t size) {
  return 2 * size + 1 + 2 * size * size + (size + 1) * (size + 1);
}

/* The columns of a lower triangle of order h that a block of eigh's
   products of its trailing matrix with a vector (xHEMV) takes on
   average, and the most blocks: h / HEMV_COLUMNS blocks, at least one
   and at most HEMV_BLOCKS, each into its own rows of y, summed after,
   block after block; one block computes in y alone. Each block reads and
   adds h more elements: on the i9-9900K's 16 threads, eigh of a 1024 ×
   1024 float64 matrix took 81.7 ms in at most 16 blocks, 85.4 in 24 and
   84.9 in 32 (h / 32); of 2048 × 2048, 503 against 525 and 521. The
   sums follow the blocks alone, whatever the thread count, so a host of
   more threads splits a product 16 ways at most. */
#define HEMV_COLUMNS 32
#define HEMV_BLOCKS 16

/* The blocks of a product of order [h]. */
static int64_t hemv_blocks(int64_t h) {
  int64_t b = h / HEMV_COLUMNS;
  return b < 1 ? 1 : b > HEMV_BLOCKS ? HEMV_BLOCKS : b;
}

/* The first column of block [b] of [blocks] of a lower triangle of order
   [h], h past the last: the blocks hold about equal areas, and the
   triangle right of column j holds (h − j)² / 2, so block b starts where
   h − j = h √(1 − b / blocks). The start is a multiple of 4, as
   hemv_block takes its columns four at a time from it, so that no group
   but the triangle's last falls short. A function of h alone, so a
   product's sums, and bits, do not depend on the threads. */
static int64_t hemv_bound(int64_t h, int64_t blocks, int64_t b) {
  if (b >= blocks) return h;
  double rest = sqrt(1 - (double)b / (double)blocks);
  return (int64_t)((double)h * (1 - rest)) / 4 * 4;
}

/* eigh's two-stage reduction (linalg_body.h): to a band of kd
   subdiagonals by products, then to the tridiagonal by bulge chasing. It
   saves the one-stage reduction's matrix-vector products, half its
   flops, and costs Q2, 2n³ more, on the vectors: from order EIGH_VALUES
   at kd = KD_VALUES without them, EIGH_VECTORS at kd = KD with them on
   reals and EIGH_COMPLEX on complex numbers; below, the one-stage
   reduction. Two-stage time over one-stage time, minima:

   - fudge's 6 P-cores (load 3-5, 3 rounds): without vectors at kd 16,
     f64 0.64 at 128, 0.42 at 256 and 1024, c128 0.70 at 256 (kd 32: f64
     0.84 at 128, c128 1.09); with them at kd 32, f64 0.95 at 128, 0.70 at
     256, 0.80 at 1024. Complex vectors stay one-stage there: at 1024 Q2
     took 69 ms where the reduction saved 48 (c128 1.09, Q1 by columns).
   - the M1's 8 P-cores (load 8-9, 3 rounds): without vectors at kd 16,
     f64 0.99 at 256, 0.89 at 512, 0.81 at 1024, c128 0.58 at 512; with
     them at kd 32, f64 1.15 at 512, 1.06 at 1024, 0.91 at 2048, c128 0.89
     at 1024 (1.00 at kd 16, 0.98 at 256). Its one-stage reduction reads
     the trailing matrix at the M1's bandwidth, so that on reals Q2 costs
     what the products save below 2048.

   Bands of 16 halve the chase's work, which the values alone favour;
   with the vectors, bands of 32 give Q2 half as many blocks. */
#if defined(__x86_64__)
#define EIGH_VALUES 128
#define EIGH_VECTORS 128
#define EIGH_COMPLEX INT64_MAX
#else
#define EIGH_VALUES 512
#define EIGH_VECTORS 2048
#define EIGH_COMPLEX 1024
#endif
#define KD_VALUES 16
#define KD 32

/* Whether eigh of order [n] takes the two stages, with or without
   [vectors], of a complex dtype under [cplx]. */
static int eigh_two(int64_t n, int vectors, int cplx) {
  if (!vectors) return n >= EIGH_VALUES;
  return n >= (cplx ? EIGH_COMPLEX : EIGH_VECTORS);
}

/* The sweeps whose reflectors make a block of Q2 for the vectors: on the
   M1, eigh-f64-1024's Q2 took 15.2 ms in blocks of 16, 18.5 of 32, 19.8
   of 8 and 27.1 of 64 (load 8-10). */
#define SWEEPS 16

/* The int64_t of a cache line, the M1's 128 bytes: the chase's sweeps
   count their tasks a line apart, so that a sweep's thread spins on a
   line that only the sweep before it writes. */
#define LINE 16

/* Σ_{i ≤ m} ⌈i / KD⌉. */
static int64_t ceil_sum(int64_t m) {
  int64_t q = m / KD, r = m % KD;
  return KD * q * (q + 1) / 2 + r * (q + 1);
}

/* The reflectors of the bulge chase's sweeps before sweep [s] on a band of
   order [n]: sweep s′ makes ⌈(n − 1 − s′) / KD⌉ of them. */
static int64_t chased(int64_t n, int64_t s) {
  return ceil_sum(n - 1) - ceil_sum(n - 1 - s);
}

/* The elements of eigh's scratch between τ and the reflectors of the
   bulge chase: the one-stage reduction's W, hemv's blocks of y, and the
   back-transform's V, T and products; or the first stage's V, T, leaf
   copy, products and U, which outnumber the band and its sweeps' counts,
   from the values' crossover, the least. */
static int64_t eigh_work(int64_t n) {
  int64_t one = (3 * n + NB) * NB + hemv_blocks(n) * n;
  int64_t first = 4 * KD * n + 2 * LEAF * (n + 1) + 5 * KD * KD;
  return !eigh_two(n, 0, 0) || one > first ? one : first;
}

/* The elements of the bulge chase's reflectors, KD a reflector, kept for
   the vectors of a complex dtype under [cplx]. */
static int64_t eigh_slots(int64_t n, int vectors, int cplx) {
  return vectors && eigh_two(n, 1, cplx) ? chased(n, n - 1) * KD : 0;
}

/* svd's two-stage reduction (linalg_body.h): to a band of kd
   superdiagonals by products, then to the bidiagonal by bulge chasing. It
   saves the one-stage reduction's matrix-vector products, half its
   flops, and costs Q2 and P2, 4k³ more, on the vectors, k the lesser
   extent: from k = SVD_VALUES at kd = KD_VALUES without them on reals
   and SVD_COMPLEX_VALUES on complex numbers, SVD_VECTORS at kd = KD with
   them on reals and SVD_COMPLEX on complex numbers; below, the one-stage
   reduction. Two-stage time over one-stage time of square matrices,
   minima:

   - fudge's 6 P-cores (load 8-14, 3 rounds): without vectors, f64 0.75
     at 128, 0.58 at 256, 0.49 at 1024, c128 0.70 at 128, 0.53 at 512;
     with them, f64 1.00 at 128, 0.95 at 256, 0.89 at 1024, c128 1.18 at
     128, 1.06 at 256, 1.02 at 512. Complex vectors stay one-stage there,
     as eigh's do.
   - the M1's 8 P-cores (load 11-15, 4 ABBA rounds): without vectors,
     f64 1.17 at 128, 0.92 at 192, 0.87 at 256, 0.79 at 512, 0.66 at
     1024, c128 0.72 at 128, 0.57 at 256, 0.47 at 512; with them, f64
     1.34 at 128, 1.15 at 256, 0.99 at 512, 0.97 at 1024, 0.73 at 1536,
     c128 1.28 at 128, 0.98 at 256, 0.85 at 512. As eigh's, its
     one-stage reduction reads the trailing matrix at the M1's
     bandwidth. */
#if defined(__x86_64__)
#define SVD_VALUES 128
#define SVD_COMPLEX_VALUES 128
#define SVD_VECTORS 256
#define SVD_COMPLEX INT64_MAX
#else
#define SVD_VALUES 192
#define SVD_COMPLEX_VALUES 128
#define SVD_VECTORS 1024
#define SVD_COMPLEX 512
#endif

/* Whether svd of order [k] takes the two stages, with or without
   [vectors], of a complex dtype under [cplx]. */
static int svd_two(int64_t k, int vectors, int cplx) {
  if (!vectors) return k >= (cplx ? SVD_COMPLEX_VALUES : SVD_VALUES);
  return k >= (cplx ? SVD_COMPLEX : SVD_VECTORS);
}

/* The elements of scratch svd's reductions of [mm] × [k] take: QR's, or
   a block's V, T and products; or the first stage's V, T, U, Y, the
   LQ's V, T and copy, B, its products and leaf copy, which outnumber the
   band and its sweeps' counts, from the values' crossover of a complex
   dtype under [cplx], its least. */
static int64_t svd_scratch(int64_t mm, int64_t k, int cplx) {
  int64_t qr = qr_scratch(mm, k), block = (3 * mm + NB + 1) * NB;
  int64_t one = qr > block ? qr : block;
  int64_t first = (4 * mm + 5 * k + 5 * KD) * KD + 2 * LEAF * (mm + 1);
  return !svd_two(k, 0, cplx) || one > first ? one : first;
}

/* The elements of the bulge chase's left and right reflectors, KD a
   reflector, kept for the vectors of a complex dtype under [cplx]. */
static int64_t svd_slots(int64_t k, int vectors, int cplx) {
  return vectors && svd_two(k, 1, cplx) ? 2 * chased(k, k - 1) * KD : 0;
}

/* The columns a recursive routine on [n] computes first: half, a
   multiple of 8 from 16 on (ReLAPACK's split). */
static int64_t split(int64_t n) { return n >= 16 ? (n + 8) / 16 * 8 : n / 2; }

/* The most operands and results of a routine. */
#define ARRAYS 5

/* A worker's buffer, grown as its matrices need; [copy], the worker's
   buffer for an operand a routine keeps while products take the first;
   [workers], every worker's where a matrix has the call's threads, else
   NULL. */
typedef struct arena {
  void *p;
  int64_t n;
  struct arena *copy, *workers;
  double factor; /* the call's routine's (factor_of) */
} arena;

/* [bytes] of [s], NULL if they cannot be allocated. */
static void *room(arena *s, int64_t bytes) {
  if (bytes <= s->n) return s->p;
  free(s->p);
  s->p = malloc((size_t)bytes);
  s->n = s->p ? bytes : 0;
  return s->p;
}

/* The routines. */
enum { LU, CHOLESKY, QR, SOLVE_TRIANGULAR, EIGH, SVD, EIG };

/* A triangular solve: x ← t⁻¹ x for the triangle t, its diagonal read
   or taken as ones; or x ← x l⁻ᴴ for the lower l. */
typedef enum { LOWER, UPPER, UNIT_LOWER, UNIT_UPPER, ADJOINT_LOWER } solve;

/* A divide and conquer merge's work on its roots, element by element,
   each the same arithmetic on whichever thread: the roots and their
   vectors (ROOTS), z's weights recomputed from them (WEIGHTS), then the
   vectors from the weights (VECTORS). */
typedef enum { ROOTS, WEIGHTS, VECTORS } secular;

/* An svd merge's, likewise: the roots, z's weights, then the left
   vectors and, once products have read them, the right ones. */
typedef enum { SVD_ROOTS, SVD_WEIGHTS, LEFT_VECTORS, RIGHT_VECTORS } singular;

/* The order a sort leaves values in. */
typedef enum { ASCENDING, DESCENDING } order;

/* How reflectors are read: as stored, or conjugated (an LQ's, which
   store vᴴ). */
typedef enum { AS_STORED, CONJUGATED } reading;

/* How a back-transform's matrix holds its vectors: as its columns, or as
   its rows, their transposes. */
typedef enum { AS_COLUMNS, AS_ROWS } holding;

/* A call: its routine and attributes, 0 where it has none: [upper] for the
   triangle Upper, [complete] for complete factors, which q's or u's and
   vh's shapes state, [adjoint] and [unit] for a solve's, [vectors] where
   eigh or eig computes v or svd u and vh; its operand's [m] rows and [n]
   columns and, for a solve, the right-hand sides' count [r]; its arrays,
   the operands then the results, whose first [batch] axes are the batch. */
typedef struct {
  int routine, dt, w, upper, complete, adjoint, unit, vectors;
  int64_t m, n, r, matrices, per; /* per unit of the job */
  int nops, nres, batch;
  const nx_array *a;
  int64_t scratch; /* elements per worker, from its kept buffers */
  arena *packs, *copies; /* per worker: the products', the solves' */
  int failed;
} call;

/* The first element of matrix [b] of array [k], in C order of the batch's
   indices. */
static uint8_t *matrix_at(const call *c, int k, int64_t b) {
  const nx_array *x = &c->a[k];
  int64_t at = x->offset;
  for (int i = c->batch - 1; i >= 0; i--) {
    at += b % x->dim[i] * x->dim[x->rank + i];
    b /= x->dim[i];
  }
  return x->base + at * (x->bits / 8);
}

/* A stage: matrix [b] of operand [k] into [d], rows [ld] apart,
   transposed under [transposed]. */
typedef struct {
  const call *c;
  int k, transposed;
  int64_t b, ld;
  uint8_t *d;
} staging;

/* Rows [lo, hi) of the stage's destination. */
static void stage_rows(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const staging *g = ctx;
  const nx_array *x = &g->c->a[g->k];
  int r = x->rank, w = x->bits / 8;
  int64_t cols = x->dim[r - 1];
  int64_t sr = x->dim[2 * r - 2], sc = x->dim[2 * r - 1];
  if (g->transposed) {
    int64_t t = sr;
    cols = x->dim[r - 2];
    sr = sc;
    sc = t;
  }
  nx_box box = {.extent = {1, hi - lo, cols},
                .first = {0, 0},
                .step = {{0, g->ld, 1}, {0, sr, sc}}};
  if (hi > lo && cols > 0)
    nx_copy_box(g->d + lo * g->ld * w,
                matrix_at(g->c, g->k, g->b) + lo * sr * w, &box, x->bits);
}

/* What a matrix routine's work takes (cpu.h, Calls and jobs, scalar; M1
   Max and fudge, 2026-10-14), by whether its routine iterates (eigh, svd,
   eig) or solves and factors. Its flops at a core's float64 speed, times a
   factor for its scalar loops, which run no faster in float32:
   Hessenberg's and QR's sweeps, the tridiagonal's and bidiagonal's
   iterations, the triangular solves of eig's vectors. The M1 Max took up
   to 3.8 times them for float64 solves and factorisations of 36 to 128
   rows, 13 for svd's values alone of 36 rows. And a matrix's set-up and
   its scratch's, 3.5 KiB an LU of 2 × 2 float32, 2.8 a Cholesky factor
   and 34 an eigh (fudge). */
#define SOLVES_FACTOR 4
#define ITERATES_FACTOR 16
#define SOLVES_MATRIX (8 * 1024)
#define ITERATES_MATRIX (64 * 1024)

/* Whether [routine] iterates. */
static int iterates(int routine) {
  return routine == EIGH || routine == SVD || routine == EIG;
}

/* The factor of [routine]'s flops. */
static double factor_of(int routine) {
  return iterates(routine) ? ITERATES_FACTOR : SOLVES_FACTOR;
}

/* The cost of work on [bytes] that computes [flops] real flops, a complex
   one four, in a routine of [factor]: the call's, past its matrices'
   set-up, and each of its jobs'. */
static double work_cost(double factor, double bytes, double flops) {
  return bytes + factor * flops / nx_cpu_core_speed(NX_FLOAT64);
}

/* The cost of a job of [units] units on [bytes] that computes [flops],
   [cplx] 1 for complex ones, as work_cost at [s]'s factor, where it takes
   the call's threads, else 0: where the matrix has them in [s], there are
   units to share and the cost asks for more than one thread. A matrix
   without them prices nothing. */
static double threaded(const arena *s, int64_t units, int64_t bytes,
                       double flops, int cplx) {
  if (s->workers == NULL || units < 2) return 0;
  double cost = work_cost(s->factor, (double)bytes, (cplx ? 4 : 1) * flops);
  return nx_cpu_threads(bytes, cost) > 1 ? cost : 0;
}

/* Runs [body] over the [units] of a copy of [bytes]: on the call's
   threads where [s] holds them and the copy pays for them, since a matrix
   the call computes alone is copied while the threads wait. */
static void copying(const arena *s, int64_t units, int64_t bytes,
                    rig_pool_body body, void *ctx) {
  double cost = threaded(s, units, 2 * bytes, 0, 0);
  if (cost == 0) body(0, units, 0, ctx);
  else nx_cpu_job(units, 2 * bytes, cost, body, ctx);
}

/* Copies matrix [b] of operand [k] into [d], rows [ld] apart, transposed
   under [transposed], by bands of rows. */
static void stage(const call *c, int k, int64_t b, void *d, int64_t ld,
                  int transposed, const arena *s) {
  const nx_array *x = &c->a[k];
  int r = x->rank;
  int64_t rows = x->dim[r - 2 + (transposed != 0)];
  int64_t bytes = rows * x->dim[r - 1 - (transposed != 0)] * (x->bits / 8);
  staging g = {.c = c, .k = k, .transposed = transposed, .b = b, .ld = ld,
               .d = d};
  copying(s, rows, bytes, stage_rows, &g);
}

/* A matrix of host memory: element (i, j) at p + (i·rs + j·cs)·w, w its
   dtype's width. */
typedef struct {
  const void *p;
  int64_t rs, cs;
} strided;

/* A matrix of a routine's dtype, op(x)[i][j] = x[i·rs + j·cs],
   conjugated under [conj] (complex dtypes alone). */
typedef struct {
  const void *p;
  int64_t rs, cs;
  int conj;
} operand;

/* c += a·b for c of [m] rows and [n] columns, rows [ldc] apart and its
   columns adjacent, and [k] products an element, in [dt], float32 or
   float64, by the contraction (gemm.c) in chain order, c its own init.
   Answers 0 if an allocation failed. */
static int product(int dt, int64_t m, int64_t n, int64_t k, strided a,
                   strided b, void *c, int64_t ldc) {
  int bits = 8 * nx_cpu_width(dt);
  nx_array x[3] = {{.base = (uint8_t *)a.p, .dtype = dt, .bits = bits},
                   {.base = (uint8_t *)b.p, .dtype = dt, .bits = bits},
                   {.base = c, .dtype = dt, .bits = bits}};
  nx_contract_view v = {.extent = {1, m, n, k}};
  v.stride[NX_VIEW_A][NX_VIEW_ROW] = a.rs;
  v.stride[NX_VIEW_A][NX_VIEW_CONTRACTED] = a.cs;
  v.stride[NX_VIEW_B][NX_VIEW_CONTRACTED] = b.rs;
  v.stride[NX_VIEW_B][NX_VIEW_COLUMN] = b.cs;
  for (int o = NX_VIEW_INIT; o <= NX_VIEW_DST; o++) {
    v.stride[o][NX_VIEW_ROW] = ldc;
    v.stride[o][NX_VIEW_COLUMN] = 1;
  }
  const nx_array *op[4] = {&x[0], &x[1], &x[2], &x[2]};
  return nx_cpu_contract_on(dt, NX_CPU_CHAIN, &v, op);
}

/* Sets the [n] reals of the dtype [dt] at [p] to NaN: a failed matrix's
   result. */
static void nans(void *p, int64_t n, int dt) {
  if (dt == NX_FLOAT32)
    for (int64_t i = 0; i < n; i++) ((float *)p)[i] = NAN;
  else
    for (int64_t i = 0; i < n; i++) ((double *)p)[i] = NAN;
}

/* The spins of wait_for before it yields the core. On the M1 at load
   23-32, the chases of eigvalsh-f64-1024 and eigh-c128-1024 took 28.3
   and 25.6 ms at least in two runs of 100 spins, 7.2-12.0 in four of
   1000, 10.1 and 13.4 in two of 10000: a sweep's next task is a few
   microseconds of the sweep before away. */
#define SPINS 1000

/* The rows and columns of a tile of an in-place transpose. On the M1 at
   load 23-32, eigh-c128-1024's two transposes took 4.9 ms in tiles of
   16, 6.2 of 32 and 8.3 of 64 (minima of two runs). */
#define TILE 16

/* Waits until [*x] reaches [v], which another thread's release store
   sets: by spinning, then by yielding the core, as the thread waited on
   may have lost its own to another program on a busy host. */
static void wait_for(const int64_t *x, int64_t v) {
  for (int i = 0; __atomic_load_n(x, __ATOMIC_ACQUIRE) < v;) {
    if (i == SPINS) {
      sched_yield();
      continue;
    }
    i++;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
  }
}

/* The routines, per dtype */

#define CAT2(a, b) a##_##b
#define CAT(a, b) CAT2(a, b)
#define F(name) CAT(name, SFX)

#define T float
#define R float
#define RDT NX_FLOAT32
#define SFX f32
#define C float complex
#define CK c64
#define CPLX 0
#define CONJ(x) (x)
#define REAL(x) (x)
#define IMAG(x) ((R)0)
#define SQRT sqrtf
#define COPYSIGN copysignf
#define LDEXP ldexpf
#define ILOGB ilogbf
#define WORD int32_t
#include "linalg_body.h"

#define T double
#define R double
#define RDT NX_FLOAT64
#define SFX f64
#define C double complex
#define CK c128
#define CPLX 0
#define CONJ(x) (x)
#define REAL(x) (x)
#define IMAG(x) ((R)0)
#define SQRT sqrt
#define COPYSIGN copysign
#define LDEXP ldexp
#define ILOGB ilogb
#define WORD int64_t
#include "linalg_body.h"

#define T float complex
#define R float
#define RDT NX_FLOAT32
#define SFX c64
#define C T
#define CK c64
#define CPLX 1
#define CONJ conjf
#define REAL crealf
#define IMAG cimagf
#define SQRT sqrtf
#define COPYSIGN copysignf
#define LDEXP ldexpf
#define ILOGB ilogbf
#define WORD int32_t
#include "linalg_body.h"

#define T double complex
#define R double
#define RDT NX_FLOAT64
#define SFX c128
#define C T
#define CK c128
#define CPLX 1
#define CONJ conj
#define REAL creal
#define IMAG cimag
#define SQRT sqrt
#define COPYSIGN copysign
#define LDEXP ldexp
#define ILOGB ilogb
#define WORD int64_t
#include "linalg_body.h"

/* The job */

/* A job's body: units [lo, hi), matrices [u·per, (u + 1)·per) of each. */
static void matrices(int64_t lo, int64_t hi, int worker, void *ctx) {
  call *c = ctx;
  arena *s = &c->packs[worker];
  void *w = nx_cpu_take((c->scratch > 0 ? c->scratch : 1) * c->w);
  if (w == NULL) {
    __atomic_store_n(&c->failed, 1, __ATOMIC_RELAXED);
    return;
  }
  int64_t end = hi * c->per < c->matrices ? hi * c->per : c->matrices;
  for (int64_t b = lo * c->per; b < end; b++) {
    int ok;
    switch (c->dt) {
      case NX_FLOAT32: ok = matrix_f32(c, b, w, s); break;
      case NX_FLOAT64: ok = matrix_f64(c, b, w, s); break;
      case NX_COMPLEX64: ok = matrix_c64(c, b, w, s); break;
      default: ok = matrix_c128(c, b, w, s); break;
    }
    if (!ok) __atomic_store_n(&c->failed, 1, __ATOMIC_RELAXED);
  }
  nx_cpu_give(w);
}

/* A matrix's flops, two a multiply-add as gemm.c counts them, with k the
   lesser extent: a solve's n² r; Cholesky's n³/3 (xPOTRF); LU's
   m n k − k³/3 (xGETRF); QR's 2 m n k − 2k³/3 for R (xGEQRF) and, for Q
   (xORGQR), 2 m k² − 2k³/3 of k columns or 4 m² k − 4 m k² + 4k³/3 of m;
   eigh's about 10/3 n³ with vectors, 4/3 n³ without (xHEEVD); svd's
   about 4 m n k without, three times as many with (xGESDD); eig's about
   25 n³ with vectors, 10 n³ without (xGEEV, Golub and Van Loan). */
static double flops(const call *c) {
  double m = c->m, n = c->n, k = m < n ? m : n, k3 = k * k * k;
  switch (c->routine) {
    case SOLVE_TRIANGULAR: return n * n * c->r;
    case CHOLESKY: return n * n * n / 3;
    case LU: return m * n * k - k3 / 3;
    case QR:
      return 2 * m * n * k - 2 * k3 / 3 +
             (c->complete ? 4 * m * m * k - 4 * m * k * k + 4 * k3 / 3
                          : 2 * m * k * k - 2 * k3 / 3);
    case EIGH: return (c->vectors ? 10 : 4) * n * n * n / 3;
    case EIG: return (c->vectors ? 25 : 10) * n * n * n;
    default: return (c->vectors ? 12 : 4) * m * n * k;
  }
}

/* The elements of scratch a matrix of [c] takes besides its products':
   a solve's triangle; QR's factored matrix, its τ and qr_scratch; eigh's
   reduced matrix, its τ, eigh_work, the bulge chase's reflectors, then as
   reals T's diagonals, Z and divide and conquer's work, and its indices;
   svd's B, QR's R, the τ, a panel's X and Y, svd_scratch, X or W, then
   as reals B's diagonals, U_B, V_B and divide and conquer's work, and its
   indices; eig's balanced matrix, H, Z, X, v on reals, Q's τ, the scratch
   of its blocks, then as reals the eigenvalues, the scales and S's column
   norms, and a complex column (linalg_body.h). */
static int64_t scratch(const call *c) {
  int64_t m = c->m, n = c->n, big = m > n ? m : n, k = m < n ? m : n;
  switch (c->routine) {
    case SOLVE_TRIANGULAR: return n * n;
    case QR: return m * (n + PAD) + k + qr_scratch(m, n);
    case EIGH:
      return 4 * n * n + 16 * n + 2 + eigh_work(n) +
             eigh_slots(n, c->vectors,
                        c->dt == NX_COMPLEX64 || c->dt == NX_COMPLEX128);
    case SVD: {
      int cplx = c->dt == NX_COMPLEX64 || c->dt == NX_COMPLEX128;
      int64_t tmp = m >= n ? k * k : big * (c->complete ? big : k);
      return big * k + 3 * k * k + 3 * (k + 1) * (k + 1) + 19 * k + 2 +
             svd_scratch(big, k, cplx) + tmp + svd_slots(k, c->vectors, cplx);
    }
    case EIG: return 5 * n * n + 7 * n + (3 * n + NB) * NB;
    default: return 0;
  }
}

/* The entries */

/* Whether the routines take the dtype [dt]: Nx_kernel.routines_take's. */
static int takes(int dt) {
  return dt == NX_FLOAT32 || dt == NX_FLOAT64 || dt == NX_COMPLEX64 ||
         dt == NX_COMPLEX128;
}

/* The results' shapes into [y], their ranks into [ranks], for the operand
   of shape [x] of rank [xr] and the right-hand sides [z] of rank [zr]:
   Nx_kernel.Shape's. Answers 0 if the operands do not fit. */
static int results(const call *c, const int64_t *x, int xr, const int64_t *z,
                   int zr, int64_t y[][NX_MAX_RANK], int *ranks) {
  if (xr < 2) return 0;
  int64_t m = x[xr - 2], n = x[xr - 1], k = m < n ? m : n;
  int square = c->routine != LU && c->routine != QR && c->routine != SVD;
  if (square && m != n) return 0;
  for (int j = 0; j < c->nres; j++) {
    memcpy(y[j], x, sizeof(int64_t) * xr);
    ranks[j] = xr;
  }
  switch (c->routine) {
    case LU:
      ranks[1] = ranks[2] = xr - 1;
      y[1][xr - 2] = k;
      y[2][xr - 2] = m;
      return 1;
    case QR: {
      int64_t q = c->complete ? m : k;
      y[0][xr - 1] = q;
      y[1][xr - 2] = q;
      return 1;
    }
    case EIGH:
    case EIG: ranks[0] = xr - 1; return 1;
    case SVD:
      ranks[0] = xr - 1;
      y[0][xr - 2] = k;
      if (c->nres == 1) return 1;
      y[1][xr - 1] = c->complete ? m : k;
      y[2][xr - 2] = c->complete ? n : k;
      return 1;
    case SOLVE_TRIANGULAR:
      if (zr != xr) return 0;
      for (int i = 0; i < xr - 1; i++)
        if (z[i] != x[i]) return 0;
      y[0][xr - 1] = z[xr - 1];
      return 1;
    default: return 1;
  }
}

/* Whether [q], a QR's left factor for an operand of shape [x] of rank
   [xr], is complete: as many columns as the operand has rows. Where the
   operand has no more rows than columns, both factors have that shape. */
static int complete(value q, const int64_t *x, int xr) {
  int64_t d[2 * NX_MAX_RANK], off;
  int r = nx_array_layout(q, d, &off);
  return xr >= 2 && r >= 1 && d[r - 1] == x[xr - 2];
}

/* Whether [u] and [vh], an SVD's factors for an operand of shape [x] of
   rank [xr], are complete: u of as many columns as the operand has rows,
   vh of as many rows as it has columns. */
static int complete_svd(value u, value vh, const int64_t *x, int xr) {
  int64_t d[2 * NX_MAX_RANK], e[2 * NX_MAX_RANK], off;
  int r = nx_array_layout(u, d, &off), s = nx_array_layout(vh, e, &off);
  return xr >= 2 && r >= 1 && s >= 2 && d[r - 1] == x[xr - 2] &&
         e[s - 2] == x[xr - 1];
}

/* Whether the array [v] has the shape [s] of rank [r]. */
static int has_shape(value v, const int64_t *s, int r) {
  int64_t d[2 * NX_MAX_RANK], off;
  if (nx_array_layout(v, d, &off) != r) return 0;
  for (int i = 0; i < r; i++)
    if (d[i] != s[i]) return 0;
  return 1;
}

/* Runs [c], its routine and attributes set, on [arrays], its operands then
   its results. The caller roots the arrays: they are read before nx_read,
   which may run OCaml code. */
static value run(call c, const value *arrays) {
  int solve = c.routine == SOLVE_TRIANGULAR;
  c.nops = 1 + solve;
  c.nres = c.routine == LU || (c.routine == SVD && c.vectors) ? 3
           : c.routine == QR || ((c.routine == EIGH || c.routine == EIG) &&
                                 c.vectors)
               ? 2
               : 1;
  c.dt = nx_array_dtype(arrays[0]);
  if (!takes(c.dt)) return Val_int(NX_DTYPE);
  int64_t x[2 * NX_MAX_RANK], z[2 * NX_MAX_RANK], y[3][NX_MAX_RANK], off;
  int ranks[3];
  int xr = nx_array_layout(arrays[0], x, &off);
  int zr = solve ? nx_array_layout(arrays[1], z, &off) : 0;
  if (c.routine == QR) c.complete = complete(arrays[1], x, xr);
  if (c.routine == SVD && c.vectors)
    c.complete = complete_svd(arrays[2], arrays[3], x, xr);
  if (!results(&c, x, xr, z, zr, y, ranks)) return Val_int(NX_SHAPE);
  for (int j = 0; j < c.nres; j++)
    if (!has_shape(arrays[c.nops + j], y[j], ranks[j]))
      return Val_int(NX_SHAPE);
  c.batch = xr - 2;
  c.m = x[xr - 2];
  c.n = x[xr - 1];
  c.r = solve ? z[zr - 1] : 0;
  c.matrices = 1;
  for (int i = 0; i < c.batch; i++) c.matrices *= x[i];
  /* The door: the operands, then the results. */
  int n = c.nops + c.nres;
  nx_operand in[ARRAYS];
  for (int k = 0; k < n; k++)
    in[k] = (nx_operand){arrays[k], nx_array_dtype(arrays[k]), k >= c.nops};
  nx_array a[ARRAYS];
  int e = nx_read(n, in, a);
  if (e) return Val_int(e);
  c.a = a;
  c.w = nx_cpu_width(c.dt);
  c.scratch = scratch(&c);
  int cplx = c.dt == NX_COMPLEX64 || c.dt == NX_COMPLEX128;
  int64_t bytes = c.matrices * c.m * (c.n + c.r) * c.w;
  double factor = factor_of(c.routine);
  double matrix = iterates(c.routine) ? ITERATES_MATRIX : SOLVES_MATRIX;
  double cost = work_cost(factor, (double)bytes,
                          (cplx ? 4 : 1) * (double)c.matrices * flops(&c)) +
                (double)c.matrices * matrix;
  int released = nx_cpu_release(cost);
  /* A matrix a unit where there are enough for the threads, else one unit
     of all, whose GEMM updates and own jobs take the threads where the
     call asks for more than one. */
  int t = nx_cpu_threads(bytes, cost);
  c.per = c.matrices >= t ? 1 : c.matrices;
  int cores = rig_pool_cores();
  c.packs = calloc((size_t)cores, sizeof(arena));
  c.copies = calloc((size_t)cores, sizeof(arena));
  if (c.packs == NULL || c.copies == NULL) {
    free(c.packs);
    free(c.copies);
    nx_cpu_reacquire(released);
    nx_done(n, a);
    caml_raise_out_of_memory();
  }
  for (int i = 0; i < cores; i++) {
    c.packs[i].copy = &c.copies[i];
    c.packs[i].factor = c.copies[i].factor = factor;
    if (c.matrices < t) c.packs[i].workers = c.packs;
  }
  if (c.matrices > 0) {
    int64_t units = c.matrices / c.per;
    nx_cpu_job(units, bytes, cost, matrices, &c);
  }
  for (int i = 0; i < cores; i++) {
    free(c.packs[i].p);
    free(c.copies[i].p);
  }
  free(c.packs);
  free(c.copies);
  nx_cpu_reacquire(released);
  nx_done(n, a);
  if (c.failed) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_cholesky(value triangle, value dst, value a) {
  CAMLparam2(dst, a);
  value arrays[] = {a, dst};
  call c = {.routine = CHOLESKY,
            .upper = Int_val(triangle) == NX_TRIANGLE_UPPER};
  CAMLreturn(run(c, arrays));
}

value nx_cpu_lu(value lu, value pivots, value perm, value a) {
  CAMLparam4(lu, pivots, perm, a);
  value arrays[] = {a, lu, pivots, perm};
  CAMLreturn(run((call){.routine = LU}, arrays));
}

value nx_cpu_qr(value q, value r, value a) {
  CAMLparam3(q, r, a);
  value arrays[] = {a, q, r};
  CAMLreturn(run((call){.routine = QR}, arrays));
}

value nx_cpu_solve_triangular(value triangle, value adjoint, value unit,
                              value dst, value a, value b) {
  CAMLparam3(dst, a, b);
  value arrays[] = {a, b, dst};
  call c = {.routine = SOLVE_TRIANGULAR,
            .upper = Int_val(triangle) == NX_TRIANGLE_UPPER,
            .adjoint = Bool_val(adjoint),
            .unit = Bool_val(unit)};
  CAMLreturn(run(c, arrays));
}

value nx_cpu_solve_triangular_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_solve_triangular(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

value nx_cpu_eigh(value w, value v, value a) {
  CAMLparam3(w, v, a);
  int vectors = Is_some(v);
  value arrays[] = {a, w, vectors ? Some_val(v) : Val_unit};
  CAMLreturn(run((call){.routine = EIGH, .vectors = vectors}, arrays));
}

value nx_cpu_eig(value w, value v, value a) {
  CAMLparam3(w, v, a);
  int vectors = Is_some(v);
  value arrays[] = {a, w, vectors ? Some_val(v) : Val_unit};
  CAMLreturn(run((call){.routine = EIG, .vectors = vectors}, arrays));
}

value nx_cpu_svd(value s, value vectors, value a) {
  CAMLparam3(s, vectors, a);
  CAMLlocal2(u, vh);
  int some = Is_some(vectors);
  if (some) {
    u = Field(Some_val(vectors), 0);
    vh = Field(Some_val(vectors), 1);
  }
  value arrays[] = {a, s, some ? u : Val_unit, some ? vh : Val_unit};
  CAMLreturn(run((call){.routine = SVD, .vectors = some}, arrays));
}
