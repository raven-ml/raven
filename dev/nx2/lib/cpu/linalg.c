/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Factorisations, triangular solves and eigendecompositions: Nx_kernel.S's
   cholesky, lu, qr, solve_triangular and eigh.

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
   named at each. */

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* The most columns a recursive routine computes by scalar loops; the
   columns of QR's blocks; the columns of Cholesky's update panels; the
   least columns of a triangular solve's units on the call's threads. */
#define LEAF 24
#define NB 32
#define HERK 128
#define SOLVE_COLUMNS 32

/* The columns a recursive routine on [n] computes first: half, a
   multiple of 8 from 16 on (ReLAPACK's split). */
static int64_t split(int64_t n) { return n >= 16 ? (n + 8) / 16 * 8 : n / 2; }

/* The most operands and results of a routine. */
#define ARRAYS 5

/* A worker's buffer, grown as its matrices need; [copy], the worker's
   buffer for a solve's copies; [workers], every worker's where a matrix
   has the call's threads, else NULL. */
typedef struct arena {
  void *p;
  int64_t n;
  struct arena *copy, *workers;
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
enum { LU, CHOLESKY, QR, SOLVE_TRIANGULAR, EIGH };

/* A triangular solve: x ← t⁻¹ x for the triangle t, its diagonal read
   or taken as ones; or x ← x l⁻ᴴ for the lower l. */
typedef enum { LOWER, UPPER, UNIT_LOWER, UNIT_UPPER, ADJOINT_LOWER } solve;

/* A call: its routine and attributes, 0 where it has none: [upper] for the
   triangle Upper, [complete] for complete factors, which q's shape states,
   [adjoint] and [unit] for a solve's, [vectors] where eigh computes v; its operand's [m] rows and [n]
   columns and, for a solve, the right-hand sides' count [r]; its arrays,
   the operands then the results, whose first [batch] axes are the batch. */
typedef struct {
  int routine, dt, w, upper, complete, adjoint, unit, vectors;
  int64_t m, n, r, matrices, per; /* per unit of the job */
  int nops, nres, batch;
  const nx_array *a;
  int64_t scratch; /* elements per worker */
  arena *ws, *packs, *copies; /* per worker: the routine's, the products',
                                 the solves' */
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

/* Copies matrix [b] of operand [k] into [d], rows [ld] apart, transposed
   under [transposed]. */
static void stage(const call *c, int k, int64_t b, void *d, int64_t ld,
                  int transposed) {
  const nx_array *x = &c->a[k];
  int r = x->rank;
  int64_t rows = x->dim[r - 2], cols = x->dim[r - 1];
  int64_t sr = x->dim[2 * r - 2], sc = x->dim[2 * r - 1];
  if (transposed) {
    int64_t t = rows;
    rows = cols;
    cols = t;
    t = sr;
    sr = sc;
    sc = t;
  }
  nx_box box = {.extent = {1, rows, cols},
                .first = {0, 0},
                .step = {{0, ld, 1}, {0, sr, sc}}};
  if (rows > 0 && cols > 0)
    nx_copy_box(d, matrix_at(c, k, b), &box, x->bits);
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
   float64, by the contraction (gemm.c), c its own init. Answers 0 if an
   allocation failed. */
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
  return nx_cpu_contract_on(dt, &v, op);
}

/* Sets the [n] reals of the dtype [dt] at [p] to NaN: a failed matrix's
   result. */
static void nans(void *p, int64_t n, int dt) {
  if (dt == NX_FLOAT32)
    for (int64_t i = 0; i < n; i++) ((float *)p)[i] = NAN;
  else
    for (int64_t i = 0; i < n; i++) ((double *)p)[i] = NAN;
}

/* The routines, per dtype */

#define CAT2(a, b) a##_##b
#define CAT(a, b) CAT2(a, b)
#define F(name) CAT(name, SFX)

#define T float
#define R float
#define RDT NX_FLOAT32
#define SFX f32
#define CPLX 0
#define CONJ(x) (x)
#define REAL(x) (x)
#define IMAG(x) ((R)0)
#define SQRT sqrtf
#include "linalg_body.h"

#define T double
#define R double
#define RDT NX_FLOAT64
#define SFX f64
#define CPLX 0
#define CONJ(x) (x)
#define REAL(x) (x)
#define IMAG(x) ((R)0)
#define SQRT sqrt
#include "linalg_body.h"

#define T float complex
#define R float
#define RDT NX_FLOAT32
#define SFX c64
#define CPLX 1
#define CONJ conjf
#define REAL crealf
#define IMAG cimagf
#define SQRT sqrtf
#include "linalg_body.h"

#define T double complex
#define R double
#define RDT NX_FLOAT64
#define SFX c128
#define CPLX 1
#define CONJ conj
#define REAL creal
#define IMAG cimag
#define SQRT sqrt
#include "linalg_body.h"

/* The job */

/* Units [lo, hi): matrices [u·per, (u + 1)·per) of each. */
static void matrices(int64_t lo, int64_t hi, int worker, void *ctx) {
  call *c = ctx;
  arena *s = &c->packs[worker];
  void *w = room(&c->ws[worker], (c->scratch > 0 ? c->scratch : 1) * c->w);
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
}

/* The elements of scratch a matrix of [c] takes besides its products':
   a solve's triangle; QR's factored matrix, its τ, a block's V and T,
   two products of a block and a row; eigh's reduced matrix, its τ, a
   panel's W or a block's V, T, two products of a block, then as reals
   T's diagonals, Z and divide and conquer's work, and its indices
   (linalg_body.h). */
static int64_t scratch(const call *c) {
  int64_t m = c->m, n = c->n, big = m > n ? m : n, k = m < n ? m : n;
  switch (c->routine) {
    case SOLVE_TRIANGULAR: return n * n;
    case QR: return m * n + k + (m + NB + 2 * big + 1) * NB;
    case EIGH: return 4 * n * n + 16 * n + 2 + (3 * n + NB) * NB;
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
  int square = c->routine != LU && c->routine != QR;
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
    case EIGH: ranks[0] = xr - 1; return 1;
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

/* Whether the array [v] has the shape [s] of rank [r]. */
static int has_shape(value v, const int64_t *s, int r) {
  int64_t d[2 * NX_MAX_RANK], off;
  if (nx_array_layout(v, d, &off) != r) return 0;
  for (int i = 0; i < r; i++)
    if (d[i] != s[i]) return 0;
  return 1;
}

/* Runs [c], its routine and attributes set, on [arrays], its operands then
   its results, on [threads] threads as cpu.h states. The caller roots the
   arrays: they are read before nx_read, which may run OCaml code. */
static value run(call c, const value *arrays, int threads) {
  int solve = c.routine == SOLVE_TRIANGULAR;
  c.nops = 1 + solve;
  c.nres = c.routine == LU                         ? 3
           : c.routine == QR || (c.routine == EIGH && c.vectors) ? 2
                                                                 : 1;
  c.dt = nx_array_dtype(arrays[0]);
  if (!takes(c.dt)) return Val_int(NX_DTYPE);
  int64_t x[2 * NX_MAX_RANK], z[2 * NX_MAX_RANK], y[3][NX_MAX_RANK], off;
  int ranks[3];
  int xr = nx_array_layout(arrays[0], x, &off);
  int zr = solve ? nx_array_layout(arrays[1], z, &off) : 0;
  if (c.routine == QR) c.complete = complete(arrays[1], x, xr);
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
  int64_t order = c.m > c.n ? c.m : c.n;
  /* eigh's flops: about 9 n³ with vectors, 4/3 n³ without (xHEEV). */
  int64_t per = solve                 ? c.r
                : c.routine != EIGH ? order / 3 + 1
                : c.vectors           ? 9 * order
                                      : 4 * order / 3 + 1;
  int64_t flops = c.matrices * c.m * c.n * per;
  int64_t bytes = c.matrices * c.m * (c.n + c.r) * c.w;
  /* A matrix a unit where there are enough for the threads, else one unit
     of all, whose GEMM updates take the threads. */
  c.per = c.matrices >= nx_cpu_threads(bytes, bytes + flops) ? 1 : c.matrices;
  int cores = rig_pool_cores();
  c.ws = calloc((size_t)cores, sizeof(arena));
  c.packs = calloc((size_t)cores, sizeof(arena));
  c.copies = calloc((size_t)cores, sizeof(arena));
  if (c.ws == NULL || c.packs == NULL || c.copies == NULL) {
    free(c.ws);
    free(c.packs);
    free(c.copies);
    nx_done(n, a);
    caml_raise_out_of_memory();
  }
  for (int i = 0; i < cores; i++) {
    c.packs[i].copy = &c.copies[i];
    if (c.per == c.matrices && threads != 1) c.packs[i].workers = c.packs;
  }
  if (c.matrices > 0) {
    int64_t units = c.matrices / c.per;
    if (threads == 1) matrices(0, units, 0, &c);
    else nx_cpu_job(units, bytes, bytes + flops, matrices, &c);
  }
  for (int i = 0; i < cores; i++) {
    free(c.ws[i].p);
    free(c.packs[i].p);
    free(c.copies[i].p);
  }
  free(c.ws);
  free(c.packs);
  free(c.copies);
  nx_done(n, a);
  if (c.failed) caml_raise_out_of_memory();
  return Val_int(NX_OK);
}

value nx_cpu_cholesky_on(int upper, value dst, value a, int threads) {
  CAMLparam2(dst, a);
  value arrays[] = {a, dst};
  CAMLreturn(run((call){.routine = CHOLESKY, .upper = upper}, arrays, threads));
}

value nx_cpu_lu_on(value lu, value pivots, value perm, value a, int threads) {
  CAMLparam4(lu, pivots, perm, a);
  value arrays[] = {a, lu, pivots, perm};
  CAMLreturn(run((call){.routine = LU}, arrays, threads));
}

value nx_cpu_qr_on(value q, value r, value a, int threads) {
  CAMLparam3(q, r, a);
  value arrays[] = {a, q, r};
  CAMLreturn(run((call){.routine = QR}, arrays, threads));
}

value nx_cpu_solve_triangular_on(int upper, int adjoint, int unit, value dst,
                                 value a, value b, int threads) {
  CAMLparam3(dst, a, b);
  value arrays[] = {a, b, dst};
  call c = {.routine = SOLVE_TRIANGULAR,
            .upper = upper,
            .adjoint = adjoint,
            .unit = unit};
  CAMLreturn(run(c, arrays, threads));
}

value nx_cpu_eigh_on(value w, value v, value a, int threads) {
  CAMLparam3(w, v, a);
  int vectors = Is_some(v);
  value arrays[] = {a, w, vectors ? Some_val(v) : Val_unit};
  CAMLreturn(run((call){.routine = EIGH, .vectors = vectors}, arrays, threads));
}

value nx_cpu_cholesky(value triangle, value dst, value a) {
  return nx_cpu_cholesky_on(Int_val(triangle) == NX_TRIANGLE_UPPER, dst, a,
                            0);
}

value nx_cpu_lu(value lu, value pivots, value perm, value a) {
  return nx_cpu_lu_on(lu, pivots, perm, a, 0);
}

value nx_cpu_qr(value q, value r, value a) {
  return nx_cpu_qr_on(q, r, a, 0);
}

value nx_cpu_solve_triangular(value triangle, value adjoint, value unit,
                              value dst, value a, value b) {
  return nx_cpu_solve_triangular_on(Int_val(triangle) == NX_TRIANGLE_UPPER,
                                    Bool_val(adjoint), Bool_val(unit), dst, a,
                                    b, 0);
}

value nx_cpu_solve_triangular_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_solve_triangular(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

value nx_cpu_eigh(value w, value v, value a) {
  return nx_cpu_eigh_on(w, v, a, 0);
}
