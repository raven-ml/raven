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

/* linalg.c's routines on matrices of one dtype, included once per dtype
   with T the element, R its real part's type, RDT that real type's dtype
   code, CPLX 1 for complex, C the complex type of R's precision and CK
   its kind in nx_kinds.h, CSQRT its square root on complex dtypes, WORD
   the signed integer of R's width, and CONJ, REAL, IMAG, SQRT, COPYSIGN,
   LDEXP and ILOGB for T and R, which it undefines at its end. F(name)
   names the dtype's instance.

   Matrices are dense and row-major, row i at a + i·ld. Each routine runs
   by scalar loops on blocks of at most LEAF columns and updates the rest
   of the matrix by products (F(gemm)). */

/* The unit roundoff, the least normal number and the least positive
   subnormal. */
#define EPS ((R)(sizeof(R) == 4 ? FLT_EPSILON / 2 : DBL_EPSILON / 2))
#define TINY ((R)(sizeof(R) == 4 ? FLT_MIN : DBL_MIN))
#define ETA ((R)(sizeof(R) == 4 ? FLT_TRUE_MIN : DBL_TRUE_MIN))

static R F(abs)(R x) { return x < 0 ? -x : x; }

/* Element [x]'s share of a lane of scan: x − x into [*nan], 0 while the
   lane's elements are finite and NaN once one is not, and its largest
   part's magnitude into [*most]. A magnitude is max(x, −x), which
   compilers make a maximum where abs's x < 0 ? −x : x stays a select;
   the two differ only on −0, whose magnitude is 0 either way. */
static inline void F(scan_one)(T x, R *nan, R *most) {
  R re = REAL(x), im = IMAG(x);
  R ar = re > -re ? re : -re, ai = im > -im ? im : -im;
  R a = ar > ai ? ar : ai;
  *nan += (re - re) + (im - im);
  *most = a > *most ? a : *most;
}

/* Whether the [n] elements at [x] are all finite, the largest magnitude
   of their parts into [*big] where it is larger. In eight lanes by the
   index mod 8, whose largest is the same in any order: no comparison
   waits on the one before, where a single running maximum waits a
   comparison and a select an element. */
static int F(scan)(const T *x, int64_t n, R *big) {
  R nan[8] = {0}, most[8] = {0};
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    for (int l = 0; l < 8; l++) F(scan_one)(x[i + l], &nan[l], &most[l]);
  for (int l = 0; i < n; i++, l++) F(scan_one)(x[i], &nan[l], &most[l]);
  int finite = 1;
  for (int l = 0; l < 8; l++) {
    finite &= nan[l] == 0;
    *big = most[l] > *big ? most[l] : *big;
  }
  return finite;
}

/* The complex number [re] + i [im] of R's precision, whatever its parts. */
static inline C F(cplx)(R re, R im) {
  C z;
  ((R *)&z)[0] = re;
  ((R *)&z)[1] = im;
  return z;
}

/* a b and a / b as nx_kinds.h's Mul and Fdiv, nx.cpu's elementwise
   ones: the textbook product, each part one fma over the other product,
   and Smith's quotient. C's operators would recover
   infinities from NaN parts (C99 Annex G) by calls that keep every loop
   scalar. MUL and DIV are T's: these on complex dtypes, C's operators on
   reals. */
static inline CAT(nx, CK) F(kind)(C a) {
  return CAT(CAT(nx, CK), of)(((R *)&a)[0], ((R *)&a)[1]);
}

static inline C F(cmul)(C a, C b) {
  CAT(nx, CK) z = CAT(nx_mul, CK)(F(kind)(a), F(kind)(b));
  return F(cplx)(z.re, z.im);
}

static inline C F(cdiv)(C a, C b) {
  CAT(nx, CK) z = CAT(nx_fdiv, CK)(F(kind)(a), F(kind)(b));
  return F(cplx)(z.re, z.im);
}

#if CPLX
#define MUL(a, b) F(cmul)(a, b)
#define DIV(a, b) F(cdiv)(a, b)
#else
#define MUL(a, b) ((a) * (b))
#define DIV(a, b) ((a) / (b))
#endif

/* Products */

#if !CPLX
/* w ← −x for the [r] × [c] x, element (i, j) at x[i·rs + j·cs], w's rows
   [c] long. A unit is a block of 8 rows, copied row by row, or column by
   column where x's rows lie adjacent, as a transposed operand's do, so
   that x is read a line at a time, once. */
typedef struct {
  const T *x;
  int64_t r, c, rs, cs;
  T *w;
} F(negation);

static void F(negate)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const F(negation) *g = ctx;
  const T *x = g->x;
  int64_t c = g->c, rs = g->rs, cs = g->cs;
  T *w = g->w;
  for (int64_t i0 = 8 * lo; i0 < 8 * hi && i0 < g->r; i0 += 8) {
    int64_t i1 = i0 + 8 < g->r ? i0 + 8 : g->r;
    if (rs != 1 || cs == 1) {
      for (int64_t i = i0; i < i1; i++)
        for (int64_t j = 0; j < c; j++) w[i * c + j] = -x[i * rs + j * cs];
      continue;
    }
    for (int64_t j = 0; j < c; j++)
      for (int64_t i = i0; i < i1; i++) w[i * c + j] = -x[i + j * cs];
  }
}

/* The negated copy of the [r] × [c] [x] into [w], on the call's threads
   where [s] holds them (copying). */
static void F(negated)(const T *x, int64_t r, int64_t c, int64_t rs,
                       int64_t cs, T *w, const arena *s) {
  F(negation) g = {.x = x, .r = r, .c = c, .rs = rs, .cs = cs, .w = w};
  copying(s, (r + 7) / 8, r * c * (int64_t)sizeof(T), F(negate), &g);
}
#else
/* A complex product's copies: a's [m] rows of [k] where [wa] takes them,
   then sign · op(b)'s [k] rows of [n] expanded into [bt] (gemm). A unit
   is a row. */
typedef struct {
  operand a, b;
  int64_t m, n, k;
  R sign;
  T *wa;
  R *bt;
} F(expansion);

static void F(expand)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const F(expansion) *g = ctx;
  const T *ap = g->a.p, *bp = g->b.p;
  int64_t ma = g->wa != NULL ? g->m : 0, n = g->n, k = g->k;
  for (int64_t u = lo; u < hi; u++) {
    if (u < ma) {
      for (int64_t p = 0; p < k; p++) {
        T x = ap[u * g->a.rs + p * g->a.cs];
        g->wa[u * k + p] = g->a.conj ? CONJ(x) : x;
      }
      continue;
    }
    int64_t p = u - ma;
    R *r0 = g->bt + 2 * p * 2 * n, *r1 = r0 + 2 * n;
    for (int64_t j = 0; j < n; j++) {
      T z = bp[p * g->b.rs + j * g->b.cs];
      R x = g->sign * REAL(z);
      R y = g->sign * (g->b.conj ? -IMAG(z) : IMAG(z));
      r0[2 * j] = x;
      r0[2 * j + 1] = y;
      r1[2 * j] = -y;
      r1[2 * j + 1] = x;
    }
  }
}
#endif

/* c += sign · op(a) op(b) for c of [m] rows and [n] columns, rows [ldc]
   apart and its columns adjacent, and [k] products an element, by the
   contraction (product). Answers 0 if an allocation failed.

   On reals, the smaller of a and b is copied negated where sign is -1.
   On complex numbers, the product is a real one, with no more flops (the
   1m method, Van Zee, 2020): a's [m] rows of [k] complex numbers are [m]
   rows of [2k] reals, and c's are [m] rows of [2n] reals,
   where b is copied into [2k] rows of [2n] reals, sign · op(b)'s element
   (p, j) = x + iy as

     row 2p:     … x  y …        so that a's re·x + im·(−y) is c's real
     row 2p + 1: … −y x …        part and re·y + im·x its imaginary one.

   a is read in place where its complex numbers lie adjacent along k and
   unconjugated, and copied otherwise. The copies run on the call's
   threads where [s] holds them. */
static int F(gemm)(int64_t m, int64_t n, int64_t k, int sign, operand a,
                   operand b, T *c, int64_t ldc, arena *s) {
  if (m == 0 || n == 0 || k == 0) return 1;
#if !CPLX
  const T *ap = a.p, *bp = b.p;
  int64_t ars = a.rs, acs = a.cs, brs = b.rs, bcs = b.cs;
  if (sign < 0 && m * k <= k * n) {
    T *w = room(s, m * k * (int64_t)sizeof(T));
    if (w == NULL) return 0;
    F(negated)(ap, m, k, ars, acs, w, s);
    ap = w;
    ars = k;
    acs = 1;
  } else if (sign < 0) {
    T *w = room(s, k * n * (int64_t)sizeof(T));
    if (w == NULL) return 0;
    F(negated)(bp, k, n, brs, bcs, w, s);
    bp = w;
    brs = n;
    bcs = 1;
  }
  return product(RDT, m, n, k, (strided){ap, ars, acs},
                 (strided){bp, brs, bcs}, c, ldc);
#else
  int copy_a = a.cs != 1 || a.conj;
  int64_t na = copy_a ? m * k : 0;
  T *w = room(s, (na + 2 * k * n) * (int64_t)sizeof(T));
  if (w == NULL) return 0;
  F(expansion) g = {.a = a, .b = b, .m = m, .n = n, .k = k, .sign = sign,
                    .wa = copy_a ? w : NULL, .bt = (R *)(w + na)};
  copying(s, (copy_a ? m : 0) + k, (na + 2 * k * n) * (int64_t)sizeof(T),
          F(expand), &g);
  return product(RDT, m, 2 * n, 2 * k,
                 (strided){copy_a ? w : a.p, 2 * (copy_a ? k : a.rs), 1},
                 (strided){g.bt, 2 * n, 1}, c, 2 * ldc);
#endif
}

/* The operand of [x], rows [rs] and columns [cs] apart. */
static operand F(op)(const T *x, int64_t rs, int64_t cs) {
  return (operand){x, rs, cs, 0};
}

/* The conjugate transpose of [x], rows [ld] apart. */
static operand F(op_h)(const T *x, int64_t ld) {
  return (operand){x, 1, ld, CPLX};
}

/* The [r] elements at [x] divided by [d]: by its real part where its
   imaginary part is zero, as Cholesky's diagonal is, which rounds once. */
static void F(divide)(T *x, int64_t r, T d) {
#if CPLX
  if (IMAG(d) == 0) {
    R e = REAL(d);
    for (int64_t j = 0; j < r; j++) x[j] /= e;
    return;
  }
#endif
  for (int64_t j = 0; j < r; j++) x[j] = DIV(x[j], d);
}

/* A leaf's columns: the [h] × [n] [a], rows [ld] apart, copied out into
   [q], column c's real parts at q[c·hw], then on complex dtypes its
   imaginary parts lp = h + PAD after, hw = lp or 2 lp, so that a column
   streams and complex products run on reals in a vector's lanes. In
   place, each of a column's elements is a line of its own, rows a power
   of two apart sharing one set of the cache. Reals are copied by
   nx_copy_box's square blocks; complex numbers row by row, each row's
   parts into its columns' two, in one pass. [q] holds n·hw reals. */
static int64_t F(leaf_pitch)(int64_t h) { return (CPLX + 1) * (h + PAD); }

static void F(leaf_in)(R *q, const T *a, int64_t h, int64_t n, int64_t ld) {
  int64_t hw = F(leaf_pitch)(h);
#if CPLX
  int64_t lp = h + PAD;
  for (int64_t i = 0; i < h; i++) {
    const R *r = (const R *)(a + i * ld);
    for (int64_t c = 0; c < n; c++) {
      q[c * hw + i] = r[2 * c];
      q[c * hw + lp + i] = r[2 * c + 1];
    }
  }
#else
  nx_box in = {.extent = {1, h, n}, .step = {{0, 1, hw}, {0, ld, 1}}};
  nx_copy_box((uint8_t *)q, (const uint8_t *)a, &in, 8 * (int)sizeof(T));
#endif
}

static void F(leaf_out)(T *a, const R *q, int64_t h, int64_t n,
                        int64_t ld) {
  int64_t hw = F(leaf_pitch)(h);
#if CPLX
  int64_t lp = h + PAD;
  for (int64_t i = 0; i < h; i++) {
    R *r = (R *)(a + i * ld);
    for (int64_t c = 0; c < n; c++) {
      r[2 * c] = q[c * hw + i];
      r[2 * c + 1] = q[c * hw + lp + i];
    }
  }
#else
  nx_box out = {.extent = {1, h, n}, .step = {{0, ld, 1}, {0, 1, hw}}};
  nx_copy_box((uint8_t *)a, (const uint8_t *)q, &out, 8 * (int)sizeof(T));
#endif
}

/* Triangular solves

   x ← l⁻¹ x for the lower triangular n × n [l] and the n × r [x], or
   u⁻¹ x for the upper [u], in place: halves of l down to LEAF, as
   LAPACK's xTRSM's left-side cases run blocked (ReLAPACK's recursive
   splitting). The lower solve: x1 of l11, then x2 of l22 and
   x2 − l21 x1; the upper one from its last half. At a leaf, substitution
   row by row in the direction of the solve,

     x[i] = (x[i] − Σ_{p before i} l[i][p] x[p]) / l[i][i],

   the sum in the order of p, with no division under [unit]: on reals by
   the target's gemv, each product fused, the rows of x before i the
   columns of its matrix; on complex numbers by their parts, as C's
   product rounds them. Answers 1, or 0 if an allocation failed. */

/* x[i] − Σ l[i][p] x[p] over the [k] rows p of x from [x0], [l] the
   first of their k elements of l's row i. */
static void F(substitute)(const T *l, const T *x0, int64_t k, int64_t ldx,
                          T *xi, int64_t r) {
#if CPLX
  R *y = (R *)xi;
  for (int64_t p = 0; p < k; p++) {
    R cr = REAL(l[p]), ci = IMAG(l[p]);
    const R *z = (const R *)(x0 + p * ldx);
    for (int64_t j = 0; j < 2 * r; j += 2) {
      R zr = z[j], zi = z[j + 1];
      y[j] -= cr * zr - ci * zi;
      y[j + 1] -= cr * zi + ci * zr;
    }
  }
#else
  R nl[LEAF];
  for (int64_t p = 0; p < k; p++) nl[p] = -l[p];
  if (k > 0) nx_cpu_table->gemv[RDT](r, k, x0, ldx, nl, xi);
#endif
}

static int F(lower)(const T *l, int64_t n, int64_t ld, T *x, int64_t r,
                    int64_t ldx, int unit, arena *s) {
  if (n <= LEAF) {
    for (int64_t i = 0; i < n; i++) {
      T *xi = x + i * ldx;
      F(substitute)(l + i * ld, x, i, ldx, xi, r);
      if (!unit) F(divide)(xi, r, l[i * ld + i]);
    }
    return 1;
  }
  int64_t n1 = split(n), n2 = n - n1;
  T *x2 = x + n1 * ldx;
  return F(lower)(l, n1, ld, x, r, ldx, unit, s) &&
         F(gemm)(n2, r, n1, -1, F(op)(l + n1 * ld, ld, 1), F(op)(x, ldx, 1),
                 x2, ldx, s) &&
         F(lower)(l + n1 * ld + n1, n2, ld, x2, r, ldx, unit, s);
}

static int F(upper)(const T *u, int64_t n, int64_t ld, T *x, int64_t r,
                    int64_t ldx, int unit, arena *s) {
  if (n <= LEAF) {
    for (int64_t i = n - 1; i >= 0; i--) {
      T *xi = x + i * ldx;
      F(substitute)(u + i * ld + i + 1, xi + ldx, n - i - 1, ldx, xi, r);
      if (!unit) F(divide)(xi, r, u[i * ld + i]);
    }
    return 1;
  }
  int64_t n1 = split(n), n2 = n - n1;
  T *x2 = x + n1 * ldx;
  return F(upper)(u + n1 * ld + n1, n2, ld, x2, r, ldx, unit, s) &&
         F(gemm)(n1, r, n2, -1, F(op)(u + n1, ld, 1), F(op)(x2, ldx, 1), x,
                 ldx, s) &&
         F(upper)(u, n1, ld, x, r, ldx, unit, s);
}

/* x ← x L⁻ᴴ for the r × n [x], rows [ldx] apart, and the lower L of
   order n at [l], rows [ld] apart, its diagonal real, as Cholesky's: as
   ReLAPACK's recursive xTRSM, x1 of L11, then x2 − x1 L21ᴴ by a product,
   then x2 of L22. A leaf of at most LEAF columns is copied out column by
   column (leaf_in), each column then taking those to its left by the
   target's gemv, each product fused, and divided by the diagonal's real
   part. On complex numbers, re y takes re x (−re l), then im x (−im l),
   and im y takes re x im l, then im x (−re l). */
static int F(rleaf)(const T *l, int64_t n, int64_t ld, T *x, int64_t r,
                    int64_t ldx, arena *s) {
  int64_t lp = r + PAD, hw = F(leaf_pitch)(r);
  R *q = room(s, n * hw * (int64_t)sizeof(R));
  if (q == NULL) return 0;
  nx_cpu_gemv gemv = nx_cpu_table->gemv[RDT];
  F(leaf_in)(q, x, r, n, ldx);
#if CPLX
  R ur[2 * LEAF], ui[2 * LEAF];
#else
  R nu[LEAF];
#endif
  for (int64_t p = 0; p < n; p++) {
    R *y = q + p * hw, d = REAL(l[p * ld + p]);
    for (int64_t c = 0; c < p; c++) {
#if CPLX
      R re = REAL(l[p * ld + c]), im = IMAG(l[p * ld + c]);
      ur[2 * c] = ui[2 * c + 1] = -re;
      ur[2 * c + 1] = -im;
      ui[2 * c] = im;
#else
      nu[c] = -l[p * ld + c];
#endif
    }
#if CPLX
    if (p > 0) {
      gemv(r, 2 * p, q, lp, ur, y);
      gemv(r, 2 * p, q, lp, ui, y + lp);
    }
    for (int64_t i = 0; i < r; i++) y[lp + i] /= d;
#else
    if (p > 0) gemv(r, p, q, lp, nu, y);
#endif
    for (int64_t i = 0; i < r; i++) y[i] /= d;
  }
  F(leaf_out)(x, q, r, n, ldx);
  return 1;
}

static int F(rband)(const T *l, int64_t n, int64_t ld, T *x, int64_t r,
                    int64_t ldx, arena *s) {
  if (n <= LEAF) return F(rleaf)(l, n, ld, x, r, ldx, s);
  int64_t n1 = split(n), n2 = n - n1;
  return F(rband)(l, n1, ld, x, r, ldx, s) &&
         F(gemm)(r, n2, n1, -1, F(op)(x, ldx, 1), F(op_h)(l + n1 * ld, ld),
                 x + n1, ldx, s) &&
         F(rband)(l + n1 * ld + n1, n2, ld, x + n1, r, ldx, s);
}

/* A solve by parts of x, which are independent, on the call's threads:
   units of SOLVE_COLUMNS columns at least, one a thread, of rows under
   ADJOINT_LOWER, each solved whole. Every product adds in chain order
   (product), whatever its shape, so each column's or row's arithmetic
   and bits are the whole solve's, whatever the units and threads. */
typedef struct {
  const T *l;
  T *x, *neg;
  int64_t n, ld, r, ldx, units;
  solve kind;
  int failed;
  arena *workers;
} F(solve_job);

/* Columns [c0, c0 + c) of x, or its rows under ADJOINT_LOWER. */
static int F(part)(const F(solve_job) *j, int64_t c0, int64_t c, arena *s) {
  int64_t n = j->n, ld = j->ld, ldx = j->ldx;
  switch (j->kind) {
    case LOWER: return F(lower)(j->l, n, ld, j->x + c0, c, ldx, 0, s);
    case UNIT_LOWER: return F(lower)(j->l, n, ld, j->x + c0, c, ldx, 1, s);
    case UPPER: return F(upper)(j->l, n, ld, j->x + c0, c, ldx, 0, s);
    case UNIT_UPPER: return F(upper)(j->l, n, ld, j->x + c0, c, ldx, 1, s);
    case ADJOINT_LOWER: break;
  }
  T *x = j->x + c0 * ldx;
  if (!F(rband)(j->l, n, ld, x, c, ldx, s)) return 0;
  for (int64_t i = 0; j->neg != NULL && i < c; i++)
    for (int64_t p = 0; p < n; p++) j->neg[(c0 + i) * n + p] = -x[i * ldx + p];
  return 1;
}

/* A unit runs on its worker's arena alone, its own work starting no job:
   a nested job's units would share worker 0's arena with the thread that
   holds it. LU's steps' units do the same. */
static void F(solve_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(solve_job) *j = ctx;
  arena *s = &j->workers[worker], *all = s->workers;
  s->workers = NULL;
  for (int64_t u = lo; u < hi; u++) {
    int64_t c0 = u * j->r / j->units, c1 = (u + 1) * j->r / j->units;
    if (!F(part)(j, c0, c1 - c0, s))
      __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
  }
  s->workers = all;
}

/* The solve [kind] by the n × n triangle [l] of the n × r [x], or of the
   r × n [x] under ADJOINT_LOWER, rows [ldx] apart, and then, where [neg]
   is not NULL, −x into [neg], rows n apart: by units where [s] holds the
   call's threads, else whole. Answers 0 if an allocation failed. */
static int F(solve)(solve kind, const T *l, int64_t n, int64_t ld, T *x,
                    int64_t r, int64_t ldx, T *neg, arena *s) {
  int64_t bytes = n * (n + r) * (int64_t)sizeof(T), units = r / SOLVE_COLUMNS;
  double cost = threaded(s, units, bytes, (double)n * n * r, RDT, CPLX);
  F(solve_job) j = {.l = l, .x = x, .neg = neg, .n = n, .ld = ld, .r = r,
                    .ldx = ldx, .kind = kind, .workers = s->workers};
  if (cost == 0) return F(part)(&j, 0, r, s);
  int threads = nx_cpu_threads(bytes, cost);
  j.units = units < threads ? units : threads;
  nx_cpu_job(j.units, bytes, cost, F(solve_unit), &j);
  return !j.failed;
}

/* Cholesky

   L with a = L Lᴴ, in place in a's lower triangle, right-looking by
   blocks of CHOL_NB columns: block k's diagonal block factored, then its
   rows below solved by L_kkᴴ (solve), which also copies them negated
   into N, then the trailing lower triangle updated by tiles of
   CHOL_NB × CHOL_NB, each c += N_i L_jᴴ from the diagonal down, so that
   no product copies an operand negated. A job on the call's threads
   runs the tiles, its first unit the next diagonal block's, which it
   then factors, beside the others' tiles. Each element takes the
   blocks' updates in their order, each a product in chain order, so no
   tiling or thread changes its bits. A diagonal block is LAPACK's
   xPOTF2 by columns, each taking those to its left at once,

     L[i][c] = (a[i][c] − Σ_{p < c} L[i][p] conj L[c][p]) / L[c][c]
     L[c][c] = sqrt (a[c][c] − Σ_{p < c} |L[c][p]|²),

   on its columns copied out (leaf_in), by the target's gemv, each
   product fused: on complex numbers re y takes re L (−re l) then
   im L (−im l), and im y takes re L im l then im L (−re l), l = L[c][p].
   A column's rows run on to a multiple of PAD, into its padding, zeroed,
   so that the gemv's last vector is whole: there, one element at a time
   waits on each of its k fused multiply-adds.

   Only the lower triangle and the diagonal's real parts are read. Answers
   1, 0 if a pivot is not positive (NaN included), -1 if an allocation
   failed. */
static int F(potf2)(T *a, int64_t n, int64_t ld, arena *s) {
  if (n == 0) return 1;
  int64_t lp = n + PAD, hw = F(leaf_pitch)(n);
  R *q = room(s, n * hw * (int64_t)sizeof(R));
  if (q == NULL) return -1;
  nx_cpu_gemv gemv = nx_cpu_table->gemv[RDT];
  F(leaf_in)(q, a, n, n, ld);
  for (int64_t c = 0; c < n; c++)
    for (int64_t h = 0; h <= CPLX; h++)
      memset(q + c * hw + h * lp + n, 0, PAD * sizeof(R));
#if CPLX
  R ur[2 * CHOL_NB], ui[2 * CHOL_NB];
#else
  R nu[CHOL_NB];
#endif
  for (int64_t c = 0; c < n; c++) {
    R *y = q + c * hw;
    int64_t m = (n - c + PAD - 1) / PAD * PAD;
    for (int64_t p = 0; p < c; p++) {
      R re = q[p * hw + c];
#if CPLX
      R im = q[p * hw + lp + c];
      ur[2 * p] = ui[2 * p + 1] = -re;
      ur[2 * p + 1] = -im;
      ui[2 * p] = im;
#else
      nu[p] = -re;
#endif
    }
#if CPLX
    if (c > 0) {
      gemv(m, 2 * c, q + c, lp, ur, y + c);
      gemv(m, 2 * c, q + c, lp, ui, y + lp + c);
    }
    y[lp + c] = 0;
#else
    if (c > 0) gemv(m, c, q + c, lp, nu, y + c);
#endif
    if (!(y[c] > 0)) return 0;
    R d = SQRT(y[c]);
    y[c] = d;
    for (int64_t i = c + 1; i < n; i++) y[i] /= d;
#if CPLX
    for (int64_t i = c + 1; i < n; i++) y[lp + i] /= d;
#endif
  }
  F(leaf_out)(a, q, n, n, ld);
  return 1;
}

/* A step's update: the trailing lower triangle of order [h] at [c], by
   N's and L's [h] rows of [kb], its tiles a job's units, and the next
   diagonal block's factorisation's answer. */
typedef struct {
  const T *n, *l;
  T *c;
  int64_t ld, h, kb;
  int next, failed;
  arena *workers;
} F(trailing);

/* Tile [u] of a lower triangle of tiles, row by row: (i, j), j <= i. */
static void F(tile_of)(int64_t u, int64_t *i, int64_t *j) {
  int64_t r = 0;
  while ((r + 1) * (r + 2) / 2 <= u) r++;
  *i = r;
  *j = u - r * (r + 1) / 2;
}

/* Tiles [lo, hi) on the arena [s], tile 0's block factored after it. */
static void F(tiles)(F(trailing) *t, int64_t lo, int64_t hi, arena *s) {
  for (int64_t u = lo; u < hi; u++) {
    int64_t i, j;
    F(tile_of)(u, &i, &j);
    int64_t r0 = i * CHOL_NB, c0 = j * CHOL_NB;
    int64_t m = t->h - r0 < CHOL_NB ? t->h - r0 : CHOL_NB;
    int64_t n = t->h - c0 < CHOL_NB ? t->h - c0 : CHOL_NB;
    T *c = t->c + r0 * t->ld + c0;
    if (!F(gemm)(m, n, t->kb, 1, F(op)(t->n + r0 * t->kb, t->kb, 1),
                 F(op_h)(t->l + c0 * t->ld, t->ld), c, t->ld, s))
      __atomic_store_n(&t->failed, 1, __ATOMIC_RELAXED);
    else if (u == 0)
      t->next = F(potf2)(c, m, t->ld, s);
  }
}

/* A unit runs on its worker's arena alone, as a solve's do. */
static void F(trailing_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(trailing) *t = ctx;
  arena *s = &t->workers[worker], *all = s->workers;
  s->workers = NULL;
  F(tiles)(t, lo, hi, s);
  s->workers = all;
}

static int F(cholesky)(T *a, int64_t n, int64_t ld, arena *s) {
  int ok = F(potf2)(a, n < CHOL_NB ? n : CHOL_NB, ld, s);
  for (int64_t k = 0; ok > 0 && k + CHOL_NB < n; k += CHOL_NB) {
    int64_t h = n - k - CHOL_NB, tiles = (h + CHOL_NB - 1) / CHOL_NB;
    T *akk = a + k * ld + k, *l = akk + CHOL_NB * ld;
    T *neg = room(s->copy, h * CHOL_NB * (int64_t)sizeof(T));
    if (neg == NULL ||
        !F(solve)(ADJOINT_LOWER, akk, CHOL_NB, ld, l, h, ld, neg, s))
      return -1;
    F(trailing) t = {.n = neg, .l = l, .c = l + CHOL_NB, .ld = ld, .h = h,
                     .kb = CHOL_NB, .workers = s->workers};
    int64_t units = tiles * (tiles + 1) / 2;
    int64_t bytes = (h * h / 2 + 2 * h * CHOL_NB) * (int64_t)sizeof(T);
    double cost =
        threaded(s, units, bytes, (double)h * h * CHOL_NB, RDT, CPLX);
    if (cost == 0) F(tiles)(&t, 0, units, s);
    else nx_cpu_job(units, bytes, cost, F(trailing_unit), &t);
    if (t.failed) return -1;
    ok = t.next;
  }
  return ok;
}

/* LU

   P a = L U with partial pivoting, in place in the m × n [a], by blocks
   of LU_NB or LU_NB / 2 columns (linalg.c), right-looking with a
   lookahead of one block: once block k is factored, block k + 1's
   columns take its update, k's interchanges, U12 = L11⁻¹ a12 and
   a22 − L21 U12, the product on the call's threads; then a job factors
   block k + 1 in its first unit while the others update the columns past
   it by tiles, so that the next panel's factorisation runs beside the
   update rather than after it. Every column takes its updates in the
   same order on every thread and tiling, so the bits do not depend on
   them. The columns left of a block take its interchanges at the end.

   A block's panel is factored as LAPACK's recursive xGETRF2 (ReLAPACK's
   xGETRF): it factors its left half, solves the top rows of its right
   half by L11, updates the rest of its right half by a22 − L21 U12, then
   factors that. Interchanges swap the block's rows as they happen, so
   that its columns all see them. A leaf of at most LEAF columns runs
   column by column in Crout's order: a column takes the updates of those
   to its left, then its pivot is found by a scan of the column from the
   diagonal down that moves to each later row whose magnitude (|re| +
   |im|) is greater than the current pivot's, NaN never greater; its row
   swaps with the diagonal's; a nonzero pivot scales the column below it
   by its reciprocal where its magnitude lies in [TINY, 1 / TINY], so
   that the reciprocal is finite and not subnormal, and divides it
   otherwise. Columns past the first k = min m n take U = L⁻¹ a, unit
   lower L. [pivots] and [perm] take k and m positions. Answers 1, or 0
   if an allocation failed. */

static R F(magnitude)(T x) {
  R re = REAL(x), im = IMAG(x);
  return (re < 0 ? -re : re) + (im < 0 ? -im : im);
}

static void F(swap)(T *x, T *y, int64_t n) {
  for (int64_t j = 0; j < n; j++) {
    T t = x[j];
    x[j] = y[j];
    y[j] = t;
  }
}

/* A vector of base's reals, and as many words: a comparison's lanes. */
#define LANES (VECTOR_BYTES / (int)sizeof(R))
typedef R F(lanes) __attribute__((vector_size(VECTOR_BYTES)));
typedef WORD F(words) __attribute__((vector_size(VECTOR_BYTES)));

/* [v] = the LANES reals at [x] without their signs. */
static void F(abs_lanes)(const R *x, F(lanes) *v) {
  const F(words) bits = (F(words)){0} + (WORD)((~(uint64_t)0) >> 1 >>
                                               (64 - 8 * sizeof(R)));
  memcpy(v, x, sizeof *v);
  *v = (F(lanes))((F(words))*v & bits);
}

/* The greater of [v] and [top] in each lane, [top] where [v] is NaN. */
static void F(greater)(F(lanes) v, F(lanes) *top) {
  F(words) g = v > *top;
  *top = (F(lanes))(((F(words))v & g) | ((F(words))*top & ~g));
}

/* The magnitude of row [i] of the leaf's column [x], its parts [lp]
   apart, and [v] = those of rows [i, i + LANES), each sign cleared. */
#if CPLX
#define MAGNITUDE(x, lp, i) (F(abs)((x)[i]) + F(abs)((x)[(lp) + (i)]))
#define MAGNITUDES(v, x, lp, i)             \
  do {                                      \
    F(lanes) im_;                           \
    F(abs_lanes)((x) + (i), &(v));          \
    F(abs_lanes)((x) + (lp) + (i), &im_);   \
    (v) += im_;                             \
  } while (0)
#else
#define MAGNITUDE(x, lp, i) F(abs)((x)[i])
#define MAGNITUDES(v, x, lp, i) F(abs_lanes)((x) + (i), &(v))
#endif

/* The leaf of columns [j0, j0 + n), n at most LEAF, as panel's, on its
   columns copied out (leaf_in). A pivot is the greatest magnitude, taken
   in vectors' lanes, which no order changes as NaN is never taken, then
   the first row that holds it, sought a vector at a time (a sign cleared
   in a lane turns −0 into +0, which compares equal). On the i9-9900K's
   16 threads a scalar scan in its place took 1.04 of lu-f64-1024's time,
   1.05 of lu-f64-1024-pivoting's and 1.14 of lu-f32-1024's (minima of 6
   ABBA rounds).

   Each column takes the updates of the columns to its left when its turn
   comes, as Crout's order does, in their order: its rows above the
   diagonal one at a time, row r from those above it, then its rows from
   the diagonal down at once, by the target's gemv (cpu.h), each product
   fused, which holds them in registers while the columns stream. Above
   the diagonal, a gemv call a row took longer than its few products. On
   complex numbers each column to the left is two of reals: re y takes
   re l (−re u) then im l im u, and im y takes re l (−im u) then im l
   (−re u). */
static int F(crout)(T *a, int64_t m, int64_t b0, int64_t b1, int64_t ld,
                    int64_t j0, int64_t n, int64_t *pivots, int64_t *perm,
                    arena *s) {
  int64_t h = m - j0, lp = h + PAD, hw = F(leaf_pitch)(h);
  R *q = room(s, n * hw * (int64_t)sizeof(R));
  if (q == NULL) return 0;
  nx_cpu_gemv gemv = nx_cpu_table->gemv[RDT];
  T *a0 = a + j0 * ld + j0;
  F(leaf_in)(q, a0, h, n, ld);
  /* −u, column j's rows above the diagonal negated: on complex numbers
     re y's coefficients and im y's, two a column to the left. */
#if CPLX
  R ur[2 * LEAF], ui[2 * LEAF];
#else
  R nu[LEAF];
#endif
  for (int64_t j = 0; j < n; j++) {
    R *x = q + j * hw;
#if CPLX
    for (int64_t r = 0; r < j; r++) {
      for (int64_t u = 0; u < 2 * r; u++) {
        x[r] += q[u * lp + r] * ur[u];
        x[lp + r] += q[u * lp + r] * ui[u];
      }
      ur[2 * r] = ui[2 * r + 1] = -x[r];
      ur[2 * r + 1] = x[lp + r];
      ui[2 * r] = -x[lp + r];
    }
    if (j > 0) {
      gemv(h - j, 2 * j, q + j, lp, ur, x + j);
      gemv(h - j, 2 * j, q + j, lp, ui, x + lp + j);
    }
#else
    for (int64_t r = 0; r < j; r++) {
      for (int64_t p = 0; p < r; p++) x[r] += q[p * lp + r] * nu[p];
      nu[r] = -x[r];
    }
    if (j > 0) gemv(h - j, j, q + j, lp, nu, x + j);
#endif
    R best = MAGNITUDE(x, lp, j);
    int64_t p = j;
    if (best == best) {
      F(lanes) top0 = (F(lanes)){0} + best, top1 = top0, v0, v1;
      int64_t i = j + 1;
      for (; i + 2 * LANES <= h; i += 2 * LANES) {
        MAGNITUDES(v0, x, lp, i);
        MAGNITUDES(v1, x, lp, i + LANES);
        F(greater)(v0, &top0);
        F(greater)(v1, &top1);
      }
      for (; i < h; i++) {
        R v = MAGNITUDE(x, lp, i);
        best = v > best ? v : best;
      }
      for (int t = 0; t < LANES; t++) {
        best = top0[t] > best ? top0[t] : best;
        best = top1[t] > best ? top1[t] : best;
      }
      for (top0 = (F(lanes)){0} + best; p + LANES <= h; p += LANES) {
        MAGNITUDES(v0, x, lp, p);
        F(words) equal = v0 == top0;
        WORD any = 0;
        for (int t = 0; t < LANES; t++) any |= equal[t];
        if (any) break;
      }
      while (MAGNITUDE(x, lp, p) != best) p++;
    }
    pivots[j0 + j] = j0 + p;
    if (p != j) {
      F(swap)(a + (j0 + j) * ld + b0, a + (j0 + p) * ld + b0, b1 - b0);
      /* Each column's parts, lp reals each. */
      for (int64_t c = 0; c < hw * n; c += lp) {
        R t = q[c + j];
        q[c + j] = q[c + p];
        q[c + p] = t;
      }
      int64_t t = perm[j0 + j];
      perm[j0 + j] = perm[j0 + p];
      perm[j0 + p] = t;
    }
    int64_t k = h - j - 1;
    R *xr = x + j + 1;
#if CPLX
    R dr = x[j], di = x[lp + j], *xi = xr + lp;
    T d = F(cplx)(dr, di);
    R mag = F(abs)(dr) + F(abs)(di);
    if (mag >= TINY && mag <= 1 / TINY) {
      T r = DIV(1, d);
      R rr = REAL(r), ri = IMAG(r);
      for (int64_t i = 0; i < k; i++) {
        R u = xr[i], v = xi[i];
        xr[i] = u * rr - v * ri;
        xi[i] = u * ri + v * rr;
      }
    } else if (d != 0) {
      for (int64_t i = 0; i < k; i++) {
        T z = DIV(F(cplx)(xr[i], xi[i]), d);
        xr[i] = REAL(z);
        xi[i] = IMAG(z);
      }
    }
#else
    R d = x[j], mag = F(abs)(d);
    if (mag >= TINY && mag <= 1 / TINY) {
      R r = 1 / d;
      for (int64_t i = 0; i < k; i++) xr[i] *= r;
    } else if (d != 0) {
      for (int64_t i = 0; i < k; i++) xr[i] /= d;
    }
#endif
  }
  F(leaf_out)(a0, q, h, n, ld);
  return 1;
}

#undef MAGNITUDE
#undef MAGNITUDES
#undef LANES

/* The panel of columns [j0, j0 + n) from row j0 down to row [m], in the
   block of columns [b0, b1) of [a], whose rows its interchanges swap. */
static int F(panel)(T *a, int64_t m, int64_t b0, int64_t b1, int64_t ld,
                    int64_t j0, int64_t n, int64_t *pivots, int64_t *perm,
                    arena *s) {
  if (n <= LEAF) return F(crout)(a, m, b0, b1, ld, j0, n, pivots, perm, s);
  int64_t n1 = split(n), n2 = n - n1, j1 = j0 + n1;
  T *a11 = a + j0 * ld + j0, *a12 = a11 + n1, *a21 = a11 + n1 * ld;
  return F(panel)(a, m, b0, b1, ld, j0, n1, pivots, perm, s) &&
         F(solve)(UNIT_LOWER, a11, n1, ld, a12, n2, ld, NULL, s) &&
         F(gemm)(m - j1, n2, n1, -1, F(op)(a21, ld, 1), F(op)(a12, ld, 1),
                 a21 + n1, ld, s) &&
         F(panel)(a, m, b0, b1, ld, j1, n2, pivots, perm, s);
}

/* The interchanges of rows i and pivots[i], i from [k0] to [k1] in turn,
   on columns [c0, c1) of [a]. */
static void F(laswp)(T *a, int64_t ld, int64_t c0, int64_t c1,
                     const int64_t *pivots, int64_t k0, int64_t k1) {
  for (int64_t i = k0; i < k1; i++)
    if (pivots[i] != i)
      F(swap)(a + i * ld + c0, a + pivots[i] * ld + c0, c1 - c0);
}

/* A step of LU: block [k, k + kb) factored and the next block's columns
   [next, next + nb) updated by it, the next block factored and the
   columns past it updated, by [units] units: the first the next block,
   the others [tile] columns each of the rest. */
typedef struct {
  T *a;
  int64_t m, n, ld, k, kb, next, nb, tile, units;
  int64_t *pivots, *perm;
  int failed;
  arena *workers;
} F(lu_step);

/* Columns [c0, c1) updated by the step's block: its interchanges, its
   L11⁻¹ and its L21. */
static int F(lu_update)(const F(lu_step) *t, int64_t c0, int64_t c1,
                        arena *s) {
  int64_t k = t->k, kb = t->kb, ld = t->ld;
  T *a = t->a, *l = a + k * ld + k;
  if (c1 <= c0) return 1;
  F(laswp)(a, ld, c0, c1, t->pivots, k, k + kb);
  return F(lower)(l, kb, ld, a + k * ld + c0, c1 - c0, ld, 1, s) &&
         F(gemm)(t->m - k - kb, c1 - c0, kb, -1, F(op)(l + kb * ld, ld, 1),
                 F(op)(a + k * ld + c0, ld, 1), a + (k + kb) * ld + c0, ld,
                 s);
}

static int F(lu_unit_on)(const F(lu_step) *t, int64_t u, arena *s) {
  int64_t next = t->next, nb = t->nb;
  if (u > 0) {
    int64_t c0 = next + nb + (u - 1) * t->tile;
    int64_t c1 = c0 + t->tile < t->n ? c0 + t->tile : t->n;
    return F(lu_update)(t, c0, c1, s);
  }
  return nb == 0 || F(panel)(t->a, t->m, next, next + nb, t->ld, next, nb,
                             t->pivots, t->perm, s);
}

/* A unit runs on its worker's arena alone, as a solve's do. */
static void F(lu_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(lu_step) *t = ctx;
  arena *s = &t->workers[worker], *all = s->workers;
  s->workers = NULL;
  for (int64_t u = lo; u < hi; u++)
    if (!F(lu_unit_on)(t, u, s))
      __atomic_store_n(&t->failed, 1, __ATOMIC_RELAXED);
  s->workers = all;
}

static int F(lu)(T *a, int64_t m, int64_t n, int64_t ld, int64_t *pivots,
                 int64_t *perm, arena *s) {
  int64_t kk = m < n ? m : n;
  int64_t bw = kk >= LU_BLOCKS * LU_NB ? LU_NB : LU_NB / 2;
  for (int64_t i = 0; i < m; i++) perm[i] = i;
  if (kk == 0) return 1;
  int64_t nb0 = kk < bw ? kk : bw;
  if (!F(panel)(a, m, 0, nb0, ld, 0, nb0, pivots, perm, s)) return 0;
  for (int64_t k = 0; k < kk; k += bw) {
    int64_t kb = kk - k < bw ? kk - k : bw, next = k + kb;
    int64_t nb = kk - next < bw ? kk - next : bw, rest = n - next - nb;
    int64_t bytes = (m - k) * (n - k) * (int64_t)sizeof(T);
    /* The next block a unit, and the rest's columns at most one each. */
    double cost = threaded(s, 1 + rest, bytes,
                           2.0 * (m - next) * (n - next) * kb, RDT, CPLX);
    int64_t threads = cost > 0 ? nx_cpu_threads(bytes, cost) : 1;
    int64_t tile = rest / (threads > 1 ? threads - 1 : 1);
    tile = tile < bw ? bw : tile;
    F(lu_step) t = {.a = a, .m = m, .n = n, .ld = ld, .k = k, .kb = kb,
                    .next = next, .nb = nb, .tile = tile,
                    .units = 1 + (rest + tile - 1) / tile, .pivots = pivots,
                    .perm = perm, .workers = s->workers};
    if (!F(lu_update)(&t, next, next + nb, s)) return 0;
    if (cost == 0) {
      for (int64_t u = 0; u < t.units; u++)
        if (!F(lu_unit_on)(&t, u, s)) return 0;
    } else {
      nx_cpu_job(t.units, bytes, cost, F(lu_unit), &t);
      if (t.failed) return 0;
    }
  }
  /* Each block's columns take the interchanges of the blocks right of
     it. */
  for (int64_t b0 = 0; b0 < kk; b0 += bw) {
    int64_t b1 = kk - b0 < bw ? kk : b0 + bw;
    F(laswp)(a, ld, b0, b1, pivots, b1, kk);
  }
  return 1;
}

/* Householder reflectors */

/* The least normal number over the unit roundoff, below which LAPACK's
   reflector rescales its norm. */
#define SAFMIN \
  ((R)(sizeof(R) == 4 ? FLT_MIN / (FLT_EPSILON / 2)                         \
                      : DBL_MIN / (DBL_EPSILON / 2)))

/* |a| with the sign of b. */
static R F(sign)(R a, R b) { return COPYSIGN(a, b); }

/* The 2-norm of the [n] elements at [x], [s] apart: the plain sum of
   squares where it lies where every square that matters is exact, else
   scaled by the largest part. */
static R F(norm)(const T *x, int64_t n, int64_t s) {
  R q = 0;
  for (int64_t i = 0; i < n; i++) {
    R re = REAL(x[i * s]), im = IMAG(x[i * s]);
    q += re * re + im * im;
  }
  if ((q >= SAFMIN && q <= 1 / SAFMIN) || q != q) return SQRT(q);
  R big = 0;
  for (int64_t i = 0; i < n; i++) {
    R re = F(abs)(REAL(x[i * s])), im = F(abs)(IMAG(x[i * s]));
    big = re > big ? re : big;
    big = im > big ? im : big;
  }
  if (big == 0) return SQRT(q);
  R t = 0;
  for (int64_t i = 0; i < n; i++) {
    R re = REAL(x[i * s]) / big, im = IMAG(x[i * s]) / big;
    t += re * re + im * im;
  }
  return big * SQRT(t);
}

/* sqrt (a² + b² + c²), scaled (LAPACK's xLAPY3). */
static R F(lapy3)(R a, R b, R c) {
  R xa = F(abs)(a), xb = F(abs)(b), xc = F(abs)(c);
  R w = xa > xb ? xa : xb;
  w = xc > w ? xc : w;
  if (w == 0 || w != w) return xa + xb + xc;
  xa /= w;
  xb /= w;
  xc /= w;
  return w * SQRT(xa * xa + xb * xb + xc * xc);
}

/* The reflector H = I − τ v vᴴ with Hᴴ [α; x] = [β; 0], β real, as LAPACK's
   xLARFG makes it: v = [1; x'], x' over the [n] elements of [x], [s]
   apart, and β over [*alpha]. Returns τ, 0 where [α; x] is [β; 0]
   already. x' is x times 1 / SAFMIN [*knt] times, then over [*scale]:
   reflector_by gives both, 0 and 1 where τ is 0. */
static T F(reflector_by)(T *alpha, T *x, int64_t n, int64_t s, int *knt,
                         T *scale) {
  R xnorm = F(norm)(x, n, s);
  R ar = REAL(*alpha), ai = IMAG(*alpha);
  *knt = 0;
  *scale = 1;
  if (xnorm == 0 && ai == 0) return 0;
  R beta = ar >= 0 ? -F(lapy3)(ar, ai, xnorm) : F(lapy3)(ar, ai, xnorm);
  if (F(abs)(beta) < SAFMIN) {
    do {
      (*knt)++;
      for (int64_t i = 0; i < n; i++) x[i * s] = x[i * s] * (1 / SAFMIN);
      beta *= 1 / SAFMIN;
      ar *= 1 / SAFMIN;
      ai *= 1 / SAFMIN;
    } while (F(abs)(beta) < SAFMIN && *knt < 20);
    xnorm = F(norm)(x, n, s);
    R an = F(lapy3)(ar, ai, xnorm);
    beta = ar >= 0 ? -an : an;
  }
#if CPLX
  T tau = (beta - ar) / beta - (ai / beta) * I;
  T sc = (ar - beta) + ai * I;
#else
  T tau = (beta - ar) / beta;
  T sc = ar - beta;
#endif
  for (int64_t i = 0; i < n; i++) x[i * s] = DIV(x[i * s], sc);
  *scale = sc;
  for (int j = 0; j < *knt; j++) beta *= SAFMIN;
  *alpha = beta;
  return tau;
}

static T F(reflector)(T *alpha, T *x, int64_t n, int64_t s) {
  int knt;
  T scale;
  return F(reflector_by)(alpha, x, n, s, &knt, &scale);
}

/* c ← (I − τ v vᴴ) c for the [m] × [n] [c], rows [ldc] apart, v's [m]
   elements [ldv] apart with v[0] taken as 1 (xLARF). [w] holds n
   elements. */
static void F(reflect)(const T *v, int64_t ldv, T tau, int64_t m, int64_t n,
                       T *c, int64_t ldc, T *w) {
  if (tau == 0) return;
  for (int64_t j = 0; j < n; j++) w[j] = c[j];
  for (int64_t i = 1; i < m; i++) {
    T vi = CONJ(v[i * ldv]);
    const T *ci = c + i * ldc;
    for (int64_t j = 0; j < n; j++) w[j] += MUL(vi, ci[j]);
  }
  for (int64_t i = 0; i < m; i++) {
    T t = i == 0 ? tau : MUL(tau, v[i * ldv]);
    T *ci = c + i * ldc;
    for (int64_t j = 0; j < n; j++) ci[j] -= MUL(t, w[j]);
  }
}

/* QR

   a = Q R by Householder reflectors, as LAPACK's blocked xGEQRT, in place
   in the m × n [a]: R on and above the diagonal, each reflector's v below
   it, its τ in [tau]. By blocks of QR_NB columns: a block's reflectors and
   their product H = I − V T Vᴴ by the recursive panel (xGEQRT3), then Hᴴ =
   I − V Tᴴ Vᴴ applied to the columns right of the block by products
   (xLARFB). Each block's T is kept for [form_q] in [ts], QR_NB × QR_NB a
   block, rows QR_NB apart. The scratch [q] holds a block's dense V, a
   leaf's copy and row, and two QR_NB × max m n products, as qr_scratch
   counts them. Answers 1, or 0 if an allocation failed. */

/* T of the block of the [jb] reflectors in the dense [h] × jb [v], into
   [t], rows [ldt] apart: T[i][i] = τ_i, T[0:i][i] = −τ_i T[0:i][0:i]
   V[:, 0:i]ᴴ v_i, zeros below (xLARFT), the products Vᴴ V by a product
   into [g], jb × jb. Answers 0 if an allocation failed. */
static int F(block_t)(const T *v, int64_t h, int64_t jb, const T *tau, T *t,
                      int64_t ldt, T *g, arena *s) {
  memset(g, 0, sizeof(T) * jb * jb);
  if (!F(gemm)(jb, jb, h, 1, F(op_h)(v, jb), F(op)(v, jb, 1), g, jb, s))
    return 0;
  for (int64_t i = 0; i < jb; i++) {
    for (int64_t p = 0; p < i; p++) {
      T x = 0;
      for (int64_t q = p; q < i; q++) x += MUL(t[p * ldt + q], g[q * jb + i]);
      t[p * ldt + i] = MUL(-tau[i], x);
    }
    t[i * ldt + i] = tau[i];
    for (int64_t p = i + 1; p < jb; p++) t[p * ldt + i] = 0;
  }
  return 1;
}

/* [v]'s [h] × jb reflectors in place, below its diagonal, rows [ld]
   apart, made the dense V: ones on the diagonal, zeros above. */
static void F(dense_v)(T *v, int64_t h, int64_t jb, int64_t ld) {
  for (int64_t r = 0; r < h && r < jb; r++) {
    for (int64_t p = r + 1; p < jb; p++) v[r * ld + p] = 0;
    v[r * ld + r] = 1;
  }
}

/* Copies [h] rows of [jb] elements, rows [ls] apart at [x], to [y], rows
   [ld] apart. */
static void F(rows)(const T *x, int64_t ls, int64_t h, int64_t jb, T *y,
                    int64_t ld) {
  for (int64_t r = 0; r < h; r++)
    memcpy(y + r * ld, x + r * ls, sizeof(T) * jb);
}

/* c ← (I − V op(T) Vᴴ) c for c of [m2] × [n2], rows [ldc] apart; [v] the
   dense m2 × jb V, rows [ldv] apart, [t] op(T): c −= V (op(T) (Vᴴ c)).
   [w] holds 2·jb·n2 elements. */
static int F(larfb)(const T *v, int64_t ldv, operand t, int64_t jb,
                    int64_t m2, int64_t n2, T *c, int64_t ldc, T *w,
                    arena *s) {
  T *w1 = w, *w2 = w + jb * n2;
  memset(w, 0, sizeof(T) * 2 * jb * n2);
  return F(gemm)(jb, n2, m2, 1, F(op_h)(v, ldv), F(op)(c, ldc, 1), w1, n2,
                 s) &&
         F(gemm)(jb, n2, jb, 1, t, F(op)(w1, n2, 1), w2, n2, s) &&
         F(gemm)(m2, n2, jb, -1, F(op)(v, ldv, 1), F(op)(w2, n2, 1), c, ldc,
                 s);
}

/* y ← y (I − V T Vᴴ)ᵀ = y − (y conj V) Tᵀ Vᵀ for the [m] × [h] y, rows
   [ldy] apart: larfb from the right on the transposes of the columns it
   takes, so that y, a complex product's a, is read in place, where as
   larfb's b it is copied. [v] the dense h × jb V, rows [ldv] apart, [t] T,
   rows [ldt] apart; [w] holds 2·m·jb elements. */
static int F(rlarfb)(const T *v, int64_t ldv, const T *t, int64_t ldt,
                     int64_t jb, int64_t h, int64_t m, T *y, int64_t ldy,
                     T *w, arena *s) {
  T *x = w, *x2 = w + m * jb;
  memset(w, 0, sizeof(T) * 2 * m * jb);
  return F(gemm)(m, jb, h, 1, F(op)(y, ldy, 1), (operand){v, ldv, 1, CPLX},
                 x, jb, s) &&
         F(gemm)(m, jb, jb, 1, F(op)(x, jb, 1), F(op)(t, 1, ldt), x2, jb,
                 s) &&
         F(gemm)(m, h, jb, -1, F(op)(x2, jb, 1), F(op)(v, 1, ldv), y, ldy, s);
}

/* The [h] × [n] panel [a], rows [ld] apart, factored: its reflectors in
   place and τ in [tau], its dense V into [v] and its T into [t], both rows
   [lv] apart, lv at least n. As LAPACK's recursive xGEQRT3: the left
   half's reflectors V1, T1, applied to the right half, then the right
   half's V2, T2 below the left half's rows, and

     T = [T1  −T1 (V1ᴴ V2) T2]
         [0    T2            ],

   since (I − V1 T1 V1ᴴ) (I − V2 T2 V2ᴴ) = I − V T Vᴴ. A panel of fewer
   than 2·LEAF columns is a leaf, whose halves would be fewer than LEAF
   columns and cost more in products than in scalar loops: on the M1, 16
   QRs of 32 × 32 float32 took 56.5 µs in halves against 35.4 as one
   leaf. A leaf takes the reflectors one at a time, each applied to the
   columns to its right (xGEQR2), on a contiguous copy in [c], which holds
   (h + 1)·2·LEAF elements, then T (xLARFT). [w] holds 2·lv·lv
   elements. */
static int F(qr_panel)(T *a, int64_t h, int64_t n, int64_t ld, T *tau,
                       T *v, T *t, int64_t lv, T *c, T *w, arena *s) {
  if (n < 2 * LEAF) {
    /* Contiguous: the panel's rows a power of two apart would share one
       set of the cache. */
    T *row = c + h * n;
    F(rows)(a, ld, h, n, c, n);
    for (int64_t j = 0; j < n; j++) {
      T *cjj = c + j * n + j;
      tau[j] = F(reflector)(cjj, cjj + n, h - j - 1, n);
      F(reflect)(cjj, n, CONJ(tau[j]), h - j, n - j - 1, cjj + 1, n, row);
    }
    F(rows)(c, n, h, n, a, ld);
    F(dense_v)(c, h, n, n);
    F(rows)(c, n, h, n, v, lv);
    return F(block_t)(c, h, n, tau, t, lv, w, s);
  }
  int64_t n1 = split(n), n2 = n - n1;
  T *g = w, *x = w + n1 * n2, *t12 = t + n1;
  if (!F(qr_panel)(a, h, n1, ld, tau, v, t, lv, c, w, s) ||
      !F(larfb)(v, lv, F(op_h)(t, lv), n1, h, n2, a + n1, ld, w, s) ||
      !F(qr_panel)(a + n1 * ld + n1, h - n1, n2, ld, tau + n1,
                   v + n1 * lv + n1, t + n1 * lv + n1, lv, c, w, s))
    return 0;
  for (int64_t r = 0; r < n1; r++) {
    memset(v + r * lv + n1, 0, sizeof(T) * n2);
    memset(t12 + r * lv, 0, sizeof(T) * n2);
  }
  for (int64_t r = n1; r < n; r++) memset(t + r * lv, 0, sizeof(T) * n1);
  memset(w, 0, sizeof(T) * 2 * n1 * n2);
  return F(gemm)(n1, n2, h - n1, 1, F(op_h)(v + n1 * lv, lv),
                 F(op)(v + n1 * lv + n1, lv, 1), g, n2, s) &&
         F(gemm)(n1, n2, n1, 1, F(op)(t, lv, 1), F(op)(g, n2, 1), x, n2,
                 s) &&
         F(gemm)(n1, n2, n2, -1, F(op)(x, n2, 1),
                 F(op)(t + n1 * lv + n1, lv, 1), t12, lv, s);
}

static int F(qr)(T *a, int64_t m, int64_t n, int64_t ld, T *tau, T *ts,
                 T *q, arena *s) {
  int64_t k = m < n ? m : n;
  T *v = q, *c = v + m * QR_NB, *w = c + (m + 1) * 2 * LEAF;
  for (int64_t j0 = 0; j0 < k; j0 += QR_NB) {
    int64_t jb = QR_NB < k - j0 ? QR_NB : k - j0, j1 = j0 + jb, h = m - j0;
    T *t = ts + j0 * QR_NB;
    if (!F(qr_panel)(a + j0 * ld + j0, h, jb, ld, tau + j0, v, t, QR_NB, c, w,
                     s))
      return 0;
    if (j1 < n &&
        !F(larfb)(v, QR_NB, F(op_h)(t, QR_NB), jb, h, n - j1,
                  a + j0 * ld + j1, ld, w, s))
      return 0;
  }
  return 1;
}

/* q ← the first [qc] columns of H_0 ⋯ H_(k−1), m × qc, rows [ldq] apart,
   from [qr]'s reflectors in [a] and its blocks' T in [ts]: from the
   identity, the blocks applied last first (LAPACK's xORGQR and xUNGQR).
   The scratch [w] as [qr] left it, the last block's dense V first, which
   saves a matrix of one block its second copy. */
static int F(form_q)(const T *a, int64_t m, int64_t n, int64_t ld,
                     const T *ts, T *q, int64_t qc, int64_t ldq, T *w,
                     arena *s) {
  int64_t k = m < n ? m : n;
  T *v = w, *ww = v + m * QR_NB;
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < qc; j++) q[i * ldq + j] = i == j;
  int64_t last = k == 0 ? 0 : (k - 1) / QR_NB * QR_NB;
  for (int64_t j0 = last; k > 0 && j0 >= 0; j0 -= QR_NB) {
    int64_t jb = QR_NB < k - j0 ? QR_NB : k - j0, h = m - j0;
    if (j0 < last) {
      F(rows)(a + j0 * ld + j0, ld, h, jb, v, QR_NB);
      F(dense_v)(v, h, jb, QR_NB);
    }
    if (!F(larfb)(v, QR_NB, F(op)(ts + j0 * QR_NB, QR_NB, 1), jb, h, qc - j0,
                  q + j0 * ldq + j0, ldq, ww, s))
      return 0;
  }
  return 1;
}

/* c ← H_0 ⋯ H_(k−1) c for c of [h] rows, rows [ldc] apart, and [n]
   columns, or under AS_ROWS its transpose, c ← c (H_0 ⋯ H_(k−1))ᵀ for c
   of [n] rows and [h] columns (rlarfb); H_j = I − τ_j v_j v_jᴴ with
   v_j[j] = 1 and v_j[r], r > j, at [x] + r·rs + j·cs, read as [how]
   says: by blocks of NB reflectors, the last first (xUNMQR, xUNMLQ). The
   scratch [w] holds (h + NB + 2n) NB elements. Answers 0 if an
   allocation failed. */
static int F(back)(const T *x, int64_t rs, int64_t cs, reading how,
                   const T *tau, int64_t k, int64_t h, int64_t n, T *c,
                   int64_t ldc, holding as, T *w, arena *s) {
  T *t = w + h * NB, *ww = t + NB * NB;
  for (int64_t j0 = k <= 0 ? -1 : (k - 1) / NB * NB; j0 >= 0; j0 -= NB) {
    int64_t jb = NB < k - j0 ? NB : k - j0, hb = h - j0;
    for (int64_t r = 0; r < hb; r++)
      for (int64_t p = 0; p < jb; p++) {
        T v = x[(j0 + r) * rs + (j0 + p) * cs];
        w[r * jb + p] = how == CONJUGATED ? CONJ(v) : v;
      }
    F(dense_v)(w, hb, jb, jb);
    if (!F(block_t)(w, hb, jb, tau + j0, t, jb, ww, s) ||
        !(as == AS_ROWS
              ? F(rlarfb)(w, jb, t, jb, jb, hb, n, c + j0, ldc, ww, s)
              : F(larfb)(w, jb, F(op)(t, jb, 1), jb, hb, n, c + j0 * ldc,
                         ldc, ww, s)))
      return 0;
  }
  return 1;
}

/* Eigh

   a = Q T Qᴴ with T real tridiagonal, then T = Z diag w Zᵀ, so that
   v = Q Z, as LAPACK's xHEEV. The matrix is column-major here, element
   (i, j) at a[i + j·ld], so that the routines follow LAPACK's index for
   index; it is staged transposed, which on a Hermitian matrix swaps the
   triangles: a's lower triangle is read. */

/* xᴴ y over [n] elements, in four parts by the index mod 4 so that no
   addition waits on the one before. */
static T F(dot)(const T *x, const T *y, int64_t n) {
  T p[4] = {0, 0, 0, 0};
  int64_t r = 0;
  for (; r + 4 <= n; r += 4)
    for (int q = 0; q < 4; q++) p[q] += MUL(CONJ(x[r + q]), y[r + q]);
  for (; r < n; r++) p[r & 3] += MUL(CONJ(x[r]), y[r]);
  return (p[0] + p[1]) + (p[2] + p[3]);
}

/* Columns [j0, j1) of y = A v for A of order [h], Hermitian, its lower
   triangle and its diagonal's real parts read, columns [ld] apart
   (xHEMV), into rows [j0, h) of [y]: the columns' products below their
   diagonal and their products with v. By four columns at a time, which
   update y and take their products with v in one pass over their rows,
   each product in two parts, by rows' parity, so that none waits on its
   own additions. */
static void F(hemv_block)(const T *restrict a, int64_t h, int64_t ld,
                          const T *restrict v, int64_t j0, int64_t j1,
                          T *restrict y) {
  for (int64_t r = j0; r < h; r++) y[r] = 0;
  for (int64_t j = j0; j < j1; j += 4) {
    int64_t e = j + 4 < j1 ? j + 4 : j1;
    const T *c[4];
    T vq[4], t[4] = {0, 0, 0, 0}, s[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    for (int q = 0; q < 4; q++) {
      c[q] = a + (j + q < e ? j + q : j) * ld;
      vq[q] = j + q < e ? v[j + q] : 0;
    }
    /* The columns' triangle, then the rows below it. */
    for (int64_t q = j; q < e; q++) {
      y[q] += REAL(c[q - j][q]) * v[q];
      for (int64_t r = q + 1; r < e; r++) {
        y[r] += MUL(c[q - j][r], v[q]);
        t[q - j] += MUL(CONJ(c[q - j][r]), v[r]);
      }
    }
    for (int64_t r = e; r < h; r += 2)
      for (int l = 0; l < 2 && r + l < h; l++) {
        T vr = v[r + l], u = y[r + l];
        for (int q = 0; q < 4; q++) {
          u += MUL(c[q][r + l], vq[q]);
          s[q][l] += MUL(CONJ(c[q][r + l]), vr);
        }
        y[r + l] = u;
      }
    for (int64_t q = j; q < e; q++)
      y[q] += t[q - j] + (s[q - j][0] + s[q - j][1]);
  }
}

typedef struct {
  const T *a, *v;
  int64_t h, ld, blocks;
  T *yp; /* a block's rows of y, h elements a block */
} F(hemv_job);

static void F(hemv_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const F(hemv_job) *j = ctx;
  for (int64_t u = lo; u < hi; u++)
    F(hemv_block)(j->a, j->h, j->ld, j->v, hemv_bound(j->h, j->blocks, u),
                  hemv_bound(j->h, j->blocks, u + 1), j->yp + u * j->h);
}

/* y = A v as hemv_block, by hemv_blocks(h) blocks of columns into [yp],
   that many times h elements, on the call's threads where [s] holds
   them, then y their sum, block after block; one block into y. Blocks of
   columns read the lower triangle down its columns, as stored, and give
   both their rows' products and their columns' in one pass; blocks of
   rows would read each row's left part across the columns. */
static void F(hemv)(const T *restrict a, int64_t h, int64_t ld,
                    const T *restrict v, T *restrict y, T *restrict yp,
                    arena *s) {
  int64_t blocks = hemv_blocks(h);
  if (blocks == 1) {
    F(hemv_block)(a, h, ld, v, 0, h, y);
    return;
  }
  F(hemv_job) j = {.a = a, .v = v, .h = h, .ld = ld, .blocks = blocks,
                   .yp = yp};
  int64_t bytes = h * h / 2 * (int64_t)sizeof(T);
  double cost = threaded(s, blocks, bytes, (double)h * h, RDT, CPLX);
  if (cost > 0)
    nx_cpu_job(blocks, bytes, cost, F(hemv_unit), &j);
  else
    F(hemv_unit)(0, blocks, 0, &j);
  for (int64_t r = 0; r < h; r++) y[r] = 0;
  for (int64_t u = 0; u < blocks; u++)
    for (int64_t r = hemv_bound(h, blocks, u); r < h; r++)
      y[r] += yp[u * h + r];
}

/* The panel of the [nb] columns of the [n] × [n] [a] by reflectors, as
   LAPACK's xLATRD (lower): column i, the products of the panel's earlier
   reflectors deferred until now applied to it, is reduced below its
   subdiagonal by H_i = I − τ_i v_i v_iᴴ, its subdiagonal into e[i], v_i
   below it with v_i[0] = 1 stored; [w], [n] × nb with columns [ldw]
   apart, gets W with the rest of a updated as a − V Wᴴ − W Vᴴ; [yp] is
   hemv's scratch. Only the lower triangle is read, and diagonals' real
   parts. */
static void F(latrd)(T *a, int64_t n, int64_t nb, int64_t ld, R *e, T *tau,
                     T *w, int64_t ldw, T *yp, arena *s) {
  for (int64_t i = 0; i < nb; i++) {
    T *ai = a + i * ld, *wi = w + i * ldw;
    for (int64_t p = 0; p < i; p++) {
      const T *ap = a + p * ld, *wp = w + p * ldw;
      T x = CONJ(wp[i]), y = CONJ(ap[i]);
      for (int64_t r = i; r < n; r++) ai[r] -= MUL(ap[r], x) + MUL(wp[r], y);
    }
    ai[i] = REAL(ai[i]);
    /* H_i from a's column i below its diagonal. */
    T alpha = ai[i + 1];
    int64_t h = n - i - 1;
    tau[i] = F(reflector)(&alpha, ai + i + 2, h - 1, 1);
    e[i] = REAL(alpha);
    ai[i + 1] = 1;
    /* W's column i, from rows i + 1 on: τ (A v − V (Wᴴ v) − W (Vᴴ v)) −
       ½ τ (that ᴴ v) v, A the not yet updated trailing matrix (xHEMV). */
    const T *v = ai + i + 1;
    T *y = wi + i + 1, *x = wi;
    F(hemv)(a + (i + 1) * ld + i + 1, h, ld, v, y, yp, s);
    for (int k = 0; k < 2; k++) {
      /* x = Uᴴ v, then y −= U' x, for (U, U') = (W, V) then (V, W). */
      const T *u = k == 0 ? w : a, *u2 = k == 0 ? a : w;
      int64_t lu = k == 0 ? ldw : ld, lu2 = k == 0 ? ld : ldw;
      for (int64_t p = 0; p < i; p++) x[p] = F(dot)(u + p * lu + i + 1, v, h);
      for (int64_t p = 0; p < i; p++) {
        const T *up = u2 + p * lu2 + i + 1;
        for (int64_t r = 0; r < h; r++) y[r] -= MUL(up[r], x[p]);
      }
    }
    T s = 0;
    for (int64_t r = 0; r < h; r++) {
      y[r] = MUL(y[r], tau[i]);
      s += MUL(CONJ(y[r]), v[r]);
    }
    T half = MUL((R)-0.5 * tau[i], s);
    for (int64_t r = 0; r < h; r++) y[r] += MUL(half, v[r]);
  }
}

/* a = Q T Qᴴ in place in the [n] × [n] [a], as LAPACK's blocked xHETRD
   (lower): T's diagonal into [d] and subdiagonal into [e], Q = H_0 ⋯
   H_(n−2) as reflectors below the subdiagonal, their τ in [tau]. By
   panels of NB columns (latrd), each followed by the rest's update
   a − V Wᴴ − W Vᴴ (xHER2K) by products on column blocks of its lower
   triangle. The scratch [w] holds n × NB elements, [yp]
   hemv_blocks(n) × n. Answers 0 if an allocation failed. */
static int F(hetrd)(T *a, int64_t n, int64_t ld, R *d, R *e, T *tau, T *w,
                    T *yp, arena *s) {
  for (int64_t i0 = 0; i0 < n - 1; i0 += NB) {
    int64_t nb = NB < n - 1 - i0 ? NB : n - 1 - i0, h = n - i0, m = h - nb;
    T *p = a + i0 * ld + i0;
    F(latrd)(p, h, nb, ld, e + i0, tau + i0, w, h, yp, s);
    for (int64_t j = 0; j < nb; j++) d[i0 + j] = REAL(p[j * ld + j]);
    /* The rest's lower triangle, as the row-major upper triangle of its
       transpose: rows j0 on of conj W Vᵀ + conj V Wᵀ. */
    T *v = p + nb, *w2 = w + nb, *c = p + nb * ld + nb;
    for (int64_t j0 = 0; j0 < m; j0 += HERK) {
      int64_t jb = m - j0 < HERK ? m - j0 : HERK;
      T *cj = c + j0 * ld + j0;
      if (!F(gemm)(jb, m - j0, nb, -1, (operand){w2 + j0, 1, h, CPLX},
                   F(op)(v + j0, ld, 1), cj, ld, s) ||
          !F(gemm)(jb, m - j0, nb, -1, (operand){v + j0, 1, ld, CPLX},
                   F(op)(w2 + j0, h, 1), cj, ld, s))
        return 0;
    }
  }
  if (n > 0) d[n - 1] = REAL(a[(n - 1) * ld + n - 1]);
  return 1;
}

/* Two stages

   Where eigh_two says, a = Q1 B Q1ᴴ with B Hermitian of kd subdiagonals,
   then B = Q2 T Q2ᴴ with T tridiagonal, as LAPACK's xHETRD_HE2HB and
   xHETRD_HB2ST. The first stage's flops are products on the threads, where
   half the one-stage reduction's are matrix-vector products that read
   the trailing matrix once a column; the second stage's O(n² kd) work
   runs on the band alone. The matrix is row-major here, element (i, j)
   at a[i·ld + j], both its triangles. */

/* The [m] × kd panel [p] of a, rows [ld] apart, below the band,
   factored (qr_panel): H = I − V T Vᴴ, its τ into [tau], V and T into
   [v] and [t], rows kd apart, [l] and [q] its leaf's and products'
   scratch. A last panel of fewer rows than kd has columns past its
   reflectors, in the band, which take Hᴴ too. */
static int F(band_panel)(T *p, int64_t m, int64_t ld, T *tau, T *v, T *t,
                         T *l, T *q, int64_t kd, arena *s) {
  int64_t kb = m < kd ? m : kd;
  return F(qr_panel)(p, m, kb, ld, tau, v, t, kd, l, q, s) &&
         (kb == kd || F(larfb)(v, kd, F(op_h)(t, kd), kb, m, kd - kb, p + kb,
                               ld, q, s));
}

/* A step's update of the trailing matrix [c], m × m, past its first kd
   columns: c −= [W V] [V W]ᴴ by blocks of kd columns, from U = [W V W],
   kb columns each, and the next panel's factorisation, whose columns are
   c's first, beside them: unit 0 the panel, unit k columns k·kd on. */
typedef struct {
  const T *u;
  T *c, *tau, *v, *t, *l, *q;
  int64_t m, kb, ld, kd;
  int failed;
  arena *workers;
} F(band_step);

static void F(band_units)(F(band_step) *j, int64_t lo, int64_t hi,
                          arena *s) {
  int64_t m = j->m, kb = j->kb, lu = 3 * kb, kd = j->kd;
  for (int64_t k = lo; k < hi; k++) {
    int64_t c0 = k * kd, cb = m - c0 < kd ? m - c0 : kd;
    if (k == 0 ? !F(band_panel)(j->c + kd * j->ld, m - kd, j->ld, j->tau,
                                j->v, j->t, j->l, j->q, kd, s)
               : !F(gemm)(m, cb, 2 * kb, -1, F(op)(j->u, lu, 1),
                          F(op_h)(j->u + kb + c0 * lu, lu), j->c + c0, j->ld,
                          s))
      __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
  }
}

/* A unit runs on its worker's arena alone, as a solve's do. */
static void F(band_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(band_step) *j = ctx;
  arena *s = &j->workers[worker], *all = s->workers;
  s->workers = NULL;
  F(band_units)(j, lo, hi, s);
  s->workers = all;
}

/* a = Q1 B Q1ᴴ in place in the [n] × [n] [a], rows [ld] apart: B's band
   in a's lower triangle, Q1 = H_0 ⋯ H_(n−kd−1) below it, v_j from row
   j + kd of column j, its τ in [tau]. By panels of kd columns: a panel's
   rows below the band factored (band_panel), H = I − V T Vᴴ, then the
   trailing matrix c ← Hᴴ c H = c − W Vᴴ − V Wᴴ, W = X − ½ V Tᴴ (Vᴴ X),
   X = c V T: c's first kd columns, then the next panel's factorisation
   beside the rest's update on the threads, as Cholesky's next diagonal
   block. Both of c's triangles are kept, 2n³ flops where xHETRD_HE2HB's
   one triangle takes 4/3 n³, so that each product reads c in place,
   where a complex product copies its b, and its update is a product of
   the square, [W V] [V W]ᴴ. The scratch [w] holds eigh_work elements.
   Answers 0 if an allocation failed. */
static int F(he2hb)(T *a, int64_t n, int64_t ld, T *tau, T *w, int64_t kd,
                    arena *s) {
  T *v = w, *t = v + n * kd, *q = t + kd * kd, *l = q + 2 * kd * kd,
    *u = l + 2 * LEAF * (n + 1), *g = u + 3 * kd * n, *h = g + kd * kd;
  if (n > kd && !F(band_panel)(a + kd * ld, n - kd, ld, tau, v, t, l, q, kd, s))
    return 0;
  for (int64_t i0 = 0; i0 + kd < n; i0 += kd) {
    int64_t m = n - i0 - kd, kb = m < kd ? m : kd, lu = 3 * kb;
    T *c = a + (i0 + kd) * (ld + 1), *x = u + 2 * kb;
    /* U = [Y V X], Y = c V and X = Y T, then [W V W]. */
    for (int64_t r = 0; r < m; r++) {
      memset(u + r * lu, 0, sizeof(T) * lu);
      memcpy(u + r * lu + kb, v + r * kd, sizeof(T) * kb);
    }
    memset(g, 0, sizeof(T) * 2 * kd * kd);
    if (!F(gemm)(m, kb, m, 1, F(op)(c, ld, 1), F(op)(v, kd, 1), u, lu, s) ||
        !F(gemm)(m, kb, kb, 1, F(op)(u, lu, 1), F(op)(t, kd, 1), x, lu, s) ||
        !F(gemm)(kb, kb, m, 1, F(op_h)(v, kd), F(op)(x, lu, 1), g, kb, s) ||
        !F(gemm)(kb, kb, kb, 1, F(op_h)(t, kd), F(op)(g, kb, 1), h, kb, s))
      return 0;
    for (int64_t i = 0; i < kb * kb; i++) h[i] *= (R)0.5;
    if (!F(gemm)(m, kb, kb, -1, F(op)(v, kd, 1), F(op)(h, kb, 1), x, lu, s))
      return 0;
    for (int64_t r = 0; r < m; r++)
      memcpy(u + r * lu, x + r * lu, sizeof(T) * kb);
    if (!F(gemm)(m, m <= kd ? m : kd, 2 * kb, -1, F(op)(u, lu, 1),
                 F(op_h)(u + kb, lu), c, ld, s))
      return 0;
    if (m <= kd) continue;
    F(band_step) j = {.u = u, .c = c, .tau = tau + i0 + kd, .v = v, .t = t,
                      .l = l, .q = q, .m = m, .kb = kb, .ld = ld, .kd = kd,
                      .workers = s->workers};
    int64_t units = (m + kd - 1) / kd, bytes = m * m * (int64_t)sizeof(T);
    double cost =
        threaded(s, units, bytes, 2.0 * m * m * 2 * kb, RDT, CPLX);
    if (cost == 0) F(band_units)(&j, 0, units, s);
    else nx_cpu_job(units, bytes, cost, F(band_unit), &j);
    if (j.failed) return 0;
  }
  return 1;
}

/* D ← Hᴴ D H, H = I − τ v vᴴ, for the Hermitian D of order [len] whose
   lower triangle's column c starts at d_c = b + (st + c)·2kd, element
   (r, c) at d_c[r − c]: D − y vᴴ − v yᴴ with y = τ D v − ½ τ (τ D v)ᴴ v · v,
   as hetrd's panels take it, and D's diagonal real. */
static void F(both_sides)(T *b, int64_t kd, int64_t st, int64_t len,
                          const T *v, T tau) {
  if (tau == 0) return;
  T y[KD];
  for (int64_t r = 0; r < len; r++) y[r] = 0;
  for (int64_t c = 0; c < len; c++) {
    const T *d = b + (st + c) * 2 * kd - c;
    for (int64_t r = c + 1; r < len; r++) y[r] += MUL(d[r], v[c]);
    y[c] += REAL(d[c]) * v[c] + F(dot)(d + c + 1, v + c + 1, len - c - 1);
  }
  for (int64_t r = 0; r < len; r++) y[r] = MUL(y[r], tau);
  T half = MUL((R)-0.5 * tau, F(dot)(y, v, len));
  for (int64_t r = 0; r < len; r++) y[r] += MUL(half, v[r]);
  for (int64_t c = 0; c < len; c++) {
    T *d = b + (st + c) * 2 * kd - c, vc = CONJ(v[c]), yc = CONJ(y[c]);
    for (int64_t r = c; r < len; r++) d[r] -= MUL(y[r], vc) + MUL(v[r], yc);
    d[c] = REAL(d[c]);
  }
}

/* A bulge chase on the band [b] of order [n]: column j's elements
   (j + d, j), d < 2kd, at b[j·2kd + d], B's kd subdiagonals and the
   bulges below them; the reflectors' slots [v], or NULL, which the
   vectors take at kd = KD; the next sweep to claim, and the tasks each
   sweep has finished. */
typedef struct {
  T *b, *v;
  int64_t n, kd, next;
  int64_t *done;
} F(chase);

/* Sweep [s], which makes column s tridiagonal by tasks (xHB2ST_KERNELS):
   task j's reflector H_j takes rows st = s + 1 + j·kd on, kd of them or
   to the last, from column s (j = 0) or from the first column of the
   bulge that H_(j−1) makes, applied from the right, below its own rows;
   H_j then takes the bulge's other columns from the left and its block
   of the diagonal from both sides. H_j's τ and v past its 1 go to its
   slot, KD elements, the sweep's after those of the sweeps before it.
   Task j touches rows and columns s + 1 + (j − 1)·kd to s + (j + 1)·kd,
   which sweep s − 1's tasks past j + 2 do not: it waits until that sweep
   has finished them. */
static void F(sweep)(F(chase) *ch, int64_t s) {
  int64_t n = ch->n, kd = ch->kd, tasks = (n - 2 - s) / kd + 1;
  int64_t before = (n - 1 - s) / kd + 1;
  T *b = ch->b, *slot = ch->v ? ch->v + chased(n, s) * KD : NULL;
  T v[KD], u[KD], tau = 0;
  for (int64_t j = 0; j < tasks; j++) {
    int64_t wait = j + 3 < before ? j + 3 : before;
    if (s > 0) wait_for(&ch->done[(s - 1) * LINE], wait);
    int64_t st = s + 1 + j * kd, len = n - st < kd ? n - st : kd;
    int64_t col = j == 0 ? s : st - kd;
    if (j > 0 && tau != 0) {
      for (int64_t r = 0; r < len; r++) u[r] = 0;
      for (int64_t c = 0; c < kd; c++) {
        const T *x = b + (col + c) * 2 * kd + st - col - c;
        for (int64_t r = 0; r < len; r++) u[r] += MUL(x[r], v[c]);
      }
      for (int64_t c = 0; c < kd; c++) {
        T *x = b + (col + c) * 2 * kd + st - col - c, f = MUL(tau, CONJ(v[c]));
        for (int64_t r = 0; r < len; r++) x[r] -= MUL(u[r], f);
      }
    }
    T *x = b + col * 2 * kd + st - col;
    tau = F(reflector)(x, x + 1, len - 1, 1);
    v[0] = 1;
    for (int64_t r = 1; r < len; r++) {
      v[r] = x[r];
      x[r] = 0;
    }
    for (int64_t c = 1; j > 0 && tau != 0 && c < kd; c++) {
      T *y = b + (col + c) * 2 * kd + st - col - c;
      T f = MUL(CONJ(tau), F(dot)(v, y, len));
      for (int64_t r = 0; r < len; r++) y[r] -= MUL(v[r], f);
    }
    F(both_sides)(b, kd, st, len, v, tau);
    if (slot) {
      slot[j * KD] = tau;
      memcpy(slot + j * KD + 1, v + 1, sizeof(T) * (len - 1));
    }
    __atomic_store_n(&ch->done[s * LINE], j + 1, __ATOMIC_RELEASE);
  }
}

/* A unit claims sweeps until none is left. */
static void F(chase_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  F(chase) *ch = ctx;
  for (int64_t s; (s = __atomic_fetch_add(&ch->next, 1, __ATOMIC_RELAXED)) <
                  ch->n - 1;)
    F(sweep)(ch, s);
}

/* T's diagonal into [d] and subdiagonal into [e] from the band of kd
   subdiagonals in the lower triangle of the [n] × [n] [a], rows [ld]
   apart, by bulge chasing (xHETRD_HB2ST) on its copy in [b], which holds
   2kd·n elements, sweep s's reflectors into [slots] unless NULL. Sweeps
   run on the call's threads, each claiming the next, so that a thread
   waits only on a sweep claimed before its own, and one thread alone runs
   them all in order; each element takes the sweeps' tasks in their
   order, whatever the threads. [done] holds n·LINE indices. */
static void F(hb2st)(const T *a, int64_t n, int64_t ld, T *b, T *slots,
                     int64_t *done, R *d, R *e, int64_t kd, arena *s) {
  memset(b, 0, sizeof(T) * 2 * kd * n);
  for (int64_t j = 0; j < n; j++) {
    for (int64_t k = 0; k <= kd && j + k < n; k++)
      b[j * 2 * kd + k] = a[(j + k) * ld + j];
    b[j * 2 * kd] = REAL(b[j * 2 * kd]);
  }
  memset(done, 0, sizeof(int64_t) * n * LINE);
  F(chase) ch = {.b = b, .v = slots, .n = n, .kd = kd, .next = 0,
                 .done = done};
  int64_t bytes = 2 * kd * n * (int64_t)sizeof(T);
  double cost = threaded(s, n - 1, bytes, 6.0 * kd * n * n, RDT, CPLX);
  if (cost == 0) F(chase_unit)(0, 1, 0, &ch);
  else nx_cpu_job(n - 1, bytes, cost, F(chase_unit), &ch);
  for (int64_t j = 0; j < n; j++) {
    d[j] = REAL(b[j * 2 * kd]);
    if (j < n - 1) e[j] = REAL(b[j * 2 * kd + 1]);
  }
}

/* c ← Q2 c for columns [c0, c1) of the [n] × n [c], rows [ldc] apart,
   from the chase's [slots]. Sweep s's reflector j takes rows s + 1 + j·KD
   on, so that reflectors j of SWEEPS consecutive sweeps make one block,
   I − V T Vᴴ with V of KD + SWEEPS − 1 rows (xLARFT, xLARFB), as PLASMA
   applies them. Reflector j of a later sweep shares rows with reflector
   j + 1 of an earlier one alone, so Q2 = Π_g (B_(g,J) ⋯ B_(g,0)) by
   groups in sweep order: c takes the last group first, its blocks j
   ascending. */
static int F(q2_cols)(const T *slots, int64_t n, T *c, int64_t ldc,
                      int64_t c0, int64_t c1, arena *s) {
  int64_t nc = c1 - c0, sweeps = n - 1, g2 = SWEEPS * SWEEPS;
  T *v = room(s->copy, ((KD + SWEEPS) * SWEEPS + 2 * g2 + SWEEPS +
                        2 * SWEEPS * nc) * (int64_t)sizeof(T));
  if (v == NULL) return 0;
  T *t = v + (KD + SWEEPS) * SWEEPS, *g = t + g2, *tau = g + g2,
    *w = tau + SWEEPS;
  for (int64_t s0 = (sweeps - 1) / SWEEPS * SWEEPS; s0 >= 0; s0 -= SWEEPS) {
    int64_t nb = sweeps - s0 < SWEEPS ? sweeps - s0 : SWEEPS;
    for (int64_t r0 = s0 + 1; r0 < n; r0 += KD) {
      int64_t h = n - r0 < KD + nb - 1 ? n - r0 : KD + nb - 1;
      int64_t jb = n - r0 < nb ? n - r0 : nb, j = (r0 - s0 - 1) / KD;
      memset(v, 0, sizeof(T) * h * jb);
      for (int64_t q = 0; q < jb; q++) {
        const T *sl = slots + (chased(n, s0 + q) + j) * KD;
        int64_t len = n - r0 - q < KD ? n - r0 - q : KD;
        tau[q] = sl[0];
        v[q * jb + q] = 1;
        for (int64_t r = 1; r < len; r++) v[(q + r) * jb + q] = sl[r];
      }
      if (!F(block_t)(v, h, jb, tau, t, jb, g, s) ||
          !F(larfb)(v, jb, F(op)(t, jb, 1), jb, h, nc, c + r0 * ldc + c0, ldc,
                    w, s))
        return 0;
    }
  }
  return 1;
}

/* Q2's application by units of columns on the call's threads; each
   column's arithmetic is the same in any unit. A block's V reads fewer
   than 2KD of c's rows, so that the vectors as c's columns give each
   block rows of c whole: as its rows (rlarfb), Q2 took 2.4 times as long
   in eigh-f64-1024 on the M1, 1.3 in eigh-c128-1024. */
typedef struct {
  const T *slots;
  T *c;
  int64_t n, ldc, units;
  int failed;
  arena *workers;
} F(q2_job);

/* A unit runs on its worker's arena alone, as a solve's do. */
static void F(q2_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(q2_job) *j = ctx;
  arena *s = &j->workers[worker], *all = s->workers;
  s->workers = NULL;
  for (int64_t u = lo; u < hi; u++)
    if (!F(q2_cols)(j->slots, j->n, j->c, j->ldc, u * j->n / j->units,
                    (u + 1) * j->n / j->units, s))
      __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
  s->workers = all;
}

/* c ← Q2 c for the [n] × n [c], rows [ldc] apart. Answers 0 if an
   allocation failed. */
static int F(q2)(const T *slots, int64_t n, T *c, int64_t ldc, arena *s) {
  int64_t bytes = n * n * (int64_t)sizeof(T), units = n / KD;
  double cost = threaded(s, units, bytes, 2.0 * n * n * n, RDT, CPLX);
  if (cost == 0) return F(q2_cols)(slots, n, c, ldc, 0, n, s);
  int threads = nx_cpu_threads(bytes, cost);
  F(q2_job) j = {.slots = slots, .c = c, .n = n, .ldc = ldc,
                 .workers = s->workers,
                 .units = units < threads ? units : threads};
  nx_cpu_job(j.units, bytes, cost, F(q2_unit), &j);
  return !j.failed;
}

/* c, s and r with [c s; −s c] [f; g] = [r; 0], as LAPACK's xLARTG. */
static void F(rotation)(R f, R g, R *c, R *s, R *r) {
  const R lo = SQRT(TINY), hi = SQRT((1 / TINY) / 2);
  R af = F(abs)(f), ag = F(abs)(g);
  if (g == 0) {
    *c = 1;
    *s = 0;
    *r = f;
    return;
  }
  if (f == 0) {
    *c = 0;
    *s = g < 0 ? -1 : 1;
    *r = ag;
    return;
  }
  R u = 1;
  if (!(af > lo && af < hi && ag > lo && ag < hi)) {
    u = af > ag ? af : ag;
    u = u < TINY ? TINY : u > 1 / TINY ? 1 / TINY : u;
    f /= u;
    g /= u;
  }
  R h = SQRT(f * f + g * g), rr = f < 0 ? -h : h;
  *c = F(abs)(f) / h;
  *s = g / rr;
  *r = rr * u;
}

/* The eigenvalues [rt1], of the larger magnitude, and [rt2] of
   [a b; b c], and (cs, sn) with [cs sn; −sn cs] [a b; b c] [cs −sn; sn cs]
   = diag (rt1, rt2), as LAPACK's xLAEV2. */
static void F(laev2)(R a, R b, R c, R *rt1, R *rt2, R *cs, R *sn) {
  R sm = a + c, df = a - c, adf = F(abs)(df), tb = b + b, ab = F(abs)(tb);
  R acmx = F(abs)(a) > F(abs)(c) ? a : c, acmn = F(abs)(a) > F(abs)(c) ? c : a;
  R rt = adf > ab   ? adf * SQRT(1 + (ab / adf) * (ab / adf))
         : adf < ab ? ab * SQRT(1 + (adf / ab) * (adf / ab))
                    : ab * SQRT((R)2);
  int sgn1 = sm < 0 ? -1 : 1;
  if (sm == 0) {
    *rt1 = (R)0.5 * rt;
    *rt2 = (R)-0.5 * rt;
  } else {
    *rt1 = (R)0.5 * (sm + sgn1 * rt);
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  }
  int sgn2 = df >= 0 ? 1 : -1;
  R cv = df + sgn2 * rt, c1, s1;
  if (F(abs)(cv) > ab) {
    R ct = -tb / cv;
    s1 = 1 / SQRT(1 + ct * ct);
    c1 = ct * s1;
  } else if (ab == 0) {
    c1 = 1;
    s1 = 0;
  } else {
    R tn = -cv / tb;
    c1 = 1 / SQRT(1 + tn * tn);
    s1 = tn * c1;
  }
  if (sgn1 == sgn2) {
    R tn = c1;
    c1 = -s1;
    s1 = tn;
  }
  *cs = c1;
  *sn = s1;
}

/* [x; y] ← [c s; −s c] [x; y] over [n] elements (xROT). */
static void F(rot)(R *x, R *y, int64_t n, R c, R s) {
  for (int64_t i = 0; i < n; i++) {
    R yi = y[i];
    y[i] = c * yi - s * x[i];
    x[i] = s * yi + c * x[i];
  }
}

/* Columns [j0, j0 + k) of the [n]-row [z], columns [ldz] apart, times the
   rotations (c[j], s[j]) of columns j0 + j and j0 + j + 1, the last first
   under [backward] (xLASR's 'R', 'V'). */
static void F(rotate)(R *z, int64_t n, int64_t ldz, int64_t k, const R *c,
                      const R *s, int backward) {
  for (int64_t t = 0; t < k - 1; t++) {
    int64_t j = backward ? k - 2 - t : t;
    if (c[j] == 1 && s[j] == 0) continue;
    F(rot)(z + j * ldz, z + (j + 1) * ldz, n, c[j], s[j]);
  }
}

/* d[l..lend] and e[l..lend) times [to] / [from]. */
static void F(scale)(R *d, R *e, int64_t l, int64_t lend, R from, R to) {
  for (int64_t i = l; i <= lend; i++) {
    d[i] = d[i] / from * to;
    if (i < lend) e[i] = e[i] / from * to;
  }
}

/* Swaps the [n] elements at [x] and [y]. */
static void F(rswap)(R *x, R *y, int64_t n) {
  for (int64_t r = 0; r < n; r++) {
    R t = x[r];
    x[r] = y[r];
    y[r] = t;
  }
}

/* d[0..n) in the order [o], by selection as
   xSTEQR, xSTEDC, xLASDQ and xBDSDC, with the [nz] rows of z's columns,
   [ldz] apart, and the [ny] rows of y's, [ldy] apart, where they are not
   NULL. */
static void F(sort)(int64_t n, R *d, order o, R *z, int64_t nz, int64_t ldz,
                    R *y, int64_t ny, int64_t ldy) {
  for (int64_t i = 0; i < n - 1; i++) {
    int64_t k = i;
    for (int64_t j = i + 1; j < n; j++)
      if (o == DESCENDING ? d[j] > d[k] : d[j] < d[k]) k = j;
    if (k == i) continue;
    R t = d[k];
    d[k] = d[i];
    d[i] = t;
    if (z) F(rswap)(z + k * ldz, z + i * ldz, nz);
    if (y) F(rswap)(y + k * ldy, y + i * ldy, ny);
  }
}

/* The eigenvalues of the symmetric tridiagonal T of the diagonal [d] and
   subdiagonal [e], [n] of them, ascending into d, and, where [z] is not
   NULL, z Z for T = Z diag d Zᵀ, z of [nz] rows and n columns [ldz] apart:
   implicit QL or QR with Wilkinson shifts, whichever chases the bulge
   from the end of larger magnitude, on each unreduced block scaled into
   range, as LAPACK's xSTEQR. Values alone run the same iteration, so they
   are the bits the vectors' call gives. [rot] holds 2n reals. Answers 0
   if the iteration takes more than 30 n sweeps. */
static int F(steqr)(int64_t n, R *d, R *e, R *z, int64_t nz, int64_t ldz,
                   R *rot) {
  const R eps2 = EPS * EPS, safmin = TINY;
  const R ssfmax = SQRT(1 / safmin) / 3, ssfmin = SQRT(safmin) / eps2;
  int64_t maxit = 30 * n, jtot = 0, l1 = 0;
  R *cs = rot, *sn = rot + n;
  while (l1 < n) {
    if (l1 > 0) e[l1 - 1] = 0;
    /* The block [l1, m]: the first negligible subdiagonal from l1. */
    int64_t m = l1;
    for (; m < n - 1; m++) {
      R t = F(abs)(e[m]);
      if (t == 0) break;
      if (t <= SQRT(F(abs)(d[m])) * SQRT(F(abs)(d[m + 1])) * EPS) {
        e[m] = 0;
        break;
      }
    }
    int64_t l = l1, lend = m, lsv = l, lendsv = lend;
    l1 = m + 1;
    if (lend == l) continue;
    R anorm = 0;
    for (int64_t i = l; i <= lend; i++) {
      R x = F(abs)(d[i]), y = i < lend ? F(abs)(e[i]) : 0;
      anorm = x > anorm ? x : anorm;
      anorm = y > anorm ? y : anorm;
    }
    if (anorm == 0) continue;
    /* Out of [ssfmin, ssfmax], the block is scaled to its nearer end. */
    R sc = anorm > ssfmax ? ssfmax : anorm < ssfmin ? ssfmin : anorm;
    if (sc != anorm) F(scale)(d, e, l, lend, anorm, sc);
    if (F(abs)(d[lend]) < F(abs)(d[l])) {
      lend = lsv;
      l = lendsv;
    }
    if (lend > l) {
      /* QL: deflate from the top. */
      while (l <= lend) {
        for (m = l; m < lend; m++) {
          R t = F(abs)(e[m]);
          if (t * t <= (eps2 * F(abs)(d[m])) * F(abs)(d[m + 1]) + safmin)
            break;
        }
        if (m < lend) e[m] = 0;
        R p = d[l];
        if (m == l) {
          l++;
          continue;
        }
        if (m == l + 1) {
          R rt1, rt2, c, s;
          F(laev2)(d[l], e[l], d[l + 1], &rt1, &rt2, &c, &s);
          if (z) F(rotate)(z + l * ldz, nz, ldz, 2, &c, &s, 1);
          d[l] = rt1;
          d[l + 1] = rt2;
          e[l] = 0;
          l += 2;
          continue;
        }
        if (jtot == maxit) return 0;
        jtot++;
        R g = (d[l + 1] - p) / (2 * e[l]), r = F(lapy3)(g, 1, 0);
        g = d[m] - p + (e[l] / (g + (g < 0 ? -r : r)));
        R s = 1, c = 1;
        p = 0;
        for (int64_t i = m - 1; i >= l; i--) {
          R f = s * e[i], b = c * e[i];
          F(rotation)(g, f, &c, &s, &r);
          if (i != m - 1) e[i + 1] = r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          cs[i] = c;
          sn[i] = -s;
        }
        if (z) F(rotate)(z + l * ldz, nz, ldz, m - l + 1, cs + l, sn + l, 1);
        d[l] -= p;
        e[l] = g;
      }
    } else {
      /* QR: deflate from the bottom. */
      while (l >= lend) {
        for (m = l; m > lend; m--) {
          R t = F(abs)(e[m - 1]);
          if (t * t <= (eps2 * F(abs)(d[m])) * F(abs)(d[m - 1]) + safmin)
            break;
        }
        if (m > lend) e[m - 1] = 0;
        R p = d[l];
        if (m == l) {
          l--;
          continue;
        }
        if (m == l - 1) {
          R rt1, rt2, c, s;
          F(laev2)(d[l - 1], e[l - 1], d[l], &rt1, &rt2, &c, &s);
          if (z) F(rotate)(z + (l - 1) * ldz, nz, ldz, 2, &c, &s, 0);
          d[l - 1] = rt1;
          d[l] = rt2;
          e[l - 1] = 0;
          l -= 2;
          continue;
        }
        if (jtot == maxit) return 0;
        jtot++;
        R g = (d[l - 1] - p) / (2 * e[l - 1]), r = F(lapy3)(g, 1, 0);
        g = d[m] - p + (e[l - 1] / (g + (g < 0 ? -r : r)));
        R s = 1, c = 1;
        p = 0;
        for (int64_t i = m; i < l; i++) {
          R f = s * e[i], b = c * e[i];
          F(rotation)(g, f, &c, &s, &r);
          if (i != m) e[i - 1] = r;
          g = d[i] - p;
          r = (d[i + 1] - g) * s + 2 * c * b;
          p = s * r;
          d[i] = g + p;
          g = c * r - b;
          cs[i] = c;
          sn[i] = s;
        }
        if (z) F(rotate)(z + m * ldz, nz, ldz, l - m + 1, cs + m, sn + m, 0);
        d[l] -= p;
        e[l - 1] = g;
      }
    }
    if (sc != anorm) F(scale)(d, e, lsv, lendsv, sc, anorm);
  }
  F(sort)(n, d, ASCENDING, z, nz, ldz, NULL, 0, 0);
  return 1;
}

/* Divide and conquer

   T's eigenvectors as LAPACK's xSTEDC computes them: T torn in two by a
   rank-one change, T = diag (T1, T2) + ρ u uᵀ, each half solved the same
   way down to blocks of at most SMLSIZ by steqr, then the halves' spectra
   merged (xLAED0, xLAED1). A merge deflates the values whose part of z
   is negligible, or which a rotation makes equal to a neighbour
   (xLAED2), finds the others as the roots of the secular equation
   1 + ρ Σ z_j² / (d_j − λ) = 0 (xLAED4), and their vectors by Gu and
   Eisenstat's recomputed z, so they stay orthogonal (xLAED3); the
   product with the halves' vectors is the merge's O(n³), by products
   (product) on their nonzero blocks. Matrices are column-major. */

#define SMLSIZ 25

/* index[0..n1 + n2) the merge, ascending, of a[0..n1), ascending by
   [s1] = 1 or descending by −1, and a[n1..n1 + n2) by [s2] (xLAMRG). */
static void F(lamrg)(int64_t n1, int64_t n2, const R *a, int s1, int s2,
                     int64_t *index) {
  int64_t i1 = s1 > 0 ? 0 : n1 - 1, i2 = s2 > 0 ? n1 : n1 + n2 - 1, i = 0;
  while (n1 > 0 && n2 > 0) {
    if (a[i1] <= a[i2]) {
      index[i++] = i1;
      i1 += s1;
      n1--;
    } else {
      index[i++] = i2;
      i2 += s2;
      n2--;
    }
  }
  for (; n1 > 0; n1--, i1 += s1) index[i++] = i1;
  for (; n2 > 0; n2--, i2 += s2) index[i++] = i2;
}

/* The root [i] of the secular equation of two values, d[0] < d[1], into
   [dlam], and delta_j = (d_j − λ) / ‖·‖ the normalised ones (xLAED5). */
static void F(laed5)(int i, const R *d, const R *z, R *delta, R rho,
                     R *dlam) {
  R del = d[1] - d[0], tau;
  R b = -del + rho * (z[0] * z[0] + z[1] * z[1]), c = rho * z[1] * z[1] * del;
  if (i == 0 &&
      1 + 2 * rho * (z[1] * z[1] - z[0] * z[0]) / del > 0) {
    b = del + rho * (z[0] * z[0] + z[1] * z[1]);
    c = rho * z[0] * z[0] * del;
    tau = 2 * c / (b + SQRT(F(abs)(b * b - 4 * c)));
    *dlam = d[0] + tau;
    delta[0] = -z[0] / tau;
    delta[1] = z[1] / (del - tau);
  } else {
    if (i == 0)
      tau = b > 0 ? -2 * c / (b + SQRT(b * b + 4 * c))
                  : (b - SQRT(b * b + 4 * c)) / 2;
    else
      tau = b > 0 ? (b + SQRT(b * b + 4 * c)) / 2
                  : 2 * c / (-b + SQRT(b * b + 4 * c));
    *dlam = d[1] + tau;
    delta[0] = -z[0] / (del + tau);
    delta[1] = -z[1] / tau;
  }
  R t = SQRT(delta[0] * delta[0] + delta[1] * delta[1]);
  delta[0] /= t;
  delta[1] /= t;
}

/* The root τ near zero of f(x) = rho + Σ z_j / (d_j − x) over three poles,
   between d[1] and d[2] under [orgati], else d[0] and d[1], f(0) =
   [finit], by Newton's method on a rational model (xLAED6). [kniter] 2
   asks for a fresh first guess. Answers 0 after 40 steps. */
static int F(laed6)(int64_t kniter, int orgati, R rho, const R *d,
                    const R *z, R finit, R *tau_) {
  R lbd = orgati ? d[1] : d[0], ubd = orgati ? d[2] : d[1], tau = 0;
  if (finit < 0) lbd = 0;
  else ubd = 0;
  if (kniter == 2) {
    R a, b, c;
    if (orgati) {
      R t = (d[2] - d[1]) / 2;
      c = rho + z[0] / ((d[0] - d[1]) - t);
      a = c * (d[1] + d[2]) + z[1] + z[2];
      b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
    } else {
      R t = (d[0] - d[1]) / 2;
      c = rho + z[2] / ((d[2] - d[1]) - t);
      a = c * (d[0] + d[1]) + z[0] + z[1];
      b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
    }
    R t = F(abs)(a) > F(abs)(b) ? F(abs)(a) : F(abs)(b);
    t = F(abs)(c) > t ? F(abs)(c) : t;
    a /= t;
    b /= t;
    c /= t;
    tau = c == 0   ? b / a
          : a <= 0 ? (a - SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c)
                   : 2 * b / (a + SQRT(F(abs)(a * a - 4 * b * c)));
    if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2;
    if (d[0] == tau || d[1] == tau || d[2] == tau) tau = 0;
    else {
      R f = finit + tau * z[0] / (d[0] * (d[0] - tau)) +
            tau * z[1] / (d[1] * (d[1] - tau)) +
            tau * z[2] / (d[2] * (d[2] - tau));
      if (f <= 0) lbd = tau;
      else ubd = tau;
      if (F(abs)(finit) <= F(abs)(f)) tau = 0;
    }
  }
  /* Poles nearer τ than the cube root of the least normal number are
     scaled away from it. */
  const R small1 = sizeof(R) == 4 ? 0x1p-42f : 0x1p-340,
          small2 = small1 * small1;
  R t = orgati ? d[1] - tau : d[0] - tau, t2 = orgati ? d[2] - tau : d[1] - tau;
  t = F(abs)(t) < F(abs)(t2) ? F(abs)(t) : F(abs)(t2);
  R sclfac = 1, sclinv = 1;
  if (t <= small1) {
    sclfac = t <= small2 ? 1 / small2 : 1 / small1;
    sclinv = t <= small2 ? small2 : small1;
  }
  R ds[3], zs[3];
  for (int i = 0; i < 3; i++) {
    ds[i] = d[i] * sclfac;
    zs[i] = z[i] * sclfac;
  }
  tau *= sclfac;
  lbd *= sclfac;
  ubd *= sclfac;
  R fc = 0, df = 0, ddf = 0;
  for (int i = 0; i < 3; i++) {
    R r = 1 / (ds[i] - tau), t1 = zs[i] * r, t2 = t1 * r;
    fc += t1 / ds[i];
    df += t2;
    ddf += t2 * r;
  }
  R f = finit + tau * fc;
  int ok = F(abs)(f) <= 0;
  if (f <= 0) lbd = tau;
  else ubd = tau;
  for (int it = 2; it <= 40 && !ok; it++) {
    R t1 = orgati ? ds[1] - tau : ds[0] - tau;
    R t2 = orgati ? ds[2] - tau : ds[1] - tau;
    R a = (t1 + t2) * f - t1 * t2 * df, b = t1 * t2 * f;
    R c = f - (t1 + t2) * df + t1 * t2 * ddf;
    R m = F(abs)(a) > F(abs)(b) ? F(abs)(a) : F(abs)(b);
    m = F(abs)(c) > m ? F(abs)(c) : m;
    a /= m;
    b /= m;
    c /= m;
    R eta = c == 0   ? b / a
            : a <= 0 ? (a - SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c)
                     : 2 * b / (a + SQRT(F(abs)(a * a - 4 * b * c)));
    if (f * eta >= 0) eta = -f / df;
    tau += eta;
    if (tau < lbd || tau > ubd) tau = (lbd + ubd) / 2;
    R err = 0;
    fc = df = ddf = 0;
    int i = 0;
    for (; i < 3 && ds[i] - tau != 0; i++) {
      R r = 1 / (ds[i] - tau), q1 = zs[i] * r, q2 = q1 * r, q4 = q1 / ds[i];
      fc += q4;
      err += F(abs)(q4);
      df += q2;
      ddf += q2 * r;
    }
    /* τ on a pole is a root. */
    if (i < 3) {
      ok = 1;
      break;
    }
    f = finit + tau * fc;
    err = 8 * (F(abs)(finit) + F(abs)(tau) * err) + F(abs)(tau) * df;
    ok = F(abs)(f) <= 4 * EPS * err || ubd - lbd <= 4 * EPS * F(abs)(tau);
    if (f <= 0) lbd = tau;
    else ubd = tau;
  }
  *tau_ = tau * sclinv;
  return ok;
}

/* Root [i] of 1 + ρ Σ z_j² / (d_j − λ) = 0 for d ascending, z without
   zeros and ρ > 0, into [dlam], and delta_j = d_j − λ, each from the pole
   nearer the root so that it keeps its accuracy (xLAED4): Newton's method
   on a rational model of the two nearest poles, or of three (laed6) where
   it would step the wrong way, in [lb, ub] bisected where it leaves them.
   Answers 0 after 29 steps. */
static int F(laed4)(int64_t n, int64_t i, const R *d, const R *z, R *delta,
                    R rho, R *dlam) {
  if (n == 1) {
    *dlam = d[0] + rho * z[0] * z[0];
    delta[0] = 1;
    return 1;
  }
  if (n == 2) {
    F(laed5)((int)i, d, z, delta, rho, dlam);
    return 1;
  }
  const R rhoinv = 1 / rho;
  int last = i == n - 1, orgati = 1;
  R tau, lb, ub, org;
  /* The first guess, from the two poles about the root and the rest at
     the midpoint. */
  if (last) {
    R midpt = rho / 2, psi = 0, del = d[n - 1] - d[n - 2];
    for (int64_t j = 0; j < n; j++) delta[j] = (d[j] - d[n - 1]) - midpt;
    for (int64_t j = 0; j < n - 2; j++) psi += z[j] * z[j] / delta[j];
    R c = rhoinv + psi;
    R w = c + z[n - 2] * z[n - 2] / delta[n - 2] +
          z[n - 1] * z[n - 1] / delta[n - 1];
    R a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
    R b = z[n - 1] * z[n - 1] * del;
    tau = a < 0 ? 2 * b / (SQRT(a * a + 4 * b * c) - a)
                : (a + SQRT(a * a + 4 * b * c)) / (2 * c);
    lb = 0;
    ub = midpt;
    if (w <= 0) {
      lb = midpt;
      ub = rho;
      if (c <= z[n - 2] * z[n - 2] / (del + rho) + z[n - 1] * z[n - 1] / rho)
        tau = rho;
    }
    org = d[n - 1];
  } else {
    R del = d[i + 1] - d[i], midpt = del / 2, psi = 0, phi = 0;
    for (int64_t j = 0; j < n; j++) delta[j] = (d[j] - d[i]) - midpt;
    for (int64_t j = 0; j < i; j++) psi += z[j] * z[j] / delta[j];
    for (int64_t j = n - 1; j > i + 1; j--) phi += z[j] * z[j] / delta[j];
    R c = rhoinv + psi + phi, zi = z[i] * z[i], zj = z[i + 1] * z[i + 1];
    R w = c + zi / delta[i] + zj / delta[i + 1];
    orgati = w > 0;
    if (orgati) {
      R a = c * del + zi + zj, b = zi * del;
      tau = a > 0 ? 2 * b / (a + SQRT(F(abs)(a * a - 4 * b * c)))
                  : (a - SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c);
      lb = 0;
      ub = midpt;
    } else {
      R a = c * del - zi - zj, b = zj * del;
      tau = a < 0 ? 2 * b / (a - SQRT(F(abs)(a * a + 4 * b * c)))
                  : -(a + SQRT(F(abs)(a * a + 4 * b * c))) / (2 * c);
      lb = -midpt;
      ub = 0;
    }
    org = orgati ? d[i] : d[i + 1];
  }
  for (int64_t j = 0; j < n; j++) delta[j] = (d[j] - org) - tau;
  /* ψ sums the poles below the root's nearer one [ii], φ those above. */
  int64_t ii = last ? n - 1 : orgati ? i : i + 1, ip = last ? n - 2 : i;
  int swtch3 = 0, swtch = 0;
  R w = 0, dw = 0, psi = 0, dpsi = 0, phi = 0, dphi = 0, err = 0, prew = 0;
  for (int64_t step = 0;; step++) {
    psi = dpsi = err = phi = dphi = 0;
    for (int64_t j = 0; j < ii; j++) {
      R t = z[j] / delta[j];
      psi += z[j] * t;
      dpsi += t * t;
      err += psi;
    }
    err = F(abs)(err);
    for (int64_t j = n - 1; j > ii; j--) {
      R t = z[j] / delta[j];
      phi += z[j] * t;
      dphi += t * t;
      err += phi;
    }
    R t = z[ii] / delta[ii];
    if (last) {
      /* The root's pole is φ's, the last. */
      phi = z[ii] * t;
      dphi = t * t;
      w = rhoinv + phi + psi;
      dw = dpsi + dphi;
      err = 8 * (-phi - psi) + err - phi + rhoinv + F(abs)(tau) * dw;
    } else {
      w = rhoinv + phi + psi;
      if (step == 0)
        swtch3 = (orgati ? w < 0 : w > 0) && ii != 0 && ii != n - 1;
      dw = dpsi + dphi + t * t;
      t *= z[ii];
      w += t;
      err = 8 * (phi - psi) + err + 2 * rhoinv + 3 * F(abs)(t) +
            F(abs)(tau) * dw;
      if (step == 1)
        swtch = orgati ? -w > F(abs)(prew) / 10 : w > F(abs)(prew) / 10;
      else if (step > 1 && w * prew > 0 && F(abs)(w) > F(abs)(prew) / 10)
        swtch = !swtch;
    }
    if (step == 29) break;
    if (F(abs)(w) <= EPS * err) {
      *dlam = org + tau;
      return 1;
    }
    if (w <= 0) lb = lb > tau ? lb : tau;
    else ub = ub < tau ? ub : tau;
    R eta, a, b, c;
    R di = delta[ip], dj = delta[ip + 1];
    if (last) {
      c = w - di * dpsi - dj * dphi;
      a = (di + dj) * w - di * dj * dw;
      b = di * dj * w;
      if (step == 0 && c < 0) c = F(abs)(c);
      if (step == 0 && c == 0) eta = -w / dw;
      else if (a >= 0) eta = (a + SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c);
      else eta = 2 * b / (a - SQRT(F(abs)(a * a - 4 * b * c)));
      if (w * eta > 0) eta = -w / dw;
    } else if (!swtch3) {
      if (swtch) {
        R q = z[ii] / delta[ii];
        if (orgati) dpsi += q * q;
        else dphi += q * q;
        c = w - di * dpsi - dj * dphi;
      } else {
        R q = orgati ? z[i] / di : z[i + 1] / dj;
        c = orgati ? w - dj * dw - (d[i] - d[i + 1]) * (q * q)
                   : w - di * dw - (d[i + 1] - d[i]) * (q * q);
      }
      a = (di + dj) * w - di * dj * dw;
      b = di * dj * w;
      if (c == 0) {
        if (a == 0)
          a = swtch    ? di * di * dpsi + dj * dj * dphi
              : orgati ? z[i] * z[i] + dj * dj * (dpsi + dphi)
                       : z[i + 1] * z[i + 1] + di * di * (dpsi + dphi);
        eta = b / a;
      } else if (a <= 0)
        eta = (a - SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c);
      else
        eta = 2 * b / (a + SQRT(F(abs)(a * a - 4 * b * c)));
      if (w * eta >= 0) eta = -w / dw;
    } else {
      R zz[3], s = rhoinv + psi + phi, dm = delta[ii - 1], dp = delta[ii + 1];
      if (swtch) {
        c = s - dm * dpsi - dp * dphi;
        zz[0] = dm * dm * dpsi;
        zz[2] = dp * dp * dphi;
      } else if (orgati) {
        R q = z[ii - 1] / dm;
        q *= q;
        c = s - dp * (dpsi + dphi) - (d[ii - 1] - d[ii + 1]) * q;
        zz[0] = z[ii - 1] * z[ii - 1];
        zz[2] = dp * dp * ((dpsi - q) + dphi);
      } else {
        R q = z[ii + 1] / dp;
        q *= q;
        c = s - dm * (dpsi + dphi) - (d[ii + 1] - d[ii - 1]) * q;
        zz[0] = dm * dm * (dpsi + (dphi - q));
        zz[2] = z[ii + 1] * z[ii + 1];
      }
      zz[1] = z[ii] * z[ii];
      if (!F(laed6)(step + 2, orgati, c, delta + ii - 1, zz, w, &eta))
        return 0;
      if (w * eta >= 0) eta = -w / dw;
    }
    /* Out of [lb, ub], half the way to the bound, or where Newton's step
       stays inside, the geometric mean of the two steps
       (Reference-LAPACK's PR 1191, on the last root's first step and the
       others' later ones). */
    if (tau + eta > ub || tau + eta < lb) {
      R half = w < 0 ? (ub - tau) / 2 : (lb - tau) / 2, newton = -w / dw;
      int mean = (last ? step == 0 : step > 0) && lb <= tau + newton &&
                 tau + newton <= ub;
      eta = mean ? F(sign)(1, newton) * SQRT(F(abs)(newton)) *
                       SQRT(F(abs)(half))
                 : half;
    }
    prew = w;
    for (int64_t j = 0; j < n; j++) delta[j] -= eta;
    tau += eta;
  }
  *dlam = org + tau;
  return 0;
}

/* c += a b for column-major c of [m] × [n], [ldc] apart, a of m × [k], [lda]
   apart, b of k × n, [ldb] apart, all of reals, by the contraction on
   cᵀ = bᵀ aᵀ. Answers 0 if an allocation failed. */
static int F(raccum)(int64_t m, int64_t n, int64_t k, const R *a,
                     int64_t lda, const R *b, int64_t ldb, R *c, int64_t ldc) {
  return k == 0 || product(RDT, n, m, k, (strided){b, ldb, 1},
                           (strided){a, lda, 1}, c, ldc);
}

/* c ← a b, as raccum. */
static int F(rgemm)(int64_t m, int64_t n, int64_t k, const R *a, int64_t lda,
                    const R *b, int64_t ldb, R *c, int64_t ldc) {
  for (int64_t j = 0; j < n; j++) memset(c + j * ldc, 0, sizeof(R) * m);
  return F(raccum)(m, n, k, a, lda, b, ldb, c, ldc);
}

/* A merge's work on its [k] roots (secular), element by element. */
typedef struct {
  secular phase;
  int64_t k, ldq;
  R *q, *dlambda, *w, *t, *d, rho;
  const int64_t *indxc;
  int failed;
} F(secular_job);

static void F(secular_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  F(secular_job) *j = ctx;
  int64_t k = j->k, ldq = j->ldq, i0 = lo * MERGE_ROOTS;
  int64_t i1 = hi * MERGE_ROOTS < k ? hi * MERGE_ROOTS : k;
  R *q = j->q, *dl = j->dlambda, *w = j->w;
  switch (j->phase) {
    case ROOTS:
      for (int64_t i = i0; i < i1; i++)
        if (!F(laed4)(k, i, dl, w, q + i * ldq, j->rho, j->d + i))
          __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
      break;
    case WEIGHTS:
      /* Down the columns, so that the rows' products are independent and
         read q as stored; each still multiplies in the columns' order. */
      for (int64_t i = i0; i < i1; i++) w[i] = q[i + i * ldq];
      for (int64_t c = 0; c < k; c++)
        for (int64_t i = i0; i < i1; i++)
          if (i != c) w[i] *= q[i + c * ldq] / (dl[i] - dl[c]);
      break;
    case VECTORS:
      for (int64_t i = i0; i < i1; i++) {
        R *qc = q + i * ldq, *t = j->t + i * k, big = 0, sum = 0;
        for (int64_t r = 0; r < k; r++) {
          t[r] = w[r] / qc[r];
          big = F(abs)(t[r]) > big ? F(abs)(t[r]) : big;
        }
        for (int64_t r = 0; r < k; r++) sum += (t[r] / big) * (t[r] / big);
        R norm = big * SQRT(sum);
        for (int64_t r = 0; r < k; r++) qc[r] = t[j->indxc[r]] / norm;
      }
      break;
  }
}

/* [unit] over a merge's [k] roots, by units of MERGE_ROOTS on the call's
   threads where [s] holds them: a phase that reads [matrices] k × k
   matrices of reals and does [flops] times k² work. */
static void F(on_roots)(int64_t k, int64_t matrices, double flops,
                        rig_pool_body unit, void *j, arena *s) {
  int64_t units = (k + MERGE_ROOTS - 1) / MERGE_ROOTS;
  int64_t bytes = matrices * k * k * (int64_t)sizeof(R);
  double cost = threaded(s, units, bytes, flops * k * k, RDT, 0);
  if (cost > 0)
    nx_cpu_job(units, bytes, cost, unit, j);
  else
    unit(0, units, 0, j);
}

/* [j]'s phase over its k roots. Answers 0 if a root did not converge. */
static int F(secular)(F(secular_job) *j, secular phase, arena *s) {
  j->phase = phase;
  F(on_roots)(j->k, 1, phase == ROOTS ? 8 : 1, F(secular_unit), j, s);
  return !j->failed;
}

/* Merges the spectra of the halves of [n] values, the first [n1], each
   ascending as indxq orders it, with the eigenvectors in the columns of
   q, [ldq] apart, torn by ρ (xLAED1): the values into d, their order into
   indxq, the vectors into q. [work] holds 3n + 2n² reals and [iwork] 4n
   indices. Answers 1, 0 if a root does not converge, or −1 if an
   allocation failed. */
static int F(merge)(int64_t n, int64_t n1, R *d, R *q, int64_t ldq,
                    int64_t *indxq, R rho, R *work, int64_t *iwork,
                    arena *sa) {
  int64_t n2 = n - n1, k = 0, k2 = n, ctot[4] = {0}, psm[4];
  R *z = work, *dlambda = z + n, *w = dlambda + n, *q2 = w + n,
    *s = q2 + n * n;
  int64_t *indx = iwork, *indxc = indx + n, *coltyp = indxc + n,
          *indxp = coltyp + n;
  /* z: the last row of the first half's vectors and the first of the
     second's, ‖z‖ = 1 once scaled by 1/√2, ρ = |2ρ|. */
  for (int64_t j = 0; j < n1; j++) z[j] = q[(n1 - 1) + j * ldq];
  for (int64_t j = n1; j < n; j++) z[j] = q[n1 + j * ldq];
  if (rho < 0)
    for (int64_t j = n1; j < n; j++) z[j] = -z[j];
  for (int64_t j = 0; j < n; j++) z[j] *= 1 / SQRT((R)2);
  rho = F(abs)(2 * rho);
  /* Deflation (xLAED2), the values in ascending order indx. */
  for (int64_t i = n1; i < n; i++) indxq[i] += n1;
  for (int64_t i = 0; i < n; i++) dlambda[i] = d[indxq[i]];
  F(lamrg)(n1, n2, dlambda, 1, 1, indxc);
  for (int64_t i = 0; i < n; i++) indx[i] = indxq[indxc[i]];
  R zmax = 0, dmax = 0;
  for (int64_t i = 0; i < n; i++) {
    zmax = F(abs)(z[i]) > zmax ? F(abs)(z[i]) : zmax;
    dmax = F(abs)(d[i]) > dmax ? F(abs)(d[i]) : dmax;
  }
  R tol = 8 * EPS * (dmax > zmax ? dmax : zmax);
  if (rho * zmax <= tol) {
    /* Every value deflates: the halves' spectra, merged. */
    for (int64_t j = 0; j < n; j++) {
      memcpy(q2 + j * n, q + indx[j] * ldq, sizeof(R) * n);
      dlambda[j] = d[indx[j]];
    }
    for (int64_t j = 0; j < n; j++) {
      memcpy(q + j * ldq, q2 + j * n, sizeof(R) * n);
      d[j] = dlambda[j];
      indxq[j] = j;
    }
    return 1;
  }
  /* Column types: 1 nonzero in the first half's rows alone, 3 in the
     second's, 2 in both (two mixed by a rotation), 4 deflated. */
  for (int64_t i = 0; i < n; i++) coltyp[i] = i < n1 ? 1 : 3;
  int64_t j = 0, pj = 0;
  for (; j < n; j++) {
    int64_t nj = indx[j];
    if (rho * F(abs)(z[nj]) > tol) {
      pj = nj;
      break;
    }
    coltyp[nj] = 4;
    indxp[--k2] = nj;
  }
  for (j++; j < n; j++) {
    int64_t nj = indx[j];
    if (rho * F(abs)(z[nj]) <= tol) {
      coltyp[nj] = 4;
      indxp[--k2] = nj;
      continue;
    }
    /* Two close values: a rotation zeroes pj's part of z where it moves
       no value by more than tol. */
    R sn = z[pj], cs = z[nj], tau = F(lapy3)(cs, sn, 0), t = d[nj] - d[pj];
    cs /= tau;
    sn = -sn / tau;
    if (F(abs)(t * cs * sn) > tol) {
      dlambda[k] = d[pj];
      w[k] = z[pj];
      indxp[k++] = pj;
      pj = nj;
      continue;
    }
    z[nj] = tau;
    z[pj] = 0;
    if (coltyp[nj] != coltyp[pj]) coltyp[nj] = 2;
    coltyp[pj] = 4;
    R *x = q + pj * ldq, *y = q + nj * ldq;
    for (int64_t r = 0; r < n; r++) {
      R xr = x[r];
      x[r] = cs * xr + sn * y[r];
      y[r] = cs * y[r] - sn * xr;
    }
    t = d[pj] * (cs * cs) + d[nj] * (sn * sn);
    d[nj] = d[pj] * (sn * sn) + d[nj] * (cs * cs);
    d[pj] = t;
    /* pj into the deflated values, indxp[k2..n), kept descending: lamrg
       reads them by −1. */
    int64_t p = --k2;
    for (; p + 1 < n && d[pj] < d[indxp[p + 1]]; p++) indxp[p] = indxp[p + 1];
    indxp[p] = pj;
    pj = nj;
  }
  dlambda[k] = d[pj];
  w[k] = z[pj];
  indxp[k++] = pj;
  /* The vectors grouped by type into q2, packed: types 1 and 2's first
     half rows, then types 2 and 3's second half rows, then the deflated
     ones whole, which go back to q's last columns. */
  for (int64_t i = 0; i < n; i++) ctot[coltyp[i] - 1]++;
  psm[0] = 0;
  for (int t = 1; t < 4; t++) psm[t] = psm[t - 1] + ctot[t - 1];
  for (int64_t i = 0; i < n; i++) {
    int64_t js = indxp[i], ct = coltyp[js] - 1;
    indx[psm[ct]] = js;
    indxc[psm[ct]++] = i;
  }
  int64_t n12 = ctot[0] + ctot[1], n23 = ctot[1] + ctot[2];
  R *top = q2, *bot = q2 + n12 * n1, *defl = bot + n23 * n2;
  for (int64_t i = 0; i < n; i++) {
    const R *col = q + indx[i] * ldq;
    z[i] = d[indx[i]];
    if (i >= k) memcpy(defl + (i - k) * n, col, sizeof(R) * n);
    else {
      if (i < n12) memcpy(top + i * n1, col, sizeof(R) * n1);
      if (i >= ctot[0])
        memcpy(bot + (i - ctot[0]) * n2, col + n1, sizeof(R) * n2);
    }
  }
  for (int64_t i = k; i < n; i++) {
    memcpy(q + i * ldq, defl + (i - k) * n, sizeof(R) * n);
    d[i] = z[i];
  }
  /* The k roots, their vectors of the secular equation in q's first k
     rows (xLAED3). */
  F(secular_job) sj = {.k = k, .ldq = ldq, .q = q, .dlambda = dlambda,
                       .w = w, .t = s, .d = d, .rho = rho, .indxc = indxc};
  if (!F(secular)(&sj, ROOTS, sa)) return 0;
  if (k == 2)
    for (int64_t i = 0; i < 2; i++) {
      R *c = q + i * ldq, w0 = c[0], w1 = c[1];
      c[0] = indxc[0] == 0 ? w0 : w1;
      c[1] = indxc[1] == 0 ? w0 : w1;
    }
  if (k > 2) {
    /* z recomputed from the roots (Gu and Eisenstat), so that the vectors
       are orthogonal to working accuracy. */
    memcpy(s, w, sizeof(R) * k);
    F(secular)(&sj, WEIGHTS, sa);
    for (int64_t i = 0; i < k; i++) w[i] = F(sign)(SQRT(-w[i]), s[i]);
    F(secular)(&sj, VECTORS, sa);
  }
  /* The halves' vectors times the secular ones, each half by the rows of
     the columns nonzero in it. */
  for (int64_t c = 0; c < k; c++)
    memcpy(s + c * n23, q + ctot[0] + c * ldq, sizeof(R) * n23);
  if (!F(rgemm)(n2, k, n23, bot, n2, s, n23, q + n1, ldq)) return -1;
  for (int64_t c = 0; c < k; c++)
    memcpy(s + c * n12, q + c * ldq, sizeof(R) * n12);
  if (!F(rgemm)(n1, k, n12, top, n1, s, n12, q, ldq)) return -1;
  F(lamrg)(k, n - k, d, 1, -1, indxq);
  return 1;
}

/* T's eigenvalues into d and its eigenvectors into the columns of q, the
   identity on entry, [ldq] apart, for T of [n] values torn in halves
   [levels] times, the leaves by steqr (xLAED0's subproblems, solved
   depth first); [indxq] orders d ascending. [work] and [iwork] as for
   merge. Answers as merge. */
static int F(laed0)(int64_t n, int levels, R *d, R *e, R *q, int64_t ldq,
                    int64_t *indxq, R *work, int64_t *iwork, arena *s) {
  if (levels == 0) {
    for (int64_t i = 0; i < n; i++) indxq[i] = i;
    return F(steqr)(n, d, e, q, n, ldq, work);
  }
  int64_t n1 = n / 2;
  R rho = e[n1 - 1];
  d[n1 - 1] -= F(abs)(rho);
  d[n1] -= F(abs)(rho);
  int ok = F(laed0)(n1, levels - 1, d, e, q, ldq, indxq, work, iwork, s);
  if (ok == 1)
    ok = F(laed0)(n - n1, levels - 1, d + n1, e + n1, q + n1 + n1 * ldq, ldq,
                  indxq + n1, work, iwork, s);
  return ok == 1 ? F(merge)(n, n1, d, q, ldq, indxq, rho, work, iwork, s)
                 : ok;
}

/* The eigenvalues of the symmetric tridiagonal T of the diagonal [d] and
   subdiagonal [e], [n] of them, ascending into d, and Z with T = Z diag d
   Zᵀ into z, the identity on entry, columns [ldz] apart, as LAPACK's
   xSTEDC: T's unreduced blocks of more than SMLSIZ values by divide and
   conquer (laed0), scaled to a largest magnitude of one, the others by
   steqr. [work] holds 3n + 2n² reals and [iwork] 5n indices. Answers 1,
   0 if an iteration does not converge, or −1 if an allocation failed. */
static int F(stedc)(int64_t n, R *d, R *e, R *z, int64_t ldz, R *work,
                    int64_t *iwork, arena *s) {
  if (n <= SMLSIZ) return F(steqr)(n, d, e, z, n, ldz, work);
  for (int64_t start = 0, end; start < n; start = end + 1) {
    for (end = start; end < n - 1; end++)
      if (F(abs)(e[end]) <=
          EPS * SQRT(F(abs)(d[end])) * SQRT(F(abs)(d[end + 1])))
        break;
    int64_t m = end - start + 1;
    R *ds = d + start, *es = e + start, *zs = z + start + start * ldz;
    if (m <= SMLSIZ) {
      if (!F(steqr)(m, ds, es, zs, m, ldz, work)) return 0;
      continue;
    }
    R big = 0;
    for (int64_t i = 0; i < m; i++) {
      R x = F(abs)(ds[i]), y = i < m - 1 ? F(abs)(es[i]) : 0;
      big = x > big ? x : big;
      big = y > big ? y : big;
    }
    F(scale)(ds, es, 0, m - 1, big, 1);
    int levels = 0;
    for (int64_t s = m; s > SMLSIZ; s = (s + 1) / 2) levels++;
    int ok =
        F(laed0)(m, levels, ds, es, zs, ldz, iwork + 4 * n, work, iwork, s);
    if (ok != 1) return ok;
    /* The block's values and vectors in ascending order. */
    for (int64_t i = 0; i < m; i++) {
      int64_t j = iwork[4 * n + i];
      work[i] = ds[j];
      memcpy(work + m + i * m, zs + j * ldz, sizeof(R) * m);
    }
    for (int64_t i = 0; i < m; i++) {
      ds[i] = work[i] * big;
      memcpy(zs + i * ldz, work + m + i * m, sizeof(R) * m);
    }
  }
  F(sort)(n, d, ASCENDING, z, n, ldz, NULL, 0, 0);
  return 1;
}

/* The [n] × [n] [v] transposed in place, by tiles of TILE × TILE, so
   that a tile's rows stay in the cache while its columns are read. */
static void F(transpose)(T *v, int64_t n) {
  for (int64_t i0 = 0; i0 < n; i0 += TILE)
    for (int64_t j0 = 0; j0 <= i0; j0 += TILE)
      for (int64_t i = i0; i < i0 + TILE && i < n; i++)
        for (int64_t j = j0; j < j0 + TILE && j < (i0 == j0 ? i : n); j++) {
          T t = v[i * n + j];
          v[i * n + j] = v[j * n + i];
          v[j * n + i] = t;
        }
}

/* y[i·ly + j] = f(x[j·lx + i]), i < [r] and j < [c]: the rows of y from
   the columns of x, by bands of TILE rows of y, which read x a line at a
   time, where a row of y at a time reads a line of x for each element. */
#define FROM_COLUMNS(x, lx, r, c, y, ly, f)                             \
  for (int64_t i0 = 0; i0 < (r); i0 += TILE) {                          \
    int64_t ib = (r) - i0 < TILE ? (r) - i0 : TILE;                     \
    for (int64_t j = 0; j < (c); j++)                                   \
      for (int64_t i = 0; i < ib; i++)                                  \
        (y)[(i0 + i) * (ly) + j] = f((x)[j * (lx) + i0 + i]);           \
  }

/* The rows of the [r] × [c] [y], [ly] apart, from the columns of the reals
   [x], [lx] apart. */
static void F(from_columns)(const R *x, int64_t lx, int64_t r, int64_t c,
                            T *y, int64_t ly) {
  FROM_COLUMNS(x, lx, r, c, y, ly, (T));
}

/* Eigh on matrix [b] of [c] in the scratch [x]: w, and v where the call
   has vectors, as LAPACK's xHEEV and xHEEVD: the matrix scaled into
   [rmin, rmax] where its largest magnitude lies outside, reduced (hetrd,
   or the two stages where eigh_two says), T's eigenvalues by steqr, or
   with Z by stedc, then v = Q Z: Q's blocks of NB reflectors applied
   last first (xUNMTR), after Q2 for the two stages. A matrix that holds
   NaN or an infinity where it is read, or on which an iteration does not
   converge, is NaN in both. Answers 0 if an allocation failed. */
static int F(eigh)(const call *c, int64_t b, T *x, arena *s) {
  int64_t n = c->n;
  T *v = c->vectors ? (T *)matrix_at(c, c->nops + 1, b) : NULL;
  int two = eigh_two(n, v != NULL, CPLX);
  int64_t kd = v ? KD : KD_VALUES;
  T *a = x, *tau = a + n * n, *w = tau + n, *slots = w + eigh_work(n);
  R *d = (R *)(slots + eigh_slots(n, c->vectors, CPLX)), *e = d + n,
    *z = e + n, *work = z + n * n;
  int64_t *iwork = (int64_t *)(((uintptr_t)(work + 3 * n + 2 * n * n) + 7) &
                               ~(uintptr_t)7);
  R *wo = (R *)matrix_at(c, c->nops, b);
  /* The lower triangle, as hetrd reads it column-major, row p of the
     stage from column p on; as the two stages read it row-major, up to
     column p. */
  stage(c, 0, b, a, n, !two, s);
  R big = 0;
  int finite = 1;
  for (int64_t p = 0; p < n; p++) {
    T *row = a + p * n;
    int64_t q0 = two ? 0 : p, q1 = two ? p + 1 : n;
    row[p] = REAL(row[p]);
    finite &= F(scan)(row + q0, q1 - q0, &big);
  }
  const R rmin = SQRT(TINY / (2 * EPS)), rmax = 1 / rmin;
  R sigma = big > 0 && big < rmin ? rmin / big : big > rmax ? rmax / big : 1;
  if (sigma != 1)
    for (int64_t p = 0; p < n; p++)
      for (int64_t q = two ? 0 : p; q < (two ? p + 1 : n); q++)
        a[p * n + q] *= sigma;
  if (finite && two) {
    for (int64_t p = 0; p < n; p++)
      for (int64_t q = 0; q < p; q++) a[q * n + p] = CONJ(a[p * n + q]);
    if (!F(he2hb)(a, n, n, tau, w, kd, s)) return 0;
    F(hb2st)(a, n, n, w, v ? slots : NULL, (int64_t *)(w + 2 * kd * n), d, e,
             kd, s);
  } else if (finite && !F(hetrd)(a, n, n, d, e, tau, w,
                                 w + (3 * n + NB) * NB, s))
    return 0;
  int ok = finite;
  if (ok && v) {
    for (int64_t i = 0; i < n * n; i++) z[i] = i % (n + 1) == 0;
    ok = F(stedc)(n, d, e, z, n, work, iwork, s);
    if (ok < 0) return 0;
  } else if (ok)
    ok = F(steqr)(n, d, e, NULL, 0, 0, work);
  if (!ok) {
    nans(wo, n, RDT);
    if (v) nans(v, n * n * (CPLX + 1), RDT);
    return 1;
  }
  for (int64_t j = 0; j < n; j++) wo[j] = d[j] * (1 / sigma) + 0;
  if (!v) return 1;
  F(from_columns)(z, n, n, n, v, n);
  if (two && !F(q2)(slots, n, v, n, s)) return 0;
  /* Q = H_0 ⋯ H_(n−k−1) on rows k on, k = 1, or KD for Q1: reflector j is
     a's column j from row j + k, so that they are the reflectors of a QR
     of rows k on. Complex vectors take Q1 from the right on their
     transposes, v's rows, where products read them in place: in
     eigh-c128-1024 on the M1, 38-41 ms against 69-74 with the transposes
     (load 23-36). The one-stage path's Q gained nothing so at 256. */
  int64_t k = two ? KD : 1;
  holding as = two && CPLX ? AS_ROWS : AS_COLUMNS;
  if (as == AS_ROWS) F(transpose)(v, n);
  if (!F(back)(two ? a + KD * n : a + 1, two ? n : 1, two ? 1 : n, AS_STORED,
               tau, n - k, n - k, n, as == AS_ROWS ? v + k : v + k * n, n, as,
               w, s))
    return 0;
  if (as == AS_ROWS) F(transpose)(v, n);
  return 1;
}

/* Singular value decomposition

   a = u diag s vh as LAPACK's xGESDD computes it. A wide matrix is
   decomposed as its conjugate transpose, which swaps u and vh; one with
   at least 11/6 as many rows as columns, 17/9 on complex numbers, is
   factored a = Q R first, and R decomposed. The matrix B is reduced to
   Qᴴ B P, upper bidiagonal with real d and e (xGEBRD), or where svd_two
   says in two stages, Q = Q1 Q2 and P = P1 P2; that one's SVD U_B diag s
   V_Bᵀ is found by divide and conquer (xBDSDC); then u = Q U_B and vh =
   (P V_B)ᴴ by the reflectors' blocks (q2, back). Values alone run
   implicit QR on the bidiagonal without vectors (xBDSQR), which for at
   most SMLSIZ values is the vectors' call's iteration. Matrices are
   column-major, but for B in QR and the two stages; V_B's vectors are its
   columns, where LAPACK keeps Vᵀ's rows, so that every rotation moves
   adjacent memory. */

/* The singular values [smin] <= [smax] of [f g; 0 h] (xLAS2). */
static void F(las2)(R f, R g, R h, R *smin, R *smax) {
  R fa = F(abs)(f), ga = F(abs)(g), ha = F(abs)(h);
  R mn = fa < ha ? fa : ha, mx = fa > ha ? fa : ha;
  if (mn == 0) {
    R lo = mx < ga ? mx : ga, hi = mx > ga ? mx : ga;
    *smin = 0;
    *smax = mx == 0 ? ga : hi * SQRT(1 + (lo / hi) * (lo / hi));
    return;
  }
  if (ga < mx) {
    R as = 1 + mn / mx, at = (mx - mn) / mx, au = (ga / mx) * (ga / mx);
    R c = 2 / (SQRT(as * as + au) + SQRT(at * at + au));
    *smin = mn * c;
    *smax = mx / c;
    return;
  }
  R au = mx / ga;
  if (au == 0) {
    *smin = (mn * mx) / ga;
    *smax = ga;
    return;
  }
  R as = 1 + mn / mx, at = (mx - mn) / mx;
  R c = 1 / (SQRT(1 + (as * au) * (as * au)) + SQRT(1 + (at * au) * (at * au)));
  *smin = 2 * ((mn * c) * au);
  *smax = ga / (c + c);
}

/* The singular values of [f g; 0 h], |smax| >= |smin|, and the rotations
   with [cl sl; −sl cl] [f g; 0 h] [cr −sr; sr cr] = diag (smax, smin)
   (xLASV2). */
static void F(lasv2)(R f, R g, R h, R *smin, R *smax, R *sr, R *cr, R *sl,
                     R *cl) {
  R ft = f, fa = F(abs)(f), ht = h, ha = F(abs)(h), gt = g, ga = F(abs)(g);
  int pmax = 1, swap = ha > fa;
  if (swap) {
    pmax = 3;
    ft = h;
    ht = f;
    fa = F(abs)(h);
    ha = F(abs)(f);
  }
  R mn = ha, mx = fa, clt = 1, crt = 1, slt = 0, srt = 0;
  if (ga != 0 && ga > fa && fa / ga < EPS) {
    /* g dominates. */
    pmax = 2;
    mx = ga;
    mn = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha;
    slt = ht / gt;
    srt = 1;
    crt = ft / gt;
  } else if (ga != 0) {
    if (ga > fa) pmax = 2;
    R dd = fa - ha, l = dd == fa ? 1 : dd / fa, m = gt / ft, t = 2 - l;
    R mm = m * m, s = SQRT(t * t + mm);
    R r = l == 0 ? F(abs)(m) : SQRT(l * l + mm), a = (s + r) / 2;
    mn = ha / a;
    mx = fa * a;
    if (mm != 0) t = (m / (s + t) + m / (r + l)) * (1 + a);
    else if (l == 0) t = F(sign)(2, ft) * F(sign)(1, gt);
    else t = gt / F(sign)(dd, ft) + m / t;
    l = SQRT(t * t + 4);
    crt = 2 / l;
    srt = t / l;
    clt = (crt + srt * m) / a;
    slt = (ht / ft) * srt / a;
  }
  *cl = swap ? srt : clt;
  *sl = swap ? crt : slt;
  *cr = swap ? slt : crt;
  *sr = swap ? clt : srt;
  R sg = pmax == 1   ? F(sign)(1, *cr) * F(sign)(1, *cl) * F(sign)(1, f)
         : pmax == 2 ? F(sign)(1, *sr) * F(sign)(1, *cl) * F(sign)(1, g)
                     : F(sign)(1, *sr) * F(sign)(1, *sl) * F(sign)(1, h);
  *smax = F(sign)(mx, sg);
  *smin = F(sign)(mn, sg * F(sign)(1, f) * F(sign)(1, h));
}

/* The singular values of the upper bidiagonal B of the diagonal [d] and
   superdiagonal [e], [n] of them, not negative, into d, by QR sweeps
   with a zero or a shifted start, each chasing the bulge from the end of
   larger magnitude (xBDSQR, whose dqds for values alone is left out): u
   ← u Q_l for the [nru] rows of [u] and v ← v Q_r for the [nrv] rows of
   [v], columns [ldu] and [ldv] apart, where B = Q_l diag d Q_rᵀ. [work]
   holds 4n reals. Answers 0 if the iteration takes more than 6 n²
   steps. */
static int F(bdsqr)(int64_t n, R *d, R *e, R *u, int64_t nru, int64_t ldu,
                    R *v, int64_t nrv, int64_t ldv, R *work) {
  R tol = (R)pow(EPS, -0.125);
  tol = (tol < 10 ? 10 : tol > 100 ? 100 : tol) * EPS;
  int64_t maxit = 6 * n, sweeps = 0, iter = -1, oldll = -1, oldm = -1;
  int dir = 0;
  R *c1 = work, *s1 = c1 + n, *c2 = s1 + n, *s2 = c2 + n;
  /* The threshold below which an e is negligible: tol times an estimate
     of the least singular value. */
  R sminoa = n > 0 ? F(abs)(d[0]) : 0, mu = sminoa;
  for (int64_t i = 1; i < n && sminoa != 0; i++) {
    mu = F(abs)(d[i]) * (mu / (mu + F(abs)(e[i - 1])));
    sminoa = mu < sminoa ? mu : sminoa;
  }
  sminoa /= SQRT((R)n);
  R thresh = 6 * ((R)n * ((R)n * TINY));
  thresh = tol * sminoa > thresh ? tol * sminoa : thresh;
  for (int64_t m = n - 1; m > 0;) {
    if (iter >= n) {
      iter -= n;
      if (++sweeps >= maxit) return 0;
    }
    /* The block [ll, m]: up from m to the first negligible e. */
    R smax = F(abs)(d[m]);
    int64_t ll = m - 1;
    for (; ll >= 0; ll--) {
      R abss = F(abs)(d[ll]), abse = F(abs)(e[ll]);
      if (abse <= thresh) break;
      smax = abss > smax ? abss : smax;
      smax = abse > smax ? abse : smax;
    }
    if (ll >= 0) {
      e[ll] = 0;
      if (ll == m - 1) {
        m--;
        continue;
      }
    }
    ll++;
    if (ll == m - 1) {
      R smin, sx, sr, cr, sl, cl;
      F(lasv2)(d[m - 1], e[m - 1], d[m], &smin, &sx, &sr, &cr, &sl, &cl);
      d[m - 1] = sx;
      e[m - 1] = 0;
      d[m] = smin;
      if (nrv) F(rot)(v + (m - 1) * ldv, v + m * ldv, nrv, cr, sr);
      if (nru) F(rot)(u + (m - 1) * ldu, u + m * ldu, nru, cl, sl);
      m -= 2;
      continue;
    }
    /* A new block chases from its end of larger magnitude. */
    if (ll > oldm || m < oldll) dir = F(abs)(d[ll]) >= F(abs)(d[m]) ? 1 : 2;
    /* An e negligible relative to its neighbours splits the block. */
    int split = 0;
    R smin = 0;
    if (dir == 1) {
      if (F(abs)(e[m - 1]) <= tol * F(abs)(d[m])) {
        e[m - 1] = 0;
        continue;
      }
      mu = smin = F(abs)(d[ll]);
      for (int64_t l = ll; l < m; l++) {
        if (F(abs)(e[l]) <= tol * mu) {
          e[l] = 0;
          split = 1;
          break;
        }
        mu = F(abs)(d[l + 1]) * (mu / (mu + F(abs)(e[l])));
        smin = mu < smin ? mu : smin;
      }
    } else {
      if (F(abs)(e[ll]) <= tol * F(abs)(d[ll])) {
        e[ll] = 0;
        continue;
      }
      mu = smin = F(abs)(d[m]);
      for (int64_t l = m - 1; l >= ll; l--) {
        if (F(abs)(e[l]) <= tol * mu) {
          e[l] = 0;
          split = 1;
          break;
        }
        mu = F(abs)(d[l]) * (mu / (mu + F(abs)(e[l])));
        smin = mu < smin ? mu : smin;
      }
    }
    if (split) continue;
    oldll = ll;
    oldm = m;
    /* The shift: the least singular value of the trailing 2 × 2, or zero
       where it would spoil the relative accuracy of the least value. */
    R shift = 0, lim = (R)0.01 * tol;
    lim = EPS > lim ? EPS : lim;
    if (!((R)n * tol * (smin / smax) <= lim)) {
      R sll, r;
      if (dir == 1) {
        sll = F(abs)(d[ll]);
        F(las2)(d[m - 1], e[m - 1], d[m], &shift, &r);
      } else {
        sll = F(abs)(d[m]);
        F(las2)(d[ll], e[ll], d[ll + 1], &shift, &r);
      }
      if (sll > 0 && (shift / sll) * (shift / sll) < EPS) shift = 0;
    }
    iter += m - ll;
    R cs = 1, sn = 0, ocs = 1, osn = 0, r, f, g;
    if (dir == 1 && shift == 0) {
      for (int64_t i = ll; i < m; i++) {
        F(rotation)(d[i] * cs, e[i], &cs, &sn, &r);
        if (i > ll) e[i - 1] = osn * r;
        F(rotation)(ocs * r, d[i + 1] * sn, &ocs, &osn, &d[i]);
        c1[i - ll] = cs;
        s1[i - ll] = sn;
        c2[i - ll] = ocs;
        s2[i - ll] = osn;
      }
      R h = d[m] * cs;
      d[m] = h * ocs;
      e[m - 1] = h * osn;
    } else if (dir == 1) {
      f = (F(abs)(d[ll]) - shift) * (F(sign)(1, d[ll]) + shift / d[ll]);
      g = e[ll];
      for (int64_t i = ll; i < m; i++) {
        F(rotation)(f, g, &cs, &sn, &r);
        if (i > ll) e[i - 1] = r;
        f = cs * d[i] + sn * e[i];
        e[i] = cs * e[i] - sn * d[i];
        g = sn * d[i + 1];
        d[i + 1] = cs * d[i + 1];
        F(rotation)(f, g, &ocs, &osn, &r);
        d[i] = r;
        f = ocs * e[i] + osn * d[i + 1];
        d[i + 1] = ocs * d[i + 1] - osn * e[i];
        if (i < m - 1) {
          g = osn * e[i + 1];
          e[i + 1] = ocs * e[i + 1];
        }
        c1[i - ll] = cs;
        s1[i - ll] = sn;
        c2[i - ll] = ocs;
        s2[i - ll] = osn;
      }
      e[m - 1] = f;
    } else if (shift == 0) {
      for (int64_t i = m; i > ll; i--) {
        F(rotation)(d[i] * cs, e[i - 1], &cs, &sn, &r);
        if (i < m) e[i] = osn * r;
        F(rotation)(ocs * r, d[i - 1] * sn, &ocs, &osn, &d[i]);
        c1[i - ll - 1] = cs;
        s1[i - ll - 1] = -sn;
        c2[i - ll - 1] = ocs;
        s2[i - ll - 1] = -osn;
      }
      R h = d[ll] * cs;
      d[ll] = h * ocs;
      e[ll] = h * osn;
    } else {
      f = (F(abs)(d[m]) - shift) * (F(sign)(1, d[m]) + shift / d[m]);
      g = e[m - 1];
      for (int64_t i = m; i > ll; i--) {
        F(rotation)(f, g, &cs, &sn, &r);
        if (i < m) e[i] = r;
        f = cs * d[i] + sn * e[i - 1];
        e[i - 1] = cs * e[i - 1] - sn * d[i];
        g = sn * d[i - 1];
        d[i - 1] = cs * d[i - 1];
        F(rotation)(f, g, &ocs, &osn, &r);
        d[i] = r;
        f = ocs * e[i - 1] + osn * d[i - 1];
        d[i - 1] = ocs * d[i - 1] - osn * e[i - 1];
        if (i > ll + 1) {
          g = osn * e[i - 2];
          e[i - 2] = ocs * e[i - 2];
        }
        c1[i - ll - 1] = cs;
        s1[i - ll - 1] = -sn;
        c2[i - ll - 1] = ocs;
        s2[i - ll - 1] = -osn;
      }
      e[ll] = f;
    }
    /* The sweep's rotations on the vectors: the right ones on v, the left
       ones on u. */
    int64_t w = m - ll + 1;
    R *cr = dir == 1 ? c1 : c2, *sr = dir == 1 ? s1 : s2,
      *cl = dir == 1 ? c2 : c1, *sl = dir == 1 ? s2 : s1;
    if (nrv) F(rotate)(v + ll * ldv, nrv, ldv, w, cr, sr, dir != 1);
    if (nru) F(rotate)(u + ll * ldu, nru, ldu, w, cl, sl, dir != 1);
    R *last = dir == 1 ? e + m - 1 : e + ll;
    if (F(abs)(*last) <= thresh) *last = 0;
  }
  for (int64_t i = 0; i < n; i++) {
    if (!(d[i] < 0)) continue;
    d[i] = -d[i];
    for (int64_t r = 0; r < nrv; r++) v[i * ldv + r] = -v[i * ldv + r];
  }
  return 1;
}

/* The singular values of the upper bidiagonal B, [n] × (n + [sqre]), of
   the diagonal [d] and superdiagonal [e], ascending into d, with u ← u
   U_B for the n rows of [u] and v ← v V_B for the n + sqre rows of [v],
   where they are not NULL, B = U_B [diag d 0] V_Bᵀ (xLASDQ): a B of
   n + 1 columns made square by rotations from the right, which leave it
   lower, then upper by rotations from the left, then bdsqr. [work] holds
   4n reals. Answers as bdsqr. */
static int F(lasdq)(int64_t n, int sqre, R *d, R *e, R *u, int64_t ldu,
                    R *v, int64_t ldv, R *work) {
  int64_t nv = n + sqre;
  R *cs = work, *sn = work + n, r;
  if (sqre) {
    for (int64_t i = 0; i < n; i++) {
      F(rotation)(d[i], e[i], &cs[i], &sn[i], &r);
      d[i] = r;
      e[i] = i + 1 < n ? sn[i] * d[i + 1] : 0;
      if (i + 1 < n) d[i + 1] = cs[i] * d[i + 1];
    }
    if (v) F(rotate)(v, nv, ldv, nv, cs, sn, 0);
    for (int64_t i = 0; i + 1 < n; i++) {
      F(rotation)(d[i], e[i], &cs[i], &sn[i], &r);
      d[i] = r;
      e[i] = sn[i] * d[i + 1];
      d[i + 1] = cs[i] * d[i + 1];
    }
    if (u) F(rotate)(u, n, ldu, n, cs, sn, 0);
  }
  if (!F(bdsqr)(n, d, e, u, u ? n : 0, ldu, v, v ? nv : 0, ldv, work))
    return 0;
  F(sort)(n, d, ASCENDING, u, n, ldu, v, nv, ldv);
  return 1;
}

/* Root [i] of the secular equation of two values, 0 <= d[0] < d[1], into
   [sigma], with delta_j = d_j − σ and work_j = d_j + σ (xLASD5). */
static void F(lasd5)(int i, const R *d, const R *z, R *delta, R rho,
                     R *sigma, R *work) {
  R del = d[1] - d[0], delsq = del * (d[1] + d[0]);
  R z0 = z[0] * z[0], z1 = z[1] * z[1], tau, b, c;
  if (i == 0 &&
      1 + 4 * rho * (z1 / (d[0] + 3 * d[1]) - z0 / (3 * d[0] + d[1])) / del >
          0) {
    b = delsq + rho * (z0 + z1);
    c = rho * z0 * delsq;
    tau = 2 * c / (b + SQRT(F(abs)(b * b - 4 * c)));
    tau = tau / (d[0] + SQRT(d[0] * d[0] + tau));
    *sigma = d[0] + tau;
    delta[0] = -tau;
    delta[1] = del - tau;
    work[0] = 2 * d[0] + tau;
    work[1] = (d[0] + tau) + d[1];
    return;
  }
  b = -delsq + rho * (z0 + z1);
  c = rho * z1 * delsq;
  if (i == 0) {
    tau = b > 0 ? -2 * c / (b + SQRT(b * b + 4 * c))
                : (b - SQRT(b * b + 4 * c)) / 2;
    tau = tau / (d[1] + SQRT(F(abs)(d[1] * d[1] + tau)));
  } else {
    tau = b > 0 ? (b + SQRT(b * b + 4 * c)) / 2
                : 2 * c / (-b + SQRT(b * b + 4 * c));
    tau = tau / (d[1] + SQRT(d[1] * d[1] + tau));
  }
  *sigma = d[1] + tau;
  delta[0] = -(del + tau);
  delta[1] = -tau;
  work[0] = d[0] + tau + d[1];
  work[1] = 2 * d[1] + tau;
}

/* Root [i] of 1 + ρ Σ z_j² / ((d_j − σ)(d_j + σ)) = 0 for d ascending and
   not negative, z without zeros and ρ > 0, into [sigma], with delta_j =
   d_j − σ and work_j = d_j + σ, each from the pole nearer the root so
   that it keeps its accuracy (xLASD4): laed4's iteration on σ², each step
   η in σ² mapped back to σ as η / (σ + √(σ² + η)). Answers 0 after 400
   steps. */
static int F(lasd4)(int64_t n, int64_t i, const R *d, const R *z, R *delta,
                    R rho, R *sigma, R *work) {
  if (n == 1) {
    *sigma = SQRT(d[0] * d[0] + rho * z[0] * z[0]);
    delta[0] = 1;
    work[0] = 1;
    return 1;
  }
  if (n == 2) {
    F(lasd5)((int)i, d, z, delta, rho, sigma, work);
    return 1;
  }
  const R rhoinv = 1 / rho;
  int last = i == n - 1, orgati = 1, geomavg = 0;
  int64_t ip1 = i + 1;
  R tau, sglb = 0, sgub = 0, delsq = 0;
  /* The first guess, from the two poles about the root and the rest at
     the midpoint. */
  if (last) {
    R dn = d[n - 1], dp = d[n - 2], t = rho / 2;
    R t1 = t / (dn + SQRT(dn * dn + t)), psi = 0;
    for (int64_t j = 0; j < n; j++) {
      work[j] = d[j] + dn + t1;
      delta[j] = (d[j] - dn) - t1;
    }
    for (int64_t j = 0; j < n - 2; j++)
      psi += z[j] * z[j] / (delta[j] * work[j]);
    R c = rhoinv + psi;
    R w = c + z[n - 2] * z[n - 2] / (delta[n - 2] * work[n - 2]) +
          z[n - 1] * z[n - 1] / (delta[n - 1] * work[n - 1]);
    R ds = (dn - dp) * (dn + dp);
    R a = -c * ds + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
    R b = z[n - 1] * z[n - 1] * ds;
    R tau2 = a < 0 ? 2 * b / (SQRT(a * a + 4 * b * c) - a)
                   : (a + SQRT(a * a + 4 * b * c)) / (2 * c);
    tau = tau2 / (dn + SQRT(dn * dn + tau2));
    if (w <= 0) {
      R t2 = SQRT(dn * dn + rho);
      R lim = z[n - 2] * z[n - 2] / ((dp + t2) * (dn - dp + rho / (dn + t2))) +
              z[n - 1] * z[n - 1] / rho;
      if (c <= lim) tau = rho;
    }
  } else {
    delsq = (d[ip1] - d[i]) * (d[ip1] + d[i]);
    R ds2 = delsq / 2, sq2 = SQRT((d[i] * d[i] + d[ip1] * d[ip1]) / 2);
    R t = ds2 / (d[i] + sq2), psi = 0, phi = 0;
    for (int64_t j = 0; j < n; j++) {
      work[j] = d[j] + d[i] + t;
      delta[j] = (d[j] - d[i]) - t;
    }
    for (int64_t j = 0; j < i; j++) psi += z[j] * z[j] / (work[j] * delta[j]);
    for (int64_t j = n - 1; j > i + 1; j--)
      phi += z[j] * z[j] / (work[j] * delta[j]);
    R c = rhoinv + psi + phi;
    R w = c + z[i] * z[i] / (work[i] * delta[i]) +
          z[ip1] * z[ip1] / (work[ip1] * delta[ip1]);
    orgati = w > 0;
    if (orgati) {
      sgub = ds2 / (d[i] + sq2);
      R a = c * delsq + z[i] * z[i] + z[ip1] * z[ip1], b = z[i] * z[i] * delsq;
      R q = SQRT(F(abs)(a * a - 4 * b * c));
      R tau2 = a > 0 ? 2 * b / (a + q) : (a - q) / (2 * c);
      tau = tau2 / (d[i] + SQRT(d[i] * d[i] + tau2));
      /* A tiny d_i with a tiny z_i: the root is near d_i's square root
         of products, which bisection by geometric means finds. */
      R e = SQRT(EPS);
      if (d[i] <= e * d[ip1] && F(abs)(z[i]) <= e && d[i] > 0) {
        tau = 10 * d[i] < sgub ? 10 * d[i] : sgub;
        geomavg = 1;
      }
    } else {
      sglb = -ds2 / (d[ip1] + sq2);
      R a = c * delsq - z[i] * z[i] - z[ip1] * z[ip1];
      R b = z[ip1] * z[ip1] * delsq;
      R q = SQRT(F(abs)(a * a + 4 * b * c));
      R tau2 = a < 0 ? 2 * b / (a - q) : -(a + q) / (2 * c);
      tau = tau2 / (d[ip1] + SQRT(F(abs)(d[ip1] * d[ip1] + tau2)));
    }
  }
  /* ψ sums the poles below the root's nearer one [ii], φ those above. */
  int64_t ii = last ? n - 1 : orgati ? i : ip1;
  R org = d[ii], s = org + tau, prew = 0;
  for (int64_t j = 0; j < n; j++) {
    delta[j] = (d[j] - org) - tau;
    work[j] = d[j] + org + tau;
  }
  int swtch3 = 0, swtch = 0;
  for (int64_t step = 0;; step++) {
    R psi = 0, dpsi = 0, phi = 0, dphi = 0, err = 0, w, dw;
    for (int64_t j = 0; j < ii; j++) {
      R t = z[j] / (work[j] * delta[j]);
      psi += z[j] * t;
      dpsi += t * t;
      err += psi;
    }
    err = F(abs)(err);
    for (int64_t j = n - 1; j > ii; j--) {
      R t = z[j] / (work[j] * delta[j]);
      phi += z[j] * t;
      dphi += t * t;
      err += phi;
    }
    R t = z[ii] / (work[ii] * delta[ii]);
    if (last) {
      /* The root's pole is φ's, the last. */
      phi = z[ii] * t;
      dphi = t * t;
      err = 8 * (-phi - psi) + err - phi + rhoinv;
      w = rhoinv + phi + psi;
      dw = dpsi + dphi;
    } else {
      w = rhoinv + phi + psi;
      if (step == 0)
        swtch3 = (orgati ? w < 0 : w > 0) && ii != 0 && ii != n - 1;
      dw = dpsi + dphi + t * t;
      t *= z[ii];
      w += t;
      err = 8 * (phi - psi) + err + 2 * rhoinv + 3 * F(abs)(t);
      if (step == 1)
        swtch = orgati ? -w > F(abs)(prew) / 10 : w > F(abs)(prew) / 10;
      else if (step > 1 && w * prew > 0 && F(abs)(w) > F(abs)(prew) / 10)
        swtch = !swtch;
    }
    if (step == 399) break;
    if (F(abs)(w) <= EPS * err) {
      *sigma = s;
      return 1;
    }
    R eta = 0, a, b, c;
    if (last) {
      R d1 = work[n - 2] * delta[n - 2], d2 = work[n - 1] * delta[n - 1];
      c = w - d1 * dpsi - d2 * dphi;
      a = (d2 + d1) * w - d2 * d1 * (dpsi + dphi);
      b = d2 * d1 * w;
      if (step == 0 && c < 0) c = F(abs)(c);
      if (step == 0 && c == 0) eta = rho - s * s;
      else if (a >= 0) eta = (a + SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c);
      else eta = 2 * b / (a - SQRT(F(abs)(a * a - 4 * b * c)));
      if (w * eta > 0) eta = -w / dw;
      if (step == 0 && eta - d2 > rho) eta = rho + d2;
      else if (step > 0 && eta - d2 <= 0) eta /= 2;
      eta = eta / (s + SQRT(eta + s * s));
    } else {
      if (w <= 0) sglb = sglb > tau ? sglb : tau;
      else sgub = sgub < tau ? sgub : tau;
      int done = 0;
      if (swtch3) {
        /* Three poles, by laed6, where two would step the wrong way. */
        R dm = work[ii - 1] * delta[ii - 1], dp = work[ii + 1] * delta[ii + 1];
        R base = rhoinv + psi + phi, zz[3], dd[3];
        if (swtch) {
          c = base - dm * dpsi - dp * dphi;
          zz[0] = dm * dm * dpsi;
          zz[2] = dp * dp * dphi;
        } else if (orgati) {
          R q = z[ii - 1] / dm;
          q *= q;
          c = base - dp * (dpsi + dphi) -
              (d[ii - 1] - d[ii + 1]) * (d[ii - 1] + d[ii + 1]) * q;
          zz[0] = z[ii - 1] * z[ii - 1];
          zz[2] = dpsi < q ? dp * dp * dphi : dp * dp * ((dpsi - q) + dphi);
        } else {
          R q = z[ii + 1] / dp;
          q *= q;
          c = base - dm * (dpsi + dphi) -
              (d[ii + 1] - d[ii - 1]) * (d[ii - 1] + d[ii + 1]) * q;
          zz[0] = dphi < q ? dm * dm * dpsi : dm * dm * (dpsi + (dphi - q));
          zz[2] = z[ii + 1] * z[ii + 1];
        }
        zz[1] = z[ii] * z[ii];
        dd[0] = dm;
        dd[1] = delta[ii] * work[ii];
        dd[2] = dp;
        done = F(laed6)(step + 2, orgati, c, dd, zz, w, &eta);
        swtch3 = done;
      }
      if (!done) {
        R di = work[i] * delta[i], dj = work[ip1] * delta[ip1];
        if (swtch) {
          R q = z[ii] / (work[ii] * delta[ii]);
          if (orgati) dpsi += q * q;
          else dphi += q * q;
          c = w - di * dpsi - dj * dphi;
        } else if (orgati) {
          R q = z[i] / di;
          c = w - dj * dw + delsq * (q * q);
        } else {
          R q = z[ip1] / dj;
          c = w - di * dw - delsq * (q * q);
        }
        a = (dj + di) * w - dj * di * dw;
        b = dj * di * w;
        if (c == 0) {
          if (a == 0)
            a = swtch    ? di * di * dpsi + dj * dj * dphi
                : orgati ? z[i] * z[i] + dj * dj * (dpsi + dphi)
                         : z[ip1] * z[ip1] + di * di * (dpsi + dphi);
          eta = b / a;
        } else if (a <= 0)
          eta = (a - SQRT(F(abs)(a * a - 4 * b * c))) / (2 * c);
        else
          eta = 2 * b / (a + SQRT(F(abs)(a * a - 4 * b * c)));
      }
      if (w * eta >= 0) eta = -w / dw;
      eta = eta / (s + SQRT(s * s + eta));
      if (tau + eta > sgub || tau + eta < sglb) {
        eta = w < 0 ? (sgub - tau) / 2 : (sglb - tau) / 2;
        if (geomavg && w < 0 && tau > 0) eta = SQRT(sgub * tau) - tau;
        else if (geomavg && w >= 0 && sglb > 0) eta = SQRT(sglb * tau) - tau;
      }
      prew = w;
    }
    tau += eta;
    s += eta;
    for (int64_t j = 0; j < n; j++) {
      work[j] += eta;
      delta[j] -= eta;
    }
  }
  *sigma = s;
  return 0;
}

/* The 2-norm of the [n] reals at [x], as norm's. */
static R F(rnorm)(const R *x, int64_t n) {
  R q = 0, big = 0, t = 0;
  for (int64_t i = 0; i < n; i++) q += x[i] * x[i];
  if ((q >= SAFMIN && q <= 1 / SAFMIN) || q != q) return SQRT(q);
  for (int64_t i = 0; i < n; i++) big = F(abs)(x[i]) > big ? F(abs)(x[i]) : big;
  if (big == 0) return 0;
  for (int64_t i = 0; i < n; i++) t += (x[i] / big) * (x[i] / big);
  return big * SQRT(t);
}

/* A merge's work on its [k] roots, as secular's: the roots' deltas into
   u's first k columns and their sums into v's, z recomputed from them,
   the left vectors normalised into q's columns, then the right ones. */
typedef struct {
  singular phase;
  int64_t k, ldu, ldv;
  R *u, *v, *q, *z, *dsigma, *d, rho;
  const int64_t *idxc;
  int failed;
} F(sd_job);

static void F(sd_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  F(sd_job) *j = ctx;
  int64_t k = j->k, ldu = j->ldu, ldv = j->ldv, i0 = lo * MERGE_ROOTS;
  int64_t i1 = hi * MERGE_ROOTS < k ? hi * MERGE_ROOTS : k;
  R *u = j->u, *v = j->v, *q = j->q, *z = j->z, *ds = j->dsigma;
  switch (j->phase) {
    case SVD_ROOTS:
      for (int64_t i = i0; i < i1; i++)
        if (!F(lasd4)(k, i, ds, z, u + i * ldu, j->rho, j->d + i,
                      v + i * ldv))
          __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
      break;
    case SVD_WEIGHTS:
      /* Down the columns, as secular's; z_i multiplies in the columns'
         order, skipping column i of dsigma. The signs are the old z's,
         in q. */
      for (int64_t i = i0; i < i1; i++)
        z[i] = u[i + (k - 1) * ldu] * v[i + (k - 1) * ldv];
      for (int64_t p = 0; p < k - 1; p++)
        for (int64_t i = i0; i < i1; i++) {
          int64_t c = p < i ? p : p + 1;
          z[i] *= u[i + p * ldu] * v[i + p * ldv] / (ds[i] - ds[c]) /
                  (ds[i] + ds[c]);
        }
      for (int64_t i = i0; i < i1; i++)
        z[i] = F(sign)(SQRT(F(abs)(z[i])), q[i]);
      break;
    case LEFT_VECTORS:
      for (int64_t i = i0; i < i1; i++) {
        R *ui = u + i * ldu, *wi = v + i * ldv;
        wi[0] = z[0] / ui[0] / wi[0];
        ui[0] = -1;
        for (int64_t p = 1; p < k; p++) {
          wi[p] = z[p] / ui[p] / wi[p];
          ui[p] = ds[p] * wi[p];
        }
        R t = 1 / F(rnorm)(ui, k);
        q[i * k] = ui[0] * t;
        for (int64_t p = 1; p < k; p++) q[p + i * k] = ui[j->idxc[p]] * t;
      }
      break;
    case RIGHT_VECTORS:
      for (int64_t i = i0; i < i1; i++) {
        R *wi = v + i * ldv, t = 1 / F(rnorm)(wi, k);
        q[i * k] = wi[0] * t;
        for (int64_t p = 1; p < k; p++) q[p + i * k] = wi[j->idxc[p]] * t;
      }
      break;
  }
}

/* [j]'s phase over its k roots. Answers 0 if a root did not converge. */
static int F(sd)(F(sd_job) *j, singular phase, arena *s) {
  j->phase = phase;
  F(on_roots)(j->k, 2, phase == SVD_ROOTS ? 8 : 1, F(sd_unit), j, s);
  return !j->failed;
}

/* Merges the SVDs of the halves of the upper bidiagonal B of [n] = nl + 1
   + nr rows and n + [sqre] columns, torn at row nl, whose (nl, nl) and
   (nl, nl + 1) elements are [alpha] and [beta] (xLASD1). The halves, nl ×
   (nl + 1) and nr × (nr + sqre), have their values in d, each ascending
   as idxq orders it, their left vectors in the diagonal blocks of the
   columns of u and their right ones in those of v, [ldu] and [ldv] apart.
   On return d, u and v hold B's, its values' order in idxq. A merge
   deflates the values whose part of z is negligible, or which a rotation
   makes equal to a neighbour (xLASD2), finds the others as the roots of
   the secular equation (lasd4) and their vectors from Gu and Eisenstat's
   recomputed z (xLASD3); the product with the halves' vectors is the
   merge's O(n³), by products on their nonzero blocks. [work] holds n + m
   + 2n² + m² reals, m = n + sqre, and [iwork] 4n indices. The roots'
   work runs on the call's threads where [sa] holds them (sd). Answers 1,
   0 if a root does not converge, or −1 if an allocation failed. */
static int F(lasd1)(int64_t nl, int64_t nr, int sqre, R *d, R alpha, R beta,
                    R *u, int64_t ldu, R *v, int64_t ldv, int64_t *idxq,
                    R *work, int64_t *iwork, arena *sa) {
  int64_t n = nl + nr + 1, m = n + sqre, k = 1, k2 = n, ctot[4] = {0}, psm[4];
  R *z = work, *dsigma = z + m, *u2 = dsigma + n, *v2 = u2 + n * n,
    *q = v2 + m * m;
  int64_t *idx = iwork, *idxc = idx + n, *coltyp = idxc + n,
          *idxp = coltyp + n;
  R big = F(abs)(alpha) > F(abs)(beta) ? F(abs)(alpha) : F(abs)(beta);
  d[nl] = 0;
  for (int64_t i = 0; i < n; i++) big = F(abs)(d[i]) > big ? F(abs)(d[i]) : big;
  for (int64_t i = 0; i < n; i++) d[i] /= big;
  alpha /= big;
  beta /= big;
  /* z: the torn row in the halves' right vectors, α times the last row
     of the first's and β times the first row of the second's; d[0] = 0
     its value, the halves' values after it. */
  R z1 = alpha * v[nl + nl * ldv];
  z[0] = z1;
  for (int64_t i = nl - 1; i >= 0; i--) {
    z[i + 1] = alpha * v[nl + i * ldv];
    d[i + 1] = d[i];
    idxq[i + 1] = idxq[i] + 1;
  }
  for (int64_t i = nl + 1; i < m; i++) z[i] = beta * v[(nl + 1) + i * ldv];
  /* Column types: 1 nonzero in the first nl rows of u alone, 2 in the
     last nr alone, 3 in both (two mixed by a rotation), 4 deflated. */
  for (int64_t i = 1; i < n; i++) coltyp[i] = i <= nl ? 1 : 2;
  for (int64_t i = nl + 1; i < n; i++) idxq[i] += nl + 1;
  for (int64_t i = 1; i < n; i++) {
    dsigma[i] = d[idxq[i]];
    u2[i] = z[idxq[i]];
    idxc[i] = coltyp[idxq[i]];
  }
  F(lamrg)(nl, nr, dsigma + 1, 1, 1, idx + 1);
  for (int64_t i = 1; i < n; i++) {
    int64_t p = 1 + idx[i];
    d[i] = dsigma[p];
    z[i] = u2[p];
    coltyp[i] = idxc[p];
  }
  /* 64 u, as Reference-LAPACK's PR 1286 has it: at 8 u, rotations of
     values that close lose the vectors' orthogonality (issue 255). */
  R tol = F(abs)(alpha) > F(abs)(beta) ? F(abs)(alpha) : F(abs)(beta);
  tol = 64 * EPS * (F(abs)(d[n - 1]) > tol ? F(abs)(d[n - 1]) : tol);
  int64_t j = 1, jprev = 0;
  for (; j < n && F(abs)(z[j]) <= tol; j++) {
    idxp[--k2] = j;
    coltyp[j] = 4;
  }
  if (j < n) {
    for (jprev = j++; j < n; j++) {
      if (F(abs)(z[j]) <= tol) {
        idxp[--k2] = j;
        coltyp[j] = 4;
        continue;
      }
      if (!(d[j] - d[jprev] <= tol)) {
        u2[k] = z[jprev];
        dsigma[k] = d[jprev];
        idxp[k++] = jprev;
        jprev = j;
        continue;
      }
      /* Two close values: a rotation zeroes jprev's part of z. */
      R s = z[jprev], c = z[j], tau = F(lapy3)(c, s, 0);
      c /= tau;
      s = -s / tau;
      z[j] = tau;
      z[jprev] = 0;
      int64_t pj = idxq[idx[jprev] + 1], nj = idxq[idx[j] + 1];
      if (pj <= nl) pj--;
      if (nj <= nl) nj--;
      F(rot)(u + pj * ldu, u + nj * ldu, n, c, s);
      F(rot)(v + pj * ldv, v + nj * ldv, m, c, s);
      if (coltyp[j] != coltyp[jprev]) coltyp[j] = 3;
      coltyp[jprev] = 4;
      /* jprev goes above the values deflated since it, which move down
         one, so that the deflated ones stay descending (Reference-LAPACK's
         PR 1140). */
      k2--;
      for (int64_t jp = jprev; jp < j; jp++) idxp[k2 + j - 1 - jp] = jp;
      jprev = j;
    }
    u2[k] = z[jprev];
    dsigma[k] = d[jprev];
    idxp[k++] = jprev;
  }
  /* The vectors grouped by type into u2 and v2, the deflated ones last,
     where they go back to u and v. */
  for (int64_t i = 1; i < n; i++) ctot[coltyp[i] - 1]++;
  psm[0] = 1;
  for (int t = 1; t < 4; t++) psm[t] = psm[t - 1] + ctot[t - 1];
  for (int64_t i = 1; i < n; i++) idxc[psm[coltyp[idxp[i]] - 1]++] = i;
  for (int64_t i = 1; i < n; i++) {
    dsigma[i] = d[idxp[i]];
    int64_t col = idxq[idx[idxp[idxc[i]]] + 1];
    if (col <= nl) col--;
    memcpy(u2 + i * n, u + col * ldu, sizeof(R) * n);
    memcpy(v2 + i * m, v + col * ldv, sizeof(R) * m);
  }
  dsigma[0] = 0;
  if (F(abs)(dsigma[1]) <= tol / 2) dsigma[1] = tol / 2;
  /* With n + 1 columns, a rotation of v's columns nl and n folds z's last
     element into its first. */
  R c = 1, s = 0;
  if (m > n) {
    z[0] = F(lapy3)(z1, z[m - 1], 0);
    if (z[0] <= tol) z[0] = tol;
    else {
      c = z1 / z[0];
      s = z[m - 1] / z[0];
    }
  } else z[0] = F(abs)(z1) <= tol ? tol : z1;
  memcpy(z + 1, u2 + 1, sizeof(R) * (k - 1));
  memset(u2, 0, sizeof(R) * n);
  u2[nl] = 1;
  if (m > n) {
    R *vl = v + nl * ldv, *vm = v + (m - 1) * ldv;
    for (int64_t i = 0; i <= nl; i++) {
      vm[i] = -s * vl[i];
      v2[i] = c * vl[i];
    }
    for (int64_t i = nl + 1; i < m; i++) {
      v2[i] = s * vm[i];
      vm[i] = c * vm[i];
    }
    memcpy(v2 + (m - 1) * m, vm, sizeof(R) * m);
  } else memcpy(v2, v + nl * ldv, sizeof(R) * m);
  for (int64_t i = k; i < n; i++) {
    d[i] = dsigma[i];
    memcpy(u + i * ldu, u2 + i * n, sizeof(R) * n);
    memcpy(v + i * ldv, v2 + i * m, sizeof(R) * m);
  }
  /* The k roots (xLASD3): deltas into u's first k columns, the sums
     d_j + σ into v's, then z recomputed from the roots (Gu and
     Eisenstat), so that the vectors are orthogonal to working
     accuracy. */
  if (k == 1) {
    d[0] = F(abs)(z[0]);
    memcpy(v, v2, sizeof(R) * m);
    for (int64_t i = 0; i < n; i++) u[i] = z[0] > 0 ? u2[i] : -u2[i];
  } else {
    memcpy(q, z, sizeof(R) * k);
    R rho = F(rnorm)(z, k);
    for (int64_t i = 0; i < k; i++) z[i] /= rho;
    rho *= rho;
    /* The secular vectors, normalised, rows by type: u's into q. */
    F(sd_job) sj = {.k = k, .ldu = ldu, .ldv = ldv, .u = u, .v = v, .q = q,
                    .z = z, .dsigma = dsigma, .d = d, .rho = rho,
                    .idxc = idxc};
    if (!F(sd)(&sj, SVD_ROOTS, sa)) return 0;
    F(sd)(&sj, SVD_WEIGHTS, sa);
    F(sd)(&sj, LEFT_VECTORS, sa);
    /* u = u2 q: its first nl rows from types 1 and 3, row nl from q's
       first row, the last nr from types 2 and 3. */
    int64_t c1 = ctot[0], c2 = ctot[1], c3 = ctot[2], k3 = 1 + c1 + c2;
    if (k == 2) {
      if (!F(rgemm)(n, k, k, u2, n, q, k, u, ldu)) return -1;
    } else {
      if (!F(rgemm)(nl, k, c1, u2 + n, n, q + 1, k, u, ldu) ||
          !F(raccum)(nl, k, c3, u2 + k3 * n, n, q + k3, k, u, ldu))
        return -1;
      for (int64_t i = 0; i < k; i++) u[nl + i * ldu] = q[i * k];
      if (!F(rgemm)(nr, k, c2 + c3, u2 + (nl + 1) + (1 + c1) * n, n,
                    q + 1 + c1, k, u + nl + 1, ldu))
        return -1;
    }
    /* v's into q, then v = v2 q likewise. */
    F(sd)(&sj, RIGHT_VECTORS, sa);
    if (k == 2) {
      if (!F(rgemm)(m, k, k, v2, m, q, k, v, ldv)) return -1;
    } else {
      if (!F(rgemm)(nl + 1, k, 1 + c1, v2, m, q, k, v, ldv) ||
          !F(raccum)(nl + 1, k, c3, v2 + k3 * m, m, q + k3, k, v, ldv))
        return -1;
      /* The last rows take column 0 and types 2 and 3: column 0 moved
         next to them, over the last of type 1. */
      if (c1 > 0) {
        for (int64_t i = 0; i < k; i++) q[c1 + i * k] = q[i * k];
        for (int64_t i = nl + 1; i < m; i++) v2[i + c1 * m] = v2[i];
      }
      if (!F(rgemm)(nr + sqre, k, 1 + c2 + c3, v2 + (nl + 1) + c1 * m, m,
                    q + c1, k, v + nl + 1, ldv))
        return -1;
    }
  }
  for (int64_t i = 0; i < n; i++) d[i] *= big;
  F(lamrg)(k, n - k, d, 1, -1, idxq);
  return 1;
}

/* A level of lasd0's tree: its nodes, the leaves' lasdq or the merges'
   lasd1, each on its own slice of work and iwork, their answers into
   [ok]. */
typedef struct {
  R *d, *e, *u, *v, *work;
  int64_t n, ldu, ldv, *idxq, *iwork, *at, *ok;
  int sqre, t, leaves;
  arena *workers;
} F(lasd_level);

static int F(lasd_node_run)(F(lasd_level) *l, int64_t j, arena *s) {
  int64_t c, size, ldu = l->ldu, ldv = l->ldv;
  int sq;
  lasd_node(l->n, l->sqre, l->t, j, &c, &size, &sq);
  R *d = l->d + c, *e = l->e + c, *u = l->u + c + c * ldu,
    *v = l->v + c + c * ldv, *w = l->work + l->at[j];
  if (l->leaves) {
    for (int64_t i = 0; i < size; i++) l->idxq[c + i] = i;
    return F(lasdq)(size, sq, d, e, u, ldu, v, ldv, w);
  }
  int64_t nl = size / 2, nr = size - nl - 1;
  return F(lasd1)(nl, nr, sq, d, d[nl], e[nl], u, ldu, v, ldv, l->idxq + c,
                  w, l->iwork + 4 * c, s);
}

/* A unit runs on its worker's arena alone, as a solve's do. */
static void F(lasd_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(lasd_level) *l = ctx;
  arena *s = &l->workers[worker], *all = s->workers;
  s->workers = NULL;
  for (int64_t j = lo; j < hi; j++) l->ok[j] = F(lasd_node_run)(l, j, s);
  s->workers = all;
}

/* The SVD of the upper bidiagonal B, [n] × (n + [sqre]), of the diagonal
   [d] and superdiagonal [e], into d, u and v as lasd1 leaves them, from
   the identity: torn in two at row n / 2 [levels] times, the first half
   of n + 1 columns, the last of its parent's, the leaves by lasdq, then
   merged (xLASD0); [idxq] orders d ascending. The tree is solved level
   by level from its leaves: a level's nodes are independent, so where it
   has at least half as many as the threads its work pays for, they run on
   the threads, one a unit, each on its slice of [work] and [iwork], and a
   level of fewer runs its nodes in turn, each on the threads. A node's
   arithmetic is the same either way.

   bdsdc of svd-f64-256, -1024 and -2048, minima, took so 1.22, 13.2 and
   61.9 ms on the M1's 8 P-cores (timing lock, load 10-18, 2 rounds) and
   0.86, 17.1 and 85.3 on fudge's 6 P-cores (load 2.4-5.7, 4 rounds).
   With levels of at least 2 nodes on the threads they took 1.10, 14.0
   and 68.3 on the M1 and 0.81, 20.2 and 106.5 on fudge; with 4, 1.26,
   13.2 and 62.3, and 0.87, 17.3 and 85.5; with 8, 1.41, 13.9 and 62.3,
   and 1.03, 17.9 and 85.5.

   [work] holds 3(n + 1)² + 4n reals and [iwork] 4n indices. Answers 1, 0
   if a root does not converge, or −1 if an allocation failed. */
static int F(lasd0)(int64_t n, int sqre, int levels, R *d, R *e, R *u,
                    int64_t ldu, R *v, int64_t ldv, int64_t *idxq, R *work,
                    int64_t *iwork, arena *s) {
  int64_t avail = 3 * (n + 1) * (n + 1) + 4 * n;
  int64_t *ok =
      room(s->copy, ((int64_t)2 << levels) * (int64_t)sizeof(int64_t));
  if (ok == NULL) return -1;
  int64_t *at = ok + ((int64_t)1 << levels);
  for (int t = levels; t >= 0; t--) {
    int64_t nodes = (int64_t)1 << t, total = 0, big = 0;
    for (int64_t j = 0; j < nodes; j++) {
      int64_t c, size;
      int sq;
      lasd_node(n, sqre, t, j, &c, &size, &sq);
      at[j] = total;
      total += lasd_work(size);
      big = size > big ? size : big;
    }
    F(lasd_level) l = {.d = d, .e = e, .u = u, .v = v, .work = work, .n = n,
                       .ldu = ldu, .ldv = ldv, .idxq = idxq, .iwork = iwork,
                       .at = at, .ok = ok, .sqre = sqre, .t = t,
                       .leaves = t == levels, .workers = s->workers};
    int64_t bytes = nodes * big * big * 2 * (int64_t)sizeof(R);
    double cost =
        total <= avail
            ? threaded(s, nodes, bytes, 4.0 * nodes * big * big * big, RDT, 0)
            : 0;
    if (cost > 0 && 2 * nodes >= nx_cpu_threads(bytes, cost))
      nx_cpu_job(nodes, bytes, cost, F(lasd_unit), &l);
    else
      /* In turn, every node on the whole of [work]: the slices fit only
         where total <= avail. */
      for (int64_t j = 0; j < nodes; j++) {
        at[j] = 0;
        ok[j] = F(lasd_node_run)(&l, j, s);
      }
    for (int64_t j = 0; j < nodes; j++)
      if (ok[j] != 1) return ok[j];
  }
  return 1;
}

/* The singular values of the upper bidiagonal B of the diagonal [d] and
   superdiagonal [e], [n] of them, descending into d, and where [u] is not
   NULL, B = U diag d Vᵀ into u and v, n × n, columns [ldu] and [ldv]
   apart, as LAPACK's xBDSDC: B's unreduced blocks, split where an e is
   below 0.9 u after B is scaled to a largest magnitude of one, by divide
   and conquer (lasd0); B of at most SMLSIZ values, or without vectors, by
   lasdq. [work] holds 3(n + 1)² + 4n reals and [iwork] 5n indices.
   Answers 1, 0 if an iteration does not converge, or −1 if an allocation
   failed. */
static int F(bdsdc)(int64_t n, R *d, R *e, R *u, int64_t ldu, R *v,
                    int64_t ldv, R *work, int64_t *iwork, arena *s) {
  for (int64_t j = 0; u && j < n; j++)
    for (int64_t i = 0; i < n; i++) {
      u[i + j * ldu] = i == j;
      v[i + j * ldv] = i == j;
    }
  if (u == NULL || n <= SMLSIZ) {
    if (!F(lasdq)(n, 0, d, e, u, ldu, v, ldv, work)) return 0;
    F(sort)(n, d, DESCENDING, u, n, ldu, v, n, ldv);
    return 1;
  }
  R big = 0;
  for (int64_t i = 0; i < n; i++) {
    R x = F(abs)(d[i]), y = i < n - 1 ? F(abs)(e[i]) : 0;
    big = x > big ? x : big;
    big = y > big ? y : big;
  }
  if (big == 0) return 1;
  F(scale)(d, e, 0, n - 1, big, 1);
  const R eps = (R)0.9 * EPS;
  for (int64_t i = 0; i < n; i++)
    if (F(abs)(d[i]) < eps) d[i] = F(sign)(eps, d[i]);
  for (int64_t start = 0, i = 0; i < n - 1; i++) {
    int small = F(abs)(e[i]) < eps;
    if (!small && i < n - 2) continue;
    int64_t size = !small ? n - start : i - start + 1;
    if (small && i == n - 2) {
      /* The last value alone. */
      u[(n - 1) + (n - 1) * ldu] = F(sign)(1, d[n - 1]);
      d[n - 1] = F(abs)(d[n - 1]);
    }
    int levels = 0;
    if (size > SMLSIZ)
      for (levels = 1; (int64_t)(SMLSIZ + 1) << levels <= size;) levels++;
    int ok = F(lasd0)(size, 0, levels, d + start, e + start,
                      u + start + start * ldu, ldu, v + start + start * ldv,
                      ldv, iwork, work, iwork + n, s);
    if (ok != 1) return ok;
    start = i + 1;
  }
  for (int64_t i = 0; i < n; i++) d[i] *= big;
  F(sort)(n, d, DESCENDING, u, n, ldu, v, n, ldv);
  return 1;
}

/* y[j] += s Σ_i conj(a[i + j·lda]) x[i] for j < [n], i < [m]: y += s aᴴ
   x for the column-major a (xGEMV's 'C'). By four columns at a time,
   which share x's loads, each product in four parts by the index mod 4
   as dot's, so that a column's bits do not depend on its neighbours. */
static void F(mvh)(int64_t m, int64_t n, const T *a, int64_t lda,
                   const T *x, T *y, R s) {
  int64_t j = 0;
  for (; j + 4 <= n; j += 4) {
    const T *c[4] = {a + j * lda, a + (j + 1) * lda, a + (j + 2) * lda,
                     a + (j + 3) * lda};
    T p[4][4] = {{0}};
    int64_t r = 0;
    for (; r + 4 <= m; r += 4)
      for (int l = 0; l < 4; l++)
        for (int q = 0; q < 4; q++) p[q][l] += MUL(CONJ(c[q][r + l]), x[r + l]);
    for (; r < m; r++)
      for (int q = 0; q < 4; q++) p[q][r & 3] += MUL(CONJ(c[q][r]), x[r]);
    for (int q = 0; q < 4; q++)
      y[j + q] += s * ((p[q][0] + p[q][1]) + (p[q][2] + p[q][3]));
  }
  for (; j < n; j++) y[j] += s * F(dot)(a + j * lda, x, m);
}

/* y[i] += s Σ_j a[i + j·lda] x[j] for i < [m], j < [n]: y += s a x for
   the column-major a (xGEMV's 'N'), by four columns at a time, which
   share y's loads and stores. */
static void F(mvn)(int64_t m, int64_t n, const T *a, int64_t lda,
                   const T *x, T *y, R s) {
  int64_t j = 0;
  for (; j + 4 <= n; j += 4) {
    const T *c0 = a + j * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;
    T t0 = s * x[j], t1 = s * x[j + 1], t2 = s * x[j + 2], t3 = s * x[j + 3];
    for (int64_t i = 0; i < m; i++) {
      T u = y[i];
      u += MUL(c0[i], t0);
      u += MUL(c1[i], t1);
      u += MUL(c2[i], t2);
      u += MUL(c3[i], t3);
      y[i] = u;
    }
  }
  for (; j < n; j++) {
    const T *aj = a + j * lda;
    T t = s * x[j];
    for (int64_t i = 0; i < m; i++) y[i] += MUL(aj[i], t);
  }
}

/* The panel of the first [nb] columns and rows of the [m] × [n] [a], m >=
   n, by reflectors, as LAPACK's xLABRD: column i, the products of the
   panel's earlier reflectors deferred until now applied to it, is reduced
   below its diagonal by H_i, its diagonal into d[i]; then row i likewise,
   on its conjugate, right of its superdiagonal by G_i, its superdiagonal
   into e[i]. v_i and u_i are stored where they reduced, their first
   elements 1. [x], m × nb, and [y], n × nb, columns [ldx] and [ldy]
   apart, get X and Y with the rest of a updated as a − V Yᴴ − X U. [row]
   holds n elements: the conjugated row, contiguous.

   Each column's work is three phases of chunks, which the call's threads
   claim in order, and two steps between them that one thread takes:

     A(i)  rows i on: X's column i − 1, a u_(i−1) from B(i − 1)'s parts
           less the panel's products, then column i's update
     H_i   column i's reflector
     B(i)  the panel's Vᴴ v_i and Xᴴ v_i, two chunks, then columns of the
           rest: aᴴ v_i, and once the two are done, Y's column i, row i's
           update, and the columns' part of a times the row, which G_i
           scales into u_i
     G_i   row i's reflector
     D(i)  columns of the rest by BRD_COLUMNS: their parts of Yᴴ u_i and
           U u_i, which A(i + 1)'s chunks each add in chunk order

   A chunk of B reads its columns of the rest once for both of their
   products with a vector, aᴴ v_i and a u_i, as Howell, Demmel, Fulton,
   Hammarling and Marmol's bidiagonalisation by BLAS 2.5 operators does
   (2008): its two calls, mvh then mvn after a wait, find the chunk's
   columns in the cache. a u_i is a's first column plus a times row i's
   rest, divided as G_i divides the rest, which is linear. LAPACK's
   xLABRD reads the rest twice a column.

   A chunk's arithmetic is the same on any thread, so the bits are a
   function of the panel alone. A thread waits only on chunks claimed
   before its own: those of earlier phases, or B's first two. One thread
   alone runs them all in order. */
typedef struct {
  T *a, *x, *y, *row, *parts;
  /* B's chunks' parts of a times the row's rest, m elements each, their
     exponents, and as G_i leaves them, the parts' powers of two and
     their divisor. */
  T *au;
  T div;
  R *auf;
  int *aue;
  R *d, *e;
  T *tauq, *taup;
  int64_t m, n, nb, lda, ldx, ldy, phases, rows; /* rows: an A chunk's */
  int64_t end[3 * NB + 1]; /* the chunks of the phases to each's end */
  /* Chunks claimed and done, the open phase and B's first two chunks
     done, a line apart. */
  int64_t sync[4 * LINE];
} F(brd);

/* The chunks of phase [p]: A, B and D of column p / 3 by p mod 3, A of
   column nb's last: X's last column alone. A(0) has no work: one empty
   chunk. */
static int64_t F(brd_chunks)(const F(brd) *b, int64_t p) {
  int64_t i = p / 3, nr = b->n - i - 1, c = 1;
  switch (p % 3) {
    case 0: c = i > 0 ? (b->m - i + b->rows - 1) / b->rows : 1; break;
    case 1: c = nr > 0 ? (nr + MV_COLUMNS - 1) / MV_COLUMNS + 2 : 1; break;
    default: c = (nr + BRD_COLUMNS - 1) / BRD_COLUMNS; break;
  }
  return c > 0 ? c : 1;
}

/* x times 2^[e], part by part. */
static inline T F(ldexp)(T x, int e) {
#if CPLX
  return F(cplx)(LDEXP(REAL(x), e), LDEXP(IMAG(x), e));
#else
  return LDEXP(x, e);
#endif
}

/* Chunk [k] of phase [p]. */
static void F(brd_chunk)(F(brd) *b, int64_t p, int64_t k) {
  int64_t i = p / 3, m = b->m, n = b->n, nr = n - i - 1;
  int64_t lda = b->lda, ldx = b->ldx, ldy = b->ldy;
  T *a = b->a, *x = b->x, *y = b->y, *row = b->row, *ai = a + i * lda;
  T *yr = y + i * ldy + i + 1, *yv = y + i * ldy, *yx = x + i * ldx;
  T t[NB + 1], w[NB];
  if (p % 3 == 0) {
    int64_t r0 = i + k * b->rows, h = m - r0 < b->rows ? m - r0 : b->rows;
    if (h <= 0) return;
    /* X's column j = i − 1: a u_j, a's column i plus B(j)'s parts of a
       times the row's rest, each times its power of two, over their
       divisor; less A V's and X U's parts, the D chunks' sums of Yᴴ u_j
       and U u_j; then times τ. B(j) had rest chunks of the rest. */
    if (i > 0 && n - i > 0) {
      int64_t j = i - 1, parts = (n - i + BRD_COLUMNS - 1) / BRD_COLUMNS;
      int64_t rest = (n - i + MV_COLUMNS - 1) / MV_COLUMNS;
      T *xr = x + j * ldx + r0;
      for (int64_t q = 0; q <= j; q++) {
        t[q] = b->parts[q];
        for (int64_t c = 1; c < parts; c++) t[q] += b->parts[c * 2 * NB + q];
      }
      for (int64_t q = 0; q < j; q++) {
        w[q] = b->parts[NB + q];
        for (int64_t c = 1; c < parts; c++)
          w[q] += b->parts[c * 2 * NB + NB + q];
      }
      for (int64_t r = 0; r < h; r++) {
        T z = b->au[r0 + r] * b->auf[0];
        for (int64_t c = 1; c < rest; c++)
          z += b->au[c * m + r0 + r] * b->auf[c];
        xr[r] = a[r0 + r + i * lda] + DIV(z, b->div);
      }
      F(mvn)(h, j + 1, a + r0, lda, t, xr, -1);
      F(mvn)(h, j, x + r0, ldx, w, xr, -1);
      for (int64_t r = 0; r < h; r++) xr[r] = MUL(xr[r], b->taup[j]);
    }
    if (i == b->nb) return;
    /* Column i less V Yᴴ's and X U's parts. */
    for (int64_t q = 0; q < i; q++) t[q] = CONJ(y[i + q * ldy]);
    F(mvn)(h, i, a + r0, lda, t, ai + r0, -1);
    F(mvn)(h, i, x + r0, ldx, ai, ai + r0, -1);
    return;
  }
  if (nr == 0) return;
  if (p % 3 == 1 && k < 2) {
    /* The panel's Vᴴ v_i above Y's column i, Xᴴ v_i above X's. */
    T *z = k == 0 ? yv : yx;
    memset(z, 0, sizeof(T) * i);
    F(mvh)(m - i, i, k == 0 ? a + i : x + i, k == 0 ? lda : ldx, ai + i, z, 1);
    __atomic_add_fetch(&b->sync[3 * LINE], 1, __ATOMIC_RELEASE);
    return;
  }
  int64_t width = p % 3 == 1 ? MV_COLUMNS : BRD_COLUMNS;
  int64_t c0 = (p % 3 == 1 ? k - 2 : k) * width;
  int64_t cols = nr - c0 < width ? nr - c0 : width;
  T *ac = ai + (c0 + 1) * lda, *yc = y + i + 1 + c0;
  if (p % 3 == 1) {
    /* Y's column i: τ_i times aᴴ v_i less Y Vᴴ v_i and Uᴴ Xᴴ v_i. Then
       row i, conjugated, less Y's and Uᴴ X's parts. */
    memset(yr + c0, 0, sizeof(T) * cols);
    F(mvh)(m - i, cols, ac + i, lda, ai + i, yr + c0, 1);
    wait_for(&b->sync[3 * LINE], 2 * (i + 1));
    F(mvn)(cols, i, yc, ldy, yv, yr + c0, -1);
    F(mvh)(i, cols, ac, lda, yx, yr + c0, -1);
    for (int64_t c = c0; c < c0 + cols; c++) {
      yr[c] = MUL(yr[c], b->tauq[i]);
      row[c] = CONJ(ai[(c + 1) * lda + i]);
    }
    for (int64_t q = 0; q <= i; q++) t[q] = CONJ(a[i + q * lda]);
    F(mvn)(cols, i + 1, yc, ldy, t, row + c0, -1);
    for (int64_t q = 0; q < i; q++) w[q] = CONJ(x[i + q * ldx]);
    F(mvh)(i, cols, ac, lda, w, row + c0, -1);
    /* The chunk's part of a times the row's rest, rows i + 1 on, the
       rest's entries first scaled by a power of two to a largest part in
       [1, 2), its exponent beside it. G_i scales a tiny rest up before it
       divides it (knt), so that u_i keeps its digits; unscaled, a tiny
       rest times a small a would flush to 0 here first. */
    int64_t first = c0 == 0;
    T sr[MV_COLUMNS], *au = b->au + (k - 2) * m + i + 1;
    R most = 0;
    for (int64_t c = first; c < cols; c++) {
      R mr = F(abs)(REAL(row[c0 + c])), mi = F(abs)(IMAG(row[c0 + c]));
      most = mr > most ? mr : most;
      most = mi > most ? mi : most;
    }
    int e = most > 0 ? ILOGB(most) : 0;
    for (int64_t c = first; c < cols; c++) sr[c] = F(ldexp)(row[c0 + c], -e);
    b->aue[k - 2] = most > 0 ? e : INT_MIN;
    memset(au, 0, sizeof(T) * (m - i - 1));
    F(mvn)(m - i - 1, cols - first, ac + first * lda + i + 1, lda, sr + first,
           au, 1);
    return;
  }
  /* The chunk's parts of Yᴴ u_i and U u_i; u_i's conjugate into row i. */
  T *py = b->parts + k * 2 * NB, *pa = py + NB;
  memset(py, 0, sizeof(T) * (i + 1));
  F(mvh)(cols, i + 1, yc, ldy, row + c0, py, 1);
  memset(pa, 0, sizeof(T) * i);
  F(mvn)(i, cols, ac, lda, row + c0, pa, 1);
  for (int64_t c = c0; c < c0 + cols; c++)
    ai[(c + 1) * lda + i] = CONJ(row[c]);
}

/* The step after phase [p], if any: H_i after A(i), G_i after B(i). */
static void F(brd_step)(F(brd) *b, int64_t p) {
  int64_t i = p / 3, nr = b->n - i - 1;
  if (p % 3 == 0 && i < b->nb) {
    T *ai = b->a + i * b->lda, alpha = ai[i];
    b->tauq[i] = F(reflector)(&alpha, ai + i + 1, b->m - i - 1, 1);
    b->d[i] = REAL(alpha);
    if (nr > 0) ai[i] = 1;
  } else if (p % 3 == 1 && nr > 0) {
    /* G_i, then a u_i's powers of two: a part's, 2^(its exponent − G_i's
       − its rescales'), at most 2 and of the size of u_i's elements; and
       their divisor, G_i's with its largest part in [1, 2). Scaled by the
       rest's largest part instead, the divisor overflows beside a rest of
       subnormals, and complex division by its infinite parts is NaN. */
    T beta = b->row[0], sc;
    int knt, top = INT_MIN;
    int64_t rest = (nr + MV_COLUMNS - 1) / MV_COLUMNS;
    b->taup[i] = F(reflector_by)(&beta, b->row + 1, nr - 1, 1, &knt, &sc);
    b->e[i] = REAL(beta);
    b->row[0] = 1;
    for (int64_t c = 0; c < rest; c++) top = b->aue[c] > top ? b->aue[c] : top;
    R sr = F(abs)(REAL(sc)), si = F(abs)(IMAG(sc));
    int es = top == INT_MIN ? 0 : ILOGB(sr > si ? sr : si);
    int shift = es + knt * ILOGB(SAFMIN);
    for (int64_t c = 0; c < rest; c++)
      b->auf[c] = b->aue[c] == INT_MIN ? 0 : LDEXP((R)1, b->aue[c] - shift);
    b->div = top == INT_MIN ? 1 : F(ldexp)(sc, -es);
  }
}

/* A unit claims chunks until none is left; the thread that finishes a
   phase's last takes the step after it and opens the next phase. */
static void F(brd_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  F(brd) *b = ctx;
  int64_t p = 0, total = b->end[b->phases - 1];
  for (int64_t g; (g = __atomic_fetch_add(&b->sync[0], 1, __ATOMIC_RELAXED)) <
                  total;) {
    while (b->end[p] <= g) p++;
    wait_for(&b->sync[2 * LINE], p);
    F(brd_chunk)(b, p, p == 0 ? g : g - b->end[p - 1]);
    if (__atomic_add_fetch(&b->sync[LINE], 1, __ATOMIC_ACQ_REL) == b->end[p]) {
      F(brd_step)(b, p);
      __atomic_store_n(&b->sync[2 * LINE], p + 1, __ATOMIC_RELEASE);
    }
  }
}

/* The phases' parts of Yᴴ u and U u take 2 NB elements a chunk of
   BRD_COLUMNS of the rest, those of a times the row m and a power of two
   and an exponent a chunk of MV_COLUMNS. Answers 0 if an allocation
   failed. */
static int F(labrd)(T *a, int64_t m, int64_t n, int64_t nb, int64_t lda,
                    R *d, R *e, T *tauq, T *taup, T *x, int64_t ldx, T *y,
                    int64_t ldy, T *row, arena *s) {
  int64_t chunks = (n + BRD_COLUMNS - 1) / BRD_COLUMNS;
  int64_t rest = (n + MV_COLUMNS - 1) / MV_COLUMNS;
  T *parts = room(s->copy, (chunks * 2 * NB + rest * m) * (int64_t)sizeof(T) +
                               rest * (int64_t)(sizeof(R) + sizeof(int)));
  if (parts == NULL) return 0;
  R *auf = (R *)(parts + chunks * 2 * NB + rest * m);
  int64_t rows = (m + MV_CHUNKS - 1) / MV_CHUNKS;
  F(brd) b = {.a = a, .x = x, .y = y, .row = row, .parts = parts,
              .au = parts + chunks * 2 * NB, .auf = auf,
              .aue = (int *)(auf + rest), .d = d, .e = e, .tauq = tauq,
              .taup = taup, .m = m, .n = n, .nb = nb, .lda = lda, .ldx = ldx,
              .ldy = ldy, .phases = 3 * nb + 1,
              .rows = rows > MV_ROWS ? rows : MV_ROWS};
  /* As many units as the largest phase has chunks: no more threads can
     work at once. */
  int64_t units = 1;
  for (int64_t p = 0; p < b.phases; p++) {
    int64_t c = F(brd_chunks)(&b, p);
    b.end[p] = (p > 0 ? b.end[p - 1] : 0) + c;
    units = c > units ? c : units;
  }
  /* Threads where one column's products pay for them, since every phase
     waits for its last chunk; else the chunks in order, without the
     claims. */
  int64_t bytes = m * n * (int64_t)sizeof(T);
  double cost = threaded(s, units, bytes, 4.0 * m * n, RDT, CPLX);
  if (cost > 0) {
    nx_cpu_job(units, bytes, cost, F(brd_unit), &b);
    return 1;
  }
  for (int64_t p = 0, g = 0; p < b.phases; p++) {
    for (; g < b.end[p]; g++)
      F(brd_chunk)(&b, p, p == 0 ? g : g - b.end[p - 1]);
    F(brd_step)(&b, p);
  }
  return 1;
}

/* a = Q B Pᴴ in place in the [m] × [n] [a], m >= n, columns [lda] apart,
   as LAPACK's blocked xGEBRD: B upper bidiagonal, its diagonal into [d]
   and superdiagonal into [e]; Q = H_0 ⋯ H_(n−1) with v_i below a's
   diagonal in column i, P = G_0 ⋯ G_(n−2) with u_i's conjugate right of
   its superdiagonal in row i, their τ in [tauq] and [taup]. By panels of
   NB columns and rows (labrd), each followed by the rest's update a − V
   Yᴴ − X U by products on its transpose, row-major. The scratch [w] holds
   (m + n) NB + n elements. Answers 0 if an allocation failed. */
static int F(gebrd)(T *a, int64_t m, int64_t n, int64_t lda, R *d, R *e,
                    T *tauq, T *taup, T *w, arena *s) {
  T *x = w, *y = x + m * NB, *row = y + n * NB;
  for (int64_t i = 0; i < n; i += NB) {
    int64_t nb = NB < n - i ? NB : n - i, m2 = m - i - nb, n2 = n - i - nb;
    T *p = a + i + i * lda, *c = p + nb + nb * lda;
    if (!F(labrd)(p, m - i, n - i, nb, lda, d + i, e + i, tauq + i, taup + i,
                  x, m, y, n, row, s))
      return 0;
    if (n2 == 0) continue;
    if (!F(gemm)(n2, m2, nb, -1, (operand){y + nb, 1, n, CPLX},
                 F(op)(p + nb, lda, 1), c, lda, s) ||
        !F(gemm)(n2, m2, nb, -1, F(op)(p + nb * lda, lda, 1),
                 F(op)(x + nb, m, 1), c, lda, s))
      return 0;
  }
  return 1;
}

/* Two stages

   Where svd_two says, B = Q1 Bb P1ᴴ with Bb upper triangular of kd
   superdiagonals, then Bb = Q2 Bd P2ᴴ with Bd upper bidiagonal, as
   PLASMA's xGE2GB and xGBBRD. The first stage's flops are products on the
   threads, where half the one-stage reduction's are matrix-vector
   products that read the trailing matrix twice a column; the second
   stage's O(k² kd) work runs on the band alone. B is row-major here,
   element (i, j) at a[i·ld + j]. */

/* A step's last update of the rows [c] below its own, m × wn, past
   their first kd columns: c −= U B by blocks of kd columns, U = [V1 Y],
   m × kb, and B = [Z; V2ᴴ], kb × wn, and the next step's columns'
   factorisation, whose columns are c's first, beside them: unit 0 the
   panel, unit k columns k·kd on. */
typedef struct {
  const T *u, *b;
  T *c, *tau, *v, *t, *l, *q;
  int64_t m, wn, kb, ld, kd;
  int failed;
  arena *workers;
} F(gb_step);

static void F(gb_units)(F(gb_step) *j, int64_t lo, int64_t hi, arena *s) {
  int64_t kd = j->kd;
  for (int64_t k = lo; k < hi; k++) {
    int64_t c0 = k * kd, cb = j->wn - c0 < kd ? j->wn - c0 : kd;
    if (k == 0 ? !F(qr_panel)(j->c, j->m, cb, j->ld, j->tau, j->v, j->t, kd,
                              j->l, j->q, s)
               : !F(gemm)(j->m, cb, j->kb, -1, F(op)(j->u, 2 * kd, 1),
                          F(op)(j->b + c0, j->wn, 1), j->c + c0, j->ld, s))
      __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
  }
}

/* A unit runs on its worker's arena alone, as a solve's do. */
static void F(gb_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(gb_step) *j = ctx;
  arena *s = &j->workers[worker], *all = s->workers;
  s->workers = NULL;
  F(gb_units)(j, lo, hi, s);
  s->workers = all;
}

/* a = Q1 Bb P1ᴴ in place in the [m] × [k] [a], m >= k, rows [ld] apart:
   Bb's band on and above the diagonal, Q1 = H_0 ⋯ H_(k−1) with v_j
   below the diagonal in column j, P1 = G_0 ⋯ G_(k−kd−1) with v_j's
   conjugate from column j + kd + 1 on in row j, their τ in [tauq] and
   [taup]. By steps of kd columns and rows, as xLABRD's panels with
   blocks of reflectors: the step's columns factored (qr_panel), H1 = I −
   V1 T1 V1ᴴ, then the columns c right of them take H1ᴴ, c − V1 Z with
   Z = T1ᴴ V1ᴴ c, on their first kd rows alone; those rows' LQ, factored
   on a conjugated transposed copy, gives H2 = I − V2 T2 V2ᴴ; the rows
   below take both, c − V1 Z − Y V2ᴴ with Y = (c V2 − V1 (Z V2)) T2, by
   one product [V1 Y] [Z; V2ᴴ], three reads of c where H1 then H2 would
   take four. The next step's columns are updated first and factored
   beside the rest's update on the threads, as Cholesky's next diagonal
   block. The scratch [w] holds svd_scratch elements. Answers 0 if an
   allocation failed. */
static int F(ge2gb)(T *a, int64_t m, int64_t k, int64_t ld, T *tauq,
                    T *taup, T *w, int64_t kd, arena *s) {
  T *v = w, *t = v + m * kd, *u = t + kd * kd, *y1 = u + 2 * m * kd,
    *v2 = y1 + m * kd, *t2 = v2 + k * kd, *y = t2 + kd * kd,
    *z1 = y + k * kd, *b = z1 + k * kd, *zv = b + 2 * k * kd,
    *q = zv + kd * kd, *l = q + 2 * kd * kd;
  if (!F(qr_panel)(a, m, k < kd ? k : kd, ld, tauq, v, t, kd, l, q, s))
    return 0;
  for (int64_t i0 = 0; i0 + kd < k; i0 += kd) {
    int64_t h = m - i0, wn = k - i0 - kd, m2 = h - kd;
    int64_t jb = wn < kd ? wn : kd, kb = kd + jb;
    T *c = a + i0 * ld + i0 + kd, *c2 = c + kd * ld;
    /* Z into B's first kd rows, then c's first kd rows take H1ᴴ. */
    memset(z1, 0, sizeof(T) * kd * wn);
    memset(b, 0, sizeof(T) * kd * wn);
    if (!F(gemm)(kd, wn, h, 1, F(op_h)(v, kd), F(op)(c, ld, 1), z1, wn, s) ||
        !F(gemm)(kd, wn, kd, 1, F(op_h)(t, kd), F(op)(z1, wn, 1), b, wn, s) ||
        !F(gemm)(kd, wn, kd, -1, F(op)(v, kd, 1), F(op)(b, wn, 1), c, ld, s))
      return 0;
    /* Their LQ, c = L H2ᴴ, as the QR of cᴴ = H2 Lᴴ. */
    for (int64_t j = 0; j < wn; j++)
      for (int64_t r = 0; r < kd; r++) y[j * kd + r] = CONJ(c[r * ld + j]);
    if (!F(qr_panel)(y, wn, jb, kd, taup + i0, v2, t2, kd, l, q, s) ||
        (jb < kd && !F(larfb)(v2, kd, F(op_h)(t2, kd), jb, wn, kd - jb,
                              y + jb, kd, u, s)))
      return 0;
    for (int64_t j = 0; j < wn; j++)
      for (int64_t r = 0; r < kd; r++) c[r * ld + j] = CONJ(y[j * kd + r]);
    /* U = [V1 Y] below the first kd rows, B's last jb rows V2ᴴ. */
    memset(zv, 0, sizeof(T) * kd * jb);
    memset(y1, 0, sizeof(T) * m2 * jb);
    if (!F(gemm)(kd, jb, wn, 1, F(op)(b, wn, 1), F(op)(v2, kd, 1), zv, jb,
                 s) ||
        !F(gemm)(m2, jb, wn, 1, F(op)(c2, ld, 1), F(op)(v2, kd, 1), y1, jb,
                 s) ||
        !F(gemm)(m2, jb, kd, -1, F(op)(v + kd * kd, kd, 1), F(op)(zv, jb, 1),
                 y1, jb, s))
      return 0;
    for (int64_t r = 0; r < m2; r++) {
      memcpy(u + r * 2 * kd, v + (kd + r) * kd, sizeof(T) * kd);
      memset(u + r * 2 * kd + kd, 0, sizeof(T) * jb);
    }
    for (int64_t r = 0; r < jb; r++)
      for (int64_t j = 0; j < wn; j++)
        b[(kd + r) * wn + j] = CONJ(v2[j * kd + r]);
    if (!F(gemm)(m2, jb, jb, 1, F(op)(y1, jb, 1), F(op)(t2, kd, 1), u + kd,
                 2 * kd, s) ||
        !F(gemm)(m2, jb, kb, -1, F(op)(u, 2 * kd, 1), F(op)(b, wn, 1), c2, ld,
                 s))
      return 0;
    F(gb_step) j = {.u = u, .b = b, .c = c2, .tau = tauq + i0 + kd, .v = v,
                    .t = t, .l = l, .q = q, .m = m2, .wn = wn, .kb = kb,
                    .ld = ld, .kd = kd, .workers = s->workers};
    int64_t units = (wn + kd - 1) / kd, bytes = m2 * wn * (int64_t)sizeof(T);
    double cost = threaded(s, units, bytes, 2.0 * m2 * wn * kb, RDT, CPLX);
    if (cost == 0) F(gb_units)(&j, 0, units, s);
    else nx_cpu_job(units, bytes, cost, F(gb_unit), &j);
    if (j.failed) return 0;
  }
  return 1;
}

/* A bulge chase on the upper band [b] of order [n]: element (i, j) at
   b[j·3kd + i − j + 2kd − 1], j − i from 1 − kd to 2kd − 1, Bb's kd
   superdiagonals and the bulges about them; the left and right
   reflectors' slots [q] and [p], or NULL, which the vectors take at kd =
   KD; the next sweep to claim, and the tasks each sweep has finished. */
typedef struct {
  T *b, *q, *p;
  int64_t n, kd, next;
  int64_t *done;
} F(bchase);

/* Sweep [s], which makes row s bidiagonal by tasks: task j's right
   reflector G_j takes columns c0 = s + 1 + j·kd on, kd of them or to the
   last, from row s (j = 0) or from the first row of the bulge that
   H_(j−1) makes, applied to the rows below that one; its left reflector
   H_j takes column c0 below the diagonal, applied to the block's other
   columns, and to the next task's columns there before G_(j+1). Their τ
   and v past its 1 go to their slots, KD elements, the sweep's after
   those of the sweeps before it. Task j touches rows c0 − kd to c0 + kd
   − 1 of columns c0 to c0 + kd − 1, which sweep s − 1's tasks past j + 1
   do not: it waits until that sweep has finished them. */
static void F(bsweep)(F(bchase) *ch, int64_t s) {
  int64_t n = ch->n, kd = ch->kd, cs = 3 * kd - 1;
  int64_t tasks = (n - 2 - s) / kd + 1, before = (n - 1 - s) / kd + 1;
  T *b = ch->b + 2 * kd - 1;
  T *q = ch->q ? ch->q + chased(n, s) * KD : NULL,
    *p = ch->p ? ch->p + chased(n, s) * KD : NULL;
  T vh[KD], vg[KD], u[2 * KD], th = 0;
  for (int64_t j = 0; j < tasks; j++) {
    int64_t wait = j + 2 < before ? j + 2 : before;
    if (s > 0) wait_for(&ch->done[(s - 1) * LINE], wait);
    int64_t c0 = s + 1 + j * kd, len = n - c0 < kd ? n - c0 : kd;
    int64_t r0 = j == 0 ? s : c0 - kd, h = c0 + len - r0 - 1;
    for (int64_t c = c0; j > 0 && th != 0 && c < c0 + len; c++) {
      T *x = b + c * cs + r0, f = MUL(CONJ(th), F(dot)(vh, x, kd));
      for (int64_t r = 0; r < kd; r++) x[r] -= MUL(vh[r], f);
    }
    for (int64_t c = 0; c < len; c++) vg[c] = CONJ(b[(c0 + c) * cs + r0]);
    T tg = F(reflector)(vg, vg + 1, len - 1, 1);
    b[c0 * cs + r0] = vg[0];
    vg[0] = 1;
    for (int64_t c = 1; c < len; c++) b[(c0 + c) * cs + r0] = 0;
    if (tg != 0) {
      for (int64_t r = 0; r < h; r++) u[r] = 0;
      for (int64_t c = 0; c < len; c++) {
        const T *x = b + (c0 + c) * cs + r0 + 1;
        for (int64_t r = 0; r < h; r++) u[r] += MUL(x[r], vg[c]);
      }
      for (int64_t c = 0; c < len; c++) {
        T *x = b + (c0 + c) * cs + r0 + 1, f = MUL(tg, CONJ(vg[c]));
        for (int64_t r = 0; r < h; r++) x[r] -= MUL(u[r], f);
      }
    }
    T *x = b + c0 * cs + c0;
    th = F(reflector)(x, x + 1, len - 1, 1);
    vh[0] = 1;
    for (int64_t r = 1; r < len; r++) {
      vh[r] = x[r];
      x[r] = 0;
    }
    for (int64_t c = c0 + 1; th != 0 && c < c0 + len; c++) {
      T *y = b + c * cs + c0, f = MUL(CONJ(th), F(dot)(vh, y, len));
      for (int64_t r = 0; r < len; r++) y[r] -= MUL(vh[r], f);
    }
    if (q) {
      q[j * KD] = th;
      memcpy(q + j * KD + 1, vh + 1, sizeof(T) * (len - 1));
      p[j * KD] = tg;
      memcpy(p + j * KD + 1, vg + 1, sizeof(T) * (len - 1));
    }
    __atomic_store_n(&ch->done[s * LINE], j + 1, __ATOMIC_RELEASE);
  }
}

/* A unit claims sweeps until none is left. */
static void F(bchase_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)lo;
  (void)hi;
  (void)worker;
  F(bchase) *ch = ctx;
  for (int64_t s; (s = __atomic_fetch_add(&ch->next, 1, __ATOMIC_RELAXED)) <
                  ch->n - 1;)
    F(bsweep)(ch, s);
}

/* Bd's diagonal into [d] and superdiagonal into [e] from the band of kd
   superdiagonals of the [n] × [n] [a], rows [ld] apart, by bulge chasing
   on its copy in [b], which holds 3kd·n elements, sweep s's left
   and right reflectors into [q] and [p] unless NULL. Sweeps run on the
   call's threads as hb2st's do. [done] holds n·LINE indices. */
static void F(gb2bd)(const T *a, int64_t n, int64_t ld, T *b, T *q, T *p,
                     int64_t *done, R *d, R *e, int64_t kd, arena *s) {
  int64_t cs = 3 * kd - 1;
  T *b0 = b + 2 * kd - 1;
  memset(b, 0, sizeof(T) * 3 * kd * n);
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = i; j <= i + kd && j < n; j++)
      b0[j * cs + i] = a[i * ld + j];
  memset(done, 0, sizeof(int64_t) * n * LINE);
  F(bchase) ch = {.b = b, .q = q, .p = p, .n = n, .kd = kd, .next = 0,
                  .done = done};
  int64_t bytes = 3 * kd * n * (int64_t)sizeof(T);
  double cost = threaded(s, n - 1, bytes, 8.0 * kd * n * n, RDT, CPLX);
  if (cost == 0) F(bchase_unit)(0, 1, 0, &ch);
  else nx_cpu_job(n - 1, bytes, cost, F(bchase_unit), &ch);
  for (int64_t j = 0; j < n; j++) {
    d[j] = REAL(b0[j * cs + j]);
    if (j < n - 1) e[j] = REAL(b0[(j + 1) * cs + j]);
  }
}

/* SVD on matrix [b] of [c] in the scratch [x]: s, and u and vh where the
   call has vectors. A matrix that holds NaN or an infinity, or on which
   an iteration does not converge, is NaN in each. Answers 0 if an
   allocation failed. */
static int F(svd)(const call *c, int64_t b, T *x, arena *s) {
  int64_t m = c->m, n = c->n, wide = m < n, mm = wide ? n : m;
  int64_t k = wide ? m : n, uc = c->complete ? m : k, vr = c->complete ? n : k;
  R *so = (R *)matrix_at(c, c->nops, b);
  T *uo = c->vectors ? (T *)matrix_at(c, c->nops + 1, b) : NULL;
  T *vo = c->vectors ? (T *)matrix_at(c, c->nops + 2, b) : NULL;
  if (k == 0) {
    for (int64_t i = 0; uo && i < m; i++)
      for (int64_t j = 0; j < uc; j++) uo[i * uc + j] = i == j;
    for (int64_t i = 0; vo && i < vr; i++)
      for (int64_t j = 0; j < n; j++) vo[i * n + j] = i == j;
    return 1;
  }
  /* B is a, or aᴴ where a is wide, mm × k, column-major, or row-major
     where QR comes first, from 11/6 k rows on reals (DGESDD's MNTHR) and
     17/9 k on complex numbers (ZGESDD's MNTHR1), or the two stages take
     it. X = Q diag (U_B, I), mm × xc, and W = P V_B, k × k, are u and
     vhᴴ, or vhᴴ and u where a is wide; [qs] and [ps] hold the chase's
     left and right reflectors for Q2 and P2. */
  int qrf = mm >= (CPLX ? k * 17 / 9 : k * 11 / 6);
  int two = svd_two(k, uo != NULL, CPLX), by_rows = qrf || two;
  int64_t xc = wide ? vr : uc, ld = by_rows ? k : mm, kd = uo ? KD : KD_VALUES;
  T *a = x, *r = a + mm * k, *tau = r + k * k, *tauq = tau + k,
    *taup = tauq + k, *ws = taup + k, *tmp = ws + svd_scratch(mm, k, CPLX);
  int64_t slots = svd_slots(k, c->vectors, CPLX);
  T *qs = tmp + (wide ? mm * xc : k * k), *ps = qs + slots / 2;
  R *d = (R *)(qs + slots), *e = d + k, *ub = e + k,
    *vb = ub + k * k, *work = vb + k * k;
  R *iw = work + 3 * (k + 1) * (k + 1) + 4 * k;
  int64_t *iwork = (int64_t *)(((uintptr_t)iw + 7) & ~(uintptr_t)7);
  stage(c, 0, b, a, ld, by_rows == wide, s);
  R big = 0;
  int finite = F(scan)(a, mm * k, &big);
  for (int64_t i = 0; wide && CPLX && i < mm * k; i++) a[i] = CONJ(a[i]);
  /* Scaled into [small, 1 / small] where its largest magnitude lies
     outside. */
  const R small = SQRT(TINY) / (2 * EPS);
  R sigma = big > 0 && big < small ? small / big
            : big > 1 / small        ? 1 / small / big
                                     : 1;
  if (sigma != 1)
    for (int64_t i = 0; i < mm * k; i++) a[i] *= sigma;
  T *bd = a;
  int64_t lb = mm;
  if (finite && qrf) {
    if (!F(qr)(a, mm, k, k, tau, ws, ws + qr_ts(k), s)) return 0;
    for (int64_t i = 0; i < k; i++)
      for (int64_t j = 0; j < k; j++)
        r[two ? i * k + j : i + j * k] = i <= j ? a[i * k + j] : 0;
    bd = r;
    lb = k;
  }
  if (finite && two) {
    int64_t *done =
        (int64_t *)(((uintptr_t)(ws + 3 * kd * k) + 7) & ~(uintptr_t)7);
    if (!F(ge2gb)(bd, lb, k, k, tauq, taup, ws, kd, s)) return 0;
    F(gb2bd)(bd, k, k, ws, uo ? qs : NULL, uo ? ps : NULL, done, d, e, kd,
             s);
  } else if (finite && !F(gebrd)(bd, lb, k, lb, d, e, tauq, taup, ws, s))
    return 0;
  int ok = finite ? F(bdsdc)(k, d, e, uo ? ub : NULL, k, uo ? vb : NULL, k,
                             work, iwork, s)
                  : 0;
  if (ok < 0) return 0;
  if (!ok) {
    nans(so, k, RDT);
    if (uo) nans(uo, m * uc * (CPLX + 1), RDT);
    if (vo) nans(vo, vr * n * (CPLX + 1), RDT);
    return 1;
  }
  for (int64_t j = 0; j < k; j++) so[j] = d[j] * (1 / sigma) + 0;
  if (!uo) return 1;
  T *xo = wide ? tmp : uo, *wo = wide ? uo : tmp;
  for (int64_t i = 0; i < mm; i++)
    for (int64_t j = i < k ? k : 0; j < xc; j++) xo[i * xc + j] = i == j;
  F(from_columns)(ub, k, k, k, xo, xc);
  F(from_columns)(vb, k, k, k, wo, k);
  /* Q is the bidiagonalisation's, Q2 first in two stages, after QR's
     where it came first; P's reflectors are B's rows right of the
     superdiagonal, P1's right of the band, conjugated. */
  int64_t p0 = two ? kd : 1;
  if (two && (!F(q2)(qs, k, xo, xc, s) || !F(q2)(ps, k, wo, k, s)))
    return 0;
  if (qrf && !F(back)(r, two ? k : 1, two ? 1 : k, AS_STORED, tauq, k, k, k,
                      xo, xc, AS_COLUMNS, ws, s))
    return 0;
  if (!F(back)(a, by_rows ? k : 1, by_rows ? 1 : mm, AS_STORED,
               qrf ? tau : tauq, k, mm, xc, xo, xc, AS_COLUMNS, ws, s) ||
      !F(back)(two ? bd + kd : bd + lb, two ? 1 : lb, two ? k : 1, CONJUGATED,
               taup, k - p0, k - p0, k, wo + p0 * k, k, AS_COLUMNS, ws, s))
    return 0;
  const T *src = wide ? xo : wo;
  int64_t rows = wide ? vr : k;
  FROM_COLUMNS(src, rows, rows, n, vo, n, CONJ);
  return 1;
}

#undef FROM_COLUMNS

/* Eig

   a v = v diag w as LAPACK's xGEEV computes it: a balanced, D⁻¹ Pᵀ a P D
   (xGEBAL), reduced to the upper Hessenberg H = Qᴴ (·) Q (xGEHD2), then
   H = Z S Zᴴ in Schur form by the implicit QR iteration (xLAHQR): S upper
   triangular by single shifts on complex numbers, quasi-triangular by
   Francis double shifts on reals, each complex conjugate pair of
   eigenvalues a 2 × 2 block. The eigenvectors X of S come by back
   substitution (xTREVC3), then v = P D Q Z X (xGEBAK), each column of
   unit norm with its largest element real (xGEEV). The matrix is
   column-major, element (i, j) at a[i + j·n], so that the routines follow
   LAPACK's index for index; v is row-major from Z X on. */

/* The relative spacing of the dtype's numbers: LAPACK's precision. */
#define ULP (2 * EPS)
#define M(x, i, j) (x)[(i) + (j) * n]

/* |re| + |im| of the complex [x]. */
static R F(cabs1)(C x) {
  R re = (R)creal(x), im = (R)cimag(x);
  return (re < 0 ? -re : re) + (im < 0 ? -im : im);
}

/* The index of the first element of greatest |re| + |im| among the [n] at
   [x], [s] apart. */
static int64_t F(iamax)(const T *x, int64_t n, int64_t s) {
  int64_t k = 0;
  R best = -1;
  for (int64_t i = 0; i < n; i++)
    if (F(magnitude)(x[i * s]) > best) {
      best = F(magnitude)(x[i * s]);
      k = i;
    }
  return k;
}

/* Swaps the [n] elements at [x] and [y], [s] apart. */
static void F(exchange)(T *x, T *y, int64_t n, int64_t s) {
  for (int64_t i = 0; i < n; i++) {
    T t = x[i * s];
    x[i * s] = y[i * s];
    y[i * s] = t;
  }
}

/* Balances the [n] × [n] [a] in place (xGEBAL's 'B'): rows and columns
   that isolate an eigenvalue are swapped to the ends, then the rows and
   columns [lo, hi] between are scaled by powers of 2 until each row's and
   column's norms are within 5% of balanced. [scale] takes, outside
   [lo, hi], the index each was swapped with, inside, its scale. */
static void F(gebal)(T *a, int64_t n, int64_t *lo, int64_t *hi, R *scale) {
  int64_t k = 0, l = n - 1;
  *lo = 0;
  *hi = n - 1;
  if (n == 0) return;
  /* A row with no element off the diagonal in columns [0, l] goes to
     row l. */
  for (int noconv = 1; noconv;) {
    noconv = 0;
    for (int64_t i = l; i >= 0; i--) {
      int alone = 1;
      for (int64_t j = 0; j <= l && alone; j++)
        alone = i == j || M(a, i, j) == 0;
      if (!alone) continue;
      scale[l] = (R)i;
      if (i != l) {
        F(exchange)(&M(a, 0, i), &M(a, 0, l), l + 1, 1);
        F(exchange)(&M(a, i, k), &M(a, l, k), n - k, n);
      }
      noconv = 1;
      if (l == 0) {
        *hi = 0;
        return;
      }
      l--;
    }
  }
  /* A column with no element off the diagonal in rows [k, l] goes to
     column k. */
  for (int noconv = 1; noconv;) {
    noconv = 0;
    for (int64_t j = k; j <= l; j++) {
      int alone = 1;
      for (int64_t i = k; i <= l && alone; i++)
        alone = i == j || M(a, i, j) == 0;
      if (!alone) continue;
      scale[k] = (R)j;
      if (j != k) {
        F(exchange)(&M(a, 0, j), &M(a, 0, k), l + 1, 1);
        F(exchange)(&M(a, j, k), &M(a, k, k), n - k, n);
      }
      noconv = 1;
      k++;
    }
  }
  *lo = k;
  *hi = l;
  for (int64_t i = k; i <= l; i++) scale[i] = 1;
  const R sfmin1 = TINY / ULP, sfmax1 = 1 / sfmin1, sfmin2 = sfmin1 * 2,
          sfmax2 = 1 / sfmin2;
  for (int noconv = 1; noconv;) {
    noconv = 0;
    for (int64_t i = k; i <= l; i++) {
      R c = F(norm)(&M(a, k, i), l - k + 1, 1);
      R r = F(norm)(&M(a, i, k), l - k + 1, n);
      T x = M(a, F(iamax)(&M(a, 0, i), l + 1, 1), i);
      T y = M(a, i, k + F(iamax)(&M(a, i, k), n - k, n));
      R ca = F(lapy3)(REAL(x), IMAG(x), 0), ra = F(lapy3)(REAL(y), IMAG(y), 0);
      if (c == 0 || r == 0) continue;
      R g = r / 2, f = 1, s = c + r;
      while (c < g && (f > c ? f : c) < sfmax2 && (ca > c ? ca : c) < sfmax2 &&
             (r < g ? r : g) > sfmin2 && ra > sfmin2) {
        f *= 2, c *= 2, ca *= 2, r /= 2, g /= 2, ra /= 2;
      }
      g = c / 2;
      while (g >= r && (r > ra ? r : ra) < sfmax2 && (f < c ? f : c) > sfmin2 &&
             (g < ca ? g : ca) > sfmin2) {
        f /= 2, c /= 2, g /= 2, ca /= 2, r *= 2, ra *= 2;
      }
      if (c + r >= (R)0.95 * s) continue;
      if (f < 1 && scale[i] < 1 && f * scale[i] <= sfmin1) continue;
      if (f > 1 && scale[i] > 1 && scale[i] >= sfmax1 / f) continue;
      scale[i] *= f;
      noconv = 1;
      for (int64_t j = k; j < n; j++) M(a, i, j) *= 1 / f;
      for (int64_t j = 0; j <= l; j++) M(a, j, i) *= f;
    }
  }
}

/* c ← (I − τ v vᴴ) c for the [m] × [k] [c], columns [n] apart, v's [m]
   elements adjacent (xLARF from the left). [w] holds k elements. */
static void F(larf_left)(const T *v, T tau, int64_t m, int64_t k, T *c,
                         int64_t n, T *w) {
  if (tau == 0) return;
  for (int64_t j = 0; j < k; j++) {
    T s = 0;
    for (int64_t i = 0; i < m; i++) s += MUL(CONJ(v[i]), M(c, i, j));
    w[j] = MUL(tau, s);
  }
  for (int64_t j = 0; j < k; j++)
    for (int64_t i = 0; i < m; i++) M(c, i, j) -= MUL(v[i], w[j]);
}

/* c ← c (I − τ v vᴴ) for the [m] × [k] [c], columns [n] apart (xLARF from
   the right). [w] holds m elements. */
static void F(larf_right)(const T *v, T tau, int64_t m, int64_t k, T *c,
                          int64_t n, T *w) {
  if (tau == 0) return;
  for (int64_t i = 0; i < m; i++) w[i] = 0;
  for (int64_t j = 0; j < k; j++)
    for (int64_t i = 0; i < m; i++) w[i] += MUL(M(c, i, j), v[j]);
  for (int64_t j = 0; j < k; j++) {
    T t = MUL(tau, CONJ(v[j]));
    for (int64_t i = 0; i < m; i++) M(c, i, j) -= MUL(w[i], t);
  }
}

/* H = Qᴴ a Q upper Hessenberg in place, Q = H_lo ⋯ H_(hi−1), H_i =
   I − τ_i v vᴴ with v[i + 1] = 1 and v[i + 2 … hi] in a's column i below
   H (xGEHD2). [tau] takes n − 1 elements, 0 outside [lo, hi); [w] holds
   n. */
static void F(gehd2)(T *a, int64_t n, int64_t lo, int64_t hi, T *tau, T *w) {
  for (int64_t i = 0; i < n - 1; i++) tau[i] = 0;
  for (int64_t i = lo; i < hi; i++) {
    T *v = &M(a, i + 1, i), alpha = *v;
    tau[i] = F(reflector)(&alpha, v + 1, hi - i - 1, 1);
    *v = 1;
    F(larf_right)(v, tau[i], hi + 1, hi - i, &M(a, 0, i + 1), n, w);
    F(larf_left)(v, CONJ(tau[i]), hi - i, n - i - 1, &M(a, i + 1, i + 1), n,
                 w);
    *v = alpha;
  }
}

/* xLAHQR's exceptional shifts: every KEXSH iterations without a
   deflation, from the bottom or the top of the active block. */
#define KEXSH 10
#define DAT1 ((R)0.75)
#define DAT2 ((R)-0.4375)

/* Whether H's subdiagonal element (k, k − 1) is negligible in the active
   block [lo, hi] (Ahues and Tisseur's test, as xLAHQR). */
static int F(negligible)(const T *h, int64_t n, int64_t k, int64_t lo,
                         int64_t hi, R smlnum) {
  if (F(magnitude)(M(h, k, k - 1)) <= smlnum) return 1;
  R tst = F(magnitude)(M(h, k - 1, k - 1)) + F(magnitude)(M(h, k, k));
  if (tst == 0) {
    if (k - 2 >= lo) tst += F(abs)(REAL(M(h, k - 1, k - 2)));
    if (k + 1 <= hi) tst += F(abs)(REAL(M(h, k + 1, k)));
  }
  if (!(F(abs)(REAL(M(h, k, k - 1))) <= ULP * tst)) return 0;
  R x = F(magnitude)(M(h, k, k - 1)), y = F(magnitude)(M(h, k - 1, k));
  R ab = x > y ? x : y, ba = x < y ? x : y;
  x = F(magnitude)(M(h, k, k));
  y = F(magnitude)(M(h, k - 1, k - 1) - M(h, k, k));
  R aa = x > y ? x : y, bb = x < y ? x : y, s = aa + ab;
  R t = ULP * (bb * (aa / s));
  return ba * (ab / s) <= (smlnum > t ? smlnum : t);
}

#if CPLX
/* H's eigenvalues [lo, hi] into wr and wi by single-shift QR sweeps
   (xLAHQR), H, columns [n] apart, made upper triangular under [wantt],
   and Z's rows [lo, hi] multiplied by the sweeps' rotations where [z] is
   not NULL. Answers 0 if an eigenvalue takes more than 30 max(10, hi −
   lo + 1) iterations. */
static int F(lahqr)(int wantt, int64_t n, int64_t lo, int64_t hi, T *h, R *wr,
                    R *wi, T *z) {
  if (n == 0) return 1;
  if (lo == hi) {
    wr[lo] = REAL(M(h, lo, lo));
    wi[lo] = IMAG(M(h, lo, lo));
    return 1;
  }
  for (int64_t j = lo; j <= hi - 3; j++) M(h, j + 2, j) = M(h, j + 3, j) = 0;
  if (lo <= hi - 2) M(h, hi, hi - 2) = 0;
  int64_t jlo = wantt ? 0 : lo, jhi = wantt ? n - 1 : hi;
  /* The subdiagonal made real by a diagonal similarity. */
  for (int64_t i = lo + 1; i <= hi; i++) {
    T x = M(h, i, i - 1);
    if (IMAG(x) == 0) continue;
    T sc = x / F(magnitude)(x);
    sc = CONJ(sc) / F(lapy3)(REAL(sc), IMAG(sc), 0);
    M(h, i, i - 1) = F(lapy3)(REAL(x), IMAG(x), 0);
    for (int64_t j = i; j <= jhi; j++) M(h, i, j) = MUL(M(h, i, j), sc);
    for (int64_t j = jlo; j <= (jhi < i + 1 ? jhi : i + 1); j++)
      M(h, j, i) = MUL(M(h, j, i), CONJ(sc));
    for (int64_t j = lo; z && j <= hi; j++) M(z, j, i) = MUL(M(z, j, i), CONJ(sc));
  }
  const R smlnum = TINY * ((R)(hi - lo + 1) / ULP);
  const int64_t itmax = 30 * (hi - lo + 1 > 10 ? hi - lo + 1 : 10);
  int64_t i1 = 0, i2 = n - 1, kdefl = 0;
  for (int64_t i = hi; i >= lo;) {
    int64_t l = lo, its = 0;
    for (; its <= itmax; its++) {
      int64_t k = i;
      while (k > l && !F(negligible)(h, n, k, lo, hi, smlnum)) k--;
      l = k;
      if (l > lo) M(h, l, l - 1) = 0;
      if (l >= i) break;
      kdefl++;
      if (!wantt) i1 = l, i2 = i;
      /* The shift: Wilkinson's, or an exceptional one. */
      T t;
      if (kdefl % (2 * KEXSH) == 0)
        t = DAT1 * F(abs)(REAL(M(h, i, i - 1))) + M(h, i, i);
      else if (kdefl % KEXSH == 0)
        t = DAT1 * F(abs)(REAL(M(h, l + 1, l))) + M(h, l, l);
      else {
        t = M(h, i, i);
        T u = MUL(CSQRT(M(h, i - 1, i)), CSQRT(M(h, i, i - 1)));
        R s = F(magnitude)(u);
        if (s != 0) {
          T x = (M(h, i - 1, i - 1) - t) / 2;
          R sx = F(magnitude)(x);
          s = s > sx ? s : sx;
          T y = s * CSQRT(MUL(x / s, x / s) + MUL(u / s, u / s));
          if (sx > 0 && REAL(x / sx) * REAL(y) + IMAG(x / sx) * IMAG(y) < 0)
            y = -y;
          t -= MUL(u, DIV(u, x + y));
        }
      }
      /* The sweep starts at the lowest m whose first column's bulge is
         negligible against H(m, m − 1). */
      int64_t m = i - 1;
      T v[2];
      for (;; m--) {
        T h11 = M(h, m, m), h22 = M(h, m + 1, m + 1), h11s = h11 - t;
        R h21 = REAL(M(h, m + 1, m)), s = F(magnitude)(h11s) + F(abs)(h21);
        h11s /= s;
        h21 /= s;
        v[0] = h11s;
        v[1] = h21;
        if (m == l) break;
        if (F(abs)(REAL(M(h, m, m - 1))) * F(abs)(h21) <=
            ULP * (F(magnitude)(h11s) *
                   (F(magnitude)(h11) + F(magnitude)(h22))))
          break;
      }
      for (int64_t k2 = m; k2 < i; k2++) {
        if (k2 > m) v[0] = M(h, k2, k2 - 1), v[1] = M(h, k2 + 1, k2 - 1);
        T t1 = F(reflector)(&v[0], &v[1], 1, 1);
        if (k2 > m) M(h, k2, k2 - 1) = v[0], M(h, k2 + 1, k2 - 1) = 0;
        T v2 = v[1];
        R t2 = REAL(MUL(t1, v2));
        for (int64_t j = k2; j <= i2; j++) {
          T sum = MUL(CONJ(t1), M(h, k2, j)) + t2 * M(h, k2 + 1, j);
          M(h, k2, j) -= sum;
          M(h, k2 + 1, j) -= MUL(sum, v2);
        }
        for (int64_t j = i1; j <= (k2 + 2 < i ? k2 + 2 : i); j++) {
          T sum = MUL(t1, M(h, j, k2)) + t2 * M(h, j, k2 + 1);
          M(h, j, k2) -= sum;
          M(h, j, k2 + 1) -= MUL(sum, CONJ(v2));
        }
        for (int64_t j = lo; z && j <= hi; j++) {
          T sum = MUL(t1, M(z, j, k2)) + t2 * M(z, j, k2 + 1);
          M(z, j, k2) -= sum;
          M(z, j, k2 + 1) -= MUL(sum, CONJ(v2));
        }
        if (k2 == m && m > l) {
          /* H(m + 1, m) made real again, as the sweep started below l. */
          T x = 1 - t1;
          x /= F(lapy3)(REAL(x), IMAG(x), 0);
          M(h, m + 1, m) = MUL(M(h, m + 1, m), CONJ(x));
          if (m + 2 <= i) M(h, m + 2, m + 1) = MUL(M(h, m + 2, m + 1), x);
          for (int64_t j = m; j <= i; j++) {
            if (j == m + 1) continue;
            for (int64_t p = j + 1; p <= i2; p++) M(h, j, p) = MUL(M(h, j, p), x);
            for (int64_t p = i1; p < j; p++) M(h, p, j) = MUL(M(h, p, j), CONJ(x));
            for (int64_t p = lo; z && p <= hi; p++) M(z, p, j) = MUL(M(z, p, j), CONJ(x));
          }
        }
      }
      /* H(i, i − 1) made real. */
      T x = M(h, i, i - 1);
      if (IMAG(x) != 0) {
        R r = F(lapy3)(REAL(x), IMAG(x), 0);
        M(h, i, i - 1) = r;
        x /= r;
        for (int64_t p = i + 1; p <= i2; p++) M(h, i, p) = MUL(M(h, i, p), CONJ(x));
        for (int64_t p = i1; p < i; p++) M(h, p, i) = MUL(M(h, p, i), x);
        for (int64_t p = lo; z && p <= hi; p++) M(z, p, i) = MUL(M(z, p, i), x);
      }
    }
    if (its > itmax) return 0;
    wr[i] = REAL(M(h, i, i));
    wi[i] = IMAG(M(h, i, i));
    kdefl = 0;
    i = l - 1;
  }
  return 1;
}
#else
/* The Schur factorisation of the 2 × 2 [a b; c d] in place: [a b; 0 d]
   for real eigenvalues, [a b; c a] with b c < 0 for a complex pair; the
   rotation [cs sn; −sn cs] that makes it, and the eigenvalues, the pair's
   one of positive imaginary part first (xLANV2). */
static void F(lanv2)(R *a, R *b, R *c, R *d, R *rt1r, R *rt1i, R *rt2r,
                     R *rt2i, R *cs, R *sn) {
  const R safmn2 = sizeof(R) == 4 ? (R)0x1p-51 : (R)0x1p-485,
          safmx2 = 1 / safmn2;
  if (*c == 0) {
    *cs = 1;
    *sn = 0;
  } else if (*b == 0) {
    *cs = 0;
    *sn = 1;
    R t = *d;
    *d = *a;
    *a = t;
    *b = -*c;
    *c = 0;
  } else if (*a - *d == 0 && F(sign)(1, *b) != F(sign)(1, *c)) {
    *cs = 1;
    *sn = 0;
  } else {
    R temp = *a - *d, p = temp / 2;
    R bcmax = F(abs)(*b) > F(abs)(*c) ? F(abs)(*b) : F(abs)(*c);
    R bcmis = (F(abs)(*b) < F(abs)(*c) ? F(abs)(*b) : F(abs)(*c)) *
              F(sign)(1, *b) * F(sign)(1, *c);
    R scale = F(abs)(p) > bcmax ? F(abs)(p) : bcmax;
    R zz = (p / scale) * p + (bcmax / scale) * bcmis;
    if (zz >= 4 * ULP) {
      /* Real eigenvalues. */
      zz = p + F(sign)(SQRT(scale) * SQRT(zz), p);
      *a = *d + zz;
      *d = *d - (bcmax / zz) * bcmis;
      R tau = F(lapy3)(*c, zz, 0);
      *cs = zz / tau;
      *sn = *c / tau;
      *b = *b - *c;
      *c = 0;
    } else {
      /* Complex or nearly equal eigenvalues: the diagonal made equal. */
      R sigma = *b + *c;
      for (int count = 1;; count++) {
        R sc = F(abs)(temp) > F(abs)(sigma) ? F(abs)(temp) : F(abs)(sigma);
        if (sc >= safmx2) {
          sigma *= safmn2;
          temp *= safmn2;
          if (count <= 20) continue;
        }
        if (sc <= safmn2) {
          sigma *= safmx2;
          temp *= safmx2;
          if (count <= 20) continue;
        }
        break;
      }
      p = temp / 2;
      R tau = F(lapy3)(sigma, temp, 0);
      *cs = SQRT((1 + F(abs)(sigma) / tau) / 2);
      *sn = -(p / (tau * *cs)) * F(sign)(1, sigma);
      R aa = *a * *cs + *b * *sn, bb = -*a * *sn + *b * *cs;
      R cc = *c * *cs + *d * *sn, dd = -*c * *sn + *d * *cs;
      *a = aa * *cs + cc * *sn;
      *b = bb * *cs + dd * *sn;
      *c = -aa * *sn + cc * *cs;
      *d = -bb * *sn + dd * *cs;
      temp = (*a + *d) / 2;
      *a = *d = temp;
      if (*c != 0 && *b == 0) {
        *b = -*c;
        *c = 0;
        temp = *cs;
        *cs = -*sn;
        *sn = temp;
      } else if (*c != 0 && F(sign)(1, *b) == F(sign)(1, *c)) {
        /* Real eigenvalues after all. */
        R sab = SQRT(F(abs)(*b)), sac = SQRT(F(abs)(*c));
        p = F(sign)(sab * sac, *c);
        tau = 1 / SQRT(F(abs)(*b + *c));
        *a = temp + p;
        *d = temp - p;
        *b = *b - *c;
        *c = 0;
        R cs1 = sab * tau, sn1 = sac * tau;
        temp = *cs * cs1 - *sn * sn1;
        *sn = *cs * sn1 + *sn * cs1;
        *cs = temp;
      }
    }
  }
  *rt1r = *a;
  *rt2r = *d;
  *rt1i = *c == 0 ? 0 : SQRT(F(abs)(*b)) * SQRT(F(abs)(*c));
  *rt2i = -*rt1i;
}

/* [x] and [y], [n] elements [s] apart, rotated: x ← c x + s y,
   y ← c y − s x. */
static void F(rots)(R *x, R *y, int64_t n, int64_t st, R c, R s) {
  for (int64_t i = 0; i < n; i++) {
    R t = c * x[i * st] + s * y[i * st];
    y[i * st] = c * y[i * st] - s * x[i * st];
    x[i * st] = t;
  }
}

/* H's eigenvalues [lo, hi] into wr and wi by Francis double-shift QR
   sweeps (xLAHQR), H, columns [n] apart, made quasi-triangular under
   [wantt] with each complex pair a 2 × 2 block in xLANV2's form, and Z's
   rows [lo, hi] multiplied by the sweeps' reflectors where [z] is not
   NULL. Answers 0 if an eigenvalue takes more than 30 max(10, hi − lo +
   1) iterations. */
static int F(lahqr)(int wantt, int64_t n, int64_t lo, int64_t hi, R *h, R *wr,
                    R *wi, R *z) {
  if (n == 0) return 1;
  if (lo == hi) {
    wr[lo] = M(h, lo, lo);
    wi[lo] = 0;
    return 1;
  }
  for (int64_t j = lo; j <= hi - 3; j++) M(h, j + 2, j) = M(h, j + 3, j) = 0;
  if (lo <= hi - 2) M(h, hi, hi - 2) = 0;
  const R smlnum = TINY * ((R)(hi - lo + 1) / ULP);
  const int64_t itmax = 30 * (hi - lo + 1 > 10 ? hi - lo + 1 : 10);
  int64_t i1 = 0, i2 = n - 1, kdefl = 0;
  for (int64_t i = hi; i >= lo;) {
    int64_t l = lo, its = 0;
    for (; its <= itmax; its++) {
      int64_t k = i;
      while (k > l && !F(negligible)(h, n, k, lo, hi, smlnum)) k--;
      l = k;
      if (l > lo) M(h, l, l - 1) = 0;
      if (l >= i - 1) break;
      kdefl++;
      if (!wantt) i1 = l, i2 = i;
      /* The shifts: the trailing 2 × 2 block's eigenvalues, or
         exceptional ones. */
      R h11, h12, h21, h22, s;
      if (kdefl % (2 * KEXSH) == 0 || kdefl % KEXSH == 0) {
        int64_t p = kdefl % (2 * KEXSH) == 0 ? i : l;
        s = kdefl % (2 * KEXSH) == 0
                ? F(abs)(M(h, i, i - 1)) + F(abs)(M(h, i - 1, i - 2))
                : F(abs)(M(h, l + 1, l)) + F(abs)(M(h, l + 2, l + 1));
        h11 = h22 = DAT1 * s + M(h, p, p);
        h12 = DAT2 * s;
        h21 = s;
      } else {
        h11 = M(h, i - 1, i - 1);
        h21 = M(h, i, i - 1);
        h12 = M(h, i - 1, i);
        h22 = M(h, i, i);
      }
      s = F(abs)(h11) + F(abs)(h12) + F(abs)(h21) + F(abs)(h22);
      R rt1r = 0, rt1i = 0, rt2r = 0, rt2i = 0;
      if (s != 0) {
        h11 /= s, h21 /= s, h12 /= s, h22 /= s;
        R tr = (h11 + h22) / 2, det = (h11 - tr) * (h22 - tr) - h12 * h21;
        R rtdisc = SQRT(F(abs)(det));
        if (det >= 0) {
          rt1r = rt2r = tr * s;
          rt1i = rtdisc * s;
          rt2i = -rt1i;
        } else {
          /* Real shifts: both the one nearer h22. */
          rt1r = tr + rtdisc;
          rt2r = tr - rtdisc;
          rt1r = rt2r = (F(abs)(rt1r - h22) <= F(abs)(rt2r - h22) ? rt1r
                                                                   : rt2r) *
                        s;
        }
      }
      /* The sweep starts at the lowest m whose first column's bulge is
         negligible against H(m, m − 1). */
      int64_t m = i - 2;
      R v[3];
      for (;; m--) {
        R h21s = M(h, m + 1, m), hmm = M(h, m, m);
        s = F(abs)(hmm - rt2r) + F(abs)(rt2i) + F(abs)(h21s);
        h21s = M(h, m + 1, m) / s;
        v[0] = h21s * M(h, m, m + 1) + (hmm - rt1r) * ((hmm - rt2r) / s) -
               rt1i * (rt2i / s);
        v[1] = h21s * (hmm + M(h, m + 1, m + 1) - rt1r - rt2r);
        v[2] = h21s * M(h, m + 2, m + 1);
        s = F(abs)(v[0]) + F(abs)(v[1]) + F(abs)(v[2]);
        v[0] /= s, v[1] /= s, v[2] /= s;
        if (m == l) break;
        if (F(abs)(M(h, m, m - 1)) * (F(abs)(v[1]) + F(abs)(v[2])) <=
            ULP * F(abs)(v[0]) *
                (F(abs)(M(h, m - 1, m - 1)) + F(abs)(hmm) +
                 F(abs)(M(h, m + 1, m + 1))))
          break;
      }
      for (int64_t k2 = m; k2 < i; k2++) {
        int nr = i - k2 + 1 < 3 ? (int)(i - k2 + 1) : 3;
        if (k2 > m)
          for (int p = 0; p < nr; p++) v[p] = M(h, k2 + p, k2 - 1);
        R t1 = F(reflector)(&v[0], &v[1], nr - 1, 1);
        if (k2 > m) {
          M(h, k2, k2 - 1) = v[0];
          M(h, k2 + 1, k2 - 1) = 0;
          if (k2 < i - 1) M(h, k2 + 2, k2 - 1) = 0;
        } else if (m > l)
          M(h, k2, k2 - 1) *= 1 - t1;
        R v2 = v[1], t2 = t1 * v2, v3 = nr == 3 ? v[2] : 0, t3 = t1 * v3;
        for (int64_t j = k2; j <= i2; j++) {
          R sum = M(h, k2, j) + v2 * M(h, k2 + 1, j) +
                  (nr == 3 ? v3 * M(h, k2 + 2, j) : 0);
          M(h, k2, j) -= sum * t1;
          M(h, k2 + 1, j) -= sum * t2;
          if (nr == 3) M(h, k2 + 2, j) -= sum * t3;
        }
        int64_t e = nr == 3 ? (k2 + 3 < i ? k2 + 3 : i) : i;
        for (int64_t j = i1; j <= e; j++) {
          R sum = M(h, j, k2) + v2 * M(h, j, k2 + 1) +
                  (nr == 3 ? v3 * M(h, j, k2 + 2) : 0);
          M(h, j, k2) -= sum * t1;
          M(h, j, k2 + 1) -= sum * t2;
          if (nr == 3) M(h, j, k2 + 2) -= sum * t3;
        }
        for (int64_t j = lo; z && j <= hi; j++) {
          R sum = M(z, j, k2) + v2 * M(z, j, k2 + 1) +
                  (nr == 3 ? v3 * M(z, j, k2 + 2) : 0);
          M(z, j, k2) -= sum * t1;
          M(z, j, k2 + 1) -= sum * t2;
          if (nr == 3) M(z, j, k2 + 2) -= sum * t3;
        }
      }
    }
    if (its > itmax) return 0;
    if (l == i) {
      wr[i] = M(h, i, i);
      wi[i] = 0;
    } else {
      /* A 2 × 2 block, put in standard form. */
      R cs, sn;
      F(lanv2)(&M(h, i - 1, i - 1), &M(h, i - 1, i), &M(h, i, i - 1),
               &M(h, i, i), &wr[i - 1], &wi[i - 1], &wr[i], &wi[i], &cs, &sn);
      if (wantt) {
        if (i2 > i)
          F(rots)(&M(h, i - 1, i + 1), &M(h, i, i + 1), i2 - i, n, cs, sn);
        F(rots)(&M(h, i1, i - 1), &M(h, i1, i), i - i1 - 1, 1, cs, sn);
      }
      if (z) F(rots)(&M(z, lo, i - 1), &M(z, lo, i), hi - lo + 1, 1, cs, sn);
    }
    kdefl = 0;
    i = l - 1;
  }
  return 1;
}
#endif

/* x[0 … k] times [f]. */
static void F(cscale)(C *x, int64_t k, R f) {
  for (int64_t i = 0; i <= k; i++) x[i] *= f;
}

/* The eigenvectors of the upper quasi-triangular S, of the eigenvalues wr
   + i wi on its diagonal, into X's columns, by back substitution as
   xTREVC3: for λ from the last, x with x = 1 at λ's place (or a 2 × 2
   block's eigenvector) and (S − λ I) x = 0 above it, a diagonal element
   or block pivot within smin of singular moved to smin, and x scaled down
   where an element would pass bignum. On reals, a complex pair's x, for
   its λ of positive imaginary part, fills two columns, its real and its
   imaginary parts. [cn] holds n reals and [x] n complex numbers. */
static void F(trevc)(int64_t n, const T *s, const R *wr, const R *wi, T *xm,
                     R *cn, C *x) {
  const R smlnum = TINY * ((R)n / ULP), bignum = (1 - ULP) / smlnum;
  for (int64_t j = 0; j < n; j++) {
    cn[j] = 0;
    for (int64_t i = 0; i < j; i++) cn[j] += F(magnitude)(M(s, i, j));
  }
  for (int64_t ki = n - 1; ki >= 0; ki--) {
    int pair = !CPLX && wi[ki] != 0;
    int64_t top = pair ? ki - 1 : ki;
    C lambda = wr[top] + wi[top] * I;
    R smin = ULP * F(cabs1)(lambda);
    smin = smin > smlnum ? smin : smlnum;
    if (!pair) {
      x[ki] = 1;
      for (int64_t k = 0; k < ki; k++) x[k] = -M(s, k, ki);
    } else {
      R b = REAL(M(s, ki - 1, ki)), c = REAL(M(s, ki, ki - 1));
      if (F(abs)(b) >= F(abs)(c)) {
        x[ki - 1] = 1;
        x[ki] = wi[top] / b * I;
      } else {
        x[ki - 1] = -wi[top] / c;
        x[ki] = I;
      }
      for (int64_t k = 0; k < top; k++)
        x[k] = -x[ki - 1] * M(s, k, ki - 1) - x[ki] * M(s, k, ki);
    }
    for (int64_t j = top - 1; j >= 0;) {
      int block = !CPLX && j > 0 && M(s, j, j - 1) != 0;
      int64_t j0 = block ? j - 1 : j;
      R xmax = 0;
      if (!block) {
        C d = M(s, j, j) - lambda;
        if (F(cabs1)(d) < smin) d = smin;
        R dm = F(cabs1)(d), xj = F(cabs1)(x[j]);
        if (dm < 1 && xj > 1 && xj > bignum * dm) F(cscale)(x, ki, 1 / xj);
        x[j] = F(cdiv)(x[j], d);
        xmax = F(cabs1)(x[j]);
      } else {
        /* The 2 × 2 block by elimination with complete pivoting
           (xLALN2). */
        C e[4] = {M(s, j0, j0) - lambda, M(s, j0, j), M(s, j, j0),
                  M(s, j, j) - lambda};
        int p = 0;
        for (int q = 1; q < 4; q++)
          if (F(cabs1)(e[q]) > F(cabs1)(e[p])) p = q;
        if (F(cabs1)(e[p]) < smin) p = 0;
        int pr = p / 2, pc = p % 2;
        C u11 = e[p], l21, u12, u22;
        if (F(cabs1)(u11) < smin) {
          /* The whole block within smin of zero: smin I. */
          u11 = smin;
          l21 = 0, u12 = 0, u22 = smin;
        } else {
          l21 = F(cdiv)(e[(1 - pr) * 2 + pc], u11);
          u12 = e[pr * 2 + 1 - pc];
          u22 = e[(1 - pr) * 2 + 1 - pc] - F(cmul)(l21, u12);
          if (F(cabs1)(u22) < smin) u22 = smin;
        }
        C b1 = x[j0 + pr], b2 = x[j0 + 1 - pr] - F(cmul)(l21, b1);
        R bm = F(cabs1)(b1) > F(cabs1)(b2) ? F(cabs1)(b1) : F(cabs1)(b2);
        R um = F(cabs1)(u22) < F(cabs1)(u11) ? F(cabs1)(u22) : F(cabs1)(u11);
        if (um < 1 && bm > 1 && bm > bignum * um) {
          F(cscale)(x, ki, 1 / bm);
          b1 /= bm;
          b2 /= bm;
        }
        C y2 = F(cdiv)(b2, u22),
          y1 = F(cdiv)(b1 - F(cmul)(u12, y2), u11);
        x[j0 + 1 - pc] = y2;
        x[j0 + pc] = y1;
        xmax = F(cabs1)(y1) > F(cabs1)(y2) ? F(cabs1)(y1) : F(cabs1)(y2);
        /* x scaled down where the block times it could overflow. */
        R emax = 0;
        for (int q = 0; q < 4; q++)
          emax = F(cabs1)(e[q]) > emax ? F(cabs1)(e[q]) : emax;
        if (xmax > 1 && emax > 1 && xmax > bignum / emax) {
          F(cscale)(x, ki, emax / bignum);
          xmax *= emax / bignum;
        }
      }
      /* The update of the elements above, x scaled down first where it
         could overflow. */
      R cmax = cn[j] > cn[j0] ? cn[j] : cn[j0];
      if (xmax > 1 && cmax > bignum / xmax) {
        F(cscale)(x, ki, 1 / xmax);
        xmax = 1;
      }
      for (int64_t k = 0; k < j0; k++) {
        x[k] -= MUL(x[j], M(s, k, j));
        if (block) x[k] -= MUL(x[j0], M(s, k, j0));
      }
      j = j0 - 1;
    }
    for (int64_t k = 0; k < n; k++) {
      C xk = k <= ki ? x[k] : 0;
#if CPLX
      M(xm, k, ki) = xk;
#else
      M(xm, k, top) = (R)creal(xk);
      if (pair) M(xm, k, ki) = (R)cimag(xk);
#endif
    }
    ki = top;
  }
}

/* Eig on matrix [b] of [c] in the scratch [x]: w, and v where the call
   has vectors. A matrix that holds NaN or an infinity, or on which the
   iteration does not converge, is NaN in both. Answers 0 if an
   allocation failed. */
static int F(eig)(const call *c, int64_t b, T *x, arena *s) {
  int64_t n = c->n, nn = n * n;
  if (n == 0) return 1;
  T *a = x, *h = a + nn, *z = h + nn, *xm = z + nn, *vr = xm + nn,
    *tau = vr + nn, *ws = tau + n;
  R *wr = (R *)(ws + (3 * n + NB) * NB), *wi = wr + n, *scale = wi + n,
    *cn = scale + n;
  C *xv = (C *)(cn + n);
  R *wo = (R *)matrix_at(c, c->nops, b);
  R *vo = c->vectors ? (R *)matrix_at(c, c->nops + 1, b) : NULL;
  stage(c, 0, b, a, n, 1, s);
  R big = 0;
  int finite = F(scan)(a, nn, &big);
  int64_t lo = 0, hi = n - 1;
  R sigma = 1;
  int ok = finite;
  if (ok) {
    /* Balanced, then scaled into [small, 1 / small] where its largest
       magnitude lies outside. xGEEV scales first, which flushes entries
       that balancing would have brought near the others: float32's
       [0 2^100; 2^-100 0] loses 2^-100, and its eigenvalues ±1 become 0,
       as float64's [1 1e300; 1e-300 1] loses 1e-300 and its 0 and 2
       become 1. Balancing's own guards keep it safe unscaled. */
    F(gebal)(a, n, &lo, &hi, scale);
    /* The largest after balancing; the matrix is finite already. */
    big = 0;
    F(scan)(a, nn, &big);
    const R small = SQRT(TINY) / ULP;
    sigma = big > 0 && big < small ? small / big
            : big > 1 / small      ? 1 / small / big
                                   : 1;
    for (int64_t i = 0; sigma != 1 && i < nn; i++) a[i] *= sigma;
    F(gehd2)(a, n, lo, hi, tau, ws);
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = 0; i < n; i++)
        M(h, i, j) = i <= j + 1 ? M(a, i, j) : 0;
    for (int64_t i = 0; i < n; i++) {
      wr[i] = REAL(M(h, i, i));
      wi[i] = IMAG(M(h, i, i));
    }
    for (int64_t i = 0; vo && i < nn; i++) z[i] = i % (n + 1) == 0;
    ok = F(lahqr)(vo != NULL, n, lo, hi, h, wr, wi, vo ? z : NULL);
  }
  if (!ok) {
    nans(wo, 2 * n, RDT);
    if (vo) nans(vo, 2 * nn, RDT);
    return 1;
  }
  for (int64_t j = 0; j < n; j++) {
    wo[2 * j] = wr[j] * (1 / sigma) + 0;
    wo[2 * j + 1] = wi[j] * (1 / sigma) + 0;
#if !CPLX
    /* A pair's imaginary part that rounds to 0 on a scaled up matrix is
       η, its sign kept, within η of the computed one: the values stay a
       pair, as their vectors are. */
    if (wi[j] != 0 && wo[2 * j + 1] == 0) wo[2 * j + 1] = F(sign)(ETA, wi[j]);
#endif
  }
  if (!vo) return 1;
  /* v = Z X, row-major, then Q from the left: reflector i is a's column i
     from row i + 1. */
  F(trevc)(n, h, wr, wi, xm, cn, xv);
  T *v = CPLX ? (T *)vo : vr;
  for (int64_t i = 0; i < nn; i++) v[i] = 0;
  if (!F(gemm)(n, n, n, 1, F(op)(z, 1, n), F(op)(xm, 1, n), v, n, s) ||
      (hi > lo &&
       !F(back)(&M(a, lo + 1, lo), 1, n, AS_STORED, tau + lo, hi - lo,
                hi - lo, n, v + (lo + 1) * n, n, AS_COLUMNS, ws, s)))
    return 0;
  /* Each column scaled to a largest |re| + |im| of 1, a pair's two
     together, as xTREVC3 leaves the back-transformed vectors, so that
     the balancing's scales cannot overflow them. */
  for (int64_t j = 0; j < n; j++) {
    int pair = !CPLX && wi[j] > 0;
    R emax = 0;
    for (int64_t i = 0; i < n; i++) {
      R e = F(magnitude)(v[i * n + j]);
      if (pair) e += F(magnitude)(v[i * n + j + 1]);
      emax = e > emax ? e : emax;
    }
    for (int64_t i = 0; emax > 0 && i < n; i++) {
      v[i * n + j] *= 1 / emax;
      if (pair) v[i * n + j + 1] *= 1 / emax;
    }
    j += pair;
  }
  /* The balancing undone (xGEBAK): rows scaled, then swapped back. */
  for (int64_t i = lo; i <= hi && lo < hi; i++)
    for (int64_t j = 0; j < n; j++) v[i * n + j] *= scale[i];
  for (int64_t p = 0; p < n; p++) {
    int64_t i = p < lo ? lo - 1 - p : p, k = (int64_t)scale[i];
    if ((i < lo || i > hi) && k != i) F(swap)(v + i * n, v + k * n, n);
  }
  /* Each column of unit norm, its element of largest magnitude real. */
  for (int64_t j = 0; j < n; j++) {
    int pair = !CPLX && wi[j] > 0;
    R nrm = F(norm)(v + j, n, n);
    if (pair) nrm = F(lapy3)(nrm, F(norm)(v + j + 1, n, n), 0);
    int64_t k = 0;
    R best = -1;
    for (int64_t i = 0; i < n; i++) {
      v[i * n + j] *= 1 / nrm;
      if (pair) v[i * n + j + 1] *= 1 / nrm;
      R re = REAL(v[i * n + j]), im = pair ? v[i * n + j + 1]
                                            : IMAG(v[i * n + j]);
      if (re * re + im * im > best) best = re * re + im * im, k = i;
    }
#if CPLX
    T t = CONJ(v[k * n + j]) / SQRT(best);
    for (int64_t i = 0; i < n; i++) v[i * n + j] = MUL(v[i * n + j], t);
    v[k * n + j] = REAL(v[k * n + j]);
#else
    if (!pair) continue;
    R cs, sn, r;
    F(rotation)(v[k * n + j], v[k * n + j + 1], &cs, &sn, &r);
    F(rots)(v + j, v + j + 1, n, n, cs, sn);
    v[k * n + j + 1] = 0;
    j++;
#endif
  }
#if !CPLX
  /* A pair's columns, its real and imaginary parts, are its two vectors,
     conjugate. */
  for (int64_t j = 0; j < n; j++)
    for (int64_t i = 0; i < n; i++) {
      R re = v[i * n + j], im = 0;
      if (wi[j] > 0) im = v[i * n + j + 1];
      if (wi[j] < 0) re = v[i * n + j - 1], im = -v[i * n + j];
      vo[2 * (i * n + j)] = re;
      vo[2 * (i * n + j) + 1] = im;
    }
#endif
  return 1;
}

#undef ULP
#undef M
#undef KEXSH
#undef DAT1
#undef DAT2

/* Matrices of a call */

/* Cholesky's factor [d] of order [n], its other triangle to be +0, or
   under [upper] U = Lᴴ in place. */
typedef struct {
  T *d;
  int64_t n;
  int upper;
} F(triangle);

/* Rows [lo, hi) of the triangle: each row's part right of the diagonal,
   and under upper the column below it, which no other row touches. */
static void F(other_triangle)(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const F(triangle) *t = ctx;
  T *d = t->d;
  int64_t n = t->n;
  for (int64_t i = lo; i < hi; i++)
    for (int64_t j = i + 1; j < n; j++) {
      if (!t->upper) {
        d[i * n + j] = 0;
        continue;
      }
      d[i * n + j] = CONJ(d[j * n + i]);
      d[j * n + i] = 0;
    }
}

/* The routine on matrix [b] of the call [c], with the scratch [w] of
   linalg.c's own elements and the arena [s]. Answers 0 if an allocation
   failed. */
static int F(matrix)(const call *c, int64_t b, T *w, arena *s) {
  int64_t m = c->m, n = c->n;
  T *d = (T *)matrix_at(c, c->nops, b);
  if (c->routine == EIGH) return F(eigh)(c, b, w, s);
  if (c->routine == SVD) return F(svd)(c, b, w, s);
  if (c->routine == EIG) return F(eig)(c, b, w, s);
  if (c->routine == QR) {
    /* a factored in w, then R and Q out. */
    int64_t k = m < n ? m : n, qc = c->complete ? m : k;
    int64_t lf = n + PAD;
    T *f = w, *tau = f + m * lf, *ts = tau + k, *rest = ts + qr_ts(k);
    T *r = (T *)matrix_at(c, c->nops + 1, b);
    stage(c, 0, b, f, lf, 0, s);
    if (!F(qr)(f, m, n, lf, tau, ts, rest, s)) return 0;
    for (int64_t i = 0; i < qc; i++)
      for (int64_t j = 0; j < n; j++) r[i * n + j] = j >= i ? f[i * lf + j] : 0;
    return F(form_q)(f, m, n, lf, ts, d, qc, qc, rest, s);
  }
  if (c->routine == LU) {
    stage(c, 0, b, d, n, 0, s);
    int64_t *pivots = (int64_t *)matrix_at(c, 2, b);
    int64_t *perm = (int64_t *)matrix_at(c, 3, b);
    return F(lu)(d, m, n, n, pivots, perm, s);
  }
  if (c->routine == CHOLESKY) {
    stage(c, 0, b, d, n, 0, s);
    int ok = F(cholesky)(d, n, n, s);
    if (ok < 0) return 0;
    if (ok == 0) {
      nans(d, n * n * (CPLX + 1), RDT);
      return 1;
    }
    F(triangle) t = {.d = d, .n = n, .upper = c->upper};
    int64_t bytes = n * n * (int64_t)sizeof(T);
    double cost = threaded(s, n, bytes, 0, RDT, CPLX);
    if (cost == 0) F(other_triangle)(0, n, 0, &t);
    else nx_cpu_job(n, bytes, cost, F(other_triangle), &t);
    return 1;
  }
  /* Solve_triangular: a read in place where op(a) is a and its rows'
     elements lie adjacent, since the solve only reads it, else op(a)
     staged into w; then x staged into d. A destination identical to a,
     which x overwrites, has a staged first. */
  int64_t r = c->r, ld = n;
  const nx_array *ta = &c->a[0];
  int64_t sr = ta->dim[2 * ta->rank - 2], sc = ta->dim[2 * ta->rank - 1];
  const T *l = w;
  if (!c->adjoint && sc == 1 && sr >= n && !ta->alias) {
    l = (const T *)matrix_at(c, 0, b);
    ld = sr;
  } else {
    stage(c, 0, b, w, n, c->adjoint, s);
#if CPLX
    if (c->adjoint)
      for (int64_t i = 0; i < n * n; i++) w[i] = CONJ(w[i]);
#endif
  }
  stage(c, 1, b, d, r, 0, s);
  int upper = c->upper != c->adjoint;
  if (!c->unit)
    for (int64_t i = 0; i < n; i++)
      if (l[i * ld + i] == 0) {
        nans(d, n * r * (CPLX + 1), RDT);
        return 1;
      }
  solve kind = upper ? (c->unit ? UNIT_UPPER : UPPER)
                     : (c->unit ? UNIT_LOWER : LOWER);
  return F(solve)(kind, l, n, ld, d, r, r, NULL, s);
}

#undef T
#undef R
#undef RDT
#undef SFX
#undef CPLX
#undef CONJ
#undef REAL
#undef IMAG
#undef SQRT
#undef COPYSIGN
#undef LDEXP
#undef ILOGB
#undef WORD
#undef C
#undef CK
#undef MUL
#undef DIV
#undef CSQRT
#undef SAFMIN
#undef EPS
#undef TINY
#undef ETA
#undef SMLSIZ
