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
   code, CPLX 1 for complex, C the complex type of R's precision, CSQRT
   its square root on complex dtypes, and CONJ, REAL, IMAG, SQRT and
   COPYSIGN for T and R, which it undefines at its end. F(name) names the
   dtype's instance.

   Matrices are dense and row-major, row i at a + i·ld. Each routine runs
   by scalar loops on blocks of at most LEAF columns and updates the rest
   of the matrix by products (F(gemm)). */

/* Products */

#if !CPLX
/* w ← −x for the [r] × [c] x, element (i, j) at x[i·rs + j·cs], w's rows
   [c] long: row by row, or by blocks of 8 rows where x's rows lie
   adjacent, as a transposed operand's do, so that x is read a line at a
   time, once. */
static void F(negated)(const T *x, int64_t r, int64_t c, int64_t rs,
                       int64_t cs, T *w) {
  if (rs != 1 || cs == 1) {
    for (int64_t i = 0; i < r; i++)
      for (int64_t j = 0; j < c; j++) w[i * c + j] = -x[i * rs + j * cs];
    return;
  }
  for (int64_t i0 = 0; i0 < r; i0 += 8) {
    int64_t i1 = i0 + 8 < r ? i0 + 8 : r;
    for (int64_t j = 0; j < c; j++)
      for (int64_t i = i0; i < i1; i++) w[i * c + j] = -x[i + j * cs];
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
   unconjugated, and copied otherwise. */
static int F(gemm)(int64_t m, int64_t n, int64_t k, int sign, operand a,
                   operand b, T *c, int64_t ldc, arena *s) {
  if (m == 0 || n == 0 || k == 0) return 1;
  const T *ap = a.p, *bp = b.p;
#if !CPLX
  int64_t ars = a.rs, acs = a.cs, brs = b.rs, bcs = b.cs;
  if (sign < 0 && m * k <= k * n) {
    T *w = room(s, m * k * (int64_t)sizeof(T));
    if (w == NULL) return 0;
    F(negated)(ap, m, k, ars, acs, w);
    ap = w;
    ars = k;
    acs = 1;
  } else if (sign < 0) {
    T *w = room(s, k * n * (int64_t)sizeof(T));
    if (w == NULL) return 0;
    F(negated)(bp, k, n, brs, bcs, w);
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
  int64_t ars = a.rs;
  if (copy_a) {
    for (int64_t i = 0; i < m; i++)
      for (int64_t p = 0; p < k; p++) {
        T x = ap[i * a.rs + p * a.cs];
        w[i * k + p] = a.conj ? CONJ(x) : x;
      }
    ap = w;
    ars = k;
  }
  R *bt = (R *)(w + na);
  for (int64_t p = 0; p < k; p++) {
    R *r0 = bt + 2 * p * 2 * n, *r1 = r0 + 2 * n;
    for (int64_t j = 0; j < n; j++) {
      T z = bp[p * b.rs + j * b.cs];
      R x = sign * REAL(z), y = sign * (b.conj ? -IMAG(z) : IMAG(z));
      r0[2 * j] = x;
      r0[2 * j + 1] = y;
      r1[2 * j] = -y;
      r1[2 * j + 1] = x;
    }
  }
  return product(RDT, m, 2 * n, 2 * k, (strided){ap, 2 * ars, 1},
                 (strided){bt, 2 * n, 1}, c, 2 * ldc);
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
  for (int64_t j = 0; j < r; j++) x[j] /= d;
}

/* Triangular solves

   x ← l⁻¹ x for the lower triangular n × n [l] and the n × r [x], or
   u⁻¹ x for the upper [u], in place: halves of l down to LEAF, as
   LAPACK's xTRSM's left-side cases run blocked (ReLAPACK's recursive
   splitting). The lower solve: x1 of l11, then x2 of l22 and
   x2 − l21 x1; the upper one from its last half. At a leaf, substitution
   row by row in the direction of the solve,

     x[i] = (x[i] − Σ_{p before i} l[i][p] x[p]) / l[i][i],

   with no division under [unit]. Answers 1, or 0 if an allocation
   failed. */
static int F(lower)(const T *l, int64_t n, int64_t ld, T *x, int64_t r,
                    int64_t ldx, int unit, arena *s) {
  if (n <= LEAF) {
    for (int64_t i = 0; i < n; i++) {
      T *xi = x + i * ldx;
      for (int64_t p = 0; p < i; p++) {
        T c = l[i * ld + p];
        const T *xp = x + p * ldx;
        for (int64_t j = 0; j < r; j++) xi[j] -= c * xp[j];
      }
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
      for (int64_t p = i + 1; p < n; p++) {
        T c = u[i * ld + p];
        const T *xp = x + p * ldx;
        for (int64_t j = 0; j < r; j++) xi[j] -= c * xp[j];
      }
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

/* A solve by units of columns of x, which are independent, on the
   call's threads: one a thread, of SOLVE_COLUMNS columns at least, each
   solved whole. Every product adds in chain order (product), whatever
   its shape, so each column's arithmetic and bits are the whole solve's,
   whatever the units and threads.

   ADJOINT_LOWER's x is r × n and solved as l t' = t for t = xᴴ, since
   x lᴴ is the old x: a unit copies its rows of x conjugate transposed into
   its worker's copy buffer, solves them there and copies them back, since
   columns of x read in place would step a row's length apart, which at a
   power of two maps every row to one set of the cache. */
typedef struct {
  const T *l;
  T *x;
  int64_t n, ld, r, ldx, units;
  solve kind;
  int failed;
  arena *workers;
} F(solve_job);

static int F(columns)(const F(solve_job) *j, int64_t c0, int64_t c,
                      arena *s) {
  int64_t n = j->n, ld = j->ld, ldx = j->ldx;
  switch (j->kind) {
    case LOWER: return F(lower)(j->l, n, ld, j->x + c0, c, ldx, 0, s);
    case UNIT_LOWER: return F(lower)(j->l, n, ld, j->x + c0, c, ldx, 1, s);
    case UPPER: return F(upper)(j->l, n, ld, j->x + c0, c, ldx, 0, s);
    case UNIT_UPPER: return F(upper)(j->l, n, ld, j->x + c0, c, ldx, 1, s);
    case ADJOINT_LOWER: break;
  }
  T *x = j->x + c0 * ldx, *t = room(s->copy, n * c * (int64_t)sizeof(T));
  if (t == NULL) return 0;
  for (int64_t i = 0; i < c; i++)
    for (int64_t p = 0; p < n; p++) t[p * c + i] = CONJ(x[i * ldx + p]);
  int ok = F(lower)(j->l, n, ld, t, c, c, 0, s);
  for (int64_t i = 0; i < c; i++)
    for (int64_t p = 0; p < n; p++) x[i * ldx + p] = CONJ(t[p * c + i]);
  return ok;
}

static void F(solve_unit)(int64_t lo, int64_t hi, int worker, void *ctx) {
  F(solve_job) *j = ctx;
  for (int64_t u = lo; u < hi; u++) {
    int64_t c0 = u * j->r / j->units, c1 = (u + 1) * j->r / j->units;
    if (!F(columns)(j, c0, c1 - c0, &j->workers[worker]))
      __atomic_store_n(&j->failed, 1, __ATOMIC_RELAXED);
  }
}

/* The solve [kind] by the n × n triangle [l] of the n × r [x], or of the
   r × n [x] under ADJOINT_LOWER, rows [ldx] apart: by units where [s]
   holds the call's threads, else whole. Answers 0 if an allocation
   failed. */
static int F(solve)(solve kind, const T *l, int64_t n, int64_t ld, T *x,
                    int64_t r, int64_t ldx, arena *s) {
  int64_t bytes = n * (n + r) * (int64_t)sizeof(T), units = r / SOLVE_COLUMNS;
  double cost = threaded(s, units, bytes, (double)n * n * r, RDT, CPLX);
  F(solve_job) j = {.l = l, .x = x, .n = n, .ld = ld, .r = r, .ldx = ldx,
                    .kind = kind, .workers = s->workers};
  if (cost == 0) return F(columns)(&j, 0, r, s);
  int threads = nx_cpu_threads(bytes, cost);
  j.units = units < threads ? units : threads;
  nx_cpu_job(j.units, bytes, cost, F(solve_unit), &j);
  return !j.failed;
}

/* Cholesky

   L with a = L Lᴴ, in place in a's lower triangle, as ReLAPACK's
   recursive xPOTRF: L11 of a11, then L21 = a21 L11⁻ᴴ, then L22 of
   a22 − L21 L21ᴴ. The update runs on column panels of at most HERK
   columns, each from the diagonal down, so that a panel's diagonal block
   is computed whole: its part above the diagonal is never read. At a
   leaf, LAPACK's xPOTF2, row by row,

     L[i][c] = (a[i][c] − Σ_{p < c} L[i][p] conj L[c][p]) / L[c][c]
     L[i][i] = sqrt (a[i][i] − Σ_{p < i} |L[i][p]|²).

   Only the lower triangle and the diagonal's real parts are read. Answers
   1, 0 if a pivot is not positive (NaN included), -1 if an allocation
   failed. */
static int F(cholesky)(T *a, int64_t n, int64_t ld, arena *s) {
  if (n <= LEAF) {
    for (int64_t i = 0; i < n; i++) {
      T *ai = a + i * ld;
      for (int64_t c = 0; c <= i; c++) {
        const T *ac = a + c * ld;
        T x = ai[c];
        for (int64_t p = 0; p < c; p++) x -= ai[p] * CONJ(ac[p]);
        if (c < i) {
          ai[c] = x / REAL(ac[c]);
          continue;
        }
        R d = REAL(x);
        if (!(d > 0)) return 0;
        ai[c] = SQRT(d);
      }
    }
    return 1;
  }
  int64_t n1 = split(n), n2 = n - n1;
  T *a21 = a + n1 * ld, *a22 = a21 + n1;
  int ok = F(cholesky)(a, n1, ld, s);
  if (ok <= 0) return ok;
  if (!F(solve)(ADJOINT_LOWER, a, n1, ld, a21, n2, ld, s)) return -1;
  for (int64_t j = 0; j < n2; j += HERK) {
    int64_t jb = n2 - j < HERK ? n2 - j : HERK;
    if (!F(gemm)(n2 - j, jb, n1, -1, F(op)(a21 + j * ld, ld, 1),
                 F(op_h)(a21 + j * ld, ld), a22 + j * ld + j, ld, s))
      return -1;
  }
  return F(cholesky)(a22, n2, ld, s);
}

/* LU

   P a = L U with partial pivoting, in place in the m × n [a], as LAPACK's
   recursive xGETRF2 (ReLAPACK's xGETRF): the panel of a block of columns
   factors its left half, solves the top rows of its right half by L11,
   updates the rest of its right half by a22 − L21 U12, then factors that.
   Interchanges swap whole rows of [a], so every column sees them as they
   happen. At a leaf of at most LEAF columns, LAPACK's xGETF2: column by
   column, the pivot is found by a scan of the column from the diagonal
   down that moves to each later row whose magnitude (|re| + |im|) is
   greater than the current pivot's, NaN never greater; its row swaps with
   the diagonal's; a nonzero pivot divides the column below it; the
   panel's columns to its right take the rank-1 update. Columns past the
   first k = min m n take U = L⁻¹ a, unit lower L. [pivots] and [perm]
   take k and m positions. Answers 1, or 0 if an allocation failed. */

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

/* The panel of columns [j0, j0 + n) from row j0 down to row [m], in the
   matrix [a] of [width] columns. */
static int F(panel)(T *a, int64_t m, int64_t width, int64_t ld, int64_t j0,
                    int64_t n, int64_t *pivots, int64_t *perm, arena *s) {
  if (n <= LEAF) {
    /* The panel's columns, copied contiguous (column c's row i at
       q[c·h + i − j0]), so that a column streams: in place, each of its
       elements is a line of its own, rows a power of two apart sharing one
       set of the cache. */
    int64_t h = m - j0;
    T *q = room(s, h * n * (int64_t)sizeof(T));
    if (q == NULL) return 0;
    for (int64_t i = 0; i < h; i++)
      for (int64_t c = 0; c < n; c++) q[c * h + i] = a[(j0 + i) * ld + j0 + c];
    for (int64_t j = 0; j < n; j++) {
      T *qj = q + j * h;
      int64_t p = j;
      R best = F(magnitude)(qj[j]);
      for (int64_t i = j + 1; i < h; i++) {
        R v = F(magnitude)(qj[i]);
        if (v > best) {
          best = v;
          p = i;
        }
      }
      pivots[j0 + j] = j0 + p;
      if (p != j) {
        F(swap)(a + (j0 + j) * ld, a + (j0 + p) * ld, width);
        for (int64_t c = 0; c < n; c++)
          F(swap)(q + c * h + j, q + c * h + p, 1);
        int64_t t = perm[j0 + j];
        perm[j0 + j] = perm[j0 + p];
        perm[j0 + p] = t;
      }
      T d = qj[j];
      if (d != 0)
        for (int64_t i = j + 1; i < h; i++) qj[i] /= d;
      for (int64_t c = j + 1; c < n; c++) {
        T *qc = q + c * h, u = qc[j];
        for (int64_t i = j + 1; i < h; i++) qc[i] -= qj[i] * u;
      }
    }
    for (int64_t i = 0; i < h; i++)
      for (int64_t c = 0; c < n; c++) a[(j0 + i) * ld + j0 + c] = q[c * h + i];
    return 1;
  }
  int64_t n1 = split(n), n2 = n - n1, j1 = j0 + n1;
  T *a11 = a + j0 * ld + j0, *a12 = a11 + n1, *a21 = a11 + n1 * ld;
  return F(panel)(a, m, width, ld, j0, n1, pivots, perm, s) &&
         F(solve)(UNIT_LOWER, a11, n1, ld, a12, n2, ld, s) &&
         F(gemm)(m - j1, n2, n1, -1, F(op)(a21, ld, 1), F(op)(a12, ld, 1),
                 a21 + n1, ld, s) &&
         F(panel)(a, m, width, ld, j1, n2, pivots, perm, s);
}

static int F(lu)(T *a, int64_t m, int64_t n, int64_t ld, int64_t *pivots,
                 int64_t *perm, arena *s) {
  int64_t k = m < n ? m : n;
  for (int64_t i = 0; i < m; i++) perm[i] = i;
  if (k == 0) return 1;
  return F(panel)(a, m, n, ld, 0, k, pivots, perm, s) &&
         F(solve)(UNIT_LOWER, a, k, ld, a + k, n - k, ld, s);
}

/* Householder reflectors */

/* The least normal number over the unit roundoff, below which LAPACK's
   reflector rescales its norm. */
#define SAFMIN \
  ((R)(sizeof(R) == 4 ? FLT_MIN / (FLT_EPSILON / 2)                         \
                      : DBL_MIN / (DBL_EPSILON / 2)))

static R F(abs)(R x) { return x < 0 ? -x : x; }

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
   already. */
static T F(reflector)(T *alpha, T *x, int64_t n, int64_t s) {
  R xnorm = F(norm)(x, n, s);
  R ar = REAL(*alpha), ai = IMAG(*alpha);
  if (xnorm == 0 && ai == 0) return 0;
  R beta = ar >= 0 ? -F(lapy3)(ar, ai, xnorm) : F(lapy3)(ar, ai, xnorm);
  int knt = 0;
  if (F(abs)(beta) < SAFMIN) {
    do {
      knt++;
      for (int64_t i = 0; i < n; i++) x[i * s] = x[i * s] * (1 / SAFMIN);
      beta *= 1 / SAFMIN;
      ar *= 1 / SAFMIN;
      ai *= 1 / SAFMIN;
    } while (F(abs)(beta) < SAFMIN && knt < 20);
    xnorm = F(norm)(x, n, s);
    R an = F(lapy3)(ar, ai, xnorm);
    beta = ar >= 0 ? -an : an;
  }
#if CPLX
  T tau = (beta - ar) / beta - (ai / beta) * I;
  T scale = (ar - beta) + ai * I;
#else
  T tau = (beta - ar) / beta;
  T scale = ar - beta;
#endif
  for (int64_t i = 0; i < n; i++) x[i * s] = x[i * s] / scale;
  for (int j = 0; j < knt; j++) beta *= SAFMIN;
  *alpha = beta;
  return tau;
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
    for (int64_t j = 0; j < n; j++) w[j] += vi * ci[j];
  }
  for (int64_t i = 0; i < m; i++) {
    T t = i == 0 ? tau : tau * v[i * ldv];
    T *ci = c + i * ldc;
    for (int64_t j = 0; j < n; j++) ci[j] -= t * w[j];
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
      for (int64_t q = p; q < i; q++) x += t[p * ldt + q] * g[q * jb + i];
      t[p * ldt + i] = -tau[i] * x;
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

/* The [h] × [n] panel [a], rows [ld] apart, factored: its reflectors in
   place and τ in [tau], its dense V into [v] and its T into [t], both rows
   QR_NB apart. As LAPACK's recursive xGEQRT3: the left half's reflectors
   V1, T1, applied to the right half, then the right half's V2, T2 below
   the left half's rows, and

     T = [T1  −T1 (V1ᴴ V2) T2]
         [0    T2            ],

   since (I − V1 T1 V1ᴴ) (I − V2 T2 V2ᴴ) = I − V T Vᴴ. A panel of fewer
   than 2·LEAF columns is a leaf, whose halves would be fewer than LEAF
   columns and cost more in products than in scalar loops: on the M1, 16
   QRs of 32 × 32 float32 took 56.5 µs in halves against 35.4 as one
   leaf. A leaf takes the reflectors one at a time, each applied to the
   columns to its right (xGEQR2), on a contiguous copy in [c], which holds
   (h + 1)·2·LEAF elements, then T (xLARFT). [w] holds 2·QR_NB·QR_NB
   elements. */
static int F(qr_panel)(T *a, int64_t h, int64_t n, int64_t ld, T *tau,
                       T *v, T *t, T *c, T *w, arena *s) {
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
    F(rows)(c, n, h, n, v, QR_NB);
    return F(block_t)(c, h, n, tau, t, QR_NB, w, s);
  }
  int64_t n1 = split(n), n2 = n - n1;
  T *g = w, *x = w + n1 * n2, *t12 = t + n1;
  if (!F(qr_panel)(a, h, n1, ld, tau, v, t, c, w, s) ||
      !F(larfb)(v, QR_NB, F(op_h)(t, QR_NB), n1, h, n2, a + n1, ld, w, s) ||
      !F(qr_panel)(a + n1 * ld + n1, h - n1, n2, ld, tau + n1,
                   v + n1 * QR_NB + n1, t + n1 * QR_NB + n1, c, w, s))
    return 0;
  for (int64_t r = 0; r < n1; r++) {
    memset(v + r * QR_NB + n1, 0, sizeof(T) * n2);
    memset(t12 + r * QR_NB, 0, sizeof(T) * n2);
  }
  for (int64_t r = n1; r < n; r++) memset(t + r * QR_NB, 0, sizeof(T) * n1);
  memset(w, 0, sizeof(T) * 2 * n1 * n2);
  return F(gemm)(n1, n2, h - n1, 1, F(op_h)(v + n1 * QR_NB, QR_NB),
                 F(op)(v + n1 * QR_NB + n1, QR_NB, 1), g, n2, s) &&
         F(gemm)(n1, n2, n1, 1, F(op)(t, QR_NB, 1), F(op)(g, n2, 1), x, n2,
                 s) &&
         F(gemm)(n1, n2, n2, -1, F(op)(x, n2, 1),
                 F(op)(t + n1 * QR_NB + n1, QR_NB, 1), t12, QR_NB, s);
}

static int F(qr)(T *a, int64_t m, int64_t n, int64_t ld, T *tau, T *ts,
                 T *q, arena *s) {
  int64_t k = m < n ? m : n;
  T *v = q, *c = v + m * QR_NB, *w = c + (m + 1) * 2 * LEAF;
  for (int64_t j0 = 0; j0 < k; j0 += QR_NB) {
    int64_t jb = QR_NB < k - j0 ? QR_NB : k - j0, j1 = j0 + jb, h = m - j0;
    T *t = ts + j0 * QR_NB;
    if (!F(qr_panel)(a + j0 * ld + j0, h, jb, ld, tau + j0, v, t, c, w, s))
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
   columns, H_j = I − τ_j v_j v_jᴴ with v_j[j] = 1 and v_j[r], r > j, at
   [x] + r·rs + j·cs, read as [how] says: by blocks of NB
   reflectors, the last first (xUNMQR, xUNMLQ). The scratch [w] holds
   (h + NB + 2n) NB elements. Answers 0 if an allocation failed. */
static int F(back)(const T *x, int64_t rs, int64_t cs, reading how,
                   const T *tau, int64_t k, int64_t h, int64_t n, T *c,
                   int64_t ldc, T *w, arena *s) {
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
        !F(larfb)(w, jb, F(op)(t, jb, 1), jb, hb, n, c + j0 * ldc, ldc, ww,
                  s))
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

/* The unit roundoff and the least normal number. */
#define EPS ((R)(sizeof(R) == 4 ? FLT_EPSILON / 2 : DBL_EPSILON / 2))
#define TINY ((R)(sizeof(R) == 4 ? FLT_MIN : DBL_MIN))

/* xᴴ y over [n] elements, in four parts by the index mod 4 so that no
   addition waits on the one before. */
static T F(dot)(const T *x, const T *y, int64_t n) {
  T p[4] = {0, 0, 0, 0};
  int64_t r = 0;
  for (; r + 4 <= n; r += 4)
    for (int q = 0; q < 4; q++) p[q] += CONJ(x[r + q]) * y[r + q];
  for (; r < n; r++) p[r & 3] += CONJ(x[r]) * y[r];
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
        y[r] += c[q - j][r] * v[q];
        t[q - j] += CONJ(c[q - j][r]) * v[r];
      }
    }
    for (int64_t r = e; r < h; r += 2)
      for (int l = 0; l < 2 && r + l < h; l++) {
        T vr = v[r + l], u = y[r + l];
        for (int q = 0; q < 4; q++) {
          u += c[q][r + l] * vq[q];
          s[q][l] += CONJ(c[q][r + l]) * vr;
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
      for (int64_t r = i; r < n; r++) ai[r] -= ap[r] * x + wp[r] * y;
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
        for (int64_t r = 0; r < h; r++) y[r] -= up[r] * x[p];
      }
    }
    T s = 0;
    for (int64_t r = 0; r < h; r++) {
      y[r] *= tau[i];
      s += CONJ(y[r]) * v[r];
    }
    T half = (R)-0.5 * tau[i] * s;
    for (int64_t r = 0; r < h; r++) y[r] += half * v[r];
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

/* [j]'s phase over its k roots, by units of MERGE_ROOTS on the call's
   threads where [s] holds them. Answers 0 if a root did not converge. */
static int F(secular)(F(secular_job) *j, secular phase, arena *s) {
  int64_t k = j->k, units = (k + MERGE_ROOTS - 1) / MERGE_ROOTS;
  int64_t bytes = k * k * (int64_t)sizeof(R);
  double cost = threaded(s, units, bytes,
                         (double)k * k * (phase == ROOTS ? 8 : 1), RDT, 0);
  j->phase = phase;
  if (cost > 0)
    nx_cpu_job(units, bytes, cost, F(secular_unit), j);
  else
    F(secular_unit)(0, units, 0, j);
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

/* Eigh on matrix [b] of [c] in the scratch [x]: w, and v where the call
   has vectors, as LAPACK's xHEEV and xHEEVD: the matrix scaled into
   [rmin, rmax] where its largest magnitude lies outside, reduced (hetrd),
   T's eigenvalues by steqr, or with Z by stedc, then v = Q Z by Q's
   blocks of NB reflectors applied last first (xUNMTR). A matrix that
   holds NaN or an infinity where it is read, or on which an iteration
   does not converge, is NaN in both. Answers 0 if an allocation
   failed. */
static int F(eigh)(const call *c, int64_t b, T *x, arena *s) {
  int64_t n = c->n;
  T *a = x, *tau = a + n * n, *w = tau + n, *yp = w + (3 * n + NB) * NB;
  R *d = (R *)(yp + hemv_blocks(n) * n), *e = d + n, *z = e + n,
    *work = z + n * n;
  int64_t *iwork = (int64_t *)(((uintptr_t)(work + 3 * n + 2 * n * n) + 7) &
                               ~(uintptr_t)7);
  T *wo = (T *)matrix_at(c, c->nops, b);
  T *v = c->vectors ? (T *)matrix_at(c, c->nops + 1, b) : NULL;
  stage(c, 0, b, a, n, 1);
  R big = 0;
  int finite = 1;
  for (int64_t j = 0; j < n; j++) {
    a[j * n + j] = REAL(a[j * n + j]);
    for (int64_t i = j; i < n; i++) {
      R re = REAL(a[j * n + i]), im = IMAG(a[j * n + i]);
      finite &= re - re == 0 && im - im == 0;
      big = F(abs)(re) > big ? F(abs)(re) : big;
      big = F(abs)(im) > big ? F(abs)(im) : big;
    }
  }
  const R rmin = SQRT(TINY / (2 * EPS)), rmax = 1 / rmin;
  R sigma = big > 0 && big < rmin ? rmin / big : big > rmax ? rmax / big : 1;
  if (sigma != 1)
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = j; i < n; i++) a[j * n + i] *= sigma;
  if (finite && !F(hetrd)(a, n, n, d, e, tau, w, yp, s)) return 0;
  int ok = finite;
  if (ok && v) {
    for (int64_t i = 0; i < n * n; i++) z[i] = i % (n + 1) == 0;
    ok = F(stedc)(n, d, e, z, n, work, iwork, s);
    if (ok < 0) return 0;
  } else if (ok)
    ok = F(steqr)(n, d, e, NULL, 0, 0, work);
  if (!ok) {
    nans(wo, n * (CPLX + 1), RDT);
    if (v) nans(v, n * n * (CPLX + 1), RDT);
    return 1;
  }
  for (int64_t j = 0; j < n; j++) wo[j] = d[j] * (1 / sigma) + 0;
  if (!v) return 1;
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < n; j++) v[i * n + j] = z[j * n + i];
  /* Q = H_0 ⋯ H_(n−2) on rows 1 on: reflector j is a's column j from row
     j + 1, so that they are the reflectors of a QR of rows 1 on. */
  return F(back)(a + 1, 1, n, AS_STORED, tau, n - 1, n - 1, n, v + n, n, w,
                 s);
}

/* Singular value decomposition

   a = u diag s vh as LAPACK's xGESDD computes it. A wide matrix is
   decomposed as its conjugate transpose, which swaps u and vh; one with
   at least 11/6 as many rows as columns, 17/9 on complex numbers, is
   factored a = Q R first, and R decomposed. The matrix B is reduced to Qᴴ B P, upper bidiagonal with
   real d and e (xGEBRD); that one's SVD U_B diag s V_Bᵀ is found by divide
   and conquer (xBDSDC); then u = Q U_B and vh = (P V_B)ᴴ by the
   reflectors' blocks (back). Values alone run implicit QR on the
   bidiagonal without vectors (xBDSQR), which for at most SMLSIZ values is
   the vectors' call's iteration. Matrices are column-major; V_B's
   vectors are its columns, where LAPACK keeps Vᵀ's rows, so that every
   rotation moves adjacent memory. */

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
   + 2n² + m² reals, m = n + sqre, and [iwork] 4n indices. Answers 1, 0 if
   a root does not converge, or −1 if an allocation failed. */
static int F(lasd1)(int64_t nl, int64_t nr, int sqre, R *d, R alpha, R beta,
                    R *u, int64_t ldu, R *v, int64_t ldv, int64_t *idxq,
                    R *work, int64_t *iwork) {
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
    for (int64_t i = 0; i < k; i++)
      if (!F(lasd4)(k, i, dsigma, z, u + i * ldu, rho, d + i, v + i * ldv))
        return 0;
    for (int64_t i = 0; i < k; i++) {
      R zi = u[i + (k - 1) * ldu] * v[i + (k - 1) * ldv];
      for (int64_t p = 0; p < i; p++)
        zi *= u[i + p * ldu] * v[i + p * ldv] / (dsigma[i] - dsigma[p]) /
              (dsigma[i] + dsigma[p]);
      for (int64_t p = i; p < k - 1; p++)
        zi *= u[i + p * ldu] * v[i + p * ldv] / (dsigma[i] - dsigma[p + 1]) /
              (dsigma[i] + dsigma[p + 1]);
      z[i] = F(sign)(SQRT(F(abs)(zi)), q[i]);
    }
    /* The secular vectors, normalised, rows by type: u's into q. */
    for (int64_t i = 0; i < k; i++) {
      R *ui = u + i * ldu, *wi = v + i * ldv;
      wi[0] = z[0] / ui[0] / wi[0];
      ui[0] = -1;
      for (int64_t p = 1; p < k; p++) {
        wi[p] = z[p] / ui[p] / wi[p];
        ui[p] = dsigma[p] * wi[p];
      }
      R t = 1 / F(rnorm)(ui, k);
      q[i * k] = ui[0] * t;
      for (int64_t p = 1; p < k; p++) q[p + i * k] = ui[idxc[p]] * t;
    }
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
    for (int64_t i = 0; i < k; i++) {
      R *wi = v + i * ldv, t = 1 / F(rnorm)(wi, k);
      q[i * k] = wi[0] * t;
      for (int64_t p = 1; p < k; p++) q[p + i * k] = wi[idxc[p]] * t;
    }
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

/* The SVD of the upper bidiagonal B, [n] × (n + [sqre]), of the diagonal
   [d] and superdiagonal [e], into d, u and v as lasd1 leaves them, from
   the identity: torn in two at row n / 2 [levels] times, the first half
   of n + 1 columns, the last of its parent's, the leaves by lasdq, then
   merged (xLASD0, its subproblems solved depth first); [idxq] orders d
   ascending. [work] and [iwork] as for lasd1. Answers as lasd1. */
static int F(lasd0)(int64_t n, int sqre, int levels, R *d, R *e, R *u,
                    int64_t ldu, R *v, int64_t ldv, int64_t *idxq, R *work,
                    int64_t *iwork) {
  if (levels == 0) {
    for (int64_t i = 0; i < n; i++) idxq[i] = i;
    return F(lasdq)(n, sqre, d, e, u, ldu, v, ldv, work);
  }
  int64_t nl = n / 2, nr = n - nl - 1, c = nl + 1;
  int ok = F(lasd0)(nl, 1, levels - 1, d, e, u, ldu, v, ldv, idxq, work,
                    iwork);
  if (ok == 1)
    ok = F(lasd0)(nr, sqre, levels - 1, d + c, e + c, u + c + c * ldu, ldu,
                  v + c + c * ldv, ldv, idxq + c, work, iwork);
  return ok == 1 ? F(lasd1)(nl, nr, sqre, d, d[nl], e[nl], u, ldu, v, ldv,
                            idxq, work, iwork)
                 : ok;
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
                    int64_t ldv, R *work, int64_t *iwork) {
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
                      ldv, iwork, work, iwork + n);
    if (ok != 1) return ok;
    start = i + 1;
  }
  for (int64_t i = 0; i < n; i++) d[i] *= big;
  F(sort)(n, d, DESCENDING, u, n, ldu, v, n, ldv);
  return 1;
}

/* y[j·ys] += s Σ_i conj(a[i + j·lda]) x[i] for j < [n], i < [m]: y += s aᴴ
   x for the column-major a (xGEMV's 'C'). By four columns at a time,
   which share x's loads, each product in four parts by the index mod 4
   as dot's, so that a column's bits do not depend on its neighbours. */
static void F(mvh)(int64_t m, int64_t n, const T *a, int64_t lda,
                   const T *x, T *y, int64_t ys, R s) {
  int64_t j = 0;
  for (; j + 4 <= n; j += 4) {
    const T *c[4] = {a + j * lda, a + (j + 1) * lda, a + (j + 2) * lda,
                     a + (j + 3) * lda};
    T p[4][4] = {{0}};
    int64_t r = 0;
    for (; r + 4 <= m; r += 4)
      for (int l = 0; l < 4; l++)
        for (int q = 0; q < 4; q++) p[q][l] += CONJ(c[q][r + l]) * x[r + l];
    for (; r < m; r++)
      for (int q = 0; q < 4; q++) p[q][r & 3] += CONJ(c[q][r]) * x[r];
    for (int q = 0; q < 4; q++)
      y[(j + q) * ys] += s * ((p[q][0] + p[q][1]) + (p[q][2] + p[q][3]));
  }
  for (; j < n; j++) y[j * ys] += s * F(dot)(a + j * lda, x, m);
}

/* y[i] += s Σ_j a[i + j·lda] x[j·xs] for i < [m], j < [n]: y += s a x for
   the column-major a (xGEMV's 'N'), by four columns at a time, which
   share y's loads and stores. */
static void F(mvn)(int64_t m, int64_t n, const T *a, int64_t lda,
                   const T *x, int64_t xs, T *y, R s) {
  int64_t j = 0;
  for (; j + 4 <= n; j += 4) {
    const T *c0 = a + j * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;
    T t0 = s * x[j * xs], t1 = s * x[(j + 1) * xs], t2 = s * x[(j + 2) * xs],
      t3 = s * x[(j + 3) * xs];
    for (int64_t i = 0; i < m; i++) {
      T u = y[i];
      u += c0[i] * t0;
      u += c1[i] * t1;
      u += c2[i] * t2;
      u += c3[i] * t3;
      y[i] = u;
    }
  }
  for (; j < n; j++) {
    const T *aj = a + j * lda;
    T t = s * x[j * xs];
    for (int64_t i = 0; i < m; i++) y[i] += aj[i] * t;
  }
}

/* The panel of the first [nb] columns and rows of the [m] × [n] [a], m >=
   n, by reflectors, as LAPACK's xLABRD: column i, the products of the
   panel's earlier reflectors deferred until now applied to it, is reduced
   below its diagonal by H_i, its diagonal into d[i]; then row i likewise,
   on its conjugate, right of its superdiagonal by G_i, its superdiagonal
   into e[i]. v_i and u_i are stored where they reduced, their first
   elements 1. [x], m × nb, and [y], n × nb, columns [ldx] and [ldy]
   apart, get X and Y with the rest of a updated as a − V Yᴴ − X U. [r]
   holds n + nb elements: the conjugated row, contiguous, and a short
   vector. */
static void F(labrd)(T *a, int64_t m, int64_t n, int64_t nb, int64_t lda,
                     R *d, R *e, T *tauq, T *taup, T *x, int64_t ldx, T *y,
                     int64_t ldy, T *r) {
  T *row = r + nb;
  for (int64_t i = 0; i < nb; i++) {
    T *ai = a + i * lda, *xi = x + i * ldx, *yi = y + i * ldy,
      *xr = xi + i + 1, *yr = yi + i + 1;
    int64_t mr = m - i - 1, nr = n - i - 1;
    for (int64_t p = 0; p < i; p++) r[p] = CONJ(y[i + p * ldy]);
    F(mvn)(m - i, i, a + i, lda, r, 1, ai + i, -1);
    F(mvn)(m - i, i, x + i, ldx, ai, 1, ai + i, -1);
    T alpha = ai[i];
    tauq[i] = F(reflector)(&alpha, ai + i + 1, mr, 1);
    d[i] = REAL(alpha);
    if (nr == 0) continue;
    /* Y's column i: τ_i times aᴴ v_i less the panel's products. */
    ai[i] = 1;
    memset(yi, 0, sizeof(T) * n);
    F(mvh)(m - i, nr, ai + lda + i, lda, ai + i, yr, 1, 1);
    F(mvh)(m - i, i, a + i, lda, ai + i, yi, 1, 1);
    F(mvn)(nr, i, y + i + 1, ldy, yi, 1, yr, -1);
    memset(yi, 0, sizeof(T) * i);
    F(mvh)(m - i, i, x + i, ldx, ai + i, yi, 1, 1);
    F(mvh)(i, nr, ai + lda, lda, yi, yr, 1, -1);
    for (int64_t p = 0; p < nr; p++) yr[p] *= tauq[i];
    /* Row i, conjugated into row, then G_i. */
    for (int64_t p = 0; p < nr; p++) row[p] = CONJ(ai[(p + 1) * lda + i]);
    for (int64_t p = 0; p <= i; p++) r[p] = CONJ(a[i + p * lda]);
    F(mvn)(nr, i + 1, y + i + 1, ldy, r, 1, row, -1);
    for (int64_t p = 0; p < i; p++) r[p] = CONJ(x[i + p * ldx]);
    F(mvh)(i, nr, ai + lda, lda, r, row, 1, -1);
    T beta = row[0];
    taup[i] = F(reflector)(&beta, row + 1, nr - 1, 1);
    e[i] = REAL(beta);
    row[0] = 1;
    /* X's column i: τ_i times a u_i less the panel's products. */
    memset(xi, 0, sizeof(T) * m);
    F(mvn)(mr, nr, ai + lda + i + 1, lda, row, 1, xr, 1);
    F(mvh)(nr, i + 1, y + i + 1, ldy, row, xi, 1, 1);
    F(mvn)(mr, i + 1, a + i + 1, lda, xi, 1, xr, -1);
    memset(xi, 0, sizeof(T) * i);
    F(mvn)(i, nr, ai + lda, lda, row, 1, xi, 1);
    F(mvn)(mr, i, x + i + 1, ldx, xi, 1, xr, -1);
    for (int64_t p = 0; p < mr; p++) xr[p] *= taup[i];
    for (int64_t p = 0; p < nr; p++) ai[(p + 1) * lda + i] = CONJ(row[p]);
  }
}

/* a = Q B Pᴴ in place in the [m] × [n] [a], m >= n, columns [lda] apart,
   as LAPACK's blocked xGEBRD: B upper bidiagonal, its diagonal into [d]
   and superdiagonal into [e]; Q = H_0 ⋯ H_(n−1) with v_i below a's
   diagonal in column i, P = G_0 ⋯ G_(n−2) with u_i's conjugate right of
   its superdiagonal in row i, their τ in [tauq] and [taup]. By panels of
   NB columns and rows (labrd), each followed by the rest's update a − V
   Yᴴ − X U by products on its transpose, row-major. The scratch [w] holds
   (m + n + 1) NB + n elements. Answers 0 if an allocation failed. */
static int F(gebrd)(T *a, int64_t m, int64_t n, int64_t lda, R *d, R *e,
                    T *tauq, T *taup, T *w, arena *s) {
  T *x = w, *y = x + m * NB, *r = y + n * NB;
  for (int64_t i = 0; i < n; i += NB) {
    int64_t nb = NB < n - i ? NB : n - i, m2 = m - i - nb, n2 = n - i - nb;
    T *p = a + i + i * lda, *c = p + nb + nb * lda;
    F(labrd)(p, m - i, n - i, nb, lda, d + i, e + i, tauq + i, taup + i, x,
             m, y, n, r);
    if (n2 == 0) continue;
    if (!F(gemm)(n2, m2, nb, -1, (operand){y + nb, 1, n, CPLX},
                 F(op)(p + nb, lda, 1), c, lda, s) ||
        !F(gemm)(n2, m2, nb, -1, F(op)(p + nb * lda, lda, 1),
                 F(op)(x + nb, m, 1), c, lda, s))
      return 0;
  }
  return 1;
}

/* SVD on matrix [b] of [c] in the scratch [x]: s, and u and vh where the
   call has vectors. A matrix that holds NaN or an infinity, or on which
   an iteration does not converge, is NaN in each. Answers 0 if an
   allocation failed. */
static int F(svd)(const call *c, int64_t b, T *x, arena *s) {
  int64_t m = c->m, n = c->n, wide = m < n, mm = wide ? n : m;
  int64_t k = wide ? m : n, uc = c->complete ? m : k, vr = c->complete ? n : k;
  T *so = (T *)matrix_at(c, c->nops, b);
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
     where QR comes first: from 11/6 k rows on reals (DGESDD's MNTHR),
     17/9 k on complex numbers (ZGESDD's MNTHR1). X = Q diag (U_B, I),
     mm × xc, and W = P V_B, k × k, are u and vhᴴ, or vhᴴ and u where a is
     wide. */
  int qrf = mm >= (CPLX ? k * 17 / 9 : k * 11 / 6);
  int64_t xc = wide ? vr : uc, ld = qrf ? k : mm;
  T *a = x, *r = a + mm * k, *tau = r + k * k, *tauq = tau + k,
    *taup = tauq + k, *ws = taup + k, *tmp = ws + svd_scratch(mm, k);
  R *d = (R *)(tmp + (wide ? mm * xc : k * k)), *e = d + k, *ub = e + k,
    *vb = ub + k * k, *work = vb + k * k;
  R *iw = work + 3 * (k + 1) * (k + 1) + 4 * k;
  int64_t *iwork = (int64_t *)(((uintptr_t)iw + 7) & ~(uintptr_t)7);
  stage(c, 0, b, a, ld, qrf == wide);
  R big = 0;
  int finite = 1;
  for (int64_t i = 0; i < mm * k; i++) {
    R re = REAL(a[i]), im = IMAG(a[i]);
    finite &= re - re == 0 && im - im == 0;
    big = F(abs)(re) > big ? F(abs)(re) : big;
    big = F(abs)(im) > big ? F(abs)(im) : big;
    if (wide) a[i] = CONJ(a[i]);
  }
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
    for (int64_t j = 0; j < k; j++)
      for (int64_t i = 0; i < k; i++) r[i + j * k] = i <= j ? a[i * k + j] : 0;
    bd = r;
    lb = k;
  }
  if (finite && !F(gebrd)(bd, lb, k, lb, d, e, tauq, taup, ws, s)) return 0;
  int ok = finite ? F(bdsdc)(k, d, e, uo ? ub : NULL, k, uo ? vb : NULL, k,
                             work, iwork)
                  : 0;
  if (ok < 0) return 0;
  if (!ok) {
    nans(so, k * (CPLX + 1), RDT);
    if (uo) nans(uo, m * uc * (CPLX + 1), RDT);
    if (vo) nans(vo, vr * n * (CPLX + 1), RDT);
    return 1;
  }
  for (int64_t j = 0; j < k; j++) so[j] = d[j] * (1 / sigma) + 0;
  if (!uo) return 1;
  T *xo = wide ? tmp : uo, *wo = wide ? uo : tmp;
  for (int64_t i = 0; i < mm; i++)
    for (int64_t j = 0; j < xc; j++)
      xo[i * xc + j] = i < k && j < k ? ub[i + j * k] : i == j;
  for (int64_t i = 0; i < k; i++)
    for (int64_t j = 0; j < k; j++) wo[i * k + j] = vb[i + j * k];
  /* Q is the bidiagonalisation's, after QR's where it came first; P's
     reflectors are B's rows right of the diagonal, conjugated. */
  if (qrf && !F(back)(r, 1, k, AS_STORED, tauq, k, k, k, xo, xc, ws, s))
    return 0;
  if (!F(back)(a, qrf ? k : 1, qrf ? 1 : mm, AS_STORED, qrf ? tau : tauq, k,
               mm, xc, xo, xc, ws, s) ||
      !F(back)(bd + lb, lb, 1, CONJUGATED, taup, k - 1, k - 1, k, wo + k, k,
               ws, s))
    return 0;
  const T *src = wide ? xo : wo;
  int64_t rows = wide ? vr : k;
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < n; j++) vo[i * n + j] = CONJ(src[j * rows + i]);
  return 1;
}

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
    for (int64_t i = 0; i < m; i++) s += CONJ(v[i]) * M(c, i, j);
    w[j] = tau * s;
  }
  for (int64_t j = 0; j < k; j++)
    for (int64_t i = 0; i < m; i++) M(c, i, j) -= v[i] * w[j];
}

/* c ← c (I − τ v vᴴ) for the [m] × [k] [c], columns [n] apart (xLARF from
   the right). [w] holds m elements. */
static void F(larf_right)(const T *v, T tau, int64_t m, int64_t k, T *c,
                          int64_t n, T *w) {
  if (tau == 0) return;
  for (int64_t i = 0; i < m; i++) w[i] = 0;
  for (int64_t j = 0; j < k; j++)
    for (int64_t i = 0; i < m; i++) w[i] += M(c, i, j) * v[j];
  for (int64_t j = 0; j < k; j++) {
    T t = tau * CONJ(v[j]);
    for (int64_t i = 0; i < m; i++) M(c, i, j) -= w[i] * t;
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
    for (int64_t j = i; j <= jhi; j++) M(h, i, j) *= sc;
    for (int64_t j = jlo; j <= (jhi < i + 1 ? jhi : i + 1); j++)
      M(h, j, i) *= CONJ(sc);
    for (int64_t j = lo; z && j <= hi; j++) M(z, j, i) *= CONJ(sc);
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
        T u = CSQRT(M(h, i - 1, i)) * CSQRT(M(h, i, i - 1));
        R s = F(magnitude)(u);
        if (s != 0) {
          T x = (M(h, i - 1, i - 1) - t) / 2;
          R sx = F(magnitude)(x);
          s = s > sx ? s : sx;
          T y = s * CSQRT((x / s) * (x / s) + (u / s) * (u / s));
          if (sx > 0 && REAL(x / sx) * REAL(y) + IMAG(x / sx) * IMAG(y) < 0)
            y = -y;
          t -= u * (u / (x + y));
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
        R t2 = REAL(t1 * v2);
        for (int64_t j = k2; j <= i2; j++) {
          T sum = CONJ(t1) * M(h, k2, j) + t2 * M(h, k2 + 1, j);
          M(h, k2, j) -= sum;
          M(h, k2 + 1, j) -= sum * v2;
        }
        for (int64_t j = i1; j <= (k2 + 2 < i ? k2 + 2 : i); j++) {
          T sum = t1 * M(h, j, k2) + t2 * M(h, j, k2 + 1);
          M(h, j, k2) -= sum;
          M(h, j, k2 + 1) -= sum * CONJ(v2);
        }
        for (int64_t j = lo; z && j <= hi; j++) {
          T sum = t1 * M(z, j, k2) + t2 * M(z, j, k2 + 1);
          M(z, j, k2) -= sum;
          M(z, j, k2 + 1) -= sum * CONJ(v2);
        }
        if (k2 == m && m > l) {
          /* H(m + 1, m) made real again, as the sweep started below l. */
          T x = 1 - t1;
          x /= F(lapy3)(REAL(x), IMAG(x), 0);
          M(h, m + 1, m) *= CONJ(x);
          if (m + 2 <= i) M(h, m + 2, m + 1) *= x;
          for (int64_t j = m; j <= i; j++) {
            if (j == m + 1) continue;
            for (int64_t p = j + 1; p <= i2; p++) M(h, j, p) *= x;
            for (int64_t p = i1; p < j; p++) M(h, p, j) *= CONJ(x);
            for (int64_t p = lo; z && p <= hi; p++) M(z, p, j) *= CONJ(x);
          }
        }
      }
      /* H(i, i − 1) made real. */
      T x = M(h, i, i - 1);
      if (IMAG(x) != 0) {
        R r = F(lapy3)(REAL(x), IMAG(x), 0);
        M(h, i, i - 1) = r;
        x /= r;
        for (int64_t p = i + 1; p <= i2; p++) M(h, i, p) *= CONJ(x);
        for (int64_t p = i1; p < i; p++) M(h, p, i) *= x;
        for (int64_t p = lo; z && p <= hi; p++) M(z, p, i) *= x;
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
        x[j] /= d;
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
          l21 = e[(1 - pr) * 2 + pc] / u11;
          u12 = e[pr * 2 + 1 - pc];
          u22 = e[(1 - pr) * 2 + 1 - pc] - l21 * u12;
          if (F(cabs1)(u22) < smin) u22 = smin;
        }
        C b1 = x[j0 + pr], b2 = x[j0 + 1 - pr] - l21 * b1;
        R bm = F(cabs1)(b1) > F(cabs1)(b2) ? F(cabs1)(b1) : F(cabs1)(b2);
        R um = F(cabs1)(u22) < F(cabs1)(u11) ? F(cabs1)(u22) : F(cabs1)(u11);
        if (um < 1 && bm > 1 && bm > bignum * um) {
          F(cscale)(x, ki, 1 / bm);
          b1 /= bm;
          b2 /= bm;
        }
        C y2 = b2 / u22, y1 = (b1 - u12 * y2) / u11;
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
        x[k] -= x[j] * M(s, k, j);
        if (block) x[k] -= x[j0] * M(s, k, j0);
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
  stage(c, 0, b, a, n, 1);
  int finite = 1;
  for (int64_t i = 0; i < nn; i++)
    finite &= REAL(a[i]) - REAL(a[i]) == 0 && IMAG(a[i]) - IMAG(a[i]) == 0;
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
    R big = 0;
    for (int64_t i = 0; i < nn; i++) {
      R re = F(abs)(REAL(a[i])), im = F(abs)(IMAG(a[i]));
      big = re > big ? re : big;
      big = im > big ? im : big;
    }
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
                hi - lo, n, v + (lo + 1) * n, n, ws, s)))
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
    for (int64_t i = 0; i < n; i++) v[i * n + j] *= t;
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
    stage(c, 0, b, f, lf, 0);
    if (!F(qr)(f, m, n, lf, tau, ts, rest, s)) return 0;
    for (int64_t i = 0; i < qc; i++)
      for (int64_t j = 0; j < n; j++) r[i * n + j] = j >= i ? f[i * lf + j] : 0;
    return F(form_q)(f, m, n, lf, ts, d, qc, qc, rest, s);
  }
  if (c->routine == LU) {
    stage(c, 0, b, d, n, 0);
    int64_t *pivots = (int64_t *)matrix_at(c, 2, b);
    int64_t *perm = (int64_t *)matrix_at(c, 3, b);
    return F(lu)(d, m, n, n, pivots, perm, s);
  }
  if (c->routine == CHOLESKY) {
    stage(c, 0, b, d, n, 0);
    int ok = F(cholesky)(d, n, n, s);
    if (ok < 0) return 0;
    if (ok == 0) {
      nans(d, n * n * (CPLX + 1), RDT);
      return 1;
    }
    /* The other triangle +0, or under Upper, U = Lᴴ in place. */
    for (int64_t i = 0; i < n; i++)
      for (int64_t j = i + 1; j < n; j++) {
        if (!c->upper) {
          d[i * n + j] = 0;
          continue;
        }
        d[i * n + j] = CONJ(d[j * n + i]);
        d[j * n + i] = 0;
      }
    return 1;
  }
  /* Solve_triangular: op(a) staged into w, x staged into d. */
  int64_t r = c->r;
  stage(c, 1, b, d, r, 0);
  stage(c, 0, b, w, n, c->adjoint);
  int upper = c->upper != c->adjoint;
#if CPLX
  if (c->adjoint)
    for (int64_t i = 0; i < n * n; i++) w[i] = CONJ(w[i]);
#endif
  if (!c->unit)
    for (int64_t i = 0; i < n; i++)
      if (w[i * n + i] == 0) {
        nans(d, n * r * (CPLX + 1), RDT);
        return 1;
      }
  solve kind = upper ? (c->unit ? UNIT_UPPER : UPPER)
                     : (c->unit ? UNIT_LOWER : LOWER);
  return F(solve)(kind, w, n, n, d, r, r, s);
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
#undef C
#undef CSQRT
#undef SAFMIN
#undef EPS
#undef TINY
#undef SMLSIZ
