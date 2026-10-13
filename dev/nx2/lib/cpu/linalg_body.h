/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* linalg.c's routines on matrices of one dtype, included once per dtype
   with T the element, R its real part's type, RDT that real type's dtype
   code, CPLX 1 for complex, and CONJ, REAL, IMAG and SQRT for T and R,
   which it undefines at its end. F(name) names the dtype's instance.

   Matrices are dense and row-major, row i at a + i·ld. Each routine runs
   by scalar loops on blocks of at most LEAF columns and updates the rest
   of the matrix by products (F(gemm)). */

/* Products */

/* c += sign · op(a) op(b) for c of [m] rows and [n] columns, rows [ldc]
   apart and its columns adjacent, and [k] products an element, by the
   contraction (product). Answers 0 if an allocation failed.

   On reals, the smaller of a and b is copied negated where sign is -1.
   On complex numbers, the product is a real one, with no more flops (the
   1m method, Van Zee and Smith, 2017): a's [m] rows of [k] complex
   numbers are [m] rows of [2k] reals, and c's are [m] rows of [2n] reals,
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
    for (int64_t i = 0; i < m; i++)
      for (int64_t p = 0; p < k; p++) w[i * k + p] = -ap[i * ars + p * acs];
    ap = w;
    ars = k;
    acs = 1;
  } else if (sign < 0) {
    T *w = room(s, k * n * (int64_t)sizeof(T));
    if (w == NULL) return 0;
    for (int64_t p = 0; p < k; p++)
      for (int64_t j = 0; j < n; j++) w[p * n + j] = -bp[p * brs + j * bcs];
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
   solved whole. gemm.c computes a product in chain order where its rows
   and columns are above LANE_MOST, 4, and its outputs at least
   CHAIN_OUTPUTS, 64: a unit's products have more than 8 rows, since the
   recursion splits above LEAF, and SOLVE_COLUMNS columns, so they keep it,
   and each column's arithmetic and bits are the whole solve's, whatever
   the units and threads.

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
  int64_t bytes = n * (n + r) * (int64_t)sizeof(T);
  double cost = (double)bytes + (double)n * n * r;
  int64_t units = nx_cpu_threads(bytes, cost);
  if (units > r / SOLVE_COLUMNS) units = r / SOLVE_COLUMNS;
  F(solve_job) j = {.l = l, .x = x, .n = n, .ld = ld, .r = r, .ldx = ldx,
                    .units = units, .kind = kind, .workers = s->workers};
  if (s->workers == NULL || units < 2) return F(columns)(&j, 0, r, s);
  nx_cpu_job(units, bytes, cost, F(solve_unit), &j);
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
        for (int64_t c = 0; c < n; c++) F(swap)(q + c * h + j, q + c * h + p, 1);
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
  ((R)(sizeof(R) == 4 ? FLT_MIN / (FLT_EPSILON / 2) : DBL_MIN / (DBL_EPSILON / 2)))

static R F(abs)(R x) { return x < 0 ? -x : x; }

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

   a = Q R by Householder reflectors, as LAPACK's blocked xGEQRF, in place
   in the m × n [a]: R on and above the diagonal, each reflector's v below
   it, its τ in [tau]. By blocks of NB columns: a block's columns by
   reflectors one at a time, each applied to the block's columns to its
   right (xGEQR2), on a contiguous copy; then the block's product H = I − V T Vᴴ (xLARFT)
   applied to the columns right of the block as Hᴴ = I − V Tᴴ Vᴴ by
   products (xLARFB). The scratch [q] holds the block's V, T, two NB × n
   products and a row: qr_scratch elements. Answers 1, or 0 if an
   allocation failed. */

/* T of the block of the [jb] reflectors in the dense [h] × jb [v]:
   T[i][i] = τ_i, T[0:i][i] = −τ_i T[0:i][0:i] V[:, 0:i]ᴴ v_i, zeros below
   (xLARFT), the products Vᴴ V by a product into [g], jb × jb. Answers 0
   if an allocation failed. */
static int F(block_t)(const T *v, int64_t h, int64_t jb, const T *tau, T *t,
                      T *g, arena *s) {
  memset(g, 0, sizeof(T) * jb * jb);
  if (!F(gemm)(jb, jb, h, 1, F(op_h)(v, jb), F(op)(v, jb, 1), g, jb, s))
    return 0;
  for (int64_t i = 0; i < jb; i++) {
    for (int64_t p = 0; p < i; p++) {
      T x = 0;
      for (int64_t q = p; q < i; q++) x += t[p * jb + q] * g[q * jb + i];
      t[p * jb + i] = -tau[i] * x;
    }
    t[i * jb + i] = tau[i];
    for (int64_t p = i + 1; p < jb; p++) t[p * jb + i] = 0;
  }
  return 1;
}

/* [v]'s [h] × jb reflectors in place, below its diagonal, made the dense
   V: ones on the diagonal, zeros above. */
static void F(dense_v)(T *v, int64_t h, int64_t jb) {
  for (int64_t r = 0; r < h && r < jb; r++) {
    for (int64_t p = r + 1; p < jb; p++) v[r * jb + p] = 0;
    v[r * jb + r] = 1;
  }
}

/* Copies [h] rows of [jb] elements, rows [ls] apart at [x], to [y], rows
   [ld] apart. */
static void F(rows)(const T *x, int64_t ls, int64_t h, int64_t jb, T *y,
                    int64_t ld) {
  for (int64_t r = 0; r < h; r++) memcpy(y + r * ld, x + r * ls, sizeof(T) * jb);
}

/* c ← (I − V op(T) Vᴴ) c for c of [m2] × [n2], rows [ldc] apart; [v] the
   dense m2 × jb V, [t] op(T): c −= V (op(T) (Vᴴ c)). [w] holds 2·jb·n2
   elements. */
static int F(larfb)(const T *v, operand t, int64_t jb, int64_t m2,
                    int64_t n2, T *c, int64_t ldc, T *w, arena *s) {
  T *w1 = w, *w2 = w + jb * n2;
  memset(w, 0, sizeof(T) * 2 * jb * n2);
  return F(gemm)(jb, n2, m2, 1, F(op_h)(v, jb), F(op)(c, ldc, 1), w1, n2,
                 s) &&
         F(gemm)(jb, n2, jb, 1, t, F(op)(w1, n2, 1), w2, n2, s) &&
         F(gemm)(m2, n2, jb, -1, F(op)(v, jb, 1), F(op)(w2, n2, 1), c, ldc,
                 s);
}

static int F(qr)(T *a, int64_t m, int64_t n, int64_t ld, T *tau, T *q,
                 arena *s) {
  int64_t k = m < n ? m : n;
  T *vb = q, *t = vb + m * NB, *w = t + NB * NB, *row = w + 2 * NB * n;
  for (int64_t j0 = 0; j0 < k; j0 += NB) {
    int64_t jb = NB < k - j0 ? NB : k - j0, j1 = j0 + jb, h = m - j0;
    /* The panel, contiguous: its rows a power of two apart would share one
       set of the cache. */
    F(rows)(a + j0 * ld + j0, ld, h, jb, vb, jb);
    for (int64_t j = 0; j < jb; j++) {
      T *ajj = vb + j * jb + j;
      tau[j0 + j] = F(reflector)(ajj, ajj + jb, h - j - 1, jb);
      F(reflect)(ajj, jb, CONJ(tau[j0 + j]), h - j, jb - j - 1, ajj + 1, jb,
                 row);
    }
    F(rows)(vb, jb, h, jb, a + j0 * ld + j0, ld);
    if (j1 == n) continue;
    F(dense_v)(vb, h, jb);
    if (!F(block_t)(vb, h, jb, tau + j0, t, w, s) ||
        !F(larfb)(vb, F(op_h)(t, jb), jb, h, n - j1, a + j0 * ld + j1, ld, w,
                  s))
      return 0;
  }
  return 1;
}

/* q ← the first [qc] columns of H_0 ⋯ H_(k−1), m × qc, rows [ldq] apart,
   from [qr]'s reflectors in [a]: from the identity, the blocks applied
   last first (LAPACK's xORGQR and xUNGQR). The scratch [w] as for [qr]. */
static int F(form_q)(const T *a, int64_t m, int64_t n, int64_t ld,
                     const T *tau, T *q, int64_t qc, int64_t ldq, T *w,
                     arena *s) {
  int64_t k = m < n ? m : n;
  T *vb = w, *t = vb + m * NB, *ww = t + NB * NB;
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < qc; j++) q[i * ldq + j] = i == j;
  for (int64_t j0 = k == 0 ? 0 : (k - 1) / NB * NB; k > 0 && j0 >= 0;
       j0 -= NB) {
    int64_t jb = NB < k - j0 ? NB : k - j0, h = m - j0;
    F(rows)(a + j0 * ld + j0, ld, h, jb, vb, jb);
    F(dense_v)(vb, h, jb);
    if (!F(block_t)(vb, h, jb, tau + j0, t, ww, s) ||
        !F(larfb)(vb, F(op)(t, jb, 1), jb, h, qc - j0, q + j0 * ldq + j0, ldq,
                  ww, s))
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

/* The panel of the [nb] columns of the [n] × [n] [a] by reflectors, as
   LAPACK's xLATRD (lower): column i, the products of the panel's earlier
   reflectors deferred until now applied to it, is reduced below its
   subdiagonal by H_i = I − τ_i v_i v_iᴴ, its subdiagonal into e[i], v_i
   below it with v_i[0] = 1 stored; [w], [n] × nb with columns [ldw]
   apart, gets W with the rest of a updated as a − V Wᴴ − W Vᴴ. Only the
   lower triangle is read, and diagonals' real parts. */
static void F(latrd)(T *a, int64_t n, int64_t nb, int64_t ld, R *e, T *tau,
                     T *w, int64_t ldw) {
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
    for (int64_t r = 0; r < h; r++) y[r] = 0;
    for (int64_t j = 0; j < h; j++) {
      const T *col = a + (i + 1 + j) * ld + i + 1;
      T vj = v[j], acc = REAL(col[j]) * vj;
      for (int64_t r = j + 1; r < h; r++) {
        y[r] += col[r] * vj;
        acc += CONJ(col[r]) * v[r];
      }
      y[j] += acc;
    }
    for (int k = 0; k < 2; k++) {
      /* x = Uᴴ v, then y −= U' x, for (U, U') = (W, V) then (V, W). */
      const T *u = k == 0 ? w : a, *u2 = k == 0 ? a : w;
      int64_t lu = k == 0 ? ldw : ld, lu2 = k == 0 ? ld : ldw;
      for (int64_t p = 0; p < i; p++) {
        const T *up = u + p * lu + i + 1;
        T s = 0;
        for (int64_t r = 0; r < h; r++) s += CONJ(up[r]) * v[r];
        x[p] = s;
      }
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
   triangle. The scratch [w] holds n × NB elements. Answers 0 if an
   allocation failed. */
static int F(hetrd)(T *a, int64_t n, int64_t ld, R *d, R *e, T *tau, T *w,
                    arena *s) {
  for (int64_t i0 = 0; i0 < n - 1; i0 += NB) {
    int64_t nb = NB < n - 1 - i0 ? NB : n - 1 - i0, h = n - i0, m = h - nb;
    T *p = a + i0 * ld + i0;
    F(latrd)(p, h, nb, ld, e + i0, tau + i0, w, h);
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

/* Columns [j0, j0 + k) of the [n]-row [z], columns [ldz] apart, times the
   rotations (c[j], s[j]) of columns j0 + j and j0 + j + 1, the last first
   under [backward] (xLASR's 'R', 'V'). */
static void F(rotate)(R *z, int64_t n, int64_t ldz, int64_t k, const R *c,
                      const R *s, int backward) {
  for (int64_t t = 0; t < k - 1; t++) {
    int64_t j = backward ? k - 2 - t : t;
    R ct = c[j], st = s[j];
    if (ct == 1 && st == 0) continue;
    R *x = z + j * ldz, *y = x + ldz;
    for (int64_t i = 0; i < n; i++) {
      R yi = y[i];
      y[i] = ct * yi - st * x[i];
      x[i] = st * yi + ct * x[i];
    }
  }
}

/* d[l..lend] and e[l..lend) times [to] / [from]. */
static void F(scale)(R *d, R *e, int64_t l, int64_t lend, R from, R to) {
  for (int64_t i = l; i <= lend; i++) {
    d[i] = d[i] / from * to;
    if (i < lend) e[i] = e[i] / from * to;
  }
}

/* The eigenvalues of the symmetric tridiagonal T of the diagonal [d] and
   subdiagonal [e], [n] of them, ascending into d, and, where [z] is not
   NULL, z Z for T = Z diag d Zᵀ, z of [nz] rows and n columns [nz] apart:
   implicit QL or QR with Wilkinson shifts, whichever chases the bulge
   from the end of larger magnitude, on each unreduced block scaled into
   range, as LAPACK's xSTEQR. Values alone run the same iteration, so they
   are the bits the vectors' call gives. [rot] holds 2n reals. Answers 0
   if the iteration takes more than 30 n sweeps. */
static int F(steqr)(int64_t n, R *d, R *e, R *z, int64_t nz, R *rot) {
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
          if (z) F(rotate)(z + l * nz, nz, nz, 2, &c, &s, 1);
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
        if (z) F(rotate)(z + l * nz, nz, nz, m - l + 1, cs + l, sn + l, 1);
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
          if (z) F(rotate)(z + (l - 1) * nz, nz, nz, 2, &c, &s, 0);
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
        if (z) F(rotate)(z + m * nz, nz, nz, l - m + 1, cs + m, sn + m, 0);
        d[l] -= p;
        e[l - 1] = g;
      }
    }
    if (sc != anorm) F(scale)(d, e, lsv, lendsv, sc, anorm);
  }
  /* Ascending, by selection as xSTEQR, z's columns with them. */
  for (int64_t i = 0; i < n - 1; i++) {
    int64_t k = i;
    for (int64_t j = i + 1; j < n; j++)
      if (d[j] < d[k]) k = j;
    if (k == i) continue;
    R t = d[k];
    d[k] = d[i];
    d[i] = t;
    if (z)
      for (int64_t r = 0; r < nz; r++) {
        t = z[k * nz + r];
        z[k * nz + r] = z[i * nz + r];
        z[i * nz + r] = t;
      }
  }
  return 1;
}

/* Eigh on matrix [b] of [c] in the scratch [x]: w, and v where the call
   has vectors, as LAPACK's xHEEV: the matrix scaled into [rmin, rmax]
   where its largest magnitude lies outside, reduced (hetrd), T's
   eigenvalues and Z by steqr, then v = Q Z by Q's blocks of NB reflectors
   applied last first (xUNMTR). A matrix that holds NaN or an infinity
   where it is read, or on which steqr does not converge, is NaN in both.
   Answers 0 if an allocation failed. */
static int F(eigh)(const call *c, int64_t b, T *x, arena *s) {
  int64_t n = c->n;
  T *a = x, *tau = a + n * n, *w = tau + n, *t = w + n * NB, *ww = t + NB * NB;
  R *d = (R *)(ww + 2 * NB * n), *e = d + n, *rot = e + n, *z = rot + 2 * n;
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
      R m = F(lapy3)(re, im, 0);
      big = m > big ? m : big;
    }
  }
  const R rmin = SQRT(TINY / (2 * EPS)), rmax = 1 / rmin;
  R sigma = big > 0 && big < rmin ? rmin / big : big > rmax ? rmax / big : 1;
  if (sigma != 1)
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = j; i < n; i++) a[j * n + i] *= sigma;
  if (finite && !F(hetrd)(a, n, n, d, e, tau, w, s)) return 0;
  if (v)
    for (int64_t i = 0; i < n * n; i++) z[i] = i % (n + 1) == 0;
  if (!finite || !F(steqr)(n, d, e, v ? z : NULL, n, rot)) {
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
  int64_t k = n - 1;
  for (int64_t j0 = k <= 0 ? -1 : (k - 1) / NB * NB; j0 >= 0; j0 -= NB) {
    int64_t jb = NB < k - j0 ? NB : k - j0, h = k - j0;
    for (int64_t r = 0; r < h; r++)
      for (int64_t p = 0; p < jb; p++)
        w[r * jb + p] = a[(1 + j0 + r) + (j0 + p) * n];
    F(dense_v)(w, h, jb);
    if (!F(block_t)(w, h, jb, tau + j0, t, ww, s) ||
        !F(larfb)(w, F(op)(t, jb, 1), jb, h, n, v + (1 + j0) * n, n, ww, s))
      return 0;
  }
  return 1;
}

/* Matrices of a call */

/* The routine on matrix [b] of the call [c], with the scratch [w] of
   linalg.c's own elements and the arena [s]. Answers 0 if an allocation
   failed. */
static int F(matrix)(const call *c, int64_t b, T *w, arena *s) {
  int64_t m = c->m, n = c->n;
  T *d = (T *)matrix_at(c, c->nops, b);
  if (c->routine == EIGH) return F(eigh)(c, b, w, s);
  if (c->routine == QR) {
    /* a factored in w, then R and Q out. */
    int64_t k = m < n ? m : n, qc = c->complete ? m : k;
    T *f = w, *tau = f + m * n, *rest = tau + k;
    T *r = (T *)matrix_at(c, c->nops + 1, b);
    stage(c, 0, b, f, n, 0);
    if (!F(qr)(f, m, n, n, tau, rest, s)) return 0;
    for (int64_t i = 0; i < qc; i++)
      for (int64_t j = 0; j < n; j++) r[i * n + j] = j >= i ? f[i * n + j] : 0;
    return F(form_q)(f, m, n, n, tau, d, qc, qc, rest, s);
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
#undef SAFMIN
#undef EPS
#undef TINY
