/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_eig.c — owned general (nonsymmetric) eigensolver, linalg tier 3.

   This file is fully self-contained: it owns its own balancing,
   Hessenberg reduction, QR iteration, and eigenvector back-substitution, sharing
   only the engine's extraction/dispatch/pool/status machinery (nx_c_engine.h).

   Interface contract (backend_intf.ml): eig ~vectors returns COMPLEX128
   eigenvalues ([batch..., n]) and, when vectors, COMPLEX128 eigenvectors
   ([batch..., n, n], columns = eigenvectors) — ALWAYS complex128 regardless of
   the input dtype (a real matrix can have complex eigenvalues). That output
   invariant is the key simplification: since we produce double _Complex anyway,
   we pack ANY input to one of two compute types and run one of two paths —
     * real input (f16/bf16/fp8/f32/f64): pack to double, run the REAL path;
     * complex input (c32/c64): pack to double _Complex, run the COMPLEX path.
   No per-dtype factorization tier, no traits X-macro (unlike nx_c_linalg.c, whose
   outputs keep the input dtype): just double and double _Complex.

   Algorithms
   ----------
   REAL path — a faithful port of the EISPACK matched set (the most-ported real
   eigensolver in existence), transcribed 1-based with (i-1) accessors so it can
   be audited line-for-line against the reference:
     balanc  — diagonal similarity by exact powers of two + isolated-eigenvalue
               permutation (dgebal-style); earns its keep on the graded fixtures.
     orthes  — Householder reduction to upper Hessenberg.
     ortran  — accumulate the Householder transforms into Z.
     hqr2    — Francis implicit DOUBLE-shift QR: trailing-2x2 shift, implicit
               first column, bulge chase, small-subdiagonal deflation,
               exceptional shifts at iterations 10 and 20, a 30*n total-iteration
               cap -> LA_ERR_NO_CONVERGE (never a silent wrong answer). 2x2 real
               Schur blocks yield exact complex-conjugate eigenvalue pairs; the
               back-substitution tail computes the quasi-triangular eigenvectors.
     balbak  — undo balancing on the eigenvectors.
   Real eigenvalues land in +0i; a conjugate pair lands as columns (j, j+1) of Z
   holding the real and imaginary parts, reassembled here into exact conjugate
   complex eigenvectors.

   COMPLEX path — native double _Complex throughout (C99 complex is far clearer
   than EISPACK's packed real/imag arithmetic), same convergence DISCIPLINE
   ported from zlahqr/comqr:
     balanc  — the same two-phase balance over complex entries.
     hess    — Householder reduction to upper Hessenberg, accumulating Z inline
               (zlarfg-style reflector: real beta, complex tau).
     qr      — explicit single-shift QR (no 2x2 blocks needed in complex
               arithmetic): Wilkinson shift from the trailing 2x2, Givens
               triangularize + RQ, small-subdiagonal deflation, exceptional
               shifts at 10 and 20, 30*(igh-low+1) cap -> LA_ERR_NO_CONVERGE.
     vec     — back-substitution on the triangular Schur form, then Z*x, then
               balbak.

   Eigenvectors are normalized to unit 2-norm (numpy's convention). No phase fix:
   the conformance/residual gate ‖Av-λv‖/(‖A‖‖v‖) is phase-invariant, and the
   eigenvalue-set gate is order-invariant. A defective (Jordan-block) matrix has
   correct eigenvalues but a deficient eigenvector basis — we return what the
   Schur back-substitution yields (a valid eigenpair for each column), matching
   LAPACK/numpy behavior; we do not detect or flag defectiveness.

   Errors follow the backend protocol: kernels return nx_c_status, the stub raises
   via nx_c.h's nx_c_raise / nx_c_raise_invalid (this TU never includes
   caml/fail.h). Non-convergence is RAISED, never inf-poisoned.

   Batches parallelize over the engine pool exactly like eigh: one
   nx_c_parallel_for over the batch, per-worker scratch pre-allocated under the
   lock and handed to the primitive as free_on_exit, bodies allocation-free and
   worker-indexed, per-worker error slots for NO_CONVERGE (nx_c_engine.h). */

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h" /* pool + status + nx_c_raise; nx_c.h comes with it */

/* Failure/precondition statuses (static strings; the stub maps them to
   exceptions). Distinct storage from nx_c_linalg.c's identically-worded strings —
   statuses are compared by content, never by pointer (nx_c.h). */
static const char EIG_ERR_NO_CONVERGE[] = "eigenvalue iteration did not converge";
static const char EIG_ERR_NOT_FLOAT[] = "eig requires a float or complex dtype";
static const char EIG_ERR_NOT_SQUARE[] = "matrix must be square";
static const char EIG_ERR_SHAPE[] = "operand shapes are incompatible";
static const char EIG_ERR_TOO_LARGE[] = "matrix dimension exceeds eig limit";

/* The EISPACK-faithful kernels index with plain int (like the reference's
   INTEGER), so an element offset (i-1)*n+(j-1) must fit int32: n*n-1 <= INT_MAX
   requires n <= 46340. The driver rejects larger n up front — a ~34 GB scratch
   allocation for such a matrix could otherwise SUCCEED and then overflow the
   index (silent corruption), so the bound is enforced, not merely assumed. */
#define EIG_MAX_N 46340

/* Per-worker error slots live on the driver stack; MUST be >= the engine pool
   cap (nx_c_engine.c NX_C_MAX_THREADS). Same literal-64 bound and reasoning as
   nx_c_linalg.c's LA_MAX_WORKERS; every driver also clamps nth to it. */
#define EIG_MAX_WORKERS 64

#define EIG_ALIGN(x) (((x) + 63) & ~(int64_t)63)

typedef double _Complex eig_cplx; /* == nx_c_complex64 */

static inline double eig_cabs1(eig_cplx z) {
  return fabs(creal(z)) + fabs(cimag(z));
}
static inline double eig_cnorm2(eig_cplx z) {
  return creal(z) * creal(z) + cimag(z) * cimag(z);
}

/* Smith's complex division (ar+i ai)/(br+i bi), overflow-guarded — the real
   path's back-substitution needs it exactly as EISPACK cdiv provides. */
static void eig_cdiv(double ar, double ai, double br, double bi, double *cr,
                     double *ci) {
  if (fabs(br) >= fabs(bi)) {
    double s = bi / br;
    double d = br + bi * s;
    *cr = (ar + ai * s) / d;
    *ci = (ai - ar * s) / d;
  } else {
    double s = br / bi;
    double d = br * s + bi;
    *cr = (ar * s + ai) / d;
    *ci = (ai * s - ar) / d;
  }
}

/* ── REAL PATH ─────────────────────────────────────────────────────────────

   Ported 1-based from EISPACK; array element A(i,j) is a[(i-1)*n+(j-1)], so the
   transcription tracks the reference and low/igh are 1-based throughout this
   section. The only 0-based touchpoints are unpack (fills a) and store (reads
   z/wr/wi) — both are plain linear memory that agrees on element (r,c) at
   [r*n+c] regardless of convention. */

/* Row j <-> m over columns [1,l], column j <-> m over rows [k,n]; record the
   isolation index. scale stores a 0-based-equivalent 1-based index j. */
static void eig_exch_r(double *a, int n, int l, int k, int j, int m,
                       double *scale) {
  scale[m - 1] = (double)j;
  if (j == m) return;
  for (int i = 1; i <= l; i++) {
    double f = a[(i - 1) * n + (j - 1)];
    a[(i - 1) * n + (j - 1)] = a[(i - 1) * n + (m - 1)];
    a[(i - 1) * n + (m - 1)] = f;
  }
  for (int i = k; i <= n; i++) {
    double f = a[(j - 1) * n + (i - 1)];
    a[(j - 1) * n + (i - 1)] = a[(m - 1) * n + (i - 1)];
    a[(m - 1) * n + (i - 1)] = f;
  }
}

static void eig_balanc_r(double *a, int n, int *plow, int *pigh, double *scale) {
#define A(i, j) a[((i) - 1) * n + ((j) - 1)]
  const double radix = 2.0, b2 = 4.0;
  int k = 1, l = n;
  /* push rows isolating an eigenvalue down (shrink l) */
  for (;;) {
    int iso = 0;
    for (int j = l; j >= 1; j--) {
      int ok = 1;
      for (int i = 1; i <= l; i++)
        if (i != j && A(j, i) != 0.0) {
          ok = 0;
          break;
        }
      if (ok) {
        iso = j;
        break;
      }
    }
    if (iso == 0) break;
    eig_exch_r(a, n, l, k, iso, l, scale);
    if (l == 1) {
      *plow = k;
      *pigh = l;
#undef A
      return;
    }
    l--;
  }
  /* push columns isolating an eigenvalue left (grow k) */
#define A(i, j) a[((i) - 1) * n + ((j) - 1)]
  for (;;) {
    int iso = 0;
    for (int j = k; j <= l; j++) {
      int ok = 1;
      for (int i = k; i <= l; i++)
        if (i != j && A(i, j) != 0.0) {
          ok = 0;
          break;
        }
      if (ok) {
        iso = j;
        break;
      }
    }
    if (iso == 0) break;
    eig_exch_r(a, n, l, k, iso, k, scale);
    k++;
  }
  for (int i = k; i <= l; i++) scale[i - 1] = 1.0;
  int noconv;
  do {
    noconv = 0;
    for (int i = k; i <= l; i++) {
      double c = 0.0, r = 0.0;
      for (int j = k; j <= l; j++)
        if (j != i) {
          c += fabs(A(j, i));
          r += fabs(A(i, j));
        }
      if (c == 0.0 || r == 0.0) continue;
      double g = r / radix, f = 1.0, s = c + r;
      while (c < g) {
        f *= radix;
        c *= b2;
      }
      g = r * radix;
      while (c >= g) {
        f /= radix;
        c /= b2;
      }
      if ((c + r) / f >= 0.95 * s) continue;
      g = 1.0 / f;
      scale[i - 1] *= f;
      noconv = 1;
      for (int j = k; j <= n; j++) A(i, j) *= g;
      for (int j = 1; j <= l; j++) A(j, i) *= f;
    }
  } while (noconv);
  *plow = k;
  *pigh = l;
#undef A
}

static void eig_orthes_r(double *a, int n, int low, int igh, double *ort) {
#define A(i, j) a[((i) - 1) * n + ((j) - 1)]
  int la = igh - 1;
  int kp1 = low + 1;
  if (la < kp1) return;
  for (int m = kp1; m <= la; m++) {
    double h = 0.0, scale = 0.0;
    ort[m - 1] = 0.0;
    for (int i = m; i <= igh; i++) scale += fabs(A(i, m - 1));
    if (scale == 0.0) continue;
    for (int i = igh; i >= m; i--) {
      ort[i - 1] = A(i, m - 1) / scale;
      h += ort[i - 1] * ort[i - 1];
    }
    double g = -copysign(sqrt(h), ort[m - 1]);
    h -= ort[m - 1] * g;
    ort[m - 1] -= g;
    for (int j = m; j <= n; j++) {
      double f = 0.0;
      for (int i = igh; i >= m; i--) f += ort[i - 1] * A(i, j);
      f /= h;
      for (int i = m; i <= igh; i++) A(i, j) -= f * ort[i - 1];
    }
    for (int i = 1; i <= igh; i++) {
      double f = 0.0;
      for (int j = igh; j >= m; j--) f += ort[j - 1] * A(i, j);
      f /= h;
      for (int j = m; j <= igh; j++) A(i, j) -= f * ort[j - 1];
    }
    ort[m - 1] = scale * ort[m - 1];
    A(m, m - 1) = scale * g;
  }
#undef A
}

static void eig_ortran_r(double *a, int n, int low, int igh, double *ort,
                         double *z) {
#define A(i, j) a[((i) - 1) * n + ((j) - 1)]
#define Z(i, j) z[((i) - 1) * n + ((j) - 1)]
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) Z(i, j) = (i == j) ? 1.0 : 0.0;
  int kl = igh - low - 1;
  if (kl < 1) return;
  for (int mm = 1; mm <= kl; mm++) {
    int mp = igh - mm;
    if (A(mp, mp - 1) == 0.0) continue;
    for (int i = mp + 1; i <= igh; i++) ort[i - 1] = A(i, mp - 1);
    for (int j = mp; j <= igh; j++) {
      double g = 0.0;
      for (int i = mp; i <= igh; i++) g += ort[i - 1] * Z(i, j);
      g = (g / ort[mp - 1]) / A(mp, mp - 1);
      for (int i = mp; i <= igh; i++) Z(i, j) += g * ort[i - 1];
    }
  }
#undef A
#undef Z
}

static nx_c_status eig_hqr2_r(double *h, int n, int low, int igh, double *wr,
                             double *wi, double *z) {
#define H(i, j) h[((i) - 1) * n + ((j) - 1)]
#define Z(i, j) z[((i) - 1) * n + ((j) - 1)]
#define WR(i) wr[(i) - 1]
#define WI(i) wi[(i) - 1]
  double norm = 0.0;
  int kk = 1;
  for (int i = 1; i <= n; i++) {
    for (int j = kk; j <= n; j++) norm += fabs(H(i, j));
    kk = i;
    if (i >= low && i <= igh) continue;
    WR(i) = H(i, i);
    WI(i) = 0.0;
  }

  int en = igh;
  double t = 0.0;
  int itn = 30 * n;
  double p = 0, q = 0, r = 0, s = 0, w = 0, x = 0, y = 0, zz = 0;

  while (en >= low) {
    int its = 0;
    int na = en - 1;
    int enm2 = na - 1;
    for (;;) {
      int l;
      for (l = en; l > low; l--) {
        s = fabs(H(l - 1, l - 1)) + fabs(H(l, l));
        if (s == 0.0) s = norm;
        double tst1 = s;
        double tst2 = tst1 + fabs(H(l, l - 1));
        if (tst2 == tst1) break;
      }
      x = H(en, en);
      if (l == en) { /* one real root */
        H(en, en) = x + t;
        WR(en) = H(en, en);
        WI(en) = 0.0;
        en = na;
        break;
      }
      y = H(na, na);
      w = H(en, na) * H(na, en);
      if (l == na) { /* two roots (2x2 block) */
        p = (y - x) / 2.0;
        q = p * p + w;
        zz = sqrt(fabs(q));
        H(en, en) = x + t;
        x = H(en, en);
        H(na, na) = y + t;
        if (q >= 0.0) { /* real pair */
          zz = p + copysign(zz, p);
          WR(na) = x + zz;
          WR(en) = WR(na);
          if (zz != 0.0) WR(en) = x - w / zz;
          WI(na) = 0.0;
          WI(en) = 0.0;
          x = H(en, na);
          s = fabs(x) + fabs(zz);
          p = x / s;
          q = zz / s;
          r = sqrt(p * p + q * q);
          p = p / r;
          q = q / r;
          for (int j = na; j <= n; j++) {
            zz = H(na, j);
            H(na, j) = q * zz + p * H(en, j);
            H(en, j) = q * H(en, j) - p * zz;
          }
          for (int i = 1; i <= en; i++) {
            zz = H(i, na);
            H(i, na) = q * zz + p * H(i, en);
            H(i, en) = q * H(i, en) - p * zz;
          }
          for (int i = low; i <= igh; i++) {
            zz = Z(i, na);
            Z(i, na) = q * zz + p * Z(i, en);
            Z(i, en) = q * Z(i, en) - p * zz;
          }
        } else { /* complex conjugate pair */
          WR(na) = x + p;
          WR(en) = x + p;
          WI(na) = zz;
          WI(en) = -zz;
        }
        en = enm2;
        break;
      }
      if (itn == 0) return EIG_ERR_NO_CONVERGE;
      if (its == 10 || its == 20) { /* exceptional shift */
        t += x;
        for (int i = low; i <= en; i++) H(i, i) -= x;
        s = fabs(H(en, na)) + fabs(H(na, enm2));
        x = 0.75 * s;
        y = x;
        w = -0.4375 * s * s;
      }
      its++;
      itn--;
      /* look for two consecutive small sub-diagonal elements */
      int m;
      for (m = enm2; m >= l; m--) {
        zz = H(m, m);
        r = x - zz;
        s = y - zz;
        p = (r * s - w) / H(m + 1, m) + H(m, m + 1);
        q = H(m + 1, m + 1) - zz - r - s;
        r = H(m + 2, m + 1);
        s = fabs(p) + fabs(q) + fabs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m == l) break;
        double tst1 = fabs(p) * (fabs(H(m - 1, m - 1)) + fabs(zz) +
                                 fabs(H(m + 1, m + 1)));
        double tst2 = tst1 + fabs(H(m, m - 1)) * (fabs(q) + fabs(r));
        if (tst2 == tst1) break;
      }
      int mp2 = m + 2;
      for (int i = mp2; i <= en; i++) {
        H(i, i - 2) = 0.0;
        if (i == mp2) continue;
        H(i, i - 3) = 0.0;
      }
      /* double QR step over rows l..en, columns m..en */
      for (int k = m; k <= na; k++) {
        int notlas = (k != na);
        if (k != m) {
          p = H(k, k - 1);
          q = H(k + 1, k - 1);
          r = 0.0;
          if (notlas) r = H(k + 2, k - 1);
          x = fabs(p) + fabs(q) + fabs(r);
          if (x == 0.0) continue;
          p = p / x;
          q = q / x;
          r = r / x;
        }
        s = copysign(sqrt(p * p + q * q + r * r), p);
        if (k == m) {
          if (l != m) H(k, k - 1) = -H(k, k - 1);
        } else {
          H(k, k - 1) = -s * x;
        }
        p = p + s;
        x = p / s;
        y = q / s;
        zz = r / s;
        q = q / p;
        r = r / p;
        if (!notlas) {
          for (int j = k; j <= n; j++) {
            p = H(k, j) + q * H(k + 1, j);
            H(k, j) -= p * x;
            H(k + 1, j) -= p * y;
          }
          int jmax = (en < k + 3) ? en : k + 3;
          for (int i = 1; i <= jmax; i++) {
            p = x * H(i, k) + y * H(i, k + 1);
            H(i, k) -= p;
            H(i, k + 1) -= p * q;
          }
          for (int i = low; i <= igh; i++) {
            p = x * Z(i, k) + y * Z(i, k + 1);
            Z(i, k) -= p;
            Z(i, k + 1) -= p * q;
          }
        } else {
          for (int j = k; j <= n; j++) {
            p = H(k, j) + q * H(k + 1, j) + r * H(k + 2, j);
            H(k, j) -= p * x;
            H(k + 1, j) -= p * y;
            H(k + 2, j) -= p * zz;
          }
          int jmax = (en < k + 3) ? en : k + 3;
          for (int i = 1; i <= jmax; i++) {
            p = x * H(i, k) + y * H(i, k + 1) + zz * H(i, k + 2);
            H(i, k) -= p;
            H(i, k + 1) -= p * q;
            H(i, k + 2) -= p * r;
          }
          for (int i = low; i <= igh; i++) {
            p = x * Z(i, k) + y * Z(i, k + 1) + zz * Z(i, k + 2);
            Z(i, k) -= p;
            Z(i, k + 1) -= p * q;
            Z(i, k + 2) -= p * r;
          }
        }
      }
      /* continue the its-loop for this en (go to 70) */
    }
  }

  /* All roots found. Back-substitute for the eigenvectors of the quasi-upper-
     triangular Schur form (stored in H), reusing H's upper part as scratch. */
  if (norm == 0.0) return NX_C_OK;
  for (int nn = 1; nn <= n; nn++) {
    en = n + 1 - nn;
    p = WR(en);
    q = WI(en);
    int na = en - 1;
    if (q == 0.0) { /* real vector */
      int m = en;
      H(en, en) = 1.0;
      if (na != 0) {
        for (int ii = 1; ii <= na; ii++) {
          int i = en - ii;
          w = H(i, i) - p;
          r = 0.0;
          for (int j = m; j <= en; j++) r += H(i, j) * H(j, en);
          if (WI(i) < 0.0) {
            zz = w;
            s = r;
            continue;
          }
          m = i;
          if (WI(i) == 0.0) {
            t = w;
            if (t == 0.0) {
              double tst1 = norm;
              t = tst1;
              double tst2;
              do {
                t = 0.01 * t;
                tst2 = norm + t;
              } while (tst2 > tst1);
            }
            H(i, en) = -r / t;
          } else { /* solve real 2x2 equations */
            x = H(i, i + 1);
            y = H(i + 1, i);
            q = (WR(i) - p) * (WR(i) - p) + WI(i) * WI(i);
            t = (x * s - zz * r) / q;
            H(i, en) = t;
            if (fabs(x) > fabs(zz))
              H(i + 1, en) = (-r - w * t) / x;
            else
              H(i + 1, en) = (-s - y * t) / zz;
          }
          t = fabs(H(i, en));
          if (t != 0.0) {
            double tst1 = t;
            double tst2 = tst1 + 1.0 / tst1;
            if (tst2 <= tst1)
              for (int j = i; j <= en; j++) H(j, en) /= t;
          }
        }
      }
    } else if (q < 0.0) { /* complex vector (last component chosen imaginary) */
      int m = na;
      if (fabs(H(en, na)) > fabs(H(na, en))) {
        H(na, na) = q / H(en, na);
        H(na, en) = -(H(en, en) - p) / H(en, na);
      } else {
        double cr, ci;
        eig_cdiv(0.0, -H(na, en), H(na, na) - p, q, &cr, &ci);
        H(na, na) = cr;
        H(na, en) = ci;
      }
      H(en, na) = 0.0;
      H(en, en) = 1.0;
      int enm2 = na - 1;
      if (enm2 != 0) {
        for (int ii = 1; ii <= enm2; ii++) {
          int i = na - ii;
          w = H(i, i) - p;
          double ra = 0.0, sa = 0.0;
          for (int j = m; j <= en; j++) {
            ra += H(i, j) * H(j, na);
            sa += H(i, j) * H(j, en);
          }
          if (WI(i) < 0.0) {
            zz = w;
            r = ra;
            s = sa;
            continue;
          }
          m = i;
          if (WI(i) == 0.0) {
            double cr, ci;
            eig_cdiv(-ra, -sa, w, q, &cr, &ci);
            H(i, na) = cr;
            H(i, en) = ci;
          } else { /* solve complex 2x2 equations */
            x = H(i, i + 1);
            y = H(i + 1, i);
            double vr = (WR(i) - p) * (WR(i) - p) + WI(i) * WI(i) - q * q;
            double vi = (WR(i) - p) * 2.0 * q;
            if (vr == 0.0 && vi == 0.0) {
              double tst1 =
                  norm * (fabs(w) + fabs(q) + fabs(x) + fabs(y) + fabs(zz));
              vr = tst1;
              double tst2;
              do {
                vr = 0.01 * vr;
                tst2 = tst1 + vr;
              } while (tst2 > tst1);
            }
            double cr, ci;
            eig_cdiv(x * r - zz * ra + q * sa, x * s - zz * sa - q * ra, vr, vi,
                     &cr, &ci);
            H(i, na) = cr;
            H(i, en) = ci;
            if (fabs(x) > fabs(zz) + fabs(q)) {
              H(i + 1, na) = (-ra - w * H(i, na) + q * H(i, en)) / x;
              H(i + 1, en) = (-sa - w * H(i, en) - q * H(i, na)) / x;
            } else {
              eig_cdiv(-r - y * H(i, na), -s - y * H(i, en), zz, q, &cr, &ci);
              H(i + 1, na) = cr;
              H(i + 1, en) = ci;
            }
          }
          double tmax = fmax(fabs(H(i, na)), fabs(H(i, en)));
          if (tmax != 0.0) {
            double tst1 = tmax;
            double tst2 = tst1 + 1.0 / tst1;
            if (tst2 <= tst1)
              for (int j = i; j <= en; j++) {
                H(j, na) /= tmax;
                H(j, en) /= tmax;
              }
          }
        }
      }
    }
  }
  /* vectors of isolated roots */
  for (int i = 1; i <= n; i++) {
    if (i >= low && i <= igh) continue;
    for (int j = i; j <= n; j++) Z(i, j) = H(i, j);
  }
  /* multiply by the accumulated transformation to give vectors of the balanced
     matrix */
  for (int jj = low; jj <= n; jj++) {
    int j = n + low - jj;
    int m = (j < igh) ? j : igh;
    for (int i = low; i <= igh; i++) {
      zz = 0.0;
      for (int k = low; k <= m; k++) zz += Z(i, k) * H(k, j);
      Z(i, j) = zz;
    }
  }
  return NX_C_OK;
#undef H
#undef Z
#undef WR
#undef WI
}

static void eig_balbak_r(double *z, int n, int low, int igh,
                         const double *scale) {
  /* When igh==low the active block is a single element that never entered the
     norm-reduction loop, so scale[low] holds a permutation index, not a scale
     factor — skip the scaling (EISPACK balbak's `if igh==low`). */
  if (igh != low)
    for (int i = low; i <= igh; i++) {
      double s = scale[i - 1];
      for (int j = 1; j <= n; j++) z[(i - 1) * n + (j - 1)] *= s;
    }
  for (int ii = 1; ii <= n; ii++) {
    int i = ii;
    if (i >= low && i <= igh) continue;
    if (i < low) i = low - ii;
    int kk = (int)scale[i - 1];
    if (kk == i) continue;
    for (int j = 1; j <= n; j++) {
      double f = z[(i - 1) * n + (j - 1)];
      z[(i - 1) * n + (j - 1)] = z[(kk - 1) * n + (j - 1)];
      z[(kk - 1) * n + (j - 1)] = f;
    }
  }
}

/* ── COMPLEX PATH (native double _Complex, 0-based) ────────────────────────*/

static void eig_exch_c(eig_cplx *a, int n, int l, int k, int j, int m,
                       double *scale) {
  scale[m] = (double)j;
  if (j == m) return;
  for (int i = 0; i <= l; i++) {
    eig_cplx f = a[i * n + j];
    a[i * n + j] = a[i * n + m];
    a[i * n + m] = f;
  }
  for (int i = k; i < n; i++) {
    eig_cplx f = a[j * n + i];
    a[j * n + i] = a[m * n + i];
    a[m * n + i] = f;
  }
}

static void eig_balanc_c(eig_cplx *a, int n, int *plow, int *pigh,
                         double *scale) {
  const double radix = 2.0, b2 = 4.0;
  int k = 0, l = n - 1;
  for (;;) {
    int iso = -1;
    for (int j = l; j >= 0; j--) {
      int ok = 1;
      for (int i = 0; i <= l; i++)
        if (i != j && (creal(a[j * n + i]) != 0.0 || cimag(a[j * n + i]) != 0.0)) {
          ok = 0;
          break;
        }
      if (ok) {
        iso = j;
        break;
      }
    }
    if (iso < 0) break;
    eig_exch_c(a, n, l, k, iso, l, scale);
    if (l == 0) {
      *plow = k;
      *pigh = l;
      return;
    }
    l--;
  }
  for (;;) {
    int iso = -1;
    for (int j = k; j <= l; j++) {
      int ok = 1;
      for (int i = k; i <= l; i++)
        if (i != j && (creal(a[i * n + j]) != 0.0 || cimag(a[i * n + j]) != 0.0)) {
          ok = 0;
          break;
        }
      if (ok) {
        iso = j;
        break;
      }
    }
    if (iso < 0) break;
    eig_exch_c(a, n, l, k, iso, k, scale);
    k++;
  }
  for (int i = k; i <= l; i++) scale[i] = 1.0;
  int noconv;
  do {
    noconv = 0;
    for (int i = k; i <= l; i++) {
      double c = 0.0, r = 0.0;
      for (int j = k; j <= l; j++)
        if (j != i) {
          c += eig_cabs1(a[j * n + i]);
          r += eig_cabs1(a[i * n + j]);
        }
      if (c == 0.0 || r == 0.0) continue;
      double g = r / radix, f = 1.0, s = c + r;
      while (c < g) {
        f *= radix;
        c *= b2;
      }
      g = r * radix;
      while (c >= g) {
        f /= radix;
        c /= b2;
      }
      if ((c + r) / f >= 0.95 * s) continue;
      g = 1.0 / f;
      scale[i] *= f;
      noconv = 1;
      for (int j = k; j < n; j++) a[i * n + j] *= g;
      for (int j = 0; j <= l; j++) a[j * n + i] *= f;
    }
  } while (noconv);
  *plow = k;
  *pigh = l;
}

/* Householder reduction of the [low,igh] window to upper Hessenberg, native
   complex, accumulating the unitary Z (so eigenvector-of-A = Z * eigenvector-of-
   Hessenberg). Reflector per zlarfg: beta real, tau complex, v[0]=1; v is the
   caller's n-length workspace.

   Convention (the load-bearing detail): zlarfg builds H = I - tau v vᴴ such that
   Hᴴ x = beta e1 — it is Hᴴ, not H, that annihilates the subcolumn. So the
   similarity is A <- Hᴴ A H: the LEFT application (which zeros the column) uses
   Hᴴ = I - conj(tau) v vᴴ, and the RIGHT application and the Z accumulation use
   H = I - tau v vᴴ. Z accumulates Q = H_{low+1} ... H_{igh-1} (right-multiplied),
   giving A_hess = Qᴴ A Q, hence eigenvector-of-A = Z * eigenvector-of-A_hess.
   This matches nx_c_linalg.c's complex QR (conj(tau) to reduce, tau to form Q).
   Swapping the two conjugations breaks the similarity and silently corrupts the
   spectrum of any non-Hessenberg complex input (the eig-review catch). */
static void eig_hess_c(eig_cplx *a, int n, int low, int igh, eig_cplx *z,
                       eig_cplx *v) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) z[i * n + j] = (i == j) ? 1.0 : 0.0;
  for (int m = low + 1; m <= igh - 1; m++) {
    int pcol = m - 1;
    double xnorm2 = 0.0;
    for (int i = m + 1; i <= igh; i++) xnorm2 += eig_cnorm2(a[i * n + pcol]);
    eig_cplx alpha = a[m * n + pcol];
    double alphr = creal(alpha), alphi = cimag(alpha);
    if (xnorm2 == 0.0) continue; /* column already reduced */
    double anorm = sqrt(eig_cnorm2(alpha) + xnorm2);
    double beta = (alphr >= 0.0) ? -anorm : anorm;
    eig_cplx tau = CMPLX((beta - alphr) / beta, -alphi / beta);
    eig_cplx scal = alpha - beta;
    v[m] = 1.0;
    for (int i = m + 1; i <= igh; i++) v[i] = a[i * n + pcol] / scal;
    a[m * n + pcol] = CMPLX(beta, 0.0);
    for (int i = m + 1; i <= igh; i++) a[i * n + pcol] = 0.0;
    /* Hᴴ A (zeros the subcolumn): rows [m,igh], columns [m,n). Hᴴ uses
       conj(tau). */
    for (int j = m; j < n; j++) {
      eig_cplx s = 0.0;
      for (int i = m; i <= igh; i++) s += conj(v[i]) * a[i * n + j];
      s *= conj(tau);
      for (int i = m; i <= igh; i++) a[i * n + j] -= v[i] * s;
    }
    /* (Hᴴ A) H : columns [m,igh], rows [0,igh]. H uses tau. */
    for (int i = 0; i <= igh; i++) {
      eig_cplx s = 0.0;
      for (int jj = m; jj <= igh; jj++) s += a[i * n + jj] * v[jj];
      s *= tau;
      for (int jj = m; jj <= igh; jj++) a[i * n + jj] -= s * conj(v[jj]);
    }
    /* Z <- Z H : columns [m,igh], all rows. H uses tau. */
    for (int i = 0; i < n; i++) {
      eig_cplx s = 0.0;
      for (int jj = m; jj <= igh; jj++) s += z[i * n + jj] * v[jj];
      s *= tau;
      for (int jj = m; jj <= igh; jj++) z[i * n + jj] -= s * conj(v[jj]);
    }
  }
}

/* Explicit single-shift QR on the complex Hessenberg H, accumulating Z. On
   success H is upper triangular (Schur form), w holds the diagonal eigenvalues.
   cs (double) and sn (complex) are n-length rotation workspaces. */
static nx_c_status eig_qr_c(eig_cplx *h, int n, int low, int igh, eig_cplx *w,
                           eig_cplx *z, double *cs, eig_cplx *sn) {
  for (int i = 0; i < n; i++)
    if (i < low || i > igh) w[i] = h[i * n + i];
  double norm = 0.0;
  for (int i = 0; i < n; i++)
    for (int j = (i > 0 ? i - 1 : 0); j < n; j++) norm += eig_cabs1(h[i * n + j]);

  int en = igh;
  int itn = 30 * (igh - low + 1);
  if (itn < 30) itn = 30;
  while (en >= low) {
    int its = 0;
    for (;;) {
      int l;
      for (l = en; l > low; l--) {
        double sdd =
            eig_cabs1(h[(l - 1) * n + (l - 1)]) + eig_cabs1(h[l * n + l]);
        if (sdd == 0.0) sdd = norm;
        if (eig_cabs1(h[l * n + (l - 1)]) <= DBL_EPSILON * sdd) break;
      }
      if (l == en) {
        w[en] = h[en * n + en];
        en--;
        break;
      }
      if (itn == 0) return EIG_ERR_NO_CONVERGE;
      eig_cplx mu;
      if (its == 10 || its == 20) {
        double bump = eig_cabs1(h[en * n + (en - 1)]);
        if (en - 1 > low) bump += eig_cabs1(h[(en - 1) * n + (en - 2)]);
        mu = h[en * n + en] + bump;
      } else {
        eig_cplx a11 = h[(en - 1) * n + (en - 1)], a12 = h[(en - 1) * n + en];
        eig_cplx a21 = h[en * n + (en - 1)], a22 = h[en * n + en];
        eig_cplx tr = a11 + a22;
        eig_cplx det = a11 * a22 - a12 * a21;
        eig_cplx disc = csqrt(tr * tr - 4.0 * det);
        eig_cplx r1 = (tr + disc) / 2.0, r2 = (tr - disc) / 2.0;
        mu = (cabs(r1 - a22) <= cabs(r2 - a22)) ? r1 : r2;
      }
      its++;
      itn--;
      /* explicit shifted QR step on block [l,en] */
      for (int i = l; i <= en; i++) h[i * n + i] -= mu;
      for (int k = l; k < en; k++) {
        eig_cplx aa = h[k * n + k], bb = h[(k + 1) * n + k];
        double amag = cabs(aa);
        double d = hypot(amag, cabs(bb));
        double c;
        eig_cplx s;
        if (d == 0.0) {
          c = 1.0;
          s = 0.0;
        } else if (amag == 0.0) {
          c = 0.0;
          s = conj(bb) / cabs(bb);
        } else {
          c = amag / d;
          s = (aa / amag) * conj(bb) / d;
        }
        cs[k] = c;
        sn[k] = s;
        for (int j = k; j < n; j++) {
          eig_cplx t1 = h[k * n + j], t2 = h[(k + 1) * n + j];
          h[k * n + j] = c * t1 + s * t2;
          h[(k + 1) * n + j] = -conj(s) * t1 + c * t2;
        }
      }
      for (int k = l; k < en; k++) {
        double c = cs[k];
        eig_cplx s = sn[k];
        int hi = (k + 2 < en) ? k + 2 : en;
        for (int i = 0; i <= hi; i++) {
          eig_cplx t1 = h[i * n + k], t2 = h[i * n + (k + 1)];
          h[i * n + k] = c * t1 + conj(s) * t2;
          h[i * n + (k + 1)] = -s * t1 + c * t2;
        }
        for (int i = 0; i < n; i++) {
          eig_cplx t1 = z[i * n + k], t2 = z[i * n + (k + 1)];
          z[i * n + k] = c * t1 + conj(s) * t2;
          z[i * n + (k + 1)] = -s * t1 + c * t2;
        }
      }
      for (int i = l; i <= en; i++) h[i * n + i] += mu;
    }
  }
  return NX_C_OK;
}

/* Undo balancing on the complex eigenvectors: scale rows [low,igh] by scale[i],
   then reverse the isolation permutation (same order as eig_balbak_r). */
static void eig_balbak_c(eig_cplx *z, int n, int low, int igh,
                         const double *scale) {
  /* igh==low: scale[low] is a permutation index, not a scale factor (see
     eig_balbak_r). */
  if (igh != low)
    for (int i = low; i <= igh; i++) {
      double s = scale[i];
      for (int j = 0; j < n; j++) z[i * n + j] *= s;
    }
  for (int ii = 0; ii < n; ii++) {
    int i;
    if (ii >= low && ii <= igh) continue;
    if (ii < low)
      i = low - ii - 1;
    else
      i = ii;
    int kk = (int)scale[i];
    if (kk == i) continue;
    for (int j = 0; j < n; j++) {
      eig_cplx f = z[i * n + j];
      z[i * n + j] = z[kk * n + j];
      z[kk * n + j] = f;
    }
  }
}

/* ── Storage dtype -> compute buffer unpack (dispatch on the input dtype) ───*/

#define EIG_UNPACK_R(sfx, storage)                                             \
  static void eig_unpack_r_##sfx(const char *src, int64_t rs, int64_t cs,      \
                                 int64_t n, double *dst) {                      \
    int64_t esz = (int64_t)sizeof(storage);                                    \
    for (int64_t i = 0; i < n; i++)                                            \
      for (int64_t k = 0; k < n; k++)                                          \
        dst[i * n + k] = (double)nx_c_ld_##sfx(src + (i * rs + k * cs) * esz);  \
  }
#define EIG_UNPACK_C(sfx, storage)                                             \
  static void eig_unpack_c_##sfx(const char *src, int64_t rs, int64_t cs,      \
                                 int64_t n, eig_cplx *dst) {                    \
    int64_t esz = (int64_t)sizeof(storage);                                    \
    for (int64_t i = 0; i < n; i++)                                            \
      for (int64_t k = 0; k < n; k++)                                          \
        dst[i * n + k] = (eig_cplx)nx_c_ld_##sfx(src + (i * rs + k * cs) * esz); \
  }
#define EIG_UNP_NX_C_CAT_FLOAT(sfx, storage) EIG_UNPACK_R(sfx, storage)
#define EIG_UNP_NX_C_CAT_COMPLEX(sfx, storage) EIG_UNPACK_C(sfx, storage)
#define EIG_UNP_NX_C_CAT_SINT(sfx, storage)
#define EIG_UNP_NX_C_CAT_UINT(sfx, storage)
#define EIG_UNP_NX_C_CAT_BOOL(sfx, storage)
#define EIG_UNP_ROW(sfx, kind, storage, compute, ld, st, cat)                  \
  EIG_UNP_##cat(sfx, storage)
NX_C_FOR_EACH_COMPUTE_DTYPE(EIG_UNP_ROW)
#undef EIG_UNP_ROW
#undef EIG_UNP_NX_C_CAT_FLOAT
#undef EIG_UNP_NX_C_CAT_COMPLEX
#undef EIG_UNP_NX_C_CAT_SINT
#undef EIG_UNP_NX_C_CAT_UINT
#undef EIG_UNP_NX_C_CAT_BOOL
#undef EIG_UNPACK_R
#undef EIG_UNPACK_C

typedef void (*eig_unpack_r_fn)(const char *, int64_t, int64_t, int64_t,
                                double *);
typedef void (*eig_unpack_c_fn)(const char *, int64_t, int64_t, int64_t,
                                eig_cplx *);

static const eig_unpack_r_fn eig_unpack_r[NX_C_DTYPE_COUNT] = {
    [NX_C_DTYPE_f16] = eig_unpack_r_f16,
    [NX_C_DTYPE_bf16] = eig_unpack_r_bf16,
    [NX_C_DTYPE_f8e4m3] = eig_unpack_r_f8e4m3,
    [NX_C_DTYPE_f8e5m2] = eig_unpack_r_f8e5m2,
    [NX_C_DTYPE_f32] = eig_unpack_r_f32,
    [NX_C_DTYPE_f64] = eig_unpack_r_f64,
};
static const eig_unpack_c_fn eig_unpack_c[NX_C_DTYPE_COUNT] = {
    [NX_C_DTYPE_c32] = eig_unpack_c_c32,
    [NX_C_DTYPE_c64] = eig_unpack_c_c64,
};

/* Is dt one we can eig? Real floats and complex; everything else is rejected
   before the blocking section (the frontend's check_float_or_complex normally
   catches it, but the driver is the single owner of the guarantee). */
static int eig_supported(nx_c_dtype dt) {
  return eig_unpack_r[dt] != NULL || eig_unpack_c[dt] != NULL;
}

/* ── Batched pooled driver ─────────────────────────────────────────────────*/

typedef struct {
  const nx_c_ndarray *in;
  const nx_c_ndarray *w;  /* complex128 [batch..., n] */
  const nx_c_ndarray *v;  /* complex128 [batch..., n, n], only when vectors */
  nx_c_dtype dt;
  int is_complex;
  int vectors;
  int64_t n, esz;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *in_bs;
  const int64_t *w_bs;
  const int64_t *v_bs;
  int64_t in_rs, in_cs, w_cs, v_rs, v_cs;
  char *scratch;
  int64_t stride;
  /* per-worker sub-buffer offsets within a scratch slot */
  int64_t off_a, off_z, off_wr, off_wi, off_scale, off_ort, off_cvec;
  int64_t off_cw, off_cV, off_cy, off_cvh, off_csn, off_ccs;
  nx_c_status *werr;
} eig_ctx;

static void eig_batch_base(int64_t bt, int nd, const int64_t *bshape,
                           const int64_t *stride, int64_t off, int64_t esz,
                           const char *data, const char **out) {
  int64_t o = off;
  int64_t rem = bt;
  for (int i = nd - 1; i >= 0; i--) {
    int64_t q = rem % bshape[i];
    rem /= bshape[i];
    o += q * stride[i];
  }
  *out = data + o * esz;
}

/* Normalize a complex column to unit 2-norm and store it as column j of the
   eigenvector output (complex128). */
static void eig_store_vcol(const eig_ctx *x, const char *vb, int64_t j,
                           eig_cplx *col) {
  int64_t n = x->n;
  double nrm2 = 0.0;
  for (int64_t i = 0; i < n; i++) nrm2 += eig_cnorm2(col[i]);
  double inv = (nrm2 > 0.0) ? 1.0 / sqrt(nrm2) : 1.0;
  for (int64_t i = 0; i < n; i++) {
    eig_cplx z = col[i] * inv;
    *(nx_c_complex64 *)(vb + (i * x->v_rs + j * x->v_cs) * 16) =
        (nx_c_complex64)z;
  }
}

static void eig_real_body(const eig_ctx *x, int worker, int64_t bt,
                          nx_c_status *slot) {
  int64_t n = x->n;
  int ni = (int)n;
  char *base = x->scratch + (int64_t)worker * x->stride;
  double *a = (double *)(base + x->off_a);
  double *z = (double *)(base + x->off_z);
  double *wr = (double *)(base + x->off_wr);
  double *wi = (double *)(base + x->off_wi);
  double *scale = (double *)(base + x->off_scale);
  double *ort = (double *)(base + x->off_ort);
  eig_cplx *cvec = (eig_cplx *)(base + x->off_cvec);

  const char *inb;
  eig_batch_base(bt, x->batch_nd, x->bshape, x->in_bs, x->in->offset, x->esz,
                 (const char *)x->in->data, &inb);
  eig_unpack_r[x->dt](inb, x->in_rs, x->in_cs, n, a);

  int low, igh;
  eig_balanc_r(a, ni, &low, &igh, scale);
  eig_orthes_r(a, ni, low, igh, ort);
  eig_ortran_r(a, ni, low, igh, ort, z);
  nx_c_status s = eig_hqr2_r(a, ni, low, igh, wr, wi, z);
  if (s != NX_C_OK) {
    if (*slot == NX_C_OK) *slot = s;
    return;
  }
  eig_balbak_r(z, ni, low, igh, scale);

  const char *wb;
  eig_batch_base(bt, x->batch_nd, x->bshape, x->w_bs, x->w->offset, 16,
                 (const char *)x->w->data, &wb);
  for (int64_t j = 0; j < n; j++)
    *(nx_c_complex64 *)(wb + j * x->w_cs * 16) = (nx_c_complex64)CMPLX(wr[j], wi[j]);

  if (!x->vectors) return;
  const char *vb;
  eig_batch_base(bt, x->batch_nd, x->bshape, x->v_bs, x->v->offset, 16,
                 (const char *)x->v->data, &vb);
  for (int64_t j = 0; j < n; j++) {
    double im = wi[j];
    if (im == 0.0)
      for (int64_t i = 0; i < n; i++) cvec[i] = CMPLX(z[i * n + j], 0.0);
    else if (im > 0.0)
      for (int64_t i = 0; i < n; i++)
        cvec[i] = CMPLX(z[i * n + j], z[i * n + j + 1]);
    else
      for (int64_t i = 0; i < n; i++)
        cvec[i] = CMPLX(z[i * n + j - 1], -z[i * n + j]);
    eig_store_vcol(x, vb, j, cvec);
  }
}

static void eig_cplx_body(const eig_ctx *x, int worker, int64_t bt,
                          nx_c_status *slot) {
  int64_t n = x->n;
  int ni = (int)n;
  char *base = x->scratch + (int64_t)worker * x->stride;
  eig_cplx *a = (eig_cplx *)(base + x->off_a);
  eig_cplx *z = (eig_cplx *)(base + x->off_z);
  eig_cplx *w = (eig_cplx *)(base + x->off_cw);
  eig_cplx *V = (eig_cplx *)(base + x->off_cV);
  eig_cplx *y = (eig_cplx *)(base + x->off_cy);
  eig_cplx *vh = (eig_cplx *)(base + x->off_cvh);
  eig_cplx *sn = (eig_cplx *)(base + x->off_csn);
  double *scale = (double *)(base + x->off_scale);
  double *ccs = (double *)(base + x->off_ccs);

  const char *inb;
  eig_batch_base(bt, x->batch_nd, x->bshape, x->in_bs, x->in->offset, x->esz,
                 (const char *)x->in->data, &inb);
  eig_unpack_c[x->dt](inb, x->in_rs, x->in_cs, n, a);

  int low, igh;
  eig_balanc_c(a, ni, &low, &igh, scale);
  eig_hess_c(a, ni, low, igh, z, vh);
  nx_c_status s = eig_qr_c(a, ni, low, igh, w, z, ccs, sn);
  if (s != NX_C_OK) {
    if (*slot == NX_C_OK) *slot = s;
    return;
  }

  const char *wb;
  eig_batch_base(bt, x->batch_nd, x->bshape, x->w_bs, x->w->offset, 16,
                 (const char *)x->w->data, &wb);
  for (int64_t j = 0; j < n; j++)
    *(nx_c_complex64 *)(wb + j * x->w_cs * 16) = (nx_c_complex64)w[j];

  if (!x->vectors) return;
  /* eigenvectors: back-substitute on the triangular Schur form, then V = Z x */
  double norm = 0.0;
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = i; j < n; j++) norm += eig_cabs1(a[i * n + j]);
  if (norm == 0.0) norm = 1.0;
  for (int64_t k = 0; k < n; k++) {
    eig_cplx lambda = w[k];
    y[k] = 1.0;
    for (int64_t j = k - 1; j >= 0; j--) {
      eig_cplx sum = 0.0;
      for (int64_t m = j + 1; m <= k; m++) sum += a[j * n + m] * y[m];
      eig_cplx denom = a[j * n + j] - lambda;
      if (eig_cabs1(denom) < DBL_EPSILON * norm)
        denom = CMPLX(DBL_EPSILON * norm, 0.0);
      y[j] = -sum / denom;
    }
    for (int64_t i = 0; i < n; i++) {
      eig_cplx acc = 0.0;
      for (int64_t m = 0; m <= k; m++) acc += z[i * n + m] * y[m];
      V[i * n + k] = acc;
    }
  }
  eig_balbak_c(V, ni, low, igh, scale);

  const char *vb;
  eig_batch_base(bt, x->batch_nd, x->bshape, x->v_bs, x->v->offset, 16,
                 (const char *)x->v->data, &vb);
  for (int64_t j = 0; j < n; j++) {
    for (int64_t i = 0; i < n; i++) y[i] = V[i * n + j];
    eig_store_vcol(x, vb, j, y);
  }
}

static void eig_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  eig_ctx *x = (eig_ctx *)vctx;
  nx_c_status *slot = &x->werr[worker];
  for (int64_t bt = lo; bt < hi; bt++) {
    if (x->is_complex)
      eig_cplx_body(x, worker, bt, slot);
    else
      eig_real_body(x, worker, bt, slot);
  }
}

static nx_c_status nx_c_eig_run(const nx_c_ndarray *in, const nx_c_ndarray *w,
                              const nx_c_ndarray *v, nx_c_dtype dt, int vectors) {
  if (in->ndim < 2) return EIG_ERR_SHAPE;
  int64_t n = in->shape[in->ndim - 1];
  if (in->shape[in->ndim - 2] != n) return EIG_ERR_NOT_SQUARE;
  if (n > EIG_MAX_N) return EIG_ERR_TOO_LARGE;
  if (w->ndim != in->ndim - 1 || w->shape[w->ndim - 1] != n) return EIG_ERR_SHAPE;
  if (vectors &&
      (v->ndim != in->ndim || v->shape[v->ndim - 1] != n ||
       v->shape[v->ndim - 2] != n))
    return EIG_ERR_SHAPE;
  if (!eig_supported(dt)) return EIG_ERR_NOT_FLOAT;
  int is_complex = nx_c_dtype_is_complex(dt);
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = in->ndim - 2;
  int64_t bshape[NX_C_MAX_NDIM], in_bs[NX_C_MAX_NDIM], w_bs[NX_C_MAX_NDIM],
      v_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (w->shape[i] != in->shape[i]) return EIG_ERR_SHAPE;
    if (vectors && v->shape[i] != in->shape[i]) return EIG_ERR_SHAPE;
    bshape[i] = in->shape[i];
    in_bs[i] = in->strides[i];
    w_bs[i] = w->strides[i];
    v_bs[i] = vectors ? v->strides[i] : 0;
    nbatch *= bshape[i];
  }
  if (n == 0 || nbatch == 0) return NX_C_OK;

  int64_t bytes = nbatch * n * n * esz;
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nbatch, n * n * n, bytes);
  if (nth > nbatch) nth = (int)nbatch;
  if (nth > EIG_MAX_WORKERS) nth = EIG_MAX_WORKERS;
  if (nth < 1) nth = 1;

  eig_ctx x;
  memset(&x, 0, sizeof(x));
  int64_t cz = (int64_t)sizeof(eig_cplx); /* 16 */
  int64_t stride;
  if (is_complex) {
    int64_t a_mat = EIG_ALIGN(n * n * cz);
    int64_t a_vec = EIG_ALIGN(n * cz);
    int64_t a_rvec = EIG_ALIGN(n * (int64_t)sizeof(double));
    x.off_a = 0;
    x.off_z = x.off_a + a_mat;
    x.off_cV = x.off_z + a_mat;
    x.off_cw = x.off_cV + a_mat;
    x.off_cy = x.off_cw + a_vec;
    x.off_cvh = x.off_cy + a_vec;
    x.off_csn = x.off_cvh + a_vec;
    x.off_scale = x.off_csn + a_vec;
    x.off_ccs = x.off_scale + a_rvec;
    stride = x.off_ccs + a_rvec;
  } else {
    int64_t a_mat = EIG_ALIGN(n * n * (int64_t)sizeof(double));
    int64_t a_rvec = EIG_ALIGN(n * (int64_t)sizeof(double));
    int64_t a_cvec = EIG_ALIGN(n * cz);
    x.off_a = 0;
    x.off_z = x.off_a + a_mat;
    x.off_wr = x.off_z + a_mat;
    x.off_wi = x.off_wr + a_rvec;
    x.off_scale = x.off_wi + a_rvec;
    x.off_ort = x.off_scale + a_rvec;
    x.off_cvec = x.off_ort + a_rvec;
    stride = x.off_cvec + a_cvec;
  }
  char *scratch = aligned_alloc(64, (size_t)stride * nth);
  if (!scratch) return NX_C_ERR_ALLOC;

  nx_c_status werr[EIG_MAX_WORKERS];
  for (int i = 0; i < nth; i++) werr[i] = NX_C_OK;

  x.in = in;
  x.w = w;
  x.v = v;
  x.dt = dt;
  x.is_complex = is_complex;
  x.vectors = vectors;
  x.n = n;
  x.esz = esz;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.in_bs = in_bs;
  x.w_bs = w_bs;
  x.v_bs = v_bs;
  x.in_rs = in->strides[in->ndim - 2];
  x.in_cs = in->strides[in->ndim - 1];
  x.w_cs = w->strides[w->ndim - 1];
  x.v_rs = vectors ? v->strides[v->ndim - 2] : 0;
  x.v_cs = vectors ? v->strides[v->ndim - 1] : 0;
  x.scratch = scratch;
  x.stride = stride;
  x.werr = werr;

  nx_c_parallel_for(nth, nbatch, bytes, eig_body, &x, scratch);

  nx_c_status err = NX_C_OK;
  for (int i = 0; i < nth; i++)
    if (werr[i] != NX_C_OK) {
      err = werr[i];
      break;
    }
  return err;
}

/* ── FFI stub ──────────────────────────────────────────────────────────────*/

static NX_C_NORETURN void eig_raise(const char *op, nx_c_status s) {
  if (strcmp(s, EIG_ERR_NOT_FLOAT) == 0 || strcmp(s, EIG_ERR_NOT_SQUARE) == 0 ||
      strcmp(s, EIG_ERR_SHAPE) == 0)
    nx_c_raise_invalid(op, s);
  nx_c_raise(op, s);
}

/* vw: complex128 eigenvalues [batch..., n]. vv: complex128 eigenvectors
   [batch..., n, n], columns = eigenvectors — allocated by the binding ONLY when
   vectors is true. ABI (mirrors eigh): when vectors is false the binding passes
   a complex128 placeholder in the vv slot; this stub never extracts or touches
   vv then. The dispatch dtype comes from the INPUT (vin), not the always-
   complex128 outputs. */
CAMLprim value caml_nx_c_eig(value vw, value vv, value vin, value vvectors) {
  CAMLparam4(vw, vv, vin, vvectors);
  int vectors = Bool_val(vvectors);
  nx_c_ndarray in, w, v;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vw, &w);
  if (s == NX_C_OK && vectors) s = nx_c_ndarray_of_value(vv, &v);
  if (s != NX_C_OK) eig_raise("eig", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt == NX_C_DTYPE_COUNT) eig_raise("eig", NX_C_ERR_BAD_KIND);
  s = nx_c_eig_run(&in, &w, vectors ? &v : NULL, dt, vectors);
  if (s != NX_C_OK) eig_raise("eig", s);
  CAMLreturn(Val_unit);
}
