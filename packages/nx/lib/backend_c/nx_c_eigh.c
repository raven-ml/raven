/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_eigh.c — symmetric/Hermitian eigendecomposition: Householder
   tridiagonalization + implicit-shift QL (linalg tier 2). Shared machinery in
   nx_c_linalg.h. */

#include "nx_c_linalg.h"

/* ── Symmetric/Hermitian eigendecomposition (eigh) ────────────────────────

   Householder tridiagonalization reduces A (lower triangle read) to a REAL
   tridiagonal (d, e): each reflector's beta is real (larfg postcondition), so
   every subdiagonal e[i]=beta is real even for complex-Hermitian input — the
   complex phase lives only in the accumulated unitary Q, not in the tridiagonal.
   Implicit-shift QL with Wilkinson shifts (la_tql2, real d/e/rotations) then
   diagonalizes, accumulating the real Givens rotations into Q — starting the
   accumulator at Q gives the eigenvectors V = Q·Z of A directly. Eigenvalues are
   real; sorted ascending with paired eigenvector-column swaps.

   la_tridiag: A → d[n], e[n] (e[i] couples d[i],d[i+1]; e[n-1]=0), reflectors in
   the sub-diagonal columns (unit on the subdiagonal), tau[n]. wv is an n-length
   compute-typed workspace for the Hermitian matvec.

   la_tridiag_range is the unblocked (BLAS-2) reduction of the trailing block from
   column j0 — the exact pre-blocking kernel body. Below LA_EIGH_NB the driver runs
   it whole (small-n path). Above it, dsytrd/zhetrd blocking: la_latrd reduces a
   LA_EIGH_NB-column panel, accumulating W so the trailing update is a Hermitian
   rank-2k (dsyr2k) update A22 -= V Wᴴ + W Vᴴ done as ONE GEMM (P = V Wᴴ, then the
   lower triangle A[r][c] -= P[r][c] + conj(P[c][r]), diagonal 2·Re — both halves
   from that single product). la_tridiag_range then finishes the last block. The
   two-sided Hermitian semantics and real-β subdiagonal are preserved: la_latrd is
   zhetd2's panel form and the reflectors/e it produces feed the untouched orgtr
   and tql2 unchanged (residual gates cover the reordered arithmetic). W, Wc (the
   Wᴴ conj panel), P (the V Wᴴ product), and the GEMM panels are caller scratch. */
#define LA_GEN_EIGH(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)             \
  static void la_tridiag_range_##sfx(void *vA, int64_t n, int64_t lda,         \
                                     int64_t j0, void *ve, void *vtau,         \
                                     void *vwv) {                              \
    T *A = (T *)vA;                                                           \
    R *e = (R *)ve;                                                           \
    T *tau = (T *)vtau;                                                       \
    T *wv = (T *)vwv;                                                         \
    for (int64_t i = j0; i < n - 1; i++) {                                    \
      R xnorm2 = (R)0;                                                        \
      for (int64_t r = i + 2; r < n; r++) xnorm2 += NORM2(A[r * lda + i]);    \
      T alpha = A[(i + 1) * lda + i];                                         \
      R alphr = REAL(alpha);                                                  \
      if (xnorm2 == (R)0 && LA_IMAG_##sfx(alpha) == (R)0) {                   \
        tau[i] = (T)0;                                                        \
        e[i] = alphr;                                                        \
        continue;                                                            \
      }                                                                       \
      R anorm = SQRT(NORM2(alpha) + xnorm2);                                  \
      R beta = alphr >= (R)0 ? -anorm : anorm;                               \
      T t = LA_MK_##sfx((beta - alphr) / beta, -LA_IMAG_##sfx(alpha) / beta); \
      tau[i] = t;                                                             \
      T scal = alpha - LA_MK_##sfx(beta, (R)0);                              \
      for (int64_t r = i + 2; r < n; r++) A[r * lda + i] = A[r * lda + i] / scal; \
      e[i] = beta;                                                           \
      A[(i + 1) * lda + i] = (T)1;                                           \
      for (int64_t j = i + 1; j < n; j++) {                                  \
        T sum = (T)0;                                                        \
        for (int64_t l = i + 1; l < n; l++) {                                \
          T ajl = l <= j ? A[j * lda + l] : CONJ(A[l * lda + j]);            \
          sum += ajl * A[l * lda + i];                                       \
        }                                                                     \
        wv[j] = t * sum;                                                     \
      }                                                                       \
      T dot = (T)0;                                                          \
      for (int64_t j = i + 1; j < n; j++) dot += CONJ(wv[j]) * A[j * lda + i]; \
      T a = LA_MK_##sfx((R)(-0.5), (R)0) * t * dot;                          \
      for (int64_t j = i + 1; j < n; j++) wv[j] += a * A[j * lda + i];       \
      for (int64_t j = i + 1; j < n; j++) {                                  \
        T vj = A[j * lda + i];                                               \
        T wj = wv[j];                                                        \
        for (int64_t l = i + 1; l <= j; l++)                                 \
          A[j * lda + l] -= vj * CONJ(wv[l]) + wj * CONJ(A[l * lda + i]);    \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  /* dlatrd/zlatrd (lower): reduce panel columns [j, j+jb), accumulating W so    \
     the trailing rank-2k update can be deferred to one GEMM. */                 \
  static void la_latrd_##sfx(void *vA, int64_t n, int64_t lda, int64_t j,       \
                             int64_t jb, void *ve, void *vtau, void *vW,        \
                             int64_t ldw) {                                     \
    T *A = (T *)vA;                                                           \
    R *e = (R *)ve;                                                           \
    T *tau = (T *)vtau;                                                       \
    T *W = (T *)vW;                                                           \
    for (int64_t c = 0; c < jb; c++) {                                       \
      int64_t i = j + c;                                                     \
      if (c > 0)                                                             \
        for (int64_t r = i; r < n; r++) {                                    \
          T s = (T)0;                                                        \
          for (int64_t l = 0; l < c; l++)                                    \
            s += A[r * lda + (j + l)] * CONJ(W[i * ldw + l]) +               \
                 W[r * ldw + l] * CONJ(A[i * lda + (j + l)]);               \
          A[r * lda + i] -= s;                                               \
        }                                                                    \
      R xnorm2 = (R)0;                                                       \
      for (int64_t r = i + 2; r < n; r++) xnorm2 += NORM2(A[r * lda + i]);   \
      T alpha = A[(i + 1) * lda + i];                                        \
      R alphr = REAL(alpha);                                                 \
      if (xnorm2 == (R)0 && LA_IMAG_##sfx(alpha) == (R)0) {                  \
        tau[i] = (T)0;                                                       \
        e[i] = alphr;                                                        \
        for (int64_t r = i + 1; r < n; r++) W[r * ldw + c] = (T)0;           \
        continue;                                                           \
      }                                                                      \
      R anorm = SQRT(NORM2(alpha) + xnorm2);                                 \
      R beta = alphr >= (R)0 ? -anorm : anorm;                              \
      T t = LA_MK_##sfx((beta - alphr) / beta, -LA_IMAG_##sfx(alpha) / beta); \
      tau[i] = t;                                                            \
      T scal = alpha - LA_MK_##sfx(beta, (R)0);                             \
      for (int64_t r = i + 2; r < n; r++) A[r * lda + i] = A[r * lda + i] / scal; \
      e[i] = beta;                                                          \
      A[(i + 1) * lda + i] = (T)1;                                          \
      for (int64_t r = i + 1; r < n; r++) {                                 \
        T s = (T)0;                                                         \
        for (int64_t l = i + 1; l < n; l++) {                               \
          T arl = l <= r ? A[r * lda + l] : CONJ(A[l * lda + r]);           \
          s += arl * A[l * lda + i];                                        \
        }                                                                    \
        W[r * ldw + c] = s;                                                 \
      }                                                                      \
      if (c > 0) {                                                          \
        T t1[LA_EIGH_NB], t2[LA_EIGH_NB];                                   \
        for (int64_t l = 0; l < c; l++) {                                   \
          T a1 = (T)0, a2 = (T)0;                                           \
          for (int64_t r = i + 1; r < n; r++) {                            \
            T vr = A[r * lda + i];                                          \
            a1 += CONJ(W[r * ldw + l]) * vr;                                \
            a2 += CONJ(A[r * lda + (j + l)]) * vr;                          \
          }                                                                  \
          t1[l] = a1;                                                       \
          t2[l] = a2;                                                       \
        }                                                                    \
        for (int64_t r = i + 1; r < n; r++) {                              \
          T s = (T)0;                                                       \
          for (int64_t l = 0; l < c; l++)                                   \
            s += A[r * lda + (j + l)] * t1[l] + W[r * ldw + l] * t2[l];     \
          W[r * ldw + c] -= s;                                              \
        }                                                                    \
      }                                                                      \
      for (int64_t r = i + 1; r < n; r++) W[r * ldw + c] = t * W[r * ldw + c]; \
      T dot = (T)0;                                                         \
      for (int64_t r = i + 1; r < n; r++)                                   \
        dot += CONJ(W[r * ldw + c]) * A[r * lda + i];                       \
      T ac = LA_MK_##sfx((R)(-0.5), (R)0) * t * dot;                        \
      for (int64_t r = i + 1; r < n; r++)                                   \
        W[r * ldw + c] += ac * A[r * lda + i];                             \
    }                                                                        \
  }                                                                          \
  static void la_tridiag_##sfx(void *vA, int64_t n, int64_t lda, void *vd,     \
                               void *ve, void *vtau, void *vwv, void *vW,     \
                               void *vWc, void *vP, void *vg) {               \
    T *A = (T *)vA;                                                          \
    R *d = (R *)vd;                                                          \
    R *e = (R *)ve;                                                          \
    if (n <= LA_EIGH_NB) {                                                   \
      la_tridiag_range_##sfx(vA, n, lda, 0, ve, vtau, vwv);                  \
    } else {                                                                 \
      T *W = (T *)vW;                                                        \
      T *Wc = (T *)vWc;                                                      \
      T *P = (T *)vP;                                                        \
      int64_t j = 0;                                                        \
      while (n - j > LA_EIGH_NB) {                                          \
        int64_t jb = LA_EIGH_NB;                                            \
        la_latrd_##sfx(vA, n, lda, j, jb, ve, vtau, vW, LA_EIGH_NB);        \
        int64_t m2 = n - (j + jb);                                          \
        for (int64_t r = 0; r < m2; r++)                                    \
          for (int64_t l = 0; l < jb; l++)                                  \
            Wc[r * jb + l] = CONJ(W[(j + jb + r) * LA_EIGH_NB + l]);        \
        nx_c_gemm2d_ct_ws(DT, m2, m2, jb,                                    \
                         (const char *)&A[(j + jb) * lda + j], lda, 1,      \
                         (const char *)Wc, 1, jb, (char *)P, m2, 1,         \
                         (char *)vg);                                       \
        for (int64_t r = 0; r < m2; r++) {                                  \
          int64_t rr = j + jb + r;                                          \
          A[rr * lda + rr] -= P[r * m2 + r] + CONJ(P[r * m2 + r]);          \
          for (int64_t cc = 0; cc < r; cc++)                               \
            A[rr * lda + (j + jb + cc)] -=                                  \
                P[r * m2 + cc] + CONJ(P[cc * m2 + r]);                      \
        }                                                                    \
        j += jb;                                                            \
      }                                                                      \
      la_tridiag_range_##sfx(vA, n, lda, j, ve, vtau, vwv);                  \
    }                                                                        \
    for (int64_t i = 0; i < n; i++) d[i] = REAL(A[i * lda + i]);            \
    if (n > 0) e[n - 1] = (R)0;                                             \
  }                                                                          \
  static void la_orgtr_##sfx(void *vA, int64_t n, int64_t lda, void *vtau,     \
                             void *vZ, int64_t ldz) {                          \
    T *A = (T *)vA;                                                           \
    T *tau = (T *)vtau;                                                       \
    T *Z = (T *)vZ;                                                           \
    for (int64_t r = 0; r < n; r++)                                           \
      for (int64_t c = 0; c < n; c++) Z[r * ldz + c] = r == c ? (T)1 : (T)0;  \
    for (int64_t jj = 0; jj < n - 1; jj++) {                                  \
      int64_t i = n - 2 - jj;                                                 \
      if (tau[i] == (T)0) continue;                                          \
      for (int64_t c = 0; c < n; c++) {                                       \
        T w = (T)0;                                                          \
        for (int64_t r = i + 1; r < n; r++)                                  \
          w += CONJ(A[r * lda + i]) * Z[r * ldz + c];                        \
        T tw = tau[i] * w;                                                    \
        for (int64_t r = i + 1; r < n; r++) Z[r * ldz + c] -= tw * A[r * lda + i]; \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  static nx_c_status la_tql2_##sfx(void *vd, void *ve, void *vZ, int64_t n,     \
                                  int64_t ldz, int want_vec) {                 \
    R *d = (R *)vd;                                                           \
    R *e = (R *)ve;                                                           \
    T *Z = (T *)vZ;                                                           \
    for (int64_t l = 0; l < n; l++) {                                         \
      int iter = 0;                                                           \
      int64_t m;                                                              \
      do {                                                                    \
        for (m = l; m < n - 1; m++) {                                         \
          R dd = LA_ABS_##sfx(d[m]) + LA_ABS_##sfx(d[m + 1]);                 \
          if (LA_ABS_##sfx(e[m]) + dd == dd) break;                          \
        }                                                                     \
        if (m != l) {                                                        \
          if (iter++ == 50) return LA_ERR_NO_CONVERGE;                       \
          R g = (d[l + 1] - d[l]) / ((R)2 * e[l]);                           \
          R r = LA_HYP_##sfx(g, (R)1);                                       \
          R sg = g >= (R)0 ? LA_ABS_##sfx(r) : -LA_ABS_##sfx(r);             \
          g = d[m] - d[l] + e[l] / (g + sg);                                 \
          R s = (R)1, c = (R)1, p = (R)0;                                    \
          int64_t i;                                                         \
          for (i = m - 1; i >= l; i--) {                                     \
            R f = s * e[i];                                                  \
            R b = c * e[i];                                                  \
            r = LA_HYP_##sfx(f, g);                                          \
            e[i + 1] = r;                                                    \
            if (r == (R)0) {                                                 \
              d[i + 1] -= p;                                                 \
              e[m] = (R)0;                                                   \
              break;                                                        \
            }                                                                 \
            s = f / r;                                                       \
            c = g / r;                                                       \
            g = d[i + 1] - p;                                                \
            r = (d[i] - g) * s + (R)2 * c * b;                               \
            p = s * r;                                                       \
            d[i + 1] = g + p;                                                \
            g = c * r - b;                                                   \
            if (want_vec)                                                    \
              for (int64_t k = 0; k < n; k++) {                             \
                T fz = Z[k * ldz + i + 1];                                   \
                Z[k * ldz + i + 1] = s * Z[k * ldz + i] + c * fz;            \
                Z[k * ldz + i] = c * Z[k * ldz + i] - s * fz;               \
              }                                                               \
          }                                                                   \
          if (r == (R)0 && i >= l) continue;                                 \
          d[l] -= p;                                                         \
          e[l] = g;                                                          \
          e[m] = (R)0;                                                       \
        }                                                                     \
      } while (m != l);                                                      \
    }                                                                         \
    return NX_C_OK;                                                          \
  }                                                                           \
  static void la_eigsort_##sfx(void *vd, void *vZ, int64_t n, int64_t ldz,     \
                               int want_vec) {                                \
    R *d = (R *)vd;                                                           \
    T *Z = (T *)vZ;                                                           \
    for (int64_t i = 0; i < n - 1; i++) {                                     \
      int64_t k = i;                                                          \
      R p = d[i];                                                             \
      for (int64_t j = i + 1; j < n; j++)                                     \
        if (d[j] < p) {                                                       \
          k = j;                                                              \
          p = d[j];                                                          \
        }                                                                     \
      if (k != i) {                                                          \
        d[k] = d[i];                                                          \
        d[i] = p;                                                            \
        if (want_vec)                                                        \
          for (int64_t r = 0; r < n; r++) {                                  \
            T tmp = Z[r * ldz + i];                                          \
            Z[r * ldz + i] = Z[r * ldz + k];                                 \
            Z[r * ldz + k] = tmp;                                            \
          }                                                                   \
      }                                                                       \
    }                                                                         \
  }
#define LA_EXPAND_EIGH(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)          \
  LA_GEN_EIGH(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_EIGH)
LA_TRAITS_double(LA_EXPAND_EIGH)
LA_TRAITS_c32(LA_EXPAND_EIGH)
LA_TRAITS_c64(LA_EXPAND_EIGH)
#undef LA_EXPAND_EIGH

/* ── Tridiagonal divide-and-conquer (dlaed0-6 structure) ──────────────────

   The with-vectors eigenproblem's cost is tql2's O(n^3) rotation accumulation
   into Q. Divide-and-conquer replaces it — tear the tridiagonal at the midpoint,
   conquer each half,
   merge via the rank-one secular equation (deflation + a BLAS-3 back-multiply of
   the deflated Q's). tql2 stays as the eigenvalues-only path and the small-n leaf.

   This core is `double` regardless of the compute type: the tridiagonal problem
   is real (larfg's real-beta subdiagonal), the eigenvalue output is float64, and
   only the final V = Q_householder·Z_tri carries the compute type (a real, or
   complex-by-real, GEMM). Running the secular solver in double also gives the f32
   path better small-eigenvalue accuracy at negligible cost (n^2 widen, n^3 solve).

   Matrices here are COLUMN-major (Q[i + j*ldq]) so the dlaed reference maps 1:1;
   the two dlaed3 back-multiply GEMMs and the final apply GEMM pass column-major
   strides to nx_c_gemm2d_ct_ws (rs=1, cs=ld). Non-convergence of the secular solver
   returns nonzero, surfaced as LA_ERR_NO_CONVERGE. The eigenvalue index arithmetic
   is 0-based throughout (the reference is 1-based). */

#define LA_DC_SMLSIZ 25 /* LAPACK DLAED0 SMLSIZ */

/* dlaed5: eigenvalue I (1 or 2) of the 2x2 rank-one modified diagonal, with the
   normalized eigenvector returned in delta. */
static void la_ed5(int i1, const double *d, const double *z, double *delta,
                   double rho, double *lam) {
  double del = d[1] - d[0];
  if (i1 == 1) {
    double w = 1.0 + 2.0 * rho * (z[1] * z[1] - z[0] * z[0]) / del;
    if (w > 0.0) {
      double b = del + rho * (z[0] * z[0] + z[1] * z[1]);
      double c = rho * z[0] * z[0] * del;
      double tau = 2.0 * c / (b + sqrt(fabs(b * b - 4.0 * c)));
      *lam = d[0] + tau;
      delta[0] = -z[0] / tau;
      delta[1] = z[1] / (del - tau);
    } else {
      double b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
      double c = rho * z[1] * z[1] * del;
      double tau;
      if (b > 0.0)
        tau = -2.0 * c / (b + sqrt(b * b + 4.0 * c));
      else
        tau = (b - sqrt(b * b + 4.0 * c)) / 2.0;
      *lam = d[1] + tau;
      delta[0] = -z[0] / (del + tau);
      delta[1] = -z[1] / tau;
    }
  } else {
    double b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
    double c = rho * z[1] * z[1] * del;
    double tau;
    if (b > 0.0)
      tau = (b + sqrt(b * b + 4.0 * c)) / 2.0;
    else
      tau = 2.0 * c / (-b + sqrt(b * b + 4.0 * c));
    *lam = d[1] + tau;
    delta[0] = -z[0] / (del + tau);
    delta[1] = -z[1] / tau;
  }
  double temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
  delta[0] /= temp;
  delta[1] /= temp;
}

/* dlaed6: Gragg-Thornton-Warner cubic-convergent root of the 3-pole rational
   equation, the SWTCH3 interpolation dlaed4 uses. d,z length 3. Returns tau via
   *tau_out; status nonzero only on non-convergence. */
static int la_ed6(int kniter, int orgati, double rho, const double *d,
                  const double *z, double finit, double *tau_out) {
  const int MAXIT = 40;
  double lbd, ubd, tau = 0.0, eta, temp, temp1, temp2, temp3, temp4;
  double a, b, c, f, fc, df, ddf, erretm;
  int niter, iter, i, info = 0, scale = 0;
  double dscale[3], zscale[3];
  double eps, base, small1, sminv1, small2, sminv2, sclfac = 0, sclinv = 0;

  if (orgati) {
    lbd = d[1];
    ubd = d[2];
  } else {
    lbd = d[0];
    ubd = d[1];
  }
  if (finit < 0.0)
    lbd = 0.0;
  else
    ubd = 0.0;
  niter = 1;
  if (kniter == 2) {
    if (orgati) {
      temp = (d[2] - d[1]) / 2.0;
      c = rho + z[0] / ((d[0] - d[1]) - temp);
      a = c * (d[1] + d[2]) + z[1] + z[2];
      b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
    } else {
      temp = (d[0] - d[1]) / 2.0;
      c = rho + z[2] / ((d[2] - d[1]) - temp);
      a = c * (d[0] + d[1]) + z[0] + z[1];
      b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
    }
    temp = fmax(fabs(a), fmax(fabs(b), fabs(c)));
    a /= temp;
    b /= temp;
    c /= temp;
    if (c == 0.0)
      tau = b / a;
    else if (a <= 0.0)
      tau = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    else
      tau = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    if (tau < lbd || tau > ubd)
      tau = (lbd + ubd) / 2.0;
    if (d[0] == tau || d[1] == tau || d[2] == tau) {
      tau = 0.0;
    } else {
      temp = finit + tau * z[0] / (d[0] * (d[0] - tau)) +
             tau * z[1] / (d[1] * (d[1] - tau)) +
             tau * z[2] / (d[2] * (d[2] - tau));
      if (temp <= 0.0)
        lbd = tau;
      else
        ubd = tau;
      if (fabs(finit) <= fabs(temp))
        tau = 0.0;
    }
  }
  eps = 0.5 * DBL_EPSILON; /* DLAMCH('E') = unit roundoff = DBL_EPSILON/2 */
  base = (double)FLT_RADIX;
  small1 = pow(base, (double)(int)(log(DBL_MIN) / log(base) / 3.0));
  sminv1 = 1.0 / small1;
  small2 = small1 * small1;
  sminv2 = sminv1 * sminv1;
  if (orgati)
    temp = fmin(fabs(d[1] - tau), fabs(d[2] - tau));
  else
    temp = fmin(fabs(d[0] - tau), fabs(d[1] - tau));
  if (temp <= small1) {
    scale = 1;
    if (temp <= small2) {
      sclfac = sminv2;
      sclinv = small2;
    } else {
      sclfac = sminv1;
      sclinv = small1;
    }
    for (i = 0; i < 3; i++) {
      dscale[i] = d[i] * sclfac;
      zscale[i] = z[i] * sclfac;
    }
    tau *= sclfac;
    lbd *= sclfac;
    ubd *= sclfac;
  } else {
    for (i = 0; i < 3; i++) {
      dscale[i] = d[i];
      zscale[i] = z[i];
    }
  }
  fc = 0.0;
  df = 0.0;
  ddf = 0.0;
  for (i = 0; i < 3; i++) {
    temp = 1.0 / (dscale[i] - tau);
    temp1 = zscale[i] * temp;
    temp2 = temp1 * temp;
    temp3 = temp2 * temp;
    fc += temp1 / dscale[i];
    df += temp2;
    ddf += temp3;
  }
  f = finit + tau * fc;
  if (fabs(f) <= 0.0)
    goto done;
  if (f <= 0.0)
    lbd = tau;
  else
    ubd = tau;
  iter = niter + 1;
  for (niter = iter; niter <= MAXIT; niter++) {
    if (orgati) {
      temp1 = dscale[1] - tau;
      temp2 = dscale[2] - tau;
    } else {
      temp1 = dscale[0] - tau;
      temp2 = dscale[1] - tau;
    }
    a = (temp1 + temp2) * f - temp1 * temp2 * df;
    b = temp1 * temp2 * f;
    c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
    temp = fmax(fabs(a), fmax(fabs(b), fabs(c)));
    a /= temp;
    b /= temp;
    c /= temp;
    if (c == 0.0)
      eta = b / a;
    else if (a <= 0.0)
      eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    else
      eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    if (f * eta >= 0.0)
      eta = -f / df;
    tau += eta;
    if (tau < lbd || tau > ubd)
      tau = (lbd + ubd) / 2.0;
    fc = 0.0;
    erretm = 0.0;
    df = 0.0;
    ddf = 0.0;
    int broke = 0;
    for (i = 0; i < 3; i++) {
      if ((dscale[i] - tau) != 0.0) {
        temp = 1.0 / (dscale[i] - tau);
        temp1 = zscale[i] * temp;
        temp2 = temp1 * temp;
        temp3 = temp2 * temp;
        temp4 = temp1 / dscale[i];
        fc += temp4;
        erretm += fabs(temp4);
        df += temp2;
        ddf += temp3;
      } else {
        broke = 1;
        break;
      }
    }
    if (broke)
      goto done;
    f = finit + tau * fc;
    erretm = 8.0 * (fabs(finit) + fabs(tau) * erretm) + fabs(tau) * df;
    if (fabs(f) <= 4.0 * eps * erretm || (ubd - lbd) <= 4.0 * eps * fabs(tau))
      goto done;
    if (f <= 0.0)
      lbd = tau;
    else
      ubd = tau;
  }
  info = 1;
done:
  if (scale)
    tau *= sclinv;
  *tau_out = tau;
  return info;
}

/* dlaed4: eigenvalue `iev` (0-based) of D + rho z z^T (n poles, ascending D),
   filling delta[j] = d[j]-lambda and returning lambda in *lam. Status nonzero on
   non-convergence. The convergence-critical rational interpolation with
   guaranteed bracketing (DLTLB/DLTUB) — ported line-for-line. */
static int la_ed4(int n, int iev, const double *d, const double *z,
                  double *delta, double rho, double *lam) {
  const int MAXIT = 30;
  const double eps = 0.5 * DBL_EPSILON; /* DLAMCH('E') = DBL_EPSILON/2 */
  double rhoinv = 1.0 / rho;
  int j, niter, iter, ii, iim1, iip1, ip1;
  double a, b, c, del, dltlb, dltub, dphi, dpsi, dw, erretm, eta, eta1, eta2,
      midpt, phi, prew, psi, tau, temp, temp1, w;
  int orgati, swtch, swtch3;
  double zz[3];

  if (n == 1) {
    *lam = d[0] + rho * z[0] * z[0];
    delta[0] = 1.0;
    return 0;
  }
  if (n == 2) {
    la_ed5(iev + 1, d, z, delta, rho, lam);
    return 0;
  }

  if (iev == n - 1) {
    ii = n - 2;
    niter = 1;
    midpt = rho / 2.0;
    for (j = 0; j < n; j++)
      delta[j] = (d[j] - d[iev]) - midpt;
    psi = 0.0;
    for (j = 0; j < n - 2; j++)
      psi += z[j] * z[j] / delta[j];
    c = rhoinv + psi;
    w = c + z[ii] * z[ii] / delta[ii] + z[n - 1] * z[n - 1] / delta[n - 1];
    if (w <= 0.0) {
      temp = z[n - 2] * z[n - 2] / (d[n - 1] - d[n - 2] + rho) +
             z[n - 1] * z[n - 1] / rho;
      if (c <= temp) {
        tau = rho;
      } else {
        del = d[n - 1] - d[n - 2];
        a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
        b = z[n - 1] * z[n - 1] * del;
        if (a < 0.0)
          tau = 2.0 * b / (sqrt(a * a + 4.0 * b * c) - a);
        else
          tau = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);
      }
      dltlb = midpt;
      dltub = rho;
    } else {
      del = d[n - 1] - d[n - 2];
      a = -c * del + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
      b = z[n - 1] * z[n - 1] * del;
      if (a < 0.0)
        tau = 2.0 * b / (sqrt(a * a + 4.0 * b * c) - a);
      else
        tau = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);
      dltlb = 0.0;
      dltub = midpt;
    }
    for (j = 0; j < n; j++)
      delta[j] = (d[j] - d[iev]) - tau;
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for (j = 0; j <= ii; j++) {
      temp = z[j] / delta[j];
      psi += z[j] * temp;
      dpsi += temp * temp;
      erretm += psi;
    }
    erretm = fabs(erretm);
    temp = z[n - 1] / delta[n - 1];
    phi = z[n - 1] * temp;
    dphi = temp * temp;
    erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv +
             fabs(tau) * (dpsi + dphi);
    w = rhoinv + phi + psi;
    if (fabs(w) <= eps * erretm) {
      *lam = d[iev] + tau;
      return 0;
    }
    if (w <= 0.0)
      dltlb = fmax(dltlb, tau);
    else
      dltub = fmin(dltub, tau);
    /* first step (with Li update and geometric-mean safeguard) */
    niter++;
    c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
    a = (delta[n - 2] + delta[n - 1]) * w -
        delta[n - 2] * delta[n - 1] * (dpsi + dphi);
    b = delta[n - 2] * delta[n - 1] * w;
    if (c < 0.0)
      c = fabs(c);
    if (c == 0.0)
      eta = -w / (dpsi + dphi);
    else if (a >= 0.0)
      eta = (a + sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    else
      eta = 2.0 * b / (a - sqrt(fabs(a * a - 4.0 * b * c)));
    if (w * eta > 0.0)
      eta = -w / (dpsi + dphi);
    temp = tau + eta;
    if (temp > dltub || temp < dltlb) {
      eta1 = -w / (dpsi + dphi);
      temp = tau + eta1;
      if (w < 0.0)
        eta2 = (dltub - tau) / 2.0;
      else
        eta2 = (dltlb - tau) / 2.0;
      if (dltlb <= temp && temp <= dltub)
        eta = copysign(1.0, eta1) * sqrt(fabs(eta1)) * sqrt(fabs(eta2));
      else
        eta = eta2;
    }
    for (j = 0; j < n; j++)
      delta[j] -= eta;
    tau += eta;
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for (j = 0; j <= ii; j++) {
      temp = z[j] / delta[j];
      psi += z[j] * temp;
      dpsi += temp * temp;
      erretm += psi;
    }
    erretm = fabs(erretm);
    temp = z[n - 1] / delta[n - 1];
    phi = z[n - 1] * temp;
    dphi = temp * temp;
    erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv +
             fabs(tau) * (dpsi + dphi);
    w = rhoinv + phi + psi;
    iter = niter + 1;
    for (niter = iter; niter <= MAXIT; niter++) {
      if (fabs(w) <= eps * erretm) {
        *lam = d[iev] + tau;
        return 0;
      }
      if (w <= 0.0)
        dltlb = fmax(dltlb, tau);
      else
        dltub = fmin(dltub, tau);
      c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
      a = (delta[n - 2] + delta[n - 1]) * w -
          delta[n - 2] * delta[n - 1] * (dpsi + dphi);
      b = delta[n - 2] * delta[n - 1] * w;
      if (a >= 0.0)
        eta = (a + sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
      else
        eta = 2.0 * b / (a - sqrt(fabs(a * a - 4.0 * b * c)));
      if (w * eta > 0.0)
        eta = -w / (dpsi + dphi);
      temp = tau + eta;
      if (temp > dltub || temp < dltlb) {
        if (w < 0.0)
          eta = (dltub - tau) / 2.0;
        else
          eta = (dltlb - tau) / 2.0;
      }
      for (j = 0; j < n; j++)
        delta[j] -= eta;
      tau += eta;
      dpsi = 0.0;
      psi = 0.0;
      erretm = 0.0;
      for (j = 0; j <= ii; j++) {
        temp = z[j] / delta[j];
        psi += z[j] * temp;
        dpsi += temp * temp;
        erretm += psi;
      }
      erretm = fabs(erretm);
      temp = z[n - 1] / delta[n - 1];
      phi = z[n - 1] * temp;
      dphi = temp * temp;
      erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv +
               fabs(tau) * (dpsi + dphi);
      w = rhoinv + phi + psi;
    }
    *lam = d[iev] + tau;
    return 1;
  }

  /* general case iev < n-1 */
  niter = 1;
  ip1 = iev + 1;
  del = d[ip1] - d[iev];
  midpt = del / 2.0;
  for (j = 0; j < n; j++)
    delta[j] = (d[j] - d[iev]) - midpt;
  psi = 0.0;
  for (j = 0; j < iev; j++)
    psi += z[j] * z[j] / delta[j];
  phi = 0.0;
  for (j = n - 1; j >= iev + 2; j--)
    phi += z[j] * z[j] / delta[j];
  c = rhoinv + psi + phi;
  w = c + z[iev] * z[iev] / delta[iev] + z[ip1] * z[ip1] / delta[ip1];
  if (w > 0.0) {
    orgati = 1;
    a = c * del + z[iev] * z[iev] + z[ip1] * z[ip1];
    b = z[iev] * z[iev] * del;
    if (a > 0.0)
      tau = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    else
      tau = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    dltlb = 0.0;
    dltub = midpt;
  } else {
    orgati = 0;
    a = c * del - z[iev] * z[iev] - z[ip1] * z[ip1];
    b = z[ip1] * z[ip1] * del;
    if (a < 0.0)
      tau = 2.0 * b / (a - sqrt(fabs(a * a + 4.0 * b * c)));
    else
      tau = -(a + sqrt(fabs(a * a + 4.0 * b * c))) / (2.0 * c);
    dltlb = -midpt;
    dltub = 0.0;
  }
  if (orgati)
    for (j = 0; j < n; j++)
      delta[j] = (d[j] - d[iev]) - tau;
  else
    for (j = 0; j < n; j++)
      delta[j] = (d[j] - d[ip1]) - tau;
  ii = orgati ? iev : iev + 1;
  iim1 = ii - 1;
  iip1 = ii + 1;
  dpsi = 0.0;
  psi = 0.0;
  erretm = 0.0;
  for (j = 0; j < ii; j++) {
    temp = z[j] / delta[j];
    psi += z[j] * temp;
    dpsi += temp * temp;
    erretm += psi;
  }
  erretm = fabs(erretm);
  dphi = 0.0;
  phi = 0.0;
  for (j = n - 1; j > ii; j--) {
    temp = z[j] / delta[j];
    phi += z[j] * temp;
    dphi += temp * temp;
    erretm += phi;
  }
  w = rhoinv + phi + psi;
  swtch3 = 0;
  if (orgati) {
    if (w < 0.0)
      swtch3 = 1;
  } else {
    if (w > 0.0)
      swtch3 = 1;
  }
  if (ii == 0 || ii == n - 1)
    swtch3 = 0;
  temp = z[ii] / delta[ii];
  dw = dpsi + dphi + temp * temp;
  temp = z[ii] * temp;
  w = w + temp;
  erretm = 8.0 * (phi - psi) + erretm + 2.0 * rhoinv + 3.0 * fabs(temp) +
           fabs(tau) * dw;
  if (fabs(w) <= eps * erretm) {
    *lam = orgati ? d[iev] + tau : d[ip1] + tau;
    return 0;
  }
  if (w <= 0.0)
    dltlb = fmax(dltlb, tau);
  else
    dltub = fmin(dltub, tau);
  niter++;
  if (!swtch3) {
    if (orgati)
      c = w - delta[ip1] * dw -
          (d[iev] - d[ip1]) * (z[iev] / delta[iev]) * (z[iev] / delta[iev]);
    else
      c = w - delta[iev] * dw -
          (d[ip1] - d[iev]) * (z[ip1] / delta[ip1]) * (z[ip1] / delta[ip1]);
    a = (delta[iev] + delta[ip1]) * w - delta[iev] * delta[ip1] * dw;
    b = delta[iev] * delta[ip1] * w;
    if (c == 0.0) {
      if (a == 0.0) {
        if (orgati)
          a = z[iev] * z[iev] + delta[ip1] * delta[ip1] * (dpsi + dphi);
        else
          a = z[ip1] * z[ip1] + delta[iev] * delta[iev] * (dpsi + dphi);
      }
      eta = b / a;
    } else if (a <= 0.0)
      eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    else
      eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
  } else {
    temp = rhoinv + psi + phi;
    if (orgati) {
      temp1 = z[iim1] / delta[iim1];
      temp1 = temp1 * temp1;
      c = temp - delta[iip1] * (dpsi + dphi) - (d[iim1] - d[iip1]) * temp1;
      zz[0] = z[iim1] * z[iim1];
      zz[2] = delta[iip1] * delta[iip1] * ((dpsi - temp1) + dphi);
    } else {
      temp1 = z[iip1] / delta[iip1];
      temp1 = temp1 * temp1;
      c = temp - delta[iim1] * (dpsi + dphi) - (d[iip1] - d[iim1]) * temp1;
      zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
      zz[2] = z[iip1] * z[iip1];
    }
    zz[1] = z[ii] * z[ii];
    if (la_ed6(niter, orgati, c, &delta[iim1], zz, w, &eta)) {
      *lam = orgati ? d[iev] + tau : d[ip1] + tau;
      return 1;
    }
  }
  if (w * eta >= 0.0)
    eta = -w / dw;
  temp = tau + eta;
  if (temp > dltub || temp < dltlb) {
    if (w < 0.0)
      eta = (dltub - tau) / 2.0;
    else
      eta = (dltlb - tau) / 2.0;
  }
  prew = w;
  for (j = 0; j < n; j++)
    delta[j] -= eta;
  dpsi = 0.0;
  psi = 0.0;
  erretm = 0.0;
  for (j = 0; j < ii; j++) {
    temp = z[j] / delta[j];
    psi += z[j] * temp;
    dpsi += temp * temp;
    erretm += psi;
  }
  erretm = fabs(erretm);
  dphi = 0.0;
  phi = 0.0;
  for (j = n - 1; j > ii; j--) {
    temp = z[j] / delta[j];
    phi += z[j] * temp;
    dphi += temp * temp;
    erretm += phi;
  }
  temp = z[ii] / delta[ii];
  dw = dpsi + dphi + temp * temp;
  temp = z[ii] * temp;
  w = rhoinv + phi + psi + temp;
  erretm = 8.0 * (phi - psi) + erretm + 2.0 * rhoinv + 3.0 * fabs(temp) +
           fabs(tau + eta) * dw;
  swtch = 0;
  if (orgati) {
    if (-w > fabs(prew) / 10.0)
      swtch = 1;
  } else {
    if (w > fabs(prew) / 10.0)
      swtch = 1;
  }
  tau += eta;
  iter = niter + 1;
  for (niter = iter; niter <= MAXIT; niter++) {
    if (fabs(w) <= eps * erretm) {
      *lam = orgati ? d[iev] + tau : d[ip1] + tau;
      return 0;
    }
    if (w <= 0.0)
      dltlb = fmax(dltlb, tau);
    else
      dltub = fmin(dltub, tau);
    if (!swtch3) {
      if (!swtch) {
        if (orgati)
          c = w - delta[ip1] * dw -
              (d[iev] - d[ip1]) * (z[iev] / delta[iev]) * (z[iev] / delta[iev]);
        else
          c = w - delta[iev] * dw -
              (d[ip1] - d[iev]) * (z[ip1] / delta[ip1]) * (z[ip1] / delta[ip1]);
      } else {
        temp = z[ii] / delta[ii];
        if (orgati)
          dpsi += temp * temp;
        else
          dphi += temp * temp;
        c = w - delta[iev] * dpsi - delta[ip1] * dphi;
      }
      a = (delta[iev] + delta[ip1]) * w - delta[iev] * delta[ip1] * dw;
      b = delta[iev] * delta[ip1] * w;
      if (c == 0.0) {
        if (a == 0.0) {
          if (!swtch) {
            if (orgati)
              a = z[iev] * z[iev] + delta[ip1] * delta[ip1] * (dpsi + dphi);
            else
              a = z[ip1] * z[ip1] + delta[iev] * delta[iev] * (dpsi + dphi);
          } else {
            a = delta[iev] * delta[iev] * dpsi + delta[ip1] * delta[ip1] * dphi;
          }
        }
        eta = b / a;
      } else if (a <= 0.0)
        eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
      else
        eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    } else {
      temp = rhoinv + psi + phi;
      if (swtch) {
        c = temp - delta[iim1] * dpsi - delta[iip1] * dphi;
        zz[0] = delta[iim1] * delta[iim1] * dpsi;
        zz[2] = delta[iip1] * delta[iip1] * dphi;
      } else {
        if (orgati) {
          temp1 = z[iim1] / delta[iim1];
          temp1 = temp1 * temp1;
          c = temp - delta[iip1] * (dpsi + dphi) - (d[iim1] - d[iip1]) * temp1;
          zz[0] = z[iim1] * z[iim1];
          zz[2] = delta[iip1] * delta[iip1] * ((dpsi - temp1) + dphi);
        } else {
          temp1 = z[iip1] / delta[iip1];
          temp1 = temp1 * temp1;
          c = temp - delta[iim1] * (dpsi + dphi) - (d[iip1] - d[iim1]) * temp1;
          zz[0] = delta[iim1] * delta[iim1] * (dpsi + (dphi - temp1));
          zz[2] = z[iip1] * z[iip1];
        }
      }
      zz[1] = z[ii] * z[ii];
      if (la_ed6(niter, orgati, c, &delta[iim1], zz, w, &eta)) {
        *lam = orgati ? d[iev] + tau : d[ip1] + tau;
        return 1;
      }
    }
    if (w * eta >= 0.0)
      eta = -w / dw;
    temp = tau + eta;
    if (temp > dltub || temp < dltlb) {
      eta1 = -w / dw;
      temp = tau + eta1;
      if (w < 0.0)
        eta2 = (dltub - tau) / 2.0;
      else
        eta2 = (dltlb - tau) / 2.0;
      if (dltlb <= temp && temp <= dltub)
        eta = copysign(1.0, eta1) * sqrt(fabs(eta1)) * sqrt(fabs(eta2));
      else
        eta = eta2;
    }
    for (j = 0; j < n; j++)
      delta[j] -= eta;
    tau += eta;
    prew = w;
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for (j = 0; j < ii; j++) {
      temp = z[j] / delta[j];
      psi += z[j] * temp;
      dpsi += temp * temp;
      erretm += psi;
    }
    erretm = fabs(erretm);
    dphi = 0.0;
    phi = 0.0;
    for (j = n - 1; j > ii; j--) {
      temp = z[j] / delta[j];
      phi += z[j] * temp;
      dphi += temp * temp;
      erretm += phi;
    }
    temp = z[ii] / delta[ii];
    dw = dpsi + dphi + temp * temp;
    temp = z[ii] * temp;
    w = rhoinv + phi + psi + temp;
    erretm = 8.0 * (phi - psi) + erretm + 2.0 * rhoinv + 3.0 * fabs(temp) +
             fabs(tau) * dw;
    if (w * prew > 0.0 && fabs(w) > fabs(prew) / 10.0)
      swtch = !swtch;
  }
  *lam = orgati ? d[iev] + tau : d[ip1] + tau;
  return 1;
}

/* ── small BLAS-ish helpers (double) ────────────────────────────────────── */
static int la_dc_idamax(int n, const double *x) {
  int im = 0;
  double mx = fabs(x[0]);
  for (int i = 1; i < n; i++)
    if (fabs(x[i]) > mx) { mx = fabs(x[i]); im = i; }
  return im;
}
static double la_dc_dnrm2(int n, const double *x) {
  double s = 0.0;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return sqrt(s);
}
/* dlamrg: merge two sorted sublists of a[] into a permutation index (0-based). */
static void la_dc_dlamrg(int n1, int n2, const double *a, int dtrd1, int dtrd2,
                      int *index) {
  int n1sv = n1, n2sv = n2, ind1, ind2, i = 0;
  ind1 = (dtrd1 > 0) ? 0 : n1 - 1;
  ind2 = (dtrd2 > 0) ? n1 : n1 + n2 - 1;
  while (n1sv > 0 && n2sv > 0) {
    if (a[ind1] <= a[ind2]) { index[i++] = ind1; ind1 += dtrd1; n1sv--; }
    else { index[i++] = ind2; ind2 += dtrd2; n2sv--; }
  }
  if (n1sv == 0) while (n2sv-- > 0) { index[i++] = ind2; ind2 += dtrd2; }
  else while (n1sv-- > 0) { index[i++] = ind1; ind1 += dtrd1; }
}

/* dlaed3: eigenvectors of the deflated rank-one system. Column-major Q (ldq);
   fills Q(:,0:k) with the merged eigenvectors via two GEMMs against the packed
   deflated blocks q2. The Gu-Eisenstat z-recomputation (loop building w from the
   computed eigenvalues) gives orthogonality without reorthogonalization. */
static int la_ed3(int k, int n, int n1, double *dout, double *Q, int ldq,
                  double rho, double *dlambda, double *q2, const int *indx,
                  const int *ctot, double *w, double *s, void *gemm) {
  int i, ii, iq2off, j, n12, n2, n23;
  double temp;
  if (k == 0) return 0;
  for (j = 0; j < k; j++) {
    int info = la_ed4(k, j, dlambda, w, &Q[j * ldq], rho, &dout[j]);
    if (info != 0) return info;
  }
  if (k == 1) goto assemble;
  if (k == 2) {
    for (j = 0; j < k; j++) {
      w[0] = Q[0 + j * ldq];
      w[1] = Q[1 + j * ldq];
      ii = indx[0];
      Q[0 + j * ldq] = w[ii];
      ii = indx[1];
      Q[1 + j * ldq] = w[ii];
    }
    goto assemble;
  }
  for (i = 0; i < k; i++) s[i] = w[i];
  for (i = 0; i < k; i++) w[i] = Q[i + i * ldq];
  for (j = 0; j < k; j++) {
    for (i = 0; i < j; i++)
      w[i] = w[i] * (Q[i + j * ldq] / (dlambda[i] - dlambda[j]));
    for (i = j + 1; i < k; i++)
      w[i] = w[i] * (Q[i + j * ldq] / (dlambda[i] - dlambda[j]));
  }
  for (i = 0; i < k; i++) w[i] = copysign(sqrt(-w[i]), s[i]);
  for (j = 0; j < k; j++) {
    for (i = 0; i < k; i++) s[i] = w[i] / Q[i + j * ldq];
    temp = la_dc_dnrm2(k, s);
    for (i = 0; i < k; i++) { ii = indx[i]; Q[i + j * ldq] = s[ii] / temp; }
  }
assemble:
  n2 = n - n1;
  n12 = ctot[0] + ctot[1];
  n23 = ctot[1] + ctot[2];
  for (j = 0; j < k; j++)
    for (i = 0; i < n23; i++) s[i + j * n23] = Q[(ctot[0] + i) + j * ldq];
  iq2off = n1 * n12;
  if (n23 != 0) {
    nx_c_status gs = nx_c_gemm2d_ct_ws(
        NX_C_DTYPE_f64, n2, k, n23, (const char *)&q2[iq2off], 1, n2,
        (const char *)s, 1, n23, (char *)&Q[n1], 1, ldq, (char *)gemm);
    if (gs != NX_C_OK) return -1;
  } else {
    for (j = 0; j < k; j++)
      for (i = 0; i < n2; i++) Q[(n1 + i) + j * ldq] = 0.0;
  }
  for (j = 0; j < k; j++)
    for (i = 0; i < n12; i++) s[i + j * n12] = Q[i + j * ldq];
  if (n12 != 0) {
    nx_c_status gs =
        nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, n1, k, n12, (const char *)&q2[0], 1, n1,
                         (const char *)s, 1, n12, (char *)&Q[0], 1, ldq,
                         (char *)gemm);
    if (gs != NX_C_OK) return -1;
  } else {
    for (j = 0; j < k; j++)
      for (i = 0; i < n1; i++) Q[i + j * ldq] = 0.0;
  }
  return 0;
}

/* dlaed2: deflate the merged problem. Two deflation tests — negligible z-comp,
   and two close eigenvalues (a Givens rotation zeroing one z-comp, DROT into Q).
   Column types 1..4 (top-only / both / bottom-only / deflated) drive the q2
   packing dlaed3 back-multiplies. All permutation arrays are 0-based here. */
static void la_ed2(int *kout, int n, int n1, double *d, double *Q, int ldq,
                   int *indxq, double *rho, double *z, double *dlambda,
                   double *w, double *q2, int *indx, int *indxc, int *indxp,
                   int *coltyp) {
  int n2, i, j, k, k2, nj, pj = 0, ct, imax, jmax, iq1, iq2, ctot[4], psm[4];
  double eps, tol, t, s, c, tau;
  if (n == 0) { *kout = 0; return; }
  n2 = n - n1;
  if (*rho < 0.0) for (i = n1; i < n; i++) z[i] = -z[i];
  t = 1.0 / sqrt(2.0);
  for (i = 0; i < n; i++) z[i] *= t;
  *rho = fabs(2.0 * (*rho));
  for (i = n1; i < n; i++) indxq[i] += n1;
  for (i = 0; i < n; i++) dlambda[i] = d[indxq[i]];
  la_dc_dlamrg(n1, n2, dlambda, 1, 1, indxc);
  for (i = 0; i < n; i++) indx[i] = indxq[indxc[i]];
  imax = la_dc_idamax(n, z);
  jmax = la_dc_idamax(n, d);
  /* LAPACK's eps is DLAMCH('E'), the unit roundoff u = DBL_EPSILON/2 (C's
     DBL_EPSILON is 2u); the reference deflation tolerance is EIGHT*EIGHT*eps. */
  eps = 0.5 * DBL_EPSILON;
  tol = 8.0 * 8.0 * eps * fmax(fabs(d[jmax]), fabs(z[imax]));
  if ((*rho) * fabs(z[imax]) <= tol) {
    k = 0;
    iq2 = 0;
    for (j = 0; j < n; j++) {
      i = indx[j];
      for (int r = 0; r < n; r++) q2[iq2 + r] = Q[r + i * ldq];
      dlambda[j] = d[i];
      iq2 += n;
    }
    for (j = 0; j < n; j++)
      for (int r = 0; r < n; r++) Q[r + j * ldq] = q2[j * n + r];
    for (i = 0; i < n; i++) d[i] = dlambda[i];
    *kout = 0;
    return;
  }
  for (i = 0; i < n1; i++) coltyp[i] = 1;
  for (i = n1; i < n; i++) coltyp[i] = 3;
  k = 0;
  k2 = n;
  j = 0;
  int found = 0;
  for (; j < n; j++) {
    nj = indx[j];
    if ((*rho) * fabs(z[nj]) <= tol) {
      k2--;
      coltyp[nj] = 4;
      indxp[k2] = nj;
      if (j == n - 1) goto L100;
    } else { pj = nj; found = 1; break; }
  }
  if (!found) goto L100;
  while (1) {
    j++;
    if (j >= n) goto L100;
    nj = indx[j];
    if ((*rho) * fabs(z[nj]) <= tol) {
      k2--;
      coltyp[nj] = 4;
      indxp[k2] = nj;
    } else {
      s = z[pj];
      c = z[nj];
      tau = hypot(c, s);
      t = d[nj] - d[pj];
      c = c / tau;
      s = -s / tau;
      if (fabs(t * c * s) <= tol) {
        z[nj] = tau;
        z[pj] = 0.0;
        if (coltyp[nj] != coltyp[pj]) coltyp[nj] = 2;
        coltyp[pj] = 4;
        for (int r = 0; r < n; r++) {
          double qp = Q[r + pj * ldq], qn = Q[r + nj * ldq];
          Q[r + pj * ldq] = c * qp + s * qn;
          Q[r + nj * ldq] = -s * qp + c * qn;
        }
        t = d[pj] * c * c + d[nj] * s * s;
        d[nj] = d[pj] * s * s + d[nj] * c * c;
        d[pj] = t;
        k2--;
        i = 1;
        while (1) {
          if (k2 + i <= n - 1) {
            if (d[pj] < d[indxp[k2 + i]]) {
              indxp[k2 + i - 1] = indxp[k2 + i];
              indxp[k2 + i] = pj;
              i++;
              continue;
            } else { indxp[k2 + i - 1] = pj; break; }
          } else { indxp[k2 + i - 1] = pj; break; }
        }
        pj = nj;
      } else {
        k++;
        dlambda[k - 1] = d[pj];
        w[k - 1] = z[pj];
        indxp[k - 1] = pj;
        pj = nj;
      }
    }
  }
L100:
  k++;
  dlambda[k - 1] = d[pj];
  w[k - 1] = z[pj];
  indxp[k - 1] = pj;
  for (j = 0; j < 4; j++) ctot[j] = 0;
  for (j = 0; j < n; j++) { ct = coltyp[j]; ctot[ct - 1]++; }
  psm[0] = 0;
  psm[1] = ctot[0];
  psm[2] = psm[1] + ctot[1];
  psm[3] = psm[2] + ctot[2];
  k = n - ctot[3];
  for (j = 0; j < n; j++) {
    int js = indxp[j];
    ct = coltyp[js];
    indx[psm[ct - 1]] = js;
    indxc[psm[ct - 1]] = j;
    psm[ct - 1]++;
  }
  i = 0;
  iq1 = 0;
  iq2 = (ctot[0] + ctot[1]) * n1;
  for (j = 0; j < ctot[0]; j++) {
    int js = indx[i];
    for (int r = 0; r < n1; r++) q2[iq1 + r] = Q[r + js * ldq];
    z[i] = d[js];
    i++;
    iq1 += n1;
  }
  for (j = 0; j < ctot[1]; j++) {
    int js = indx[i];
    for (int r = 0; r < n1; r++) q2[iq1 + r] = Q[r + js * ldq];
    for (int r = 0; r < n2; r++) q2[iq2 + r] = Q[(n1 + r) + js * ldq];
    z[i] = d[js];
    i++;
    iq1 += n1;
    iq2 += n2;
  }
  for (j = 0; j < ctot[2]; j++) {
    int js = indx[i];
    for (int r = 0; r < n2; r++) q2[iq2 + r] = Q[(n1 + r) + js * ldq];
    z[i] = d[js];
    i++;
    iq2 += n2;
  }
  iq1 = iq2;
  for (j = 0; j < ctot[3]; j++) {
    int js = indx[i];
    for (int r = 0; r < n; r++) q2[iq2 + r] = Q[r + js * ldq];
    iq2 += n;
    z[i] = d[js];
    i++;
  }
  if (k < n) {
    for (j = 0; j < ctot[3]; j++)
      for (int r = 0; r < n; r++) Q[r + (k + j) * ldq] = q2[iq1 + j * n + r];
    for (i = 0; i < n - k; i++) d[k + i] = z[k + i];
  }
  for (j = 0; j < 4; j++) coltyp[j] = ctot[j];
  *kout = k;
}

/* dlaed1: merge two adjacent solved subproblems joined by rho at cutpnt. */
static int la_ed1(int n, double *d, double *Q, int ldq, int *indxq, double rho,
                  int cutpnt, double *work, int *iwork, void *gemm) {
  int k, n1, n2;
  double *z = work, *dlmda = work + n, *wv = work + 2 * n, *q2 = work + 3 * n;
  int *indx = iwork, *indxc = iwork + n, *coltyp = iwork + 2 * n,
      *indxp = iwork + 3 * n;
  if (n == 0) return 0;
  for (int i = 0; i < cutpnt; i++) z[i] = Q[(cutpnt - 1) + i * ldq];
  for (int i = 0; i < n - cutpnt; i++)
    z[cutpnt + i] = Q[cutpnt + (cutpnt + i) * ldq];
  double rho2 = rho;
  la_ed2(&k, n, cutpnt, d, Q, ldq, indxq, &rho2, z, dlmda, wv, q2, indx, indxc,
         indxp, coltyp);
  if (k != 0) {
    int isoff =
        (coltyp[0] + coltyp[1]) * cutpnt + (coltyp[1] + coltyp[2]) * (n - cutpnt);
    double *s = q2 + isoff;
    int info = la_ed3(k, n, cutpnt, d, Q, ldq, rho2, dlmda, q2, indxc, coltyp,
                      wv, s, gemm);
    if (info != 0) return info;
    n1 = k;
    n2 = n - k;
    la_dc_dlamrg(n1, n2, d, 1, -1, indxq);
  } else {
    for (int i = 0; i < n; i++) indxq[i] = i;
  }
  return 0;
}

/* Double base solver: identity-seeded QL (Wilkinson shift) on the tridiagonal,
   column-major Z, then an ascending sort permuting Z's columns. The dlaed0
   SMLSIZ leaf. */
static int la_dc_steqr(double *d, double *e, double *Z, int ldz, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) Z[i + j * ldz] = (i == j) ? 1.0 : 0.0;
  for (int l = 0; l < n; l++) {
    int iter = 0, m;
    do {
      for (m = l; m < n - 1; m++) {
        double dd = fabs(d[m]) + fabs(d[m + 1]);
        if (fabs(e[m]) + dd == dd) break;
      }
      if (m != l) {
        if (iter++ == 50) return 1;
        double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        double r = hypot(g, 1.0);
        double sg = (g >= 0.0) ? fabs(r) : -fabs(r);
        g = d[m] - d[l] + e[l] / (g + sg);
        double s = 1.0, c = 1.0, p = 0.0;
        int i;
        for (i = m - 1; i >= l; i--) {
          double f = s * e[i], b = c * e[i];
          r = hypot(f, g);
          e[i + 1] = r;
          if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; break; }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          for (int kk = 0; kk < n; kk++) {
            double fz = Z[kk + (i + 1) * ldz];
            Z[kk + (i + 1) * ldz] = s * Z[kk + i * ldz] + c * fz;
            Z[kk + i * ldz] = c * Z[kk + i * ldz] - s * fz;
          }
        }
        if (r == 0.0 && i >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m != l);
  }
  for (int i = 0; i < n - 1; i++) {
    int k = i;
    double p = d[i];
    for (int j = i + 1; j < n; j++)
      if (d[j] < p) { k = j; p = d[j]; }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int r = 0; r < n; r++) {
        double t = Z[r + i * ldz];
        Z[r + i * ldz] = Z[r + k * ldz];
        Z[r + k * ldz] = t;
      }
    }
  }
  return 0;
}

/* dlaed0-style recursive driver: tear at n/2, conquer, merge. Q is n×n
   column-major eigenvectors of the tridiagonal (d,e). toplevel!=0 applies the
   final ascending permutation to d and Q. */
static int la_stedc(double *d, double *e, double *Q, int ldq, int n, int *indxq,
                    double *work, int *iwork, void *gemm, int toplevel) {
  if (n == 0) return 0;
  if (n <= LA_DC_SMLSIZ) {
    int info = la_dc_steqr(d, e, Q, ldq, n);
    if (info) return info;
    for (int i = 0; i < n; i++) indxq[i] = i;
    return 0;
  }
  int n1 = n / 2, n2 = n - n1;
  double rho = e[n1 - 1];
  d[n1 - 1] -= fabs(rho);
  d[n1] -= fabs(rho);
  int info = la_stedc(d, e, Q, ldq, n1, indxq, work, iwork, gemm, 0);
  if (info) return info;
  info = la_stedc(d + n1, e + n1, &Q[n1 + n1 * ldq], ldq, n2, indxq + n1, work,
                  iwork, gemm, 0);
  if (info) return info;
  info = la_ed1(n, d, Q, ldq, indxq, rho, n1, work, iwork, gemm);
  if (info) return info;
  if (toplevel) {
    double *dt = work, *qt = work + n;
    for (int i = 0; i < n; i++) {
      int jj = indxq[i];
      dt[i] = d[jj];
      for (int r = 0; r < n; r++) qt[r + i * n] = Q[r + jj * ldq];
    }
    for (int i = 0; i < n; i++) d[i] = dt[i];
    for (int i = 0; i < n; i++)
      for (int r = 0; r < n; r++) Q[r + i * ldq] = qt[r + i * n];
  }
  return 0;
}

/* Form the eigenvectors V = Q_householder · Z_tri (compute-typed), the D&C's
   BLAS-3 back-multiply. The Householder Q that reduced A to tridiagonal form is
   applied WITHOUT the O(n^3) BLAS-2 la_orgtr: dorgtr's identity — for the lower
   reduction, Q = [[1,0],[0,Q']] where Q' is the QR form-Q of the subdiagonal
   reflectors shifted up one row (rows/cols [1,n) of A). So build the shifted
   reflector panel AR, form Q' through the shared compact-WY nx_c_la_qrq (BLAS-3),
   then V's row 0 is Z_tri's row 0 and V's rows [1,n) are Q'·Z_tri[1:,:] — one
   GEMM. Z_tri arrives real (double, column-major); it is lifted to the compute
   type in zc as the first step. Every buffer is caller scratch; AR/Qp/tauR/qV/qVc
   reuse the (post-tridiag idle) latrd panels, qT/qW/qP are D&C-owned. */
#define LA_GEN_FORMV(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)             \
  static void la_dc_formv_##sfx(                                               \
      void *vwork, const void *vtau, const double *ztri, int64_t n, void *vzc, \
      void *vAR, void *vQp, void *vtauR, void *qV, void *qVc, void *qT,        \
      void *qW, void *qP, void *gemm) {                                        \
    T *work = (T *)vwork;                                                      \
    const T *tau = (const T *)vtau;                                            \
    T *zc = (T *)vzc;                                                          \
    T *AR = (T *)vAR;                                                          \
    T *Qp = (T *)vQp;                                                          \
    T *tauR = (T *)vtauR;                                                      \
    for (int64_t i = 0; i < n * n; i++) zc[i] = LA_MK_##sfx((R)ztri[i], (R)0); \
    int64_t m = n - 1;                                                         \
    for (int64_t c = 0; c < m; c++) {                                          \
      for (int64_t r = c + 1; r < m; r++) AR[r * m + c] = work[(r + 1) * n + c]; \
      tauR[c] = tau[c];                                                        \
    }                                                                          \
    nx_c_la_qrq_##sfx(AR, m, m, m, tauR, Qp, m, m, qV, qVc, qT, qW, qP, gemm);  \
    for (int64_t j = 0; j < n; j++) work[j] = zc[j * n];                       \
    nx_c_gemm2d_ct_ws(DT, m, n, m, (const char *)Qp, m, 1, (const char *)&zc[1], \
                     1, n, (char *)&work[n], n, 1, (char *)gemm);              \
  }
#define LA_EXPAND_FORMV(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)          \
  LA_GEN_FORMV(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_FORMV)
LA_TRAITS_double(LA_EXPAND_FORMV)
LA_TRAITS_c32(LA_EXPAND_FORMV)
LA_TRAITS_c64(LA_EXPAND_FORMV)
#undef LA_EXPAND_FORMV

typedef void (*la_dc_formv_fn)(void *, const void *, const double *, int64_t,
                               void *, void *, void *, void *, void *, void *,
                               void *, void *, void *, void *);
static const la_dc_formv_fn la_dc_formv[LA_NCOMPUTE] = {
    [LA_F32] = la_dc_formv_f32, [LA_F64] = la_dc_formv_f64,
    [LA_C32] = la_dc_formv_c32, [LA_C64] = la_dc_formv_c64};

static const la_compute_desc la_desc[LA_NCOMPUTE] = {
    [LA_F32] = {.csize = sizeof(float), .gemm_dt = NX_C_DTYPE_f32,
                .tridiag = la_tridiag_f32, .orgtr = la_orgtr_f32,
                .tql2 = la_tql2_f32, .eigsort = la_eigsort_f32},
    [LA_F64] = {.csize = sizeof(double), .gemm_dt = NX_C_DTYPE_f64,
                .tridiag = la_tridiag_f64, .orgtr = la_orgtr_f64,
                .tql2 = la_tql2_f64, .eigsort = la_eigsort_f64},
    [LA_C32] = {.csize = sizeof(nx_c_complex32), .gemm_dt = NX_C_DTYPE_c32,
                .tridiag = la_tridiag_c32, .orgtr = la_orgtr_c32,
                .tql2 = la_tql2_c32, .eigsort = la_eigsort_c32},
    [LA_C64] = {.csize = sizeof(nx_c_complex64), .gemm_dt = NX_C_DTYPE_c64,
                .tridiag = la_tridiag_c64, .orgtr = la_orgtr_c64,
                .tql2 = la_tql2_c64, .eigsort = la_eigsort_c64},
};

/* ── eigh driver: batched, pooled ────────────────────────────────────────
   in is [batch, n, n] symmetric/Hermitian (LOWER triangle read). Eigenvalues w
   are [batch, n] float64 ascending; eigenvectors v (only when vectors) are
   [batch, n, n] input-dtype, columns = eigenvectors. Each worker owns a combined
   scratch block: work (n×n), Z (n×n, the Q/eigenvector accumulator), tau/wv
   (n each) compute-typed, and d/e (n each) real. Non-convergence in the QL
   sweep → LA_ERR_NO_CONVERGE, reported per worker and raised. */
typedef struct {
  const nx_c_ndarray *in;
  const nx_c_ndarray *w;
  const nx_c_ndarray *v;
  nx_c_dtype dt;
  la_compute lc;
  int is_double;
  int64_t n, esz;
  int vectors;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *in_bs;
  const int64_t *w_bs;
  const int64_t *v_bs;
  int64_t in_rs, in_cs, w_cs, v_rs, v_cs;
  char *scratch;
  int64_t stride, off_z, off_tau, off_wv, off_d, off_e, off_tW, off_tWc,
      off_tP, off_tg;
  /* divide-and-conquer scratch (only touched on the vectors && n>SMLSIZ path):
     widened double d/e, the real eigenvector matrix Ztri, its compute-typed
     copy Zc, the merge permutation indxq, the dlaed workspace + iwork, and the
     back-multiply GEMM panels. */
  int use_dc;
  int64_t off_dd, off_de, off_ztri, off_zc, off_indxq, off_dcwork, off_dciwork,
      off_dcgemm, off_qt, off_qw, off_qp;
  nx_c_status *werr;
} la_eigh_ctx;

static void la_eigh_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  la_eigh_ctx *x = (la_eigh_ctx *)vctx;
  const la_compute_desc *cd = &la_desc[x->lc];
  const la_move_desc *mv = &la_move[x->dt];
  char *base = x->scratch + (int64_t)worker * x->stride;
  void *work = base;
  void *Z = base + x->off_z;
  void *tau = base + x->off_tau;
  void *wv = base + x->off_wv;
  void *d = base + x->off_d;
  void *e = base + x->off_e;
  void *tW = base + x->off_tW;
  void *tWc = base + x->off_tWc;
  void *tP = base + x->off_tP;
  void *tg = base + x->off_tg;
  int64_t n = x->n;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *inb;
    const char *wb;
    la_batch_base(bt, x->batch_nd, x->bshape, x->in_bs, x->in->offset, x->esz,
                  (const char *)x->in->data, &inb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->w_bs, x->w->offset,
                  (int64_t)sizeof(double), (const char *)x->w->data, &wb);
    mv->unpack(inb, x->in_rs, x->in_cs, n, n, work, n);
    cd->tridiag(work, n, n, d, e, tau, wv, tW, tWc, tP, tg);
    if (x->use_dc) {
      /* divide-and-conquer with-vectors path (n > SMLSIZ). Solve the real
         tridiagonal in double, then V = Q_householder · Z_tri. */
      double *dd = (double *)(base + x->off_dd);
      double *de = (double *)(base + x->off_de);
      double *ztri = (double *)(base + x->off_ztri);
      void *zc = base + x->off_zc;
      int *indxq = (int *)(base + x->off_indxq);
      double *dcwork = (double *)(base + x->off_dcwork);
      int *dciwork = (int *)(base + x->off_dciwork);
      void *dcgemm = base + x->off_dcgemm;
      if (x->is_double) {
        const double *dv = (const double *)d, *ev = (const double *)e;
        for (int64_t i = 0; i < n; i++) {
          dd[i] = dv[i];
          de[i] = ev[i];
        }
      } else {
        const float *dv = (const float *)d, *ev = (const float *)e;
        for (int64_t i = 0; i < n; i++) {
          dd[i] = dv[i];
          de[i] = ev[i];
        }
      }
      memset(ztri, 0, (size_t)n * n * sizeof(double));
      int info = la_stedc(dd, de, ztri, (int)n, (int)n, indxq, dcwork, dciwork,
                          dcgemm, 1);
      if (info != 0) {
        if (x->werr[worker] == NX_C_OK) x->werr[worker] = LA_ERR_NO_CONVERGE;
        continue;
      }
      /* V = Q_householder · Z_tri via the blocked shifted-reflector form-Q; the
         AR/Qp/tauR/qV/qVc buffers reuse the now-idle latrd panels tP/Z/wv/tW/tWc,
         so only qT/qW/qP are D&C-owned scratch. Fills V (compute type) into work,
         also lifting Z_tri to the compute type in zc. */
      la_dc_formv[x->lc](work, tau, ztri, n, zc, tP, Z, wv, tW, tWc,
                         base + x->off_qt, base + x->off_qw, base + x->off_qp,
                         dcgemm);
      double *wd = (double *)wb;
      for (int64_t i = 0; i < n; i++) wd[i * x->w_cs] = dd[i];
      const char *vb;
      la_batch_base(bt, x->batch_nd, x->bshape, x->v_bs, x->v->offset, x->esz,
                    (const char *)x->v->data, &vb);
      mv->packfull(work, n, n, n, (char *)vb, x->v_rs, x->v_cs);
      continue;
    }
    if (x->vectors) cd->orgtr(work, n, n, tau, Z, n);
    nx_c_status s = cd->tql2(d, e, Z, n, n, x->vectors);
    if (s != NX_C_OK) {
      if (x->werr[worker] == NX_C_OK) x->werr[worker] = s;
      continue;
    }
    cd->eigsort(d, Z, n, n, x->vectors);
    double *wd = (double *)wb;
    if (x->is_double) {
      const double *dv = (const double *)d;
      for (int64_t i = 0; i < n; i++) wd[i * x->w_cs] = dv[i];
    } else {
      const float *dv = (const float *)d;
      for (int64_t i = 0; i < n; i++) wd[i * x->w_cs] = (double)dv[i];
    }
    if (x->vectors) {
      const char *vb;
      la_batch_base(bt, x->batch_nd, x->bshape, x->v_bs, x->v->offset, x->esz,
                    (const char *)x->v->data, &vb);
      mv->packfull(Z, n, n, n, (char *)vb, x->v_rs, x->v_cs);
    }
  }
}

static nx_c_status nx_c_eigh_run(const nx_c_ndarray *in, const nx_c_ndarray *w,
                               const nx_c_ndarray *v, nx_c_dtype dt, int vectors) {
  if (in->ndim < 2) return LA_ERR_SHAPE_LA;
  int64_t n = in->shape[in->ndim - 1];
  if (in->shape[in->ndim - 2] != n) return LA_ERR_NOT_SQUARE;
  if (w->ndim != in->ndim - 1 || w->shape[w->ndim - 1] != n)
    return LA_ERR_SHAPE_LA;
  if (vectors &&
      (v->ndim != in->ndim || v->shape[v->ndim - 1] != n ||
       v->shape[v->ndim - 2] != n))
    return LA_ERR_SHAPE_LA;
  la_compute lc = la_compute_of(dt);
  if (lc == LA_NCOMPUTE) return LA_ERR_NOT_FLOAT;
  const la_compute_desc *cd = &la_desc[lc];
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = in->ndim - 2;
  int64_t bshape[NX_C_MAX_NDIM], in_bs[NX_C_MAX_NDIM], w_bs[NX_C_MAX_NDIM],
      v_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (w->shape[i] != in->shape[i]) return LA_ERR_SHAPE_LA;
    if (vectors && v->shape[i] != in->shape[i]) return LA_ERR_SHAPE_LA;
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
  if (nth > LA_MAX_WORKERS) nth = LA_MAX_WORKERS;
  if (nth < 1) nth = 1;

  int64_t rsize = (lc == LA_F64 || lc == LA_C64) ? 8 : 4;
#define LA_ALN(b) (((b) + 63) & ~(int64_t)63)
  int64_t a_mat = LA_ALN(n * n * cd->csize);
  int64_t a_vec = LA_ALN(n * cd->csize);
  int64_t a_rvec = LA_ALN(n * rsize);
  /* latrd panel W (n×NB), its conj panel Wc for the rank-2k GEMM's Bᴴ operand,
     the V·Wᴴ product P (n×n), and the GEMM packing panels — unused below the
     LA_EIGH_NB crossover, always allocated. */
  int64_t a_tW = LA_ALN(n * (int64_t)LA_EIGH_NB * cd->csize);
  int64_t tg = nx_c_gemm2d_ct_scratch(cd->gemm_dt, n, n, LA_EIGH_NB);
  int64_t a_tg = LA_ALN(tg);
  int64_t off_z = a_mat, off_tau = off_z + a_mat, off_wv = off_tau + a_vec;
  int64_t off_d = off_wv + a_vec, off_e = off_d + a_rvec;
  int64_t off_tW = off_e + a_rvec, off_tWc = off_tW + a_tW,
          off_tP = off_tWc + a_tW;
  int64_t off_tg = off_tP + a_mat;
  /* D&C scratch (0 unless the with-vectors D&C path runs): widened d/e (double),
     the real eigenvector matrix Ztri (double), its compute-typed copy Zc, the
     merge permutation indxq, the dlaed workspace (n^2+4n doubles) + iwork (4n),
     and the back-multiply GEMM panels sized for the largest merge/apply GEMM
     (the dlaed3 GEMM is f64, the apply GEMM the compute dtype — take the max). */
  int use_dc = vectors && n > LA_DC_SMLSIZ;
  int64_t a_dvec = use_dc ? LA_ALN(n * (int64_t)sizeof(double)) : 0;
  int64_t a_ztri = use_dc ? LA_ALN(n * n * (int64_t)sizeof(double)) : 0;
  int64_t a_zc = use_dc ? a_mat : 0;
  int64_t a_indxq = use_dc ? LA_ALN(n * (int64_t)sizeof(int)) : 0;
  int64_t a_dcwork =
      use_dc ? LA_ALN((n * n + 4 * n) * (int64_t)sizeof(double)) : 0;
  int64_t a_dciwork = use_dc ? LA_ALN(4 * n * (int64_t)sizeof(int)) : 0;
  int64_t dcg_f64 = use_dc ? nx_c_gemm2d_ct_scratch(NX_C_DTYPE_f64, n, n, n) : 0;
  int64_t dcg_ct = use_dc ? nx_c_gemm2d_ct_scratch(cd->gemm_dt, n, n, n) : 0;
  int64_t a_dcgemm = LA_ALN(dcg_f64 > dcg_ct ? dcg_f64 : dcg_ct);
  /* form-Q compact-WY scratch owned here (qV/qVc/AR/Qp/tauR reuse tW/tWc/tP/Z/wv):
     the block factor qT, the Vᴴ·C product qW, and the V·(TᴴW) product qP. */
  int64_t a_qt = use_dc ? LA_ALN((int64_t)LA_QR_NB * LA_QR_NB * cd->csize) : 0;
  int64_t a_qw = use_dc ? LA_ALN(n * (int64_t)LA_QR_NB * cd->csize) : 0;
  int64_t a_qp = use_dc ? a_mat : 0;
  int64_t off_dd = off_tg + a_tg, off_de = off_dd + a_dvec,
          off_ztri = off_de + a_dvec, off_zc = off_ztri + a_ztri;
  int64_t off_indxq = off_zc + a_zc, off_dcwork = off_indxq + a_indxq,
          off_dciwork = off_dcwork + a_dcwork,
          off_dcgemm = off_dciwork + a_dciwork;
  int64_t off_qt = off_dcgemm + a_dcgemm, off_qw = off_qt + a_qt,
          off_qp = off_qw + a_qw;
  int64_t stride = off_qp + a_qp;
#undef LA_ALN
  char *scratch = aligned_alloc(64, (size_t)stride * nth);
  if (!scratch) return NX_C_ERR_ALLOC;

  nx_c_status werr[LA_MAX_WORKERS];
  for (int i = 0; i < nth; i++) werr[i] = NX_C_OK;

  la_eigh_ctx x;
  x.in = in;
  x.w = w;
  x.v = v;
  x.dt = dt;
  x.lc = lc;
  x.is_double = (lc == LA_F64 || lc == LA_C64);
  x.n = n;
  x.esz = esz;
  x.vectors = vectors;
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
  x.off_z = off_z;
  x.off_tau = off_tau;
  x.off_wv = off_wv;
  x.off_d = off_d;
  x.off_e = off_e;
  x.off_tW = off_tW;
  x.off_tWc = off_tWc;
  x.off_tP = off_tP;
  x.off_tg = off_tg;
  x.use_dc = use_dc;
  x.off_dd = off_dd;
  x.off_de = off_de;
  x.off_ztri = off_ztri;
  x.off_zc = off_zc;
  x.off_indxq = off_indxq;
  x.off_dcwork = off_dcwork;
  x.off_dciwork = off_dciwork;
  x.off_dcgemm = off_dcgemm;
  x.off_qt = off_qt;
  x.off_qw = off_qw;
  x.off_qp = off_qp;
  x.werr = werr;

  nx_c_parallel_for(nth, nbatch, bytes, la_eigh_body, &x, scratch);

  nx_c_status err = NX_C_OK;
  for (int i = 0; i < nth; i++)
    if (werr[i] != NX_C_OK) {
      err = werr[i];
      break;
    }
  return err;
}

/* vw: float64 eigenvalues [batch, n] (ascending). vv: input-dtype eigenvectors
   [batch, n, n], columns = eigenvectors — allocated by the binding ONLY when
   vectors is true. ABI: when vectors is false the binding re-passes the values
   tensor in the vv slot; this stub must not extract or touch vv then (it is read
   only inside the `if (vectors)` guard below). */
CAMLprim value caml_nx_c_eigh(value vw, value vv, value vin, value vvectors) {
  CAMLparam4(vw, vv, vin, vvectors);
  int vectors = Bool_val(vvectors);
  nx_c_ndarray in, w, v;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vw, &w);
  if (s == NX_C_OK && vectors) s = nx_c_ndarray_of_value(vv, &v);
  if (s != NX_C_OK) la_raise("eigh", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt == NX_C_DTYPE_COUNT) la_raise("eigh", NX_C_ERR_BAD_KIND);
  s = nx_c_eigh_run(&in, &w, vectors ? &v : NULL, dt, vectors);
  if (s != NX_C_OK) la_raise("eigh", s);
  CAMLreturn(Val_unit);
}
