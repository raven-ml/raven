/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_svd.c — singular value decomposition: Golub–Kahan bidiagonalization +
   divide-and-conquer bidiagonal SVD (dbdsdc/dlasd, above SMLSIZ) with the
   Demmel–Kahan implicit-QR sweep as the small-n path and D&C leaf solver
   (linalg tier 3). U formation reuses the QR compact-WY form-Q
   (nx_c_la_qrq_*, nx_c_linalg.h). Shared machinery there. */

#include "nx_c_linalg.h"

/* ── SVD: Golub–Kahan bidiagonalization + Demmel–Kahan implicit QR ─────────

   A = U Σ Vᴴ via a direct bidiagonal path instead of a Jordan–Wielandt
   embedding, which would run a full eigh on the 2n-order
   Hermitian [[0,A],[Aᴴ,0]]: ~64n³/3 flops and a (2n)² dense matrix, versus
   ~20n³/3 and a few n² buffers here (≈3× fewer flops, ≈4× less working memory).

   The tall/square working matrix P (pr×pc, pr>=pc) is A when m>=n; for m<n we
   bidiagonalize Aᴴ (pr=n, pc=m) and swap the roles of U and V at the end
   (LAPACK's transpose trick — one upper-bidiagonal path serves both). Two-sided
   Householder (la_gebrd) reduces P to a REAL upper bidiagonal (d, e): the left
   reflectors are la_qr's column reflectors (real beta on the diagonal), and the
   right reflectors are generated from the CONJUGATE of each row so their beta is
   real too — so B is real for any input dtype and the phase lives entirely in the
   accumulated unitaries (the larfg real-beta property eigh's tridiag also uses).
   la_qrq forms Q_U (the left transform); la_formp forms Q_Vᴴ (the right). The
   real bidiagonal SVD (la_bdsvd) is the dbdsqr sweep: implicit zero-shift QR for
   high relative accuracy on tiny singular values, Wilkinson-shifted QR when the
   relative gap is wide, both chase directions, 2×2 deflation via la_lasv2, a
   negligible e deflated by a scale-aware threshold; it accumulates the real Givens
   into the compute-typed U/Vᴴ (T·R, as tql2 does) and returns the singular values
   descending, sign-folded nonnegative. Non-convergence (>6n² rotations) RAISES.

   The rotation scalars are computed in double for every compute type (the f32/c32
   bidiagonal QR gains accuracy at no material cost); only d/e/S carry the real
   scalar R, and the vectors the compute type T. */

/* Robust real plane rotation: [cs sn; -sn cs]·[f;g] = [r;0] with cs >= 0. */
static void la_lartg(double f, double g, double *cs, double *sn, double *r) {
  if (g == 0.0) {
    *cs = 1.0;
    *sn = 0.0;
    *r = f;
  } else if (f == 0.0) {
    *cs = 0.0;
    *sn = 1.0;
    *r = g;
  } else {
    double d = hypot(f, g);
    double rr = f >= 0.0 ? d : -d;
    *cs = f / rr;
    *sn = g / rr;
    *r = rr;
  }
}

/* Smaller singular value of the 2×2 upper-triangular [[f,g],[0,h]] (LAPACK dlas2);
   the Wilkinson shift for the bidiagonal QR. */
static double la_las2(double f, double g, double h) {
  double fa = fabs(f), ga = fabs(g), ha = fabs(h);
  double fhmn = fa < ha ? fa : ha;
  double fhmx = fa < ha ? ha : fa;
  if (fhmn == 0.0) return 0.0;
  if (ga < fhmx) {
    double as = 1.0 + fhmn / fhmx;
    double at = (fhmx - fhmn) / fhmx;
    double au = (ga / fhmx) * (ga / fhmx);
    double c = 2.0 / (sqrt(as * as + au) + sqrt(at * at + au));
    return fhmn * c;
  } else {
    double au = fhmx / ga;
    if (au == 0.0) return (fhmn * fhmx) / ga;
    double as = 1.0 + fhmn / fhmx;
    double at = (fhmx - fhmn) / fhmx;
    double c = 1.0 / (sqrt(1.0 + (as * au) * (as * au)) +
                      sqrt(1.0 + (at * au) * (at * au)));
    double smin = (fhmn * c) * au;
    return smin + smin;
  }
}

/* Full 2×2 SVD of [[f,g],[0,h]] (LAPACK dlasv2): singular values + the left/right
   rotations that diagonalize the trailing 2×2 block for exact deflation. */
static void la_lasv2(double f, double g, double h, double *ssmin, double *ssmax,
                     double *snr, double *csr, double *snl, double *csl) {
  double ft = f, ht = h, gt = g;
  double fa = fabs(ft), ha = fabs(ht), ga = fabs(gt);
  int pmax = 1;
  int swap = (ha > fa);
  if (swap) {
    pmax = 3;
    double t = ft;
    ft = ht;
    ht = t;
    t = fa;
    fa = ha;
    ha = t;
  }
  double clt = 0, crt = 0, slt = 0, srt = 0;
  if (ga == 0.0) {
    *ssmin = ha;
    *ssmax = fa;
    clt = 1.0;
    crt = 1.0;
    slt = 0.0;
    srt = 0.0;
  } else {
    int gasmal = 1;
    if (ga > fa) {
      pmax = 2;
      if ((fa / ga) < DBL_EPSILON) {
        gasmal = 0;
        *ssmax = ga;
        if (ha > 1.0)
          *ssmin = fa / (ga / ha);
        else
          *ssmin = (fa / ga) * ha;
        clt = 1.0;
        slt = ht / gt;
        srt = 1.0;
        crt = ft / gt;
      }
    }
    if (gasmal) {
      double d = fa - ha;
      double l = (d == fa) ? 1.0 : d / fa;
      double m = gt / ft;
      double t = 2.0 - l;
      double mm = m * m, tt = t * t;
      double s = sqrt(tt + mm);
      double r = (l == 0.0) ? fabs(m) : sqrt(l * l + mm);
      double a = 0.5 * (s + r);
      *ssmin = ha / a;
      *ssmax = fa * a;
      if (mm == 0.0) {
        if (l == 0.0)
          t = copysign(2.0, ft) * copysign(1.0, gt);
        else
          t = gt / copysign(d, ft) + m / t;
      } else {
        t = (m / (s + t) + m / (r + l)) * (1.0 + a);
      }
      double l2 = sqrt(t * t + 4.0);
      crt = 2.0 / l2;
      srt = t / l2;
      clt = (crt + srt * m) / a;
      slt = (ht / ft) * srt / a;
    }
  }
  if (swap) {
    *csl = srt;
    *snl = crt;
    *csr = slt;
    *snr = clt;
  } else {
    *csl = clt;
    *snl = slt;
    *csr = crt;
    *snr = srt;
  }
  double tsign = 0;
  if (pmax == 1) tsign = copysign(1.0, *csr) * copysign(1.0, *csl) * copysign(1.0, f);
  if (pmax == 2) tsign = copysign(1.0, *snr) * copysign(1.0, *csl) * copysign(1.0, g);
  if (pmax == 3) tsign = copysign(1.0, *snr) * copysign(1.0, *snl) * copysign(1.0, h);
  *ssmax = copysign(*ssmax, tsign);
  *ssmin = copysign(*ssmin, tsign * copysign(1.0, f) * copysign(1.0, h));
}

/* la_gebrd: Golub–Kahan two-sided Householder bidiagonalization of a tall/square
   pr×pc (pr>=pc) matrix P in place. On return d[pc], e[pc-1] hold the REAL upper
   bidiagonal, the left reflectors sit below the diagonal (tauq, la_qr layout so
   la_qrq forms Q_U), and the right reflectors sit right of the superdiagonal
   (taup, applied by la_formp). la_formp: forms VT = Q_Vᴴ (pc×pc) from the stored
   right reflectors — Q_Vᴴ = ∏ G_iᴴ applied left-to-right, the initial VT that
   la_bdsvd premultiplies into V_pᴴ.

   la_cpc / la_cpct: plain and conjugate-transpose compute-buffer copies used to
   build P from the unpacked A and to assemble the outputs in the m<n case.

   la_gebrd_unb is the unblocked (BLAS-2) reduction — the exact pre-blocking
   kernel, and the small-n / trailing-block path. la_gebrd blocks it (dgebrd/
   dlabrd) once pc exceeds LA_SVD_NB: la_labrd reduces a LA_SVD_NB-wide panel,
   accumulating the two auxiliary matrices X (pr×nb) and Y (pc×nb) so that the
   trailing update A22 -= V·Yᴴ + X·Uᴴ is TWO GEMMs; la_gebrd_unb finishes the last
   block. la_labrd generates each reflector with the same code as the unblocked
   kernel (so d/e/tauq/taup and the reflector storage that la_qrq/la_formp/la_bdsvd
   consume are unchanged — residual gates cover the reordered arithmetic). X, Y,
   the conj panel Yc for the Yᴴ GEMM operand, the product Pm, and the GEMM panels
   are caller scratch. */
#define LA_GEN_SVD(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)               \
  static void la_gebrd_unb_##sfx(void *vP, int64_t pr, int64_t pc, int64_t ld,  \
                                 void *vd, void *ve, void *vtauq, void *vtaup) { \
    T *P = (T *)vP;                                                             \
    R *d = (R *)vd;                                                            \
    R *e = (R *)ve;                                                            \
    T *tauq = (T *)vtauq;                                                      \
    T *taup = (T *)vtaup;                                                      \
    for (int64_t i = 0; i < pc; i++) {                                         \
      R xn = (R)0;                                                             \
      for (int64_t r = i + 1; r < pr; r++) xn += NORM2(P[r * ld + i]);         \
      T alpha = P[i * ld + i];                                                 \
      R alr = REAL(alpha);                                                     \
      if (xn == (R)0 && LA_IMAG_##sfx(alpha) == (R)0) {                        \
        tauq[i] = (T)0;                                                        \
        d[i] = alr;                                                            \
      } else {                                                                 \
        R an = SQRT(NORM2(alpha) + xn);                                        \
        R beta = alr >= (R)0 ? -an : an;                                       \
        tauq[i] =                                                             \
            LA_MK_##sfx((beta - alr) / beta, -LA_IMAG_##sfx(alpha) / beta);    \
        T scal = alpha - LA_MK_##sfx(beta, (R)0);                              \
        for (int64_t r = i + 1; r < pr; r++)                                   \
          P[r * ld + i] = P[r * ld + i] / scal;                               \
        d[i] = beta;                                                           \
        for (int64_t c = i + 1; c < pc; c++) {                                 \
          T w = P[i * ld + c];                                                 \
          for (int64_t r = i + 1; r < pr; r++)                                 \
            w += CONJ(P[r * ld + i]) * P[r * ld + c];                          \
          T tw = CONJ(tauq[i]) * w;                                            \
          P[i * ld + c] -= tw;                                                 \
          for (int64_t r = i + 1; r < pr; r++)                                 \
            P[r * ld + c] -= tw * P[r * ld + i];                               \
        }                                                                      \
      }                                                                        \
      if (i < pc - 1) {                                                        \
        R xnr = (R)0;                                                          \
        for (int64_t c = i + 2; c < pc; c++) xnr += NORM2(P[i * ld + c]);      \
        T al = P[i * ld + (i + 1)];                                            \
        R alre = REAL(al);                                                     \
        if (xnr == (R)0 && LA_IMAG_##sfx(al) == (R)0) {                        \
          taup[i] = (T)0;                                                      \
          e[i] = alre;                                                         \
        } else {                                                               \
          R an = SQRT(NORM2(al) + xnr);                                        \
          R beta = alre >= (R)0 ? -an : an;                                    \
          T tauc = LA_MK_##sfx((beta - alre) / beta, LA_IMAG_##sfx(al) / beta); \
          taup[i] = tauc;                                                      \
          T scal = CONJ(al) - LA_MK_##sfx(beta, (R)0);                         \
          e[i] = beta;                                                         \
          for (int64_t c = i + 2; c < pc; c++)                                 \
            P[i * ld + c] = CONJ(P[i * ld + c]) / scal;                        \
          for (int64_t a = i + 1; a < pr; a++) {                               \
            T s = P[a * ld + (i + 1)];                                         \
            for (int64_t c = i + 2; c < pc; c++)                               \
              s += P[a * ld + c] * P[i * ld + c];                              \
            s = tauc * s;                                                      \
            P[a * ld + (i + 1)] -= s;                                          \
            for (int64_t c = i + 2; c < pc; c++)                               \
              P[a * ld + c] -= s * CONJ(P[i * ld + c]);                        \
          }                                                                    \
          P[i * ld + (i + 1)] = LA_MK_##sfx(beta, (R)0);                       \
        }                                                                      \
      } else {                                                                 \
        taup[i] = (T)0;                                                        \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  /* dlabrd/zlabrd (pr>=pc): reduce the first `nb` columns/rows of the trailing  \
     block P[off:,off:], accumulating X (mp×nb) and Y (np×nb) so the trailing     \
     rank-updates defer to two GEMMs. Reflector generation and storage mirror     \
     la_gebrd_unb exactly (left tail in the column, right tail conj/scal in the    \
     row); Y[cl][i] = tauq·conj(wᴸ) and X carries the right coefficients, so the   \
     deferred update is A22 -= V·Yᴴ + X·conj(U). */                               \
  static void la_labrd_##sfx(void *vP, int64_t pr, int64_t pc, int64_t ld,      \
                             int64_t off, int64_t nb, void *vd, void *ve,        \
                             void *vtauq, void *vtaup, void *vX, void *vY) {      \
    T *P = (T *)vP;                                                             \
    R *d = (R *)vd;                                                            \
    R *e = (R *)ve;                                                            \
    T *tauq = (T *)vtauq;                                                      \
    T *taup = (T *)vtaup;                                                      \
    T *X = (T *)vX;                                                            \
    T *Y = (T *)vY;                                                            \
    int64_t mp = pr - off, np = pc - off;                                      \
    T ytmp[LA_SVD_NB];                                                         \
    for (int64_t i = 0; i < nb; i++) {                                         \
      int64_t gi = off + i;                                                    \
      for (int64_t rl = i; rl < mp; rl++) {                                    \
        T s = (T)0;                                                            \
        for (int64_t l = 0; l < i; l++)                                        \
          s += P[(off + rl) * ld + (off + l)] * CONJ(Y[i * nb + l]) +          \
               X[rl * nb + l] * CONJ(P[(off + l) * ld + gi]);                  \
        P[(off + rl) * ld + gi] -= s;                                          \
      }                                                                        \
      R xn = (R)0;                                                             \
      for (int64_t r = gi + 1; r < pr; r++) xn += NORM2(P[r * ld + gi]);       \
      T alpha = P[gi * ld + gi];                                               \
      R alr = REAL(alpha);                                                     \
      if (xn == (R)0 && LA_IMAG_##sfx(alpha) == (R)0) {                        \
        tauq[gi] = (T)0;                                                       \
        d[gi] = alr;                                                           \
      } else {                                                                 \
        R an = SQRT(NORM2(alpha) + xn);                                        \
        R beta = alr >= (R)0 ? -an : an;                                       \
        tauq[gi] =                                                            \
            LA_MK_##sfx((beta - alr) / beta, -LA_IMAG_##sfx(alpha) / beta);    \
        T scal = alpha - LA_MK_##sfx(beta, (R)0);                             \
        for (int64_t r = gi + 1; r < pr; r++)                                 \
          P[r * ld + gi] = P[r * ld + gi] / scal;                             \
        d[gi] = beta;                                                          \
        P[gi * ld + gi] = (T)1;                                               \
      }                                                                        \
      /* Y(i+1:np, i) = tauq · conj(A(i:mp, i+1:np)ᴴ vᴸ) with the panel-column  \
         corrections; ytmp is the shared (i-length) GEMV temp. */              \
      for (int64_t cl = i + 1; cl < np; cl++) {                                \
        T s = (T)0;                                                            \
        for (int64_t rl = i; rl < mp; rl++)                                    \
          s += CONJ(P[(off + rl) * ld + (off + cl)]) * P[(off + rl) * ld + gi]; \
        Y[cl * nb + i] = s;                                                    \
      }                                                                        \
      for (int64_t l = 0; l < i; l++) {                                        \
        T s = (T)0;                                                            \
        for (int64_t rl = i; rl < mp; rl++)                                    \
          s += CONJ(P[(off + rl) * ld + (off + l)]) * P[(off + rl) * ld + gi]; \
        ytmp[l] = s;                                                           \
      }                                                                        \
      for (int64_t cl = i + 1; cl < np; cl++) {                                \
        T s = (T)0;                                                            \
        for (int64_t l = 0; l < i; l++) s += Y[cl * nb + l] * ytmp[l];         \
        Y[cl * nb + i] -= s;                                                   \
      }                                                                        \
      for (int64_t l = 0; l < i; l++) {                                        \
        T s = (T)0;                                                            \
        for (int64_t rl = i; rl < mp; rl++)                                    \
          s += CONJ(X[rl * nb + l]) * P[(off + rl) * ld + gi];                 \
        ytmp[l] = s;                                                           \
      }                                                                        \
      for (int64_t cl = i + 1; cl < np; cl++) {                                \
        T s = (T)0;                                                            \
        for (int64_t l = 0; l < i; l++)                                        \
          s += P[(off + l) * ld + (off + cl)] * ytmp[l];                       \
        Y[cl * nb + i] -= s;                                                   \
      }                                                                        \
      for (int64_t cl = i + 1; cl < np; cl++)                                  \
        Y[cl * nb + i] = tauq[gi] * Y[cl * nb + i];                            \
      /* Update row gi (cols i+1:np) with the left + right deferred pieces. */  \
      for (int64_t cl = i + 1; cl < np; cl++) {                                \
        T s = (T)0;                                                            \
        for (int64_t l = 0; l <= i; l++)                                       \
          s += CONJ(Y[cl * nb + l]) * P[gi * ld + (off + l)];                  \
        for (int64_t l = 0; l < i; l++)                                        \
          s += X[i * nb + l] * CONJ(P[(off + l) * ld + (off + cl)]);           \
        P[gi * ld + (off + cl)] -= s;                                          \
      }                                                                        \
      /* Right reflector P_i on row gi (cols gi+1:pc) [our storage]. */         \
      R xnr = (R)0;                                                            \
      for (int64_t cl = i + 2; cl < np; cl++)                                  \
        xnr += NORM2(P[gi * ld + (off + cl)]);                                 \
      T al = P[gi * ld + (gi + 1)];                                            \
      R alre = REAL(al);                                                       \
      if (xnr == (R)0 && LA_IMAG_##sfx(al) == (R)0) {                          \
        taup[gi] = (T)0;                                                       \
        e[gi] = alre;                                                          \
        for (int64_t rl = i + 1; rl < mp; rl++) X[rl * nb + i] = (T)0;         \
      } else {                                                                 \
        R an = SQRT(NORM2(al) + xnr);                                          \
        R beta = alre >= (R)0 ? -an : an;                                      \
        T tauc =                                                              \
            LA_MK_##sfx((beta - alre) / beta, LA_IMAG_##sfx(al) / beta);       \
        taup[gi] = tauc;                                                       \
        T scal = CONJ(al) - LA_MK_##sfx(beta, (R)0);                           \
        e[gi] = beta;                                                          \
        for (int64_t cl = i + 2; cl < np; cl++)                               \
          P[gi * ld + (off + cl)] = CONJ(P[gi * ld + (off + cl)]) / scal;      \
        P[gi * ld + (gi + 1)] = (T)1;                                          \
        /* X(i+1:mp, i) = taup · (A(i+1:mp, i+1:np) v_R with corrections). */   \
        for (int64_t rl = i + 1; rl < mp; rl++) {                             \
          T s = (T)0;                                                          \
          for (int64_t cl = i + 1; cl < np; cl++)                             \
            s += P[(off + rl) * ld + (off + cl)] * P[gi * ld + (off + cl)];    \
          X[rl * nb + i] = s;                                                  \
        }                                                                      \
        for (int64_t l = 0; l <= i; l++) {                                    \
          T s = (T)0;                                                          \
          for (int64_t cl = i + 1; cl < np; cl++)                             \
            s += CONJ(Y[cl * nb + l]) * P[gi * ld + (off + cl)];               \
          ytmp[l] = s;                                                         \
        }                                                                      \
        for (int64_t rl = i + 1; rl < mp; rl++) {                             \
          T s = (T)0;                                                          \
          for (int64_t l = 0; l <= i; l++)                                    \
            s += P[(off + rl) * ld + (off + l)] * ytmp[l];                     \
          X[rl * nb + i] -= s;                                                 \
        }                                                                      \
        for (int64_t l = 0; l < i; l++) {                                     \
          T s = (T)0;                                                          \
          for (int64_t cl = i + 1; cl < np; cl++)                             \
            s += CONJ(P[(off + l) * ld + (off + cl)]) * P[gi * ld + (off + cl)]; \
          ytmp[l] = s;                                                         \
        }                                                                      \
        for (int64_t rl = i + 1; rl < mp; rl++) {                             \
          T s = (T)0;                                                          \
          for (int64_t l = 0; l < i; l++) s += X[rl * nb + l] * ytmp[l];       \
          X[rl * nb + i] -= s;                                                 \
        }                                                                      \
        for (int64_t rl = i + 1; rl < mp; rl++)                               \
          X[rl * nb + i] = tauc * X[rl * nb + i];                              \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  static void la_gebrd_##sfx(void *vP, int64_t pr, int64_t pc, int64_t ld,      \
                             void *vd, void *ve, void *vtauq, void *vtaup,       \
                             void *vX, void *vY, void *vMc, void *vPm,           \
                             void *vg) {                                        \
    T *P = (T *)vP;                                                            \
    if (pc <= LA_SVD_NB) {                                                     \
      la_gebrd_unb_##sfx(vP, pr, pc, ld, vd, ve, vtauq, vtaup);                \
      return;                                                                  \
    }                                                                          \
    T *X = (T *)vX;                                                            \
    T *Y = (T *)vY;                                                            \
    T *Mc = (T *)vMc;                                                          \
    T *Pm = (T *)vPm;                                                          \
    int64_t off = 0;                                                           \
    while (pc - off > LA_SVD_NB) {                                             \
      int64_t nb = LA_SVD_NB;                                                  \
      la_labrd_##sfx(vP, pr, pc, ld, off, nb, vd, ve, vtauq, vtaup, X, Y);      \
      int64_t mt = pr - off - nb, nt = pc - off - nb;                          \
      if (mt > 0 && nt > 0) {                                                  \
        /* GEMM 1: A22 -= V · Yᴴ (materialize conj(Y[nb:]) as Mc, transposed). */ \
        for (int64_t b = 0; b < nt; b++)                                       \
          for (int64_t il = 0; il < nb; il++)                                  \
            Mc[b * nb + il] = CONJ(Y[(nb + b) * nb + il]);                      \
        nx_c_gemm2d_ct_ws(DT, mt, nt, nb,                                       \
                         (const char *)&P[(off + nb) * ld + off], ld, 1,       \
                         (const char *)Mc, 1, nb, (char *)Pm, nt, 1,           \
                         (char *)vg);                                          \
        for (int64_t a = 0; a < mt; a++)                                       \
          for (int64_t b = 0; b < nt; b++)                                     \
            P[(off + nb + a) * ld + (off + nb + b)] -= Pm[a * nt + b];         \
        /* GEMM 2: A22 -= X · conj(U) (materialize conj(U) rows as Mc). */      \
        for (int64_t il = 0; il < nb; il++)                                    \
          for (int64_t b = 0; b < nt; b++)                                     \
            Mc[il * nt + b] = CONJ(P[(off + il) * ld + (off + nb + b)]);       \
        nx_c_gemm2d_ct_ws(DT, mt, nt, nb, (const char *)&X[nb * LA_SVD_NB],     \
                         LA_SVD_NB, 1, (const char *)Mc, nt, 1, (char *)Pm, nt, \
                         1, (char *)vg);                                       \
        for (int64_t a = 0; a < mt; a++)                                       \
          for (int64_t b = 0; b < nt; b++)                                     \
            P[(off + nb + a) * ld + (off + nb + b)] -= Pm[a * nt + b];         \
      }                                                                        \
      off += nb;                                                               \
    }                                                                          \
    la_gebrd_unb_##sfx((char *)P + (off * ld + off) * (int64_t)sizeof(T),      \
                       pr - off, pc - off, ld, (R *)vd + off, (R *)ve + off,    \
                       (T *)vtauq + off, (T *)vtaup + off);                     \
  }                                                                            \
  static void la_formp_##sfx(void *vP, int64_t pc, int64_t ld, void *vtaup,     \
                             void *vVT, int64_t ldvt) {                         \
    T *P = (T *)vP;                                                            \
    T *taup = (T *)vtaup;                                                      \
    T *VT = (T *)vVT;                                                          \
    for (int64_t r = 0; r < pc; r++)                                           \
      for (int64_t c = 0; c < pc; c++) VT[r * ldvt + c] = r == c ? (T)1 : (T)0; \
    for (int64_t i = 0; i < pc - 1; i++) {                                     \
      if (taup[i] == (T)0) continue;                                          \
      for (int64_t c = 0; c < pc; c++) {                                       \
        T w = VT[(i + 1) * ldvt + c];                                          \
        for (int64_t s = i + 2; s < pc; s++)                                   \
          w += CONJ(P[i * ld + s]) * VT[s * ldvt + c];                         \
        T tw = CONJ(taup[i]) * w;                                              \
        VT[(i + 1) * ldvt + c] -= tw;                                          \
        for (int64_t s = i + 2; s < pc; s++)                                   \
          VT[s * ldvt + c] -= tw * P[i * ld + s];                              \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  static nx_c_status la_bdsvd_##sfx(void *vd, void *ve, void *vU, int64_t pr,    \
                                   int64_t ldu, void *vVT, int64_t ldvt,        \
                                   int64_t pc) {                                \
    R *d = (R *)vd;                                                            \
    R *e = (R *)ve;                                                            \
    T *U = (T *)vU;                                                            \
    T *VT = (T *)vVT;                                                          \
    int64_t n = pc;                                                            \
    if (n == 0) return NX_C_OK;                                                 \
    if (n == 1) {                                                              \
      if (d[0] == (R)0) d[0] = (R)0; /* avoid -0 (dbdsqr) */                   \
      if (d[0] < (R)0) {                                                       \
        d[0] = -d[0];                                                          \
        for (int64_t c = 0; c < n; c++) VT[c] = -VT[c];                        \
      }                                                                        \
      return NX_C_OK;                                                           \
    }                                                                          \
    const double eps = sizeof(R) == 4 ? (double)FLT_EPSILON : DBL_EPSILON;     \
    const double unfl = sizeof(R) == 4 ? (double)FLT_MIN : DBL_MIN;            \
    const int MAXITR = 6;                                                      \
    double tolmul = pow(eps, -0.125);                                          \
    if (tolmul < 10.0) tolmul = 10.0;                                          \
    if (tolmul > 100.0) tolmul = 100.0;                                        \
    double tol = tolmul * eps;                                                 \
    double sminoa = fabs((double)d[0]);                                        \
    if (sminoa != 0.0) {                                                       \
      double mu = sminoa;                                                      \
      for (int64_t i = 1; i < n; i++) {                                        \
        mu = fabs((double)d[i]) * (mu / (mu + fabs((double)e[i - 1])));        \
        if (mu < sminoa) sminoa = mu;                                          \
        if (sminoa == 0.0) break;                                             \
      }                                                                        \
    }                                                                          \
    sminoa = sminoa / sqrt((double)n);                                         \
    double thresh = tol * sminoa;                                             \
    double flr = (double)(MAXITR * n) * (double)n * unfl;                      \
    if (flr > thresh) thresh = flr;                                            \
    int64_t maxit = (int64_t)MAXITR * n * n;                                   \
    int64_t iter = 0;                                                          \
    int64_t m = n - 1;                                                         \
    int64_t oldll = -1, oldm = -1;                                             \
    int idir = 0;                                                              \
    double sminl = 0.0;                                                        \
    while (1) {                                                                \
      if (m <= 0) break;                                                       \
      if (iter > maxit) return LA_ERR_NO_CONVERGE;                             \
      double bsmax = fabs((double)d[m]);                                       \
      int64_t ll = 0;                                                          \
      int split_bottom = 0;                                                    \
      for (int64_t lll = 1; lll <= m; lll++) {                                 \
        int64_t c = m - lll;                                                   \
        double abss = fabs((double)d[c]);                                      \
        double abse = fabs((double)e[c]);                                      \
        if (abse <= thresh) {                                                  \
          e[c] = (R)0;                                                         \
          if (c == m - 1) {                                                    \
            m = m - 1;                                                         \
            split_bottom = 1;                                                  \
          } else                                                               \
            ll = c + 1;                                                        \
          break;                                                              \
        }                                                                      \
        if (abss > bsmax) bsmax = abss;                                        \
        if (abse > bsmax) bsmax = abse;                                        \
      }                                                                        \
      if (split_bottom) continue;                                             \
      if (ll == m - 1) {                                                       \
        double sigmn, sigmx, snr, csr, snl, csl;                               \
        la_lasv2((double)d[m - 1], (double)e[m - 1], (double)d[m], &sigmn,      \
                 &sigmx, &snr, &csr, &snl, &csl);                              \
        d[m - 1] = (R)sigmx;                                                   \
        e[m - 1] = (R)0;                                                       \
        d[m] = (R)sigmn;                                                       \
        for (int64_t c = 0; c < n; c++) {                                      \
          T t1 = VT[(m - 1) * ldvt + c], t2 = VT[m * ldvt + c];                \
          VT[(m - 1) * ldvt + c] = (T)(csr * t1 + snr * t2);                   \
          VT[m * ldvt + c] = (T)(csr * t2 - snr * t1);                         \
        }                                                                      \
        for (int64_t r = 0; r < pr; r++) {                                     \
          T t1 = U[r * ldu + (m - 1)], t2 = U[r * ldu + m];                    \
          U[r * ldu + (m - 1)] = (T)(csl * t1 + snl * t2);                     \
          U[r * ldu + m] = (T)(csl * t2 - snl * t1);                           \
        }                                                                      \
        m = m - 2;                                                             \
        continue;                                                             \
      }                                                                        \
      /* choose the chase direction only on a NEW submatrix (dbdsqr's          \
         OLDLL/OLDM latch); re-deciding every sweep can flip-flop on           \
         borderline blocks and diverge from the reference's iteration count. */ \
      if (ll > oldm || m < oldll)                                              \
        idir = fabs((double)d[ll]) >= fabs((double)d[m]) ? 1 : 2;              \
      if (idir == 1) {                                                         \
        /* dbdsqr's second disjunct (|e| <= thresh) is TOL<0-only in the       \
           reference; tol > 0 always here and the split scan zeroed those. */  \
        if (fabs((double)e[m - 1]) <= tol * fabs((double)d[m])) {              \
          e[m - 1] = (R)0;                                                     \
          continue;                                                           \
        }                                                                      \
        double mu = fabs((double)d[ll]);                                       \
        sminl = mu;                                                            \
        int conv = 0;                                                          \
        for (int64_t lll = ll; lll < m; lll++) {                               \
          if (fabs((double)e[lll]) <= tol * mu) {                             \
            e[lll] = (R)0;                                                     \
            conv = 1;                                                          \
            break;                                                            \
          }                                                                    \
          mu = fabs((double)d[lll + 1]) * (mu / (mu + fabs((double)e[lll])));  \
          if (mu < sminl) sminl = mu;                                          \
        }                                                                      \
        if (conv) continue;                                                   \
      } else {                                                                 \
        if (fabs((double)e[ll]) <= tol * fabs((double)d[ll])) {               \
          e[ll] = (R)0;                                                        \
          continue;                                                           \
        }                                                                      \
        double mu = fabs((double)d[m]);                                        \
        sminl = mu;                                                            \
        int conv = 0;                                                          \
        for (int64_t lll = m - 1; lll >= ll; lll--) {                          \
          if (fabs((double)e[lll]) <= tol * mu) {                             \
            e[lll] = (R)0;                                                     \
            conv = 1;                                                          \
            break;                                                            \
          }                                                                    \
          mu = fabs((double)d[lll]) * (mu / (mu + fabs((double)e[lll])));      \
          if (mu < sminl) sminl = mu;                                          \
        }                                                                      \
        if (conv) continue;                                                   \
      }                                                                        \
      oldll = ll;                                                              \
      oldm = m;                                                                \
      double shift = 0.0;                                                      \
      double thr2 = 0.01 * tol;                                                \
      if (eps > thr2) thr2 = eps;                                              \
      if ((double)n * tol * (sminl / bsmax) <= thr2) {                         \
        shift = 0.0;                                                           \
      } else {                                                                 \
        double sll;                                                            \
        if (idir == 1) {                                                       \
          sll = fabs((double)d[ll]);                                           \
          shift = la_las2((double)d[m - 1], (double)e[m - 1], (double)d[m]);    \
        } else {                                                               \
          sll = fabs((double)d[m]);                                            \
          shift = la_las2((double)d[ll], (double)e[ll], (double)d[ll + 1]);     \
        }                                                                      \
        if (sll > 0.0) {                                                       \
          double q = shift / sll;                                              \
          if (q * q < eps) shift = 0.0;                                        \
        }                                                                      \
      }                                                                        \
      iter += (m - ll);                                                       \
      if (shift == 0.0) {                                                      \
        if (idir == 1) {                                                       \
          double cs = 1.0, oldcs = 1.0, sn = 0.0, oldsn = 0.0, r;              \
          for (int64_t i = ll; i < m; i++) {                                   \
            la_lartg((double)d[i] * cs, (double)e[i], &cs, &sn, &r);           \
            if (i > ll) e[i - 1] = (R)(oldsn * r);                             \
            la_lartg(oldcs * r, (double)d[i + 1] * sn, &oldcs, &oldsn, &r);    \
            d[i] = (R)r;                                                       \
            for (int64_t c = 0; c < n; c++) {                                  \
              T t1 = VT[i * ldvt + c], t2 = VT[(i + 1) * ldvt + c];            \
              VT[i * ldvt + c] = (T)(cs * t1 + sn * t2);                       \
              VT[(i + 1) * ldvt + c] = (T)(cs * t2 - sn * t1);                 \
            }                                                                  \
            for (int64_t rr = 0; rr < pr; rr++) {                             \
              T t1 = U[rr * ldu + i], t2 = U[rr * ldu + (i + 1)];              \
              U[rr * ldu + i] = (T)(oldcs * t1 + oldsn * t2);                  \
              U[rr * ldu + (i + 1)] = (T)(oldcs * t2 - oldsn * t1);            \
            }                                                                  \
          }                                                                    \
          double h = (double)d[m] * cs;                                        \
          d[m] = (R)(h * oldcs);                                               \
          e[m - 1] = (R)(h * oldsn);                                           \
          if (fabs((double)e[m - 1]) <= thresh) e[m - 1] = (R)0;              \
        } else {                                                               \
          double cs = 1.0, oldcs = 1.0, sn = 0.0, oldsn = 0.0, r;              \
          for (int64_t i = m; i > ll; i--) {                                   \
            la_lartg((double)d[i] * cs, (double)e[i - 1], &cs, &sn, &r);       \
            if (i < m) e[i] = (R)(oldsn * r);                                  \
            la_lartg(oldcs * r, (double)d[i - 1] * sn, &oldcs, &oldsn, &r);    \
            d[i] = (R)r;                                                       \
            /* Backward chase pairs the rotations with U/Vᴴ opposite to the    \
               forward sweep: cs → U, oldcs → Vᴴ (dbdsqr's idir==2 DLASR        \
               argument swap). */                                              \
            for (int64_t c = 0; c < n; c++) {                                  \
              T t1 = VT[(i - 1) * ldvt + c], t2 = VT[i * ldvt + c];            \
              VT[(i - 1) * ldvt + c] = (T)(oldcs * t1 - oldsn * t2);           \
              VT[i * ldvt + c] = (T)(oldcs * t2 + oldsn * t1);                 \
            }                                                                  \
            for (int64_t rr = 0; rr < pr; rr++) {                             \
              T t1 = U[rr * ldu + (i - 1)], t2 = U[rr * ldu + i];              \
              U[rr * ldu + (i - 1)] = (T)(cs * t1 - sn * t2);                  \
              U[rr * ldu + i] = (T)(cs * t2 + sn * t1);                        \
            }                                                                  \
          }                                                                    \
          double h = (double)d[ll] * cs;                                       \
          d[ll] = (R)(h * oldcs);                                              \
          e[ll] = (R)(h * oldsn);                                              \
          if (fabs((double)e[ll]) <= thresh) e[ll] = (R)0;                    \
        }                                                                      \
      } else {                                                                 \
        if (idir == 1) {                                                       \
          double dl = (double)d[ll];                                           \
          double f = (fabs(dl) - shift) * (copysign(1.0, dl) + shift / dl);    \
          double g = (double)e[ll];                                            \
          double cosr, sinr, cosl, sinl, r;                                    \
          for (int64_t i = ll; i < m; i++) {                                   \
            la_lartg(f, g, &cosr, &sinr, &r);                                  \
            if (i > ll) e[i - 1] = (R)r;                                       \
            f = cosr * (double)d[i] + sinr * (double)e[i];                     \
            e[i] = (R)(cosr * (double)e[i] - sinr * (double)d[i]);             \
            g = sinr * (double)d[i + 1];                                       \
            d[i + 1] = (R)(cosr * (double)d[i + 1]);                           \
            la_lartg(f, g, &cosl, &sinl, &r);                                  \
            d[i] = (R)r;                                                       \
            f = cosl * (double)e[i] + sinl * (double)d[i + 1];                 \
            d[i + 1] = (R)(cosl * (double)d[i + 1] - sinl * (double)e[i]);     \
            if (i < m - 1) {                                                   \
              g = sinl * (double)e[i + 1];                                     \
              e[i + 1] = (R)(cosl * (double)e[i + 1]);                         \
            }                                                                  \
            for (int64_t c = 0; c < n; c++) {                                  \
              T t1 = VT[i * ldvt + c], t2 = VT[(i + 1) * ldvt + c];            \
              VT[i * ldvt + c] = (T)(cosr * t1 + sinr * t2);                   \
              VT[(i + 1) * ldvt + c] = (T)(cosr * t2 - sinr * t1);             \
            }                                                                  \
            for (int64_t rr = 0; rr < pr; rr++) {                             \
              T u1 = U[rr * ldu + i], u2 = U[rr * ldu + (i + 1)];              \
              U[rr * ldu + i] = (T)(cosl * u1 + sinl * u2);                    \
              U[rr * ldu + (i + 1)] = (T)(cosl * u2 - sinl * u1);              \
            }                                                                  \
          }                                                                    \
          e[m - 1] = (R)f;                                                     \
          if (fabs((double)e[m - 1]) <= thresh) e[m - 1] = (R)0;              \
        } else {                                                               \
          double dm = (double)d[m];                                           \
          double f = (fabs(dm) - shift) * (copysign(1.0, dm) + shift / dm);    \
          double g = (double)e[m - 1];                                         \
          double cosr, sinr, cosl, sinl, r;                                    \
          for (int64_t i = m; i > ll; i--) {                                   \
            la_lartg(f, g, &cosr, &sinr, &r);                                  \
            if (i < m) e[i] = (R)r;                                            \
            f = cosr * (double)d[i] + sinr * (double)e[i - 1];                 \
            e[i - 1] = (R)(cosr * (double)e[i - 1] - sinr * (double)d[i]);     \
            g = sinr * (double)d[i - 1];                                       \
            d[i - 1] = (R)(cosr * (double)d[i - 1]);                           \
            la_lartg(f, g, &cosl, &sinl, &r);                                  \
            d[i] = (R)r;                                                       \
            f = cosl * (double)e[i - 1] + sinl * (double)d[i - 1];             \
            d[i - 1] = (R)(cosl * (double)d[i - 1] - sinl * (double)e[i - 1]); \
            if (i > ll + 1) {                                                  \
              g = sinl * (double)e[i - 2];                                     \
              e[i - 2] = (R)(cosl * (double)e[i - 2]);                         \
            }                                                                  \
            /* idir==2 pairing: cosr → U, cosl → Vᴴ (see the zero-shift         \
               backward sweep above). */                                      \
            for (int64_t c = 0; c < n; c++) {                                  \
              T t1 = VT[(i - 1) * ldvt + c], t2 = VT[i * ldvt + c];            \
              VT[(i - 1) * ldvt + c] = (T)(cosl * t1 - sinl * t2);             \
              VT[i * ldvt + c] = (T)(cosl * t2 + sinl * t1);                   \
            }                                                                  \
            for (int64_t rr = 0; rr < pr; rr++) {                             \
              T u1 = U[rr * ldu + (i - 1)], u2 = U[rr * ldu + i];              \
              U[rr * ldu + (i - 1)] = (T)(cosr * u1 - sinr * u2);              \
              U[rr * ldu + i] = (T)(cosr * u2 + sinr * u1);                    \
            }                                                                  \
          }                                                                    \
          e[ll] = (R)f;                                                        \
          if (fabs((double)e[ll]) <= thresh) e[ll] = (R)0;                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int64_t i = 0; i < n; i++) {                                          \
      if (d[i] == (R)0) d[i] = (R)0; /* avoid -0 (dbdsqr) */                   \
      if (d[i] < (R)0) {                                                       \
        d[i] = -d[i];                                                          \
        for (int64_t c = 0; c < n; c++)                                        \
          VT[i * ldvt + c] = -VT[i * ldvt + c];                               \
      }                                                                        \
    }                                                                          \
    for (int64_t i = 0; i < n - 1; i++) {                                      \
      int64_t mx = i;                                                          \
      for (int64_t j = i + 1; j < n; j++)                                      \
        if (d[j] > d[mx]) mx = j;                                             \
      if (mx != i) {                                                          \
        R t = d[i];                                                            \
        d[i] = d[mx];                                                          \
        d[mx] = t;                                                            \
        for (int64_t c = 0; c < n; c++) {                                      \
          T tv = VT[i * ldvt + c];                                            \
          VT[i * ldvt + c] = VT[mx * ldvt + c];                               \
          VT[mx * ldvt + c] = tv;                                             \
        }                                                                      \
        for (int64_t r = 0; r < pr; r++) {                                     \
          T tu = U[r * ldu + i];                                              \
          U[r * ldu + i] = U[r * ldu + mx];                                   \
          U[r * ldu + mx] = tu;                                               \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    return NX_C_OK;                                                             \
  }                                                                            \
  static void la_cpc_##sfx(const void *vsrc, int64_t sld, void *vdst,           \
                           int64_t dld, int64_t rows, int64_t cols) {           \
    const T *src = (const T *)vsrc;                                            \
    T *dst = (T *)vdst;                                                        \
    for (int64_t i = 0; i < rows; i++)                                         \
      for (int64_t j = 0; j < cols; j++) dst[i * dld + j] = src[i * sld + j];  \
  }                                                                            \
  static void la_cpct_##sfx(const void *vsrc, int64_t sld, void *vdst,          \
                            int64_t dld, int64_t rows, int64_t cols) {          \
    const T *src = (const T *)vsrc;                                            \
    T *dst = (T *)vdst;                                                        \
    for (int64_t i = 0; i < rows; i++)                                         \
      for (int64_t j = 0; j < cols; j++)                                       \
        dst[i * dld + j] = CONJ(src[j * sld + i]);                            \
  }
#define LA_EXPAND_SVD(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)            \
  LA_GEN_SVD(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_SVD)
LA_TRAITS_double(LA_EXPAND_SVD)
LA_TRAITS_c32(LA_EXPAND_SVD)
LA_TRAITS_c64(LA_EXPAND_SVD)
#undef LA_EXPAND_SVD

/* ── Bidiagonal divide-and-conquer (dbdsdc / dlasd0-5 structure) ──────────

   The with-vectors bidiagonal SVD cost is la_bdsvd's Givens accumulation into
   U/Vᴴ: O(n³) BLAS-1 (the tql2 analog the eigh D&C removed). Divide-and-conquer
   replaces it on the pc > LA_SD_SMLSIZ path: tear the bidiagonal at the middle
   row, conquer each half, merge through the rank-one sqrt-form secular equation
   (deflation + BLAS-3 back-multiplies), then apply the result to the assembled
   transforms with two GEMMs — U = Q_U·U_s and Vᴴ = V_sᴴ·Q_Vᴴ, the latter with
   Q_Vᴴ itself formed BLAS-3 (see la_sd_apply). la_bdsvd + la_formp stay as the
   small-pc path.

   The core is double regardless of the compute type (the bidiagonal is real —
   gebrd's larfg betas make d/e real for complex inputs too — and S is float64
   anyway; the f32 path gains secular accuracy at negligible cost). Matrices in
   the core are COLUMN-major so the dlasd reference maps 1:1; the la_sd_apply
   glue lifts the results back into the row-major compute-typed world. Index
   arithmetic is 0-based throughout (the reference is 1-based). la_ed6,
   la_dc_dnrm2 and la_dc_dlamrg are duplicated verbatim from nx_c_eigh.c
   (TU-private statics there). Non-convergence -> LA_ERR_NO_CONVERGE. */

/* dlaed6: Gragg-Thornton-Warner cubic-convergent root of the 3-pole rational
   equation, the SWTCH3 interpolation la_sd4 uses (dlasd4 reuses dlaed's
   kernel). d,z length 3. Returns tau via *tau_out; nonzero only on
   non-convergence. */
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

#define LA_SD_SMLSIZ 25 /* LAPACK DBDSDC SMLSIZ (>=3) */

/* dlasd5: sqrt of the I-th (1 or 2) updated singular value of the 2x2 secular
   equation. d[2] singular values (d[0]<d[1]), z[2], rho. Returns dsigma and the
   delta/work pair (work[j]=d[j]+dsigma, delta[j]=d[j]-dsigma). */
static void la_sd5(int i1, const double *d, const double *z, double *delta,
                   double rho, double *dsigma, double *work) {
  double del = d[1] - d[0];
  double delsq = del * (d[1] + d[0]);
  double b, c, tau;
  if (i1 == 1) {
    double w = 1.0 +
               4.0 * rho *
                   (z[1] * z[1] / (d[0] + 3.0 * d[1]) -
                    z[0] * z[0] / (3.0 * d[0] + d[1])) /
                   del;
    if (w > 0.0) {
      b = delsq + rho * (z[0] * z[0] + z[1] * z[1]);
      c = rho * z[0] * z[0] * delsq;
      tau = 2.0 * c / (b + sqrt(fabs(b * b - 4.0 * c))); /* sigma^2 - d[0]^2 */
      tau = tau / (d[0] + sqrt(d[0] * d[0] + tau));      /* sigma - d[0] */
      *dsigma = d[0] + tau;
      delta[0] = -tau;
      delta[1] = del - tau;
      work[0] = 2.0 * d[0] + tau;
      work[1] = (d[0] + tau) + d[1];
    } else {
      b = -delsq + rho * (z[0] * z[0] + z[1] * z[1]);
      c = rho * z[1] * z[1] * delsq;
      if (b > 0.0)
        tau = -2.0 * c / (b + sqrt(b * b + 4.0 * c));
      else
        tau = (b - sqrt(b * b + 4.0 * c)) / 2.0;
      tau = tau / (d[1] + sqrt(fabs(d[1] * d[1] + tau)));
      *dsigma = d[1] + tau;
      delta[0] = -(del + tau);
      delta[1] = -tau;
      work[0] = d[0] + tau + d[1];
      work[1] = 2.0 * d[1] + tau;
    }
  } else {
    b = -delsq + rho * (z[0] * z[0] + z[1] * z[1]);
    c = rho * z[1] * z[1] * delsq;
    if (b > 0.0)
      tau = (b + sqrt(b * b + 4.0 * c)) / 2.0;
    else
      tau = 2.0 * c / (-b + sqrt(b * b + 4.0 * c));
    tau = tau / (d[1] + sqrt(d[1] * d[1] + tau));
    *dsigma = d[1] + tau;
    delta[0] = -(del + tau);
    delta[1] = -tau;
    work[0] = d[0] + tau + d[1];
    work[1] = 2.0 * d[1] + tau;
  }
}

/* dlasd4: the I-th updated singular value SIGMA of the sqrt-form secular equation
   for D^2 + rho z z^T (n poles d, ascending, non-negative). Fills delta[j]=d[j]-
   sigma and work[j]=d[j]+sigma (so work[j]*delta[j] = d[j]^2 - sigma^2, the
   pole distances). Status nonzero on non-convergence. Reuses la_ed6 for the
   3-pole interpolation. Interval bracketing SGLB/SGUB + GEOMAVG — the convergence-
   critical piece, ported line-for-line (0-based indices). Requires la_ed6 in scope. */
static int la_sd4(int n, int iev, const double *d, const double *z,
                  double *delta, double rho, double *sigma, double *work) {
  const int MAXIT = 400;
  const double eps = 0.5 * DBL_EPSILON; /* DLAMCH('E') = DBL_EPSILON/2 */
  double rhoinv = 1.0 / rho;
  int j, niter, iter, ii, iim1, iip1, ip1;
  int orgati, swtch, swtch3, geomavg;
  double a, b, c, delsq, delsq2, sq2, dphi, dpsi, dtiim, dtiip, dtipsq, dtisq;
  double dtnsq, dtnsq1;
  double dw, erretm, eta, phi, prew, psi, sglb, sgub, tau, tau2, temp, temp1,
      temp2, w;
  double dd[3], zz[3];

  if (n == 1) {
    *sigma = sqrt(d[0] * d[0] + rho * z[0] * z[0]);
    delta[0] = 1.0;
    work[0] = 1.0;
    return 0;
  }
  if (n == 2) {
    la_sd5(iev + 1, d, z, delta, rho, sigma, work);
    return 0;
  }

  tau2 = 0.0;
  niter = 1;

  if (iev == n - 1) {
    /* The case I == N: the largest singular value, bracketed by
       d[n-1]^2 < sigma^2 <= d[n-1]^2 + rho; last-two-pole iteration. */
    ii = n - 2;
    temp = rho / 2.0;
    temp1 = temp / (d[n - 1] + sqrt(d[n - 1] * d[n - 1] + temp));
    for (j = 0; j < n; j++) {
      work[j] = d[j] + d[n - 1] + temp1;
      delta[j] = (d[j] - d[n - 1]) - temp1;
    }
    psi = 0.0;
    for (j = 0; j < n - 2; j++) psi += z[j] * z[j] / (delta[j] * work[j]);
    c = rhoinv + psi;
    w = c + z[ii] * z[ii] / (delta[ii] * work[ii]) +
        z[n - 1] * z[n - 1] / (delta[n - 1] * work[n - 1]);
    if (w <= 0.0) {
      temp1 = sqrt(d[n - 1] * d[n - 1] + rho);
      temp = z[n - 2] * z[n - 2] /
                 ((d[n - 2] + temp1) *
                  (d[n - 1] - d[n - 2] + rho / (d[n - 1] + temp1))) +
             z[n - 1] * z[n - 1] / rho;
      if (c <= temp) {
        tau = rho;
      } else {
        delsq = (d[n - 1] - d[n - 2]) * (d[n - 1] + d[n - 2]);
        a = -c * delsq + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
        b = z[n - 1] * z[n - 1] * delsq;
        if (a < 0.0)
          tau2 = 2.0 * b / (sqrt(a * a + 4.0 * b * c) - a);
        else
          tau2 = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);
        tau = tau2 / (d[n - 1] + sqrt(d[n - 1] * d[n - 1] + tau2));
      }
    } else {
      delsq = (d[n - 1] - d[n - 2]) * (d[n - 1] + d[n - 2]);
      a = -c * delsq + z[n - 2] * z[n - 2] + z[n - 1] * z[n - 1];
      b = z[n - 1] * z[n - 1] * delsq;
      if (a < 0.0)
        tau2 = 2.0 * b / (sqrt(a * a + 4.0 * b * c) - a);
      else
        tau2 = (a + sqrt(a * a + 4.0 * b * c)) / (2.0 * c);
      tau = tau2 / (d[n - 1] + sqrt(d[n - 1] * d[n - 1] + tau2));
    }
    *sigma = d[n - 1] + tau;
    for (j = 0; j < n; j++) {
      delta[j] = (d[j] - d[n - 1]) - tau;
      work[j] = d[j] + d[n - 1] + tau;
    }
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for (j = 0; j <= ii; j++) {
      temp = z[j] / (delta[j] * work[j]);
      psi += z[j] * temp;
      dpsi += temp * temp;
      erretm += psi;
    }
    erretm = fabs(erretm);
    temp = z[n - 1] / (delta[n - 1] * work[n - 1]);
    phi = z[n - 1] * temp;
    dphi = temp * temp;
    erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv;
    /* + |tau2|*(dpsi+dphi) omitted per the reference (commented out there) */
    w = rhoinv + phi + psi;
    if (fabs(w) <= eps * erretm) return 0;
    niter++;
    dtnsq1 = work[n - 2] * delta[n - 2];
    dtnsq = work[n - 1] * delta[n - 1];
    c = w - dtnsq1 * dpsi - dtnsq * dphi;
    a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
    b = dtnsq * dtnsq1 * w;
    if (c < 0.0) c = fabs(c);
    if (c == 0.0)
      eta = rho - *sigma * *sigma;
    else if (a >= 0.0)
      eta = (a + sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    else
      eta = 2.0 * b / (a - sqrt(fabs(a * a - 4.0 * b * c)));
    if (w * eta > 0.0) eta = -w / (dpsi + dphi);
    temp = eta - dtnsq;
    if (temp > rho) eta = rho + dtnsq;
    eta = eta / (*sigma + sqrt(eta + *sigma * *sigma));
    tau += eta;
    *sigma += eta;
    for (j = 0; j < n; j++) {
      delta[j] -= eta;
      work[j] += eta;
    }
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for (j = 0; j <= ii; j++) {
      temp = z[j] / (work[j] * delta[j]);
      psi += z[j] * temp;
      dpsi += temp * temp;
      erretm += psi;
    }
    erretm = fabs(erretm);
    tau2 = work[n - 1] * delta[n - 1];
    temp = z[n - 1] / tau2;
    phi = z[n - 1] * temp;
    dphi = temp * temp;
    erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv;
    w = rhoinv + phi + psi;
    iter = niter + 1;
    for (niter = iter; niter <= MAXIT; niter++) {
      if (fabs(w) <= eps * erretm) return 0;
      dtnsq1 = work[n - 2] * delta[n - 2];
      dtnsq = work[n - 1] * delta[n - 1];
      c = w - dtnsq1 * dpsi - dtnsq * dphi;
      a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
      b = dtnsq1 * dtnsq * w;
      if (a >= 0.0)
        eta = (a + sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
      else
        eta = 2.0 * b / (a - sqrt(fabs(a * a - 4.0 * b * c)));
      if (w * eta > 0.0) eta = -w / (dpsi + dphi);
      temp = eta - dtnsq;
      if (temp <= 0.0) eta = eta / 2.0;
      eta = eta / (*sigma + sqrt(eta + *sigma * *sigma));
      tau += eta;
      *sigma += eta;
      for (j = 0; j < n; j++) {
        delta[j] -= eta;
        work[j] += eta;
      }
      dpsi = 0.0;
      psi = 0.0;
      erretm = 0.0;
      for (j = 0; j <= ii; j++) {
        temp = z[j] / (work[j] * delta[j]);
        psi += z[j] * temp;
        dpsi += temp * temp;
        erretm += psi;
      }
      erretm = fabs(erretm);
      tau2 = work[n - 1] * delta[n - 1];
      temp = z[n - 1] / tau2;
      phi = z[n - 1] * temp;
      dphi = temp * temp;
      erretm = 8.0 * (-phi - psi) + erretm - phi + rhoinv;
      w = rhoinv + phi + psi;
    }
    return 1;
  }

  /* The case I < N: interior singular value. */
  ip1 = iev + 1;
  delsq = (d[ip1] - d[iev]) * (d[ip1] + d[iev]);
  delsq2 = delsq / 2.0;
  sq2 = sqrt((d[iev] * d[iev] + d[ip1] * d[ip1]) / 2.0);
  temp = delsq2 / (d[iev] + sq2);
  for (j = 0; j < n; j++) {
    work[j] = d[j] + d[iev] + temp;
    delta[j] = (d[j] - d[iev]) - temp;
  }
  psi = 0.0;
  for (j = 0; j < iev; j++) psi += z[j] * z[j] / (work[j] * delta[j]);
  phi = 0.0;
  for (j = n - 1; j >= iev + 2; j--) phi += z[j] * z[j] / (work[j] * delta[j]);
  c = rhoinv + psi + phi;
  w = c + z[iev] * z[iev] / (work[iev] * delta[iev]) +
      z[ip1] * z[ip1] / (work[ip1] * delta[ip1]);
  geomavg = 0;
  if (w > 0.0) {
    orgati = 1;
    ii = iev;
    sglb = 0.0;
    sgub = delsq2 / (d[iev] + sq2);
    a = c * delsq + z[iev] * z[iev] + z[ip1] * z[ip1];
    b = z[iev] * z[iev] * delsq;
    if (a > 0.0)
      tau2 = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    else
      tau2 = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    tau = tau2 / (d[iev] + sqrt(d[iev] * d[iev] + tau2));
    temp = sqrt(eps);
    if (d[iev] <= temp * d[ip1] && fabs(z[iev]) <= temp && d[iev] > 0.0) {
      tau = fmin(10.0 * d[iev], sgub);
      geomavg = 1;
    }
  } else {
    orgati = 0;
    ii = ip1;
    sglb = -delsq2 / (d[ii] + sq2);
    sgub = 0.0;
    a = c * delsq - z[iev] * z[iev] - z[ip1] * z[ip1];
    b = z[ip1] * z[ip1] * delsq;
    if (a < 0.0)
      tau2 = 2.0 * b / (a - sqrt(fabs(a * a + 4.0 * b * c)));
    else
      tau2 = -(a + sqrt(fabs(a * a + 4.0 * b * c))) / (2.0 * c);
    tau = tau2 / (d[ip1] + sqrt(fabs(d[ip1] * d[ip1] + tau2)));
  }
  *sigma = d[ii] + tau;
  for (j = 0; j < n; j++) {
    work[j] = d[j] + d[ii] + tau;
    delta[j] = (d[j] - d[ii]) - tau;
  }
  iim1 = ii - 1;
  iip1 = ii + 1;
  dpsi = 0.0;
  psi = 0.0;
  erretm = 0.0;
  for (j = 0; j < ii; j++) {
    temp = z[j] / (work[j] * delta[j]);
    psi += z[j] * temp;
    dpsi += temp * temp;
    erretm += psi;
  }
  erretm = fabs(erretm);
  dphi = 0.0;
  phi = 0.0;
  for (j = n - 1; j >= iip1; j--) {
    temp = z[j] / (work[j] * delta[j]);
    phi += z[j] * temp;
    dphi += temp * temp;
    erretm += phi;
  }
  w = rhoinv + phi + psi;
  swtch3 = 0;
  if (orgati) {
    if (w < 0.0) swtch3 = 1;
  } else {
    if (w > 0.0) swtch3 = 1;
  }
  if (ii == 0 || ii == n - 1) swtch3 = 0;
  temp = z[ii] / (work[ii] * delta[ii]);
  dw = dpsi + dphi + temp * temp;
  temp = z[ii] * temp;
  w = w + temp;
  erretm =
      8.0 * (phi - psi) + erretm + 2.0 * rhoinv + 3.0 * fabs(temp);
  /* + |tau2|*dw omitted per the reference (commented out there) */
  if (fabs(w) <= eps * erretm) goto done;
  if (w <= 0.0)
    sglb = fmax(sglb, tau);
  else
    sgub = fmin(sgub, tau);
  niter++;
  if (!swtch3) {
    dtipsq = work[ip1] * delta[ip1];
    dtisq = work[iev] * delta[iev];
    if (orgati)
      c = w - dtipsq * dw + delsq * (z[iev] / dtisq) * (z[iev] / dtisq);
    else
      c = w - dtisq * dw - delsq * (z[ip1] / dtipsq) * (z[ip1] / dtipsq);
    a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
    b = dtipsq * dtisq * w;
    if (c == 0.0) {
      if (a == 0.0) {
        if (orgati)
          a = z[iev] * z[iev] + dtipsq * dtipsq * (dpsi + dphi);
        else
          a = z[ip1] * z[ip1] + dtisq * dtisq * (dpsi + dphi);
      }
      eta = b / a;
    } else if (a <= 0.0)
      eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
    else
      eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
  } else {
    dtiim = work[iim1] * delta[iim1];
    dtiip = work[iip1] * delta[iip1];
    temp = rhoinv + psi + phi;
    if (orgati) {
      temp1 = z[iim1] / dtiim;
      temp1 = temp1 * temp1;
      c = (temp - dtiip * (dpsi + dphi)) -
          (d[iim1] - d[iip1]) * (d[iim1] + d[iip1]) * temp1;
      zz[0] = z[iim1] * z[iim1];
      if (dpsi < temp1)
        zz[2] = dtiip * dtiip * dphi;
      else
        zz[2] = dtiip * dtiip * ((dpsi - temp1) + dphi);
    } else {
      temp1 = z[iip1] / dtiip;
      temp1 = temp1 * temp1;
      c = (temp - dtiim * (dpsi + dphi)) -
          (d[iip1] - d[iim1]) * (d[iim1] + d[iip1]) * temp1;
      if (dphi < temp1)
        zz[0] = dtiim * dtiim * dpsi;
      else
        zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
      zz[2] = z[iip1] * z[iip1];
    }
    zz[1] = z[ii] * z[ii];
    dd[0] = dtiim;
    dd[1] = delta[ii] * work[ii];
    dd[2] = dtiip;
    if (la_ed6(niter, orgati, c, dd, zz, w, &eta)) {
      /* dlaed6 failed -> fall back to 2-pole (recompute as the !swtch3 branch) */
      swtch3 = 0;
      dtipsq = work[ip1] * delta[ip1];
      dtisq = work[iev] * delta[iev];
      if (orgati)
        c = w - dtipsq * dw + delsq * (z[iev] / dtisq) * (z[iev] / dtisq);
      else
        c = w - dtisq * dw - delsq * (z[ip1] / dtipsq) * (z[ip1] / dtipsq);
      a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
      b = dtipsq * dtisq * w;
      if (c == 0.0) {
        if (a == 0.0) {
          if (orgati)
            a = z[iev] * z[iev] + dtipsq * dtipsq * (dpsi + dphi);
          else
            a = z[ip1] * z[ip1] + dtisq * dtisq * (dpsi + dphi);
        }
        eta = b / a;
      } else if (a <= 0.0)
        eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
      else
        eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    }
  }
  if (w * eta >= 0.0) eta = -w / dw;
  eta = eta / (*sigma + sqrt(*sigma * *sigma + eta));
  temp = tau + eta;
  if (temp > sgub || temp < sglb) {
    if (w < 0.0)
      eta = (sgub - tau) / 2.0;
    else
      eta = (sglb - tau) / 2.0;
    if (geomavg) {
      if (w < 0.0) {
        if (tau > 0.0) eta = sqrt(sgub * tau) - tau;
      } else {
        if (sglb > 0.0) eta = sqrt(sglb * tau) - tau;
      }
    }
  }
  prew = w;
  tau += eta;
  *sigma += eta;
  for (j = 0; j < n; j++) {
    work[j] += eta;
    delta[j] -= eta;
  }
  dpsi = 0.0;
  psi = 0.0;
  erretm = 0.0;
  for (j = 0; j < ii; j++) {
    temp = z[j] / (work[j] * delta[j]);
    psi += z[j] * temp;
    dpsi += temp * temp;
    erretm += psi;
  }
  erretm = fabs(erretm);
  dphi = 0.0;
  phi = 0.0;
  for (j = n - 1; j >= iip1; j--) {
    temp = z[j] / (work[j] * delta[j]);
    phi += z[j] * temp;
    dphi += temp * temp;
    erretm += phi;
  }
  tau2 = work[ii] * delta[ii];
  temp = z[ii] / tau2;
  dw = dpsi + dphi + temp * temp;
  temp = z[ii] * temp;
  w = rhoinv + phi + psi + temp;
  erretm = 8.0 * (phi - psi) + erretm + 2.0 * rhoinv + 3.0 * fabs(temp);
  swtch = 0;
  if (orgati) {
    if (-w > fabs(prew) / 10.0) swtch = 1;
  } else {
    if (w > fabs(prew) / 10.0) swtch = 1;
  }
  iter = niter + 1;
  for (niter = iter; niter <= MAXIT; niter++) {
    if (fabs(w) <= eps * erretm) goto done;
    if (w <= 0.0)
      sglb = fmax(sglb, tau);
    else
      sgub = fmin(sgub, tau);
    if (!swtch3) {
      dtipsq = work[ip1] * delta[ip1];
      dtisq = work[iev] * delta[iev];
      if (!swtch) {
        if (orgati)
          c = w - dtipsq * dw + delsq * (z[iev] / dtisq) * (z[iev] / dtisq);
        else
          c = w - dtisq * dw - delsq * (z[ip1] / dtipsq) * (z[ip1] / dtipsq);
      } else {
        temp = z[ii] / (work[ii] * delta[ii]);
        if (orgati)
          dpsi += temp * temp;
        else
          dphi += temp * temp;
        c = w - dtisq * dpsi - dtipsq * dphi;
      }
      a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
      b = dtipsq * dtisq * w;
      if (c == 0.0) {
        if (a == 0.0) {
          if (!swtch) {
            if (orgati)
              a = z[iev] * z[iev] + dtipsq * dtipsq * (dpsi + dphi);
            else
              a = z[ip1] * z[ip1] + dtisq * dtisq * (dpsi + dphi);
          } else {
            a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
          }
        }
        eta = b / a;
      } else if (a <= 0.0)
        eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
      else
        eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
    } else {
      dtiim = work[iim1] * delta[iim1];
      dtiip = work[iip1] * delta[iip1];
      temp = rhoinv + psi + phi;
      if (swtch) {
        c = temp - dtiim * dpsi - dtiip * dphi;
        zz[0] = dtiim * dtiim * dpsi;
        zz[2] = dtiip * dtiip * dphi;
      } else {
        if (orgati) {
          temp1 = z[iim1] / dtiim;
          temp1 = temp1 * temp1;
          temp2 = (d[iim1] - d[iip1]) * (d[iim1] + d[iip1]) * temp1;
          c = temp - dtiip * (dpsi + dphi) - temp2;
          zz[0] = z[iim1] * z[iim1];
          if (dpsi < temp1)
            zz[2] = dtiip * dtiip * dphi;
          else
            zz[2] = dtiip * dtiip * ((dpsi - temp1) + dphi);
        } else {
          temp1 = z[iip1] / dtiip;
          temp1 = temp1 * temp1;
          temp2 = (d[iip1] - d[iim1]) * (d[iim1] + d[iip1]) * temp1;
          c = temp - dtiim * (dpsi + dphi) - temp2;
          if (dphi < temp1)
            zz[0] = dtiim * dtiim * dpsi;
          else
            zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
          zz[2] = z[iip1] * z[iip1];
        }
      }
      dd[0] = dtiim;
      dd[1] = delta[ii] * work[ii];
      dd[2] = dtiip;
      if (la_ed6(niter, orgati, c, dd, zz, w, &eta)) {
        swtch3 = 0;
        dtipsq = work[ip1] * delta[ip1];
        dtisq = work[iev] * delta[iev];
        if (!swtch) {
          if (orgati)
            c = w - dtipsq * dw + delsq * (z[iev] / dtisq) * (z[iev] / dtisq);
          else
            c = w - dtisq * dw - delsq * (z[ip1] / dtipsq) * (z[ip1] / dtipsq);
        } else {
          temp = z[ii] / (work[ii] * delta[ii]);
          if (orgati)
            dpsi += temp * temp;
          else
            dphi += temp * temp;
          c = w - dtisq * dpsi - dtipsq * dphi;
        }
        a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
        b = dtipsq * dtisq * w;
        if (c == 0.0) {
          if (a == 0.0) {
            if (!swtch) {
              if (orgati)
                a = z[iev] * z[iev] + dtipsq * dtipsq * (dpsi + dphi);
              else
                a = z[ip1] * z[ip1] + dtisq * dtisq * (dpsi + dphi);
            } else {
              a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
            }
          }
          eta = b / a;
        } else if (a <= 0.0)
          eta = (a - sqrt(fabs(a * a - 4.0 * b * c))) / (2.0 * c);
        else
          eta = 2.0 * b / (a + sqrt(fabs(a * a - 4.0 * b * c)));
      }
    }
    if (w * eta >= 0.0) eta = -w / dw;
    eta = eta / (*sigma + sqrt(*sigma * *sigma + eta));
    temp = tau + eta;
    if (temp > sgub || temp < sglb) {
      if (w < 0.0)
        eta = (sgub - tau) / 2.0;
      else
        eta = (sglb - tau) / 2.0;
      if (geomavg) {
        if (w < 0.0) {
          if (tau > 0.0) eta = sqrt(sgub * tau) - tau;
        } else {
          if (sglb > 0.0) eta = sqrt(sglb * tau) - tau;
        }
      }
    }
    prew = w;
    tau += eta;
    *sigma += eta;
    for (j = 0; j < n; j++) {
      work[j] += eta;
      delta[j] -= eta;
    }
    dpsi = 0.0;
    psi = 0.0;
    erretm = 0.0;
    for (j = 0; j < ii; j++) {
      temp = z[j] / (work[j] * delta[j]);
      psi += z[j] * temp;
      dpsi += temp * temp;
      erretm += psi;
    }
    erretm = fabs(erretm);
    dphi = 0.0;
    phi = 0.0;
    for (j = n - 1; j >= iip1; j--) {
      temp = z[j] / (work[j] * delta[j]);
      phi += z[j] * temp;
      dphi += temp * temp;
      erretm += phi;
    }
    tau2 = work[ii] * delta[ii];
    temp = z[ii] / tau2;
    dw = dpsi + dphi + temp * temp;
    temp = z[ii] * temp;
    w = rhoinv + phi + psi + temp;
    erretm = 8.0 * (phi - psi) + erretm + 2.0 * rhoinv + 3.0 * fabs(temp);
    if (w * prew > 0.0 && fabs(w) > fabs(prew) / 10.0) swtch = !swtch;
  }
  /* not converged */
  return 1;
done:
  return 0;
}

/* la_sd_bdsqr: double dbdsqr for the D&C leaves — the Demmel–Kahan sweep of
   la_bdsvd (same TU) specialized to double, COLUMN-major U (nru×n) and VT
   (n×ncvt), with independent vector dims. Output: d nonnegative ASCENDING with
   matching U column / VT row order (dlasdq sorts ascending for the merge).
   Requires la_lartg/la_las2/la_lasv2 in scope. Nonzero on non-convergence. */
static int la_sd_bdsqr(int n, int ncvt, int nru, double *d, double *e,
                       double *VT, int ldvt, double *U, int ldu) {
  if (n == 0) return 0;
  if (n > 1) {
    const double eps = 0.5 * DBL_EPSILON; /* DLAMCH('E') */
    const double unfl = DBL_MIN;
    const int MAXITR = 6;
    double tolmul = pow(eps, -0.125);
    if (tolmul < 10.0) tolmul = 10.0;
    if (tolmul > 100.0) tolmul = 100.0;
    double tol = tolmul * eps;
    double sminoa = fabs(d[0]);
    if (sminoa != 0.0) {
      double mu = sminoa;
      for (int i = 1; i < n; i++) {
        mu = fabs(d[i]) * (mu / (mu + fabs(e[i - 1])));
        if (mu < sminoa) sminoa = mu;
        if (sminoa == 0.0) break;
      }
    }
    sminoa = sminoa / sqrt((double)n);
    double thresh = tol * sminoa;
    double flr = (double)(MAXITR * n) * (double)n * unfl;
    if (flr > thresh) thresh = flr;
    long maxit = (long)MAXITR * n * n;
    long iter = 0;
    int m = n - 1;
    int oldll = -1, oldm = -1;
    int idir = 0;
    double sminl = 0.0;
    while (1) {
      if (m <= 0) break;
      if (iter > maxit) return 1;
      double bsmax = fabs(d[m]);
      int ll = 0;
      int split_bottom = 0;
      for (int lll = 1; lll <= m; lll++) {
        int c = m - lll;
        double abss = fabs(d[c]);
        double abse = fabs(e[c]);
        if (abse <= thresh) {
          e[c] = 0.0;
          if (c == m - 1) {
            m = m - 1;
            split_bottom = 1;
          } else
            ll = c + 1;
          break;
        }
        if (abss > bsmax) bsmax = abss;
        if (abse > bsmax) bsmax = abse;
      }
      if (split_bottom) continue;
      if (ll == m - 1) {
        double sigmn, sigmx, snr, csr, snl, csl;
        la_lasv2(d[m - 1], e[m - 1], d[m], &sigmn, &sigmx, &snr, &csr, &snl,
                 &csl);
        d[m - 1] = sigmx;
        e[m - 1] = 0.0;
        d[m] = sigmn;
        for (int c = 0; c < ncvt; c++) {
          double t1 = VT[(m - 1) + c * ldvt], t2 = VT[m + c * ldvt];
          VT[(m - 1) + c * ldvt] = csr * t1 + snr * t2;
          VT[m + c * ldvt] = csr * t2 - snr * t1;
        }
        for (int r = 0; r < nru; r++) {
          double t1 = U[r + (m - 1) * ldu], t2 = U[r + m * ldu];
          U[r + (m - 1) * ldu] = csl * t1 + snl * t2;
          U[r + m * ldu] = csl * t2 - snl * t1;
        }
        m = m - 2;
        continue;
      }
      /* choose the chase direction only on a NEW submatrix (dbdsqr's
         OLDLL/OLDM latch); re-deciding every sweep can flip-flop on borderline
         blocks and diverge from the reference's iteration count. */
      if (ll > oldm || m < oldll)
        idir = fabs(d[ll]) >= fabs(d[m]) ? 1 : 2;
      if (idir == 1) {
        /* dbdsqr's second disjunct (|e| <= thresh) is TOL<0-only in the
           reference; tol > 0 always here and the split scan zeroed those. */
        if (fabs(e[m - 1]) <= tol * fabs(d[m])) {
          e[m - 1] = 0.0;
          continue;
        }
        double mu = fabs(d[ll]);
        sminl = mu;
        int conv = 0;
        for (int lll = ll; lll < m; lll++) {
          if (fabs(e[lll]) <= tol * mu) {
            e[lll] = 0.0;
            conv = 1;
            break;
          }
          mu = fabs(d[lll + 1]) * (mu / (mu + fabs(e[lll])));
          if (mu < sminl) sminl = mu;
        }
        if (conv) continue;
      } else {
        if (fabs(e[ll]) <= tol * fabs(d[ll])) {
          e[ll] = 0.0;
          continue;
        }
        double mu = fabs(d[m]);
        sminl = mu;
        int conv = 0;
        for (int lll = m - 1; lll >= ll; lll--) {
          if (fabs(e[lll]) <= tol * mu) {
            e[lll] = 0.0;
            conv = 1;
            break;
          }
          mu = fabs(d[lll]) * (mu / (mu + fabs(e[lll])));
          if (mu < sminl) sminl = mu;
        }
        if (conv) continue;
      }
      oldll = ll;
      oldm = m;
      double shift = 0.0;
      double thr2 = 0.01 * tol;
      if (eps > thr2) thr2 = eps;
      if ((double)n * tol * (sminl / bsmax) <= thr2) {
        shift = 0.0;
      } else {
        double sll;
        if (idir == 1) {
          sll = fabs(d[ll]);
          shift = la_las2(d[m - 1], e[m - 1], d[m]);
        } else {
          sll = fabs(d[m]);
          shift = la_las2(d[ll], e[ll], d[ll + 1]);
        }
        if (sll > 0.0) {
          double q = shift / sll;
          if (q * q < eps) shift = 0.0;
        }
      }
      iter += (m - ll);
      if (shift == 0.0) {
        if (idir == 1) {
          double cs = 1.0, oldcs = 1.0, sn = 0.0, oldsn = 0.0, r;
          for (int i = ll; i < m; i++) {
            la_lartg(d[i] * cs, e[i], &cs, &sn, &r);
            if (i > ll) e[i - 1] = oldsn * r;
            la_lartg(oldcs * r, d[i + 1] * sn, &oldcs, &oldsn, &r);
            d[i] = r;
            for (int c = 0; c < ncvt; c++) {
              double t1 = VT[i + c * ldvt], t2 = VT[(i + 1) + c * ldvt];
              VT[i + c * ldvt] = cs * t1 + sn * t2;
              VT[(i + 1) + c * ldvt] = cs * t2 - sn * t1;
            }
            for (int rr = 0; rr < nru; rr++) {
              double t1 = U[rr + i * ldu], t2 = U[rr + (i + 1) * ldu];
              U[rr + i * ldu] = oldcs * t1 + oldsn * t2;
              U[rr + (i + 1) * ldu] = oldcs * t2 - oldsn * t1;
            }
          }
          double h = d[m] * cs;
          d[m] = h * oldcs;
          e[m - 1] = h * oldsn;
          if (fabs(e[m - 1]) <= thresh) e[m - 1] = 0.0;
        } else {
          double cs = 1.0, oldcs = 1.0, sn = 0.0, oldsn = 0.0, r;
          for (int i = m; i > ll; i--) {
            la_lartg(d[i] * cs, e[i - 1], &cs, &sn, &r);
            if (i < m) e[i] = oldsn * r;
            la_lartg(oldcs * r, d[i - 1] * sn, &oldcs, &oldsn, &r);
            d[i] = r;
            /* Backward chase pairs the rotations with U/Vᴴ opposite to the
               forward sweep: cs → U, oldcs → Vᴴ (dbdsqr's idir==2 DLASR
               argument swap). */
            for (int c = 0; c < ncvt; c++) {
              double t1 = VT[(i - 1) + c * ldvt], t2 = VT[i + c * ldvt];
              VT[(i - 1) + c * ldvt] = oldcs * t1 - oldsn * t2;
              VT[i + c * ldvt] = oldcs * t2 + oldsn * t1;
            }
            for (int rr = 0; rr < nru; rr++) {
              double t1 = U[rr + (i - 1) * ldu], t2 = U[rr + i * ldu];
              U[rr + (i - 1) * ldu] = cs * t1 - sn * t2;
              U[rr + i * ldu] = cs * t2 + sn * t1;
            }
          }
          double h = d[ll] * cs;
          d[ll] = h * oldcs;
          e[ll] = h * oldsn;
          if (fabs(e[ll]) <= thresh) e[ll] = 0.0;
        }
      } else {
        if (idir == 1) {
          double dl = d[ll];
          double f = (fabs(dl) - shift) * (copysign(1.0, dl) + shift / dl);
          double g = e[ll];
          double cosr, sinr, cosl, sinl, r;
          for (int i = ll; i < m; i++) {
            la_lartg(f, g, &cosr, &sinr, &r);
            if (i > ll) e[i - 1] = r;
            f = cosr * d[i] + sinr * e[i];
            e[i] = cosr * e[i] - sinr * d[i];
            g = sinr * d[i + 1];
            d[i + 1] = cosr * d[i + 1];
            la_lartg(f, g, &cosl, &sinl, &r);
            d[i] = r;
            f = cosl * e[i] + sinl * d[i + 1];
            d[i + 1] = cosl * d[i + 1] - sinl * e[i];
            if (i < m - 1) {
              g = sinl * e[i + 1];
              e[i + 1] = cosl * e[i + 1];
            }
            for (int c = 0; c < ncvt; c++) {
              double t1 = VT[i + c * ldvt], t2 = VT[(i + 1) + c * ldvt];
              VT[i + c * ldvt] = cosr * t1 + sinr * t2;
              VT[(i + 1) + c * ldvt] = cosr * t2 - sinr * t1;
            }
            for (int rr = 0; rr < nru; rr++) {
              double u1 = U[rr + i * ldu], u2 = U[rr + (i + 1) * ldu];
              U[rr + i * ldu] = cosl * u1 + sinl * u2;
              U[rr + (i + 1) * ldu] = cosl * u2 - sinl * u1;
            }
          }
          e[m - 1] = f;
          if (fabs(e[m - 1]) <= thresh) e[m - 1] = 0.0;
        } else {
          double dm = d[m];
          double f = (fabs(dm) - shift) * (copysign(1.0, dm) + shift / dm);
          double g = e[m - 1];
          double cosr, sinr, cosl, sinl, r;
          for (int i = m; i > ll; i--) {
            la_lartg(f, g, &cosr, &sinr, &r);
            if (i < m) e[i] = r;
            f = cosr * d[i] + sinr * e[i - 1];
            e[i - 1] = cosr * e[i - 1] - sinr * d[i];
            g = sinr * d[i - 1];
            d[i - 1] = cosr * d[i - 1];
            la_lartg(f, g, &cosl, &sinl, &r);
            d[i] = r;
            f = cosl * e[i - 1] + sinl * d[i - 1];
            d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
            if (i > ll + 1) {
              g = sinl * e[i - 2];
              e[i - 2] = cosl * e[i - 2];
            }
            /* idir==2 pairing: cosr → U, cosl → Vᴴ (see the zero-shift
               backward sweep above). */
            for (int c = 0; c < ncvt; c++) {
              double t1 = VT[(i - 1) + c * ldvt], t2 = VT[i + c * ldvt];
              VT[(i - 1) + c * ldvt] = cosl * t1 - sinl * t2;
              VT[i + c * ldvt] = cosl * t2 + sinl * t1;
            }
            for (int rr = 0; rr < nru; rr++) {
              double u1 = U[rr + (i - 1) * ldu], u2 = U[rr + i * ldu];
              U[rr + (i - 1) * ldu] = cosr * u1 - sinr * u2;
              U[rr + i * ldu] = cosr * u2 + sinr * u1;
            }
          }
          e[ll] = f;
          if (fabs(e[ll]) <= thresh) e[ll] = 0.0;
        }
      }
    }
  }
  for (int i = 0; i < n; i++) {
    if (d[i] == 0.0) d[i] = 0.0; /* avoid -0 (dbdsqr) */
    if (d[i] < 0.0) {
      d[i] = -d[i];
      for (int c = 0; c < ncvt; c++) VT[i + c * ldvt] = -VT[i + c * ldvt];
    }
  }
  /* ascending selection sort (dlasdq order; la_bdsvd's descending is the
     driver-level convention, applied once at the very top by la_sd_dc) */
  for (int i = 0; i < n - 1; i++) {
    int mn = i;
    for (int j = i + 1; j < n; j++)
      if (d[j] < d[mn]) mn = j;
    if (mn != i) {
      double t = d[i];
      d[i] = d[mn];
      d[mn] = t;
      for (int c = 0; c < ncvt; c++) {
        double tv = VT[i + c * ldvt];
        VT[i + c * ldvt] = VT[mn + c * ldvt];
        VT[mn + c * ldvt] = tv;
      }
      for (int r = 0; r < nru; r++) {
        double tu = U[r + i * ldu];
        U[r + i * ldu] = U[r + mn * ldu];
        U[r + mn * ldu] = tu;
      }
    }
  }
  return 0;
}

/* la_sdq (dlasdq, upper, always with vectors): SVD of the n×(n+sqre) upper
   bidiagonal (d, e) whose U (n cols over n rows) and VT (m=n+sqre rows over m
   cols) sub-blocks the caller seeded (identity at the D&C leaves). sqre==1 is
   the NL×(NL+1) leaf: chase the extra column off with right Givens (folded into
   VT's rows), then re-chase lower→upper with left Givens (folded into U's
   columns), then the square dbdsqr. Output ascending nonnegative. work: 2m
   doubles for the rotation pairs. */
static int la_sdq(int n, int sqre, double *d, double *e, double *U, int ldu,
                  double *VT, int ldvt, double *work) {
  if (n == 0) return 0;
  int m = n + sqre;
  double *cs_a = work, *sn_a = work + m;
  if (sqre == 1) {
    double cs, sn, r;
    for (int i = 0; i < n - 1; i++) {
      la_lartg(d[i], e[i], &cs, &sn, &r);
      d[i] = r;
      e[i] = sn * d[i + 1];
      d[i + 1] = cs * d[i + 1];
      cs_a[i] = cs;
      sn_a[i] = sn;
    }
    la_lartg(d[n - 1], e[n - 1], &cs, &sn, &r);
    d[n - 1] = r;
    e[n - 1] = 0.0;
    cs_a[n - 1] = cs;
    sn_a[n - 1] = sn;
    /* DLASR('L','V','F', m, m): forward row rotations on VT */
    for (int j = 0; j < m - 1; j++) {
      double c = cs_a[j], s = sn_a[j];
      if (c == 1.0 && s == 0.0) continue;
      for (int col = 0; col < m; col++) {
        double t = VT[(j + 1) + col * ldvt];
        VT[(j + 1) + col * ldvt] = c * t - s * VT[j + col * ldvt];
        VT[j + col * ldvt] = s * t + c * VT[j + col * ldvt];
      }
    }
    /* now lower bidiagonal n×n; rotate back to upper with left Givens */
    for (int i = 0; i < n - 1; i++) {
      la_lartg(d[i], e[i], &cs, &sn, &r);
      d[i] = r;
      e[i] = sn * d[i + 1];
      d[i + 1] = cs * d[i + 1];
      cs_a[i] = cs;
      sn_a[i] = sn;
    }
    /* DLASR('R','V','F', n, n): forward column rotations on U */
    for (int j = 0; j < n - 1; j++) {
      double c = cs_a[j], s = sn_a[j];
      if (c == 1.0 && s == 0.0) continue;
      for (int row = 0; row < n; row++) {
        double t = U[row + (j + 1) * ldu];
        U[row + (j + 1) * ldu] = c * t - s * U[row + j * ldu];
        U[row + j * ldu] = s * t + c * U[row + j * ldu];
      }
    }
  }
  return la_sd_bdsqr(n, m, n, d, e, VT, ldvt, U, ldu);
}

/* la_sd2 (dlasd2): merge the two solved subproblems' singular values and
   deflate. n = nl+nr+1 rows, m = n+sqre cols. d/z/U/VT column-major; on exit
   the first *kout entries of dsigma (ascending, dsigma[0]=0) and z drive the
   secular problem, U2/VT2 hold the deflated-ordered vectors grouped by column
   type (1 top-only / 2 bottom-only / 3 dense / 4 deflated), and coltyp[0..3]
   returns the group counts. idxq on entry: the two children's ascending
   permutations (dlasd1 adjusts them here). All indices 0-based. */
static void la_sd2(int nl, int nr, int sqre, int *kout, double *d, double *z,
                   double alpha, double beta, double *U, int ldu, double *VT,
                   int ldvt, double *dsigma, double *U2, int ldu2, double *VT2,
                   int ldvt2, int *idxp, int *idx, int *idxc, int *idxq,
                   int *coltyp) {
  int n = nl + nr + 1;
  int m = n + sqre;
  double cs = 1.0, sn = 0.0;
  /* first part of z from column nl of VT; shift d/idxq back one slot */
  double z1 = alpha * VT[nl + (size_t)nl * ldvt];
  for (int i = nl - 1; i >= 0; i--) {
    z[i + 1] = alpha * VT[i + (size_t)nl * ldvt];
    d[i + 1] = d[i];
    idxq[i + 1] = idxq[i] + 1;
  }
  z[0] = z1;
  for (int i = nl + 1; i < m; i++) z[i] = beta * VT[i + (size_t)(nl + 1) * ldvt];
  for (int i = 1; i <= nl; i++) coltyp[i] = 1;
  for (int i = nl + 1; i < n; i++) coltyp[i] = 2;
  for (int i = nl + 1; i < n; i++) idxq[i] += nl + 1;
  /* dsigma / U2 col 0 / idxc as scratch for the merge sort */
  for (int i = 1; i < n; i++) {
    dsigma[i] = d[idxq[i]];
    U2[i] = z[idxq[i]];
    idxc[i] = coltyp[idxq[i]];
  }
  la_dc_dlamrg(nl, nr, &dsigma[1], 1, 1, &idx[1]);
  for (int i = 1; i < n; i++) {
    int idxi = 1 + idx[i];
    d[i] = dsigma[idxi];
    z[i] = U2[idxi];
    coltyp[i] = idxc[idxi];
  }
  /* deflation tolerance (eps = DLAMCH('E') = DBL_EPSILON/2) */
  double eps = 0.5 * DBL_EPSILON;
  double tol = fmax(fabs(alpha), fabs(beta));
  tol = 8.0 * 8.0 * eps * fmax(fabs(d[n - 1]), tol);
  int k = 1;  /* nondeflated count, slot 0 always participates */
  int k2 = n; /* 0-based boundary: deflated occupy idxp[k2..n-1] */
  int jprev = 0, j, found = 0;
  for (j = 1; j < n; j++) {
    if (fabs(z[j]) <= tol) {
      k2--;
      idxp[k2] = j;
      coltyp[j] = 4;
      if (j == n - 1) goto deflated_all;
    } else {
      jprev = j;
      found = 1;
      break;
    }
  }
  if (found) {
    j = jprev;
    while (1) {
      j++;
      if (j >= n) break;
      if (fabs(z[j]) <= tol) {
        k2--;
        idxp[k2] = j;
        coltyp[j] = 4;
      } else if (d[j] - d[jprev] <= tol) {
        /* two close singular values: rotate z mass into slot j, deflate jprev
           (Givens applied to BOTH U columns and VT rows) */
        double s = z[jprev];
        double c = z[j];
        double tau = hypot(c, s);
        c = c / tau;
        s = -s / tau;
        z[j] = tau;
        z[jprev] = 0.0;
        int idxjp = idxq[idx[jprev] + 1];
        int idxj = idxq[idx[j] + 1];
        if (idxjp <= nl) idxjp--;
        if (idxj <= nl) idxj--;
        for (int r = 0; r < n; r++) {
          double t1 = U[r + (size_t)idxjp * ldu], t2 = U[r + (size_t)idxj * ldu];
          U[r + (size_t)idxjp * ldu] = c * t1 + s * t2;
          U[r + (size_t)idxj * ldu] = c * t2 - s * t1;
        }
        for (int cc = 0; cc < m; cc++) {
          double t1 = VT[idxjp + (size_t)cc * ldvt],
                 t2 = VT[idxj + (size_t)cc * ldvt];
          VT[idxjp + (size_t)cc * ldvt] = c * t1 + s * t2;
          VT[idxj + (size_t)cc * ldvt] = c * t2 - s * t1;
        }
        if (coltyp[j] != coltyp[jprev]) coltyp[j] = 3;
        coltyp[jprev] = 4;
        k2--;
        for (int jp = jprev; jp <= j - 1; jp++) idxp[k2 + (j - 1) - jp] = jp;
        jprev = j;
      } else {
        k++;
        U2[k - 1] = z[jprev];
        dsigma[k - 1] = d[jprev];
        idxp[k - 1] = jprev;
        jprev = j;
      }
    }
    /* record the last singular value */
    k++;
    U2[k - 1] = z[jprev];
    dsigma[k - 1] = d[jprev];
    idxp[k - 1] = jprev;
  }
deflated_all:;
  /* count column types and build the grouping permutation idxc */
  int ctot[4] = {0, 0, 0, 0}, psm[4];
  for (int jj = 1; jj < n; jj++) ctot[coltyp[jj] - 1]++;
  psm[0] = 1;
  psm[1] = 1 + ctot[0];
  psm[2] = psm[1] + ctot[1];
  psm[3] = psm[2] + ctot[2];
  for (int jj = 1; jj < n; jj++) {
    int jp = idxp[jj];
    int ct = coltyp[jp];
    idxc[psm[ct - 1]] = jj;
    psm[ct - 1]++;
  }
  /* sort singular values/vectors into dsigma/U2/VT2 */
  for (int jj = 1; jj < n; jj++) {
    int jp = idxp[jj];
    dsigma[jj] = d[jp];
    int idxj = idxq[idx[idxp[idxc[jj]]] + 1];
    if (idxj <= nl) idxj--;
    for (int r = 0; r < n; r++)
      U2[r + (size_t)jj * ldu2] = U[r + (size_t)idxj * ldu];
    for (int cc = 0; cc < m; cc++)
      VT2[jj + (size_t)cc * ldvt2] = VT[idxj + (size_t)cc * ldvt];
  }
  /* dsigma[0], dsigma[1] floor, and z[0] */
  dsigma[0] = 0.0;
  double hlftol = tol / 2.0;
  if (fabs(dsigma[1]) <= hlftol) dsigma[1] = hlftol;
  if (m > n) {
    z[0] = hypot(z1, z[m - 1]);
    if (z[0] <= tol) {
      cs = 1.0;
      sn = 0.0;
      z[0] = tol;
    } else {
      cs = z1 / z[0];
      sn = z[m - 1] / z[0];
    }
  } else {
    if (fabs(z1) <= tol)
      z[0] = tol;
    else
      z[0] = z1;
  }
  for (int i = 1; i < k; i++) z[i] = U2[i];
  /* first column of U2, first row of VT2, last row of VT */
  for (int r = 0; r < n; r++) U2[r] = 0.0;
  U2[nl] = 1.0;
  if (m > n) {
    for (int i = 0; i <= nl; i++) {
      VT[(m - 1) + (size_t)i * ldvt] = -sn * VT[nl + (size_t)i * ldvt];
      VT2[0 + (size_t)i * ldvt2] = cs * VT[nl + (size_t)i * ldvt];
    }
    for (int i = nl + 1; i < m; i++) {
      VT2[0 + (size_t)i * ldvt2] = sn * VT[(m - 1) + (size_t)i * ldvt];
      VT[(m - 1) + (size_t)i * ldvt] = cs * VT[(m - 1) + (size_t)i * ldvt];
    }
  } else {
    for (int cc = 0; cc < m; cc++)
      VT2[0 + (size_t)cc * ldvt2] = VT[nl + (size_t)cc * ldvt];
  }
  if (m > n) {
    for (int cc = 0; cc < m; cc++)
      VT2[(m - 1) + (size_t)cc * ldvt2] = VT[(m - 1) + (size_t)cc * ldvt];
  }
  /* deflated tail back into d, U, VT */
  if (n > k) {
    for (int i = k; i < n; i++) d[i] = dsigma[i];
    for (int jj = k; jj < n; jj++) {
      for (int r = 0; r < n; r++)
        U[r + (size_t)jj * ldu] = U2[r + (size_t)jj * ldu2];
      for (int cc = 0; cc < m; cc++)
        VT[jj + (size_t)cc * ldvt] = VT2[jj + (size_t)cc * ldvt2];
    }
  }
  for (int jj = 0; jj < 4; jj++) coltyp[jj] = ctot[jj];
  *kout = k;
}

/* la_sd3 (dlasd3): solve the k secular equations, recompute z (Gu–Eisenstat, in
   the sqrt form with the d^2 - sigma^2 pole distances carried as work·delta
   pairs stored in U/VT columns), normalize the dual vector sets, and
   back-multiply against the deflation-grouped U2/VT2 blocks — the BLAS-3 heart.
   Q is k×k (ldq=k); acc is an n×m accumulation temp (the reference's beta=1
   GEMMs; nx_c_gemm2d overwrites, so accumulate by hand). LA_ERR_NO_CONVERGE on
   secular non-convergence; a failing GEMM propagates its own status. */
static nx_c_status la_sd3(int nl, int nr, int sqre, int k, double *d, double *Q,
                         int ldq, double *dsigma, double *U, int ldu,
                         double *U2, int ldu2, double *VT, int ldvt,
                         double *VT2, int ldvt2, const int *idxc,
                         const int *ctot, double *z, double *acc, void *gemm) {
  int n = nl + nr + 1;
  int m = n + sqre;
  nx_c_status gst;
  if (k == 1) {
    d[0] = fabs(z[0]);
    for (int cc = 0; cc < m; cc++)
      VT[0 + (size_t)cc * ldvt] = VT2[0 + (size_t)cc * ldvt2];
    if (z[0] > 0.0)
      for (int r = 0; r < n; r++) U[r] = U2[r];
    else
      for (int r = 0; r < n; r++) U[r] = -U2[r];
    return NX_C_OK;
  }
  /* keep a copy of z in Q column 0; normalize z */
  for (int i = 0; i < k; i++) Q[i] = z[i];
  double rho = la_dc_dnrm2(k, z);
  for (int i = 0; i < k; i++) z[i] /= rho;
  rho = rho * rho;
  /* secular roots; U column j holds delta, VT column j holds work */
  for (int jj = 0; jj < k; jj++) {
    if (la_sd4(k, jj, dsigma, z, &U[(size_t)jj * ldu], rho, &d[jj],
               &VT[(size_t)jj * ldvt]))
      return LA_ERR_NO_CONVERGE;
  }
  /* updated z */
  for (int i = 0; i < k; i++) {
    double zi = U[i + (size_t)(k - 1) * ldu] * VT[i + (size_t)(k - 1) * ldvt];
    for (int jj = 0; jj < i; jj++)
      zi *= U[i + (size_t)jj * ldu] * VT[i + (size_t)jj * ldvt] /
            (dsigma[i] - dsigma[jj]) / (dsigma[i] + dsigma[jj]);
    for (int jj = i; jj < k - 1; jj++)
      zi *= U[i + (size_t)jj * ldu] * VT[i + (size_t)jj * ldvt] /
            (dsigma[i] - dsigma[jj + 1]) / (dsigma[i] + dsigma[jj + 1]);
    z[i] = copysign(sqrt(fabs(zi)), Q[i]);
  }
  /* left singular vectors of the deflated problem (rows permuted by idxc into
     the U2 grouping) and the VT-column ingredients for the right vectors */
  for (int i = 0; i < k; i++) {
    VT[0 + (size_t)i * ldvt] =
        z[0] / U[0 + (size_t)i * ldu] / VT[0 + (size_t)i * ldvt];
    U[0 + (size_t)i * ldu] = -1.0;
    for (int jj = 1; jj < k; jj++) {
      VT[jj + (size_t)i * ldvt] =
          z[jj] / U[jj + (size_t)i * ldu] / VT[jj + (size_t)i * ldvt];
      U[jj + (size_t)i * ldu] = dsigma[jj] * VT[jj + (size_t)i * ldvt];
    }
    double temp = la_dc_dnrm2(k, &U[(size_t)i * ldu]);
    Q[0 + (size_t)i * ldq] = U[0 + (size_t)i * ldu] / temp;
    for (int jj = 1; jj < k; jj++) {
      int jc = idxc[jj];
      Q[jj + (size_t)i * ldq] = U[jc + (size_t)i * ldu] / temp;
    }
  }
  /* update the left singular vector matrix */
  if (k == 2) {
    gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, n, k, k, (const char *)U2, 1, ldu2,
                           (const char *)Q, 1, ldq, (char *)U, 1, ldu,
                           (char *)gemm);
    if (gst != NX_C_OK) return gst;
    goto rightvec;
  }
  if (ctot[0] > 0) {
    gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, nl, k, ctot[0],
                           (const char *)&U2[(size_t)1 * ldu2], 1, ldu2,
                           (const char *)&Q[1], 1, ldq, (char *)U, 1, ldu,
                           (char *)gemm);
    if (gst != NX_C_OK) return gst;
    if (ctot[2] > 0) {
      int kt = 1 + ctot[0] + ctot[1];
      gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, nl, k, ctot[2],
                             (const char *)&U2[(size_t)kt * ldu2], 1, ldu2,
                             (const char *)&Q[kt], 1, ldq, (char *)acc, 1, nl,
                             (char *)gemm);
      if (gst != NX_C_OK) return gst;
      for (int jj = 0; jj < k; jj++)
        for (int r = 0; r < nl; r++)
          U[r + (size_t)jj * ldu] += acc[r + (size_t)jj * nl];
    }
  } else if (ctot[2] > 0) {
    int kt = 1 + ctot[0] + ctot[1];
    gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, nl, k, ctot[2],
                           (const char *)&U2[(size_t)kt * ldu2], 1, ldu2,
                           (const char *)&Q[kt], 1, ldq, (char *)U, 1, ldu,
                           (char *)gemm);
    if (gst != NX_C_OK) return gst;
  } else {
    for (int jj = 0; jj < k; jj++)
      for (int r = 0; r < nl; r++)
        U[r + (size_t)jj * ldu] = U2[r + (size_t)jj * ldu2];
  }
  for (int jj = 0; jj < k; jj++) U[nl + (size_t)jj * ldu] = Q[(size_t)jj * ldq];
  {
    int kt2 = 1 + ctot[0];
    int ctemp = ctot[1] + ctot[2];
    if (ctemp > 0) {
      gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, nr, k, ctemp,
                             (const char *)&U2[(nl + 1) + (size_t)kt2 * ldu2],
                             1, ldu2, (const char *)&Q[kt2], 1, ldq,
                             (char *)&U[nl + 1], 1, ldu, (char *)gemm);
      if (gst != NX_C_OK) return gst;
    } else {
      for (int jj = 0; jj < k; jj++)
        for (int r = 0; r < nr; r++) U[(nl + 1) + r + (size_t)jj * ldu] = 0.0;
    }
  }
rightvec:
  /* right singular vectors of the deflated problem, into the ROWS of Q */
  for (int i = 0; i < k; i++) {
    double temp = la_dc_dnrm2(k, &VT[(size_t)i * ldvt]);
    Q[i] = VT[0 + (size_t)i * ldvt] / temp;
    for (int jj = 1; jj < k; jj++) {
      int jc = idxc[jj];
      Q[i + (size_t)jj * ldq] = VT[jc + (size_t)i * ldvt] / temp;
    }
  }
  /* update the right singular vector matrix */
  if (k == 2) {
    return nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, k, m, k, (const char *)Q, 1, ldq,
                            (const char *)VT2, 1, ldvt2, (char *)VT, 1, ldvt,
                            (char *)gemm);
  }
  {
    int ktemp = 1 + ctot[0];
    gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, k, nl + 1, ktemp, (const char *)Q, 1,
                           ldq, (const char *)VT2, 1, ldvt2, (char *)VT, 1,
                           ldvt, (char *)gemm);
    if (gst != NX_C_OK) return gst;
    int kt3 = 1 + ctot[0] + ctot[1];
    if (kt3 + 1 <= ldvt2 && ctot[2] > 0) {
      gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, k, nl + 1, ctot[2],
                             (const char *)&Q[(size_t)kt3 * ldq], 1, ldq,
                             (const char *)&VT2[kt3], 1, ldvt2, (char *)acc, 1,
                             k, (char *)gemm);
      if (gst != NX_C_OK) return gst;
      for (int cc = 0; cc <= nl; cc++)
        for (int r = 0; r < k; r++)
          VT[r + (size_t)cc * ldvt] += acc[r + (size_t)cc * k];
    }
    int ktc = ctot[0];
    int nrp1 = nr + sqre;
    if (ktc > 0) {
      for (int i = 0; i < k; i++) Q[i + (size_t)ktc * ldq] = Q[i];
      for (int i = nl + 1; i < m; i++)
        VT2[ktc + (size_t)i * ldvt2] = VT2[0 + (size_t)i * ldvt2];
    }
    int ctemp2 = 1 + ctot[1] + ctot[2];
    gst = nx_c_gemm2d_ct_ws(NX_C_DTYPE_f64, k, nrp1, ctemp2,
                           (const char *)&Q[(size_t)ktc * ldq], 1, ldq,
                           (const char *)&VT2[ktc + (size_t)(nl + 1) * ldvt2],
                           1, ldvt2, (char *)&VT[(size_t)(nl + 1) * ldvt], 1,
                           ldvt, (char *)gemm);
    if (gst != NX_C_OK) return gst;
  }
  return NX_C_OK;
}

/* la_sd1 (dlasd1): merge the two solved halves joined by the torn middle row
   (alpha = its diagonal, beta = its superdiagonal). work: m + n + n² + m² + n²
   + n·m doubles (z, dsigma, U2, VT2, Q, acc); iwork: 4n ints. idxq in/out as
   in la_sd2. */
static nx_c_status la_sd1(int nl, int nr, int sqre, double *d, double alpha,
                         double beta, double *U, int ldu, double *VT, int ldvt,
                         int *idxq, int *iwork, double *work, void *gemm) {
  int n = nl + nr + 1;
  int m = n + sqre;
  double *z = work;
  double *dsigma = z + m;
  double *U2 = dsigma + n;
  double *VT2 = U2 + (size_t)n * n;
  double *Q = VT2 + (size_t)m * m;
  double *acc = Q + (size_t)n * n;
  int *idx = iwork, *idxc = idx + n, *coltyp = idxc + n, *idxp = coltyp + n;
  double orgnrm = fmax(fabs(alpha), fabs(beta));
  d[nl] = 0.0;
  for (int i = 0; i < n; i++)
    if (fabs(d[i]) > orgnrm) orgnrm = fabs(d[i]);
  for (int i = 0; i < n; i++) d[i] /= orgnrm;
  alpha /= orgnrm;
  beta /= orgnrm;
  int k;
  la_sd2(nl, nr, sqre, &k, d, z, alpha, beta, U, ldu, VT, ldvt, dsigma, U2, n,
         VT2, m, idxp, idx, idxc, idxq, coltyp);
  nx_c_status st = la_sd3(nl, nr, sqre, k, d, Q, k, dsigma, U, ldu, U2, n, VT,
                         ldvt, VT2, m, idxc, coltyp, z, acc, gemm);
  if (st != NX_C_OK) return st;
  for (int i = 0; i < n; i++) d[i] *= orgnrm;
  la_dc_dlamrg(k, n - k, d, 1, -1, idxq);
  return NX_C_OK;
}

/* la_sd0 (dlasd0): conquer the n×(n+sqre) upper bidiagonal by direct recursion
   — tear at the middle row ic = n/2 (alpha = d[ic], beta = e[ic]); the left
   child is always nl×(nl+1) (sqre 1), the right child inherits sqre (so the
   rightmost chain carries the top-level shape, as dlasdt's tree does). Leaves
   go to la_sdq on the caller-seeded identity blocks. Direct recursion replaces
   the reference's explicit dlasdt tree walk; the split rule, leaf solver, and
   merge order are the same (post-order). */
static nx_c_status la_sd0(int n, int sqre, double *d, double *e, double *U,
                         int ldu, double *VT, int ldvt, int *idxq, double *work,
                         int *iwork, void *gemm) {
  if (n <= LA_SD_SMLSIZ) {
    if (la_sdq(n, sqre, d, e, U, ldu, VT, ldvt, work))
      return LA_ERR_NO_CONVERGE;
    for (int i = 0; i < n; i++) idxq[i] = i;
    return NX_C_OK;
  }
  int nl = n / 2, nr = n - nl - 1, ic = nl;
  nx_c_status st = la_sd0(nl, 1, d, e, U, ldu, VT, ldvt, idxq, work, iwork,
                         gemm);
  if (st != NX_C_OK) return st;
  st = la_sd0(nr, sqre, d + ic + 1, e + ic + 1,
              &U[(ic + 1) + (size_t)(ic + 1) * ldu], ldu,
              &VT[(ic + 1) + (size_t)(ic + 1) * ldvt], ldvt, idxq + ic + 1,
              work, iwork, gemm);
  if (st != NX_C_OK) return st;
  double alpha = d[ic], beta = e[ic];
  return la_sd1(nl, nr, sqre, d, alpha, beta, U, ldu, VT, ldvt, idxq, iwork,
                work, gemm);
}

/* la_sd_dc (dbdsdc, COMPQ='I', upper): SVD of the square n×n upper bidiagonal
   (d, e): B = U Σ VTᵀ with U/VT n×n column-major (seeded to identity here),
   d overwritten with the singular values, nonnegative DESCENDING with matched
   U columns / VT rows. Splits at negligible e (|e| < 0.9·eps after the
   max-norm scaling, with tiny d floored to ±eps as the reference does) into
   independent la_sd0 subproblems. work: 4n²+8n+8 doubles; iwork: 5n ints.
   LA_ERR_NO_CONVERGE on non-convergence; a failing GEMM propagates its own
   status. */
static nx_c_status la_sd_dc(int n, double *d, double *e, double *U, int ldu,
                           double *VT, int ldvt, double *work, int *iwork,
                           void *gemm) {
  if (n <= 0) return NX_C_OK;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      U[i + (size_t)j * ldu] = (i == j) ? 1.0 : 0.0;
      VT[i + (size_t)j * ldvt] = (i == j) ? 1.0 : 0.0;
    }
  }
  if (n <= LA_SD_SMLSIZ) {
    if (la_sdq(n, 0, d, e, U, ldu, VT, ldvt, work)) return LA_ERR_NO_CONVERGE;
  } else {
    double orgnrm = 0.0;
    for (int i = 0; i < n; i++)
      if (fabs(d[i]) > orgnrm) orgnrm = fabs(d[i]);
    for (int i = 0; i < n - 1; i++)
      if (fabs(e[i]) > orgnrm) orgnrm = fabs(e[i]);
    if (orgnrm == 0.0) return NX_C_OK;
    for (int i = 0; i < n; i++) d[i] /= orgnrm;
    for (int i = 0; i < n - 1; i++) e[i] /= orgnrm;
    const double eps = 0.9 * (0.5 * DBL_EPSILON); /* 0.9 * DLAMCH('E') */
    int *idxq = iwork;
    int *iwk = iwork + n;
    for (int i = 0; i < n; i++)
      if (fabs(d[i]) < eps) d[i] = copysign(eps, d[i]);
    int start = 0;
    for (int i = 0; i < n - 1; i++) {
      if (fabs(e[i]) < eps || i == n - 2) {
        int nsize;
        if (i < n - 2) {
          nsize = i - start + 1;
        } else if (fabs(e[i]) >= eps) {
          nsize = n - start;
        } else {
          /* e[n-2] negligible: 1×1 subproblem at row n-1, then [start, n-2] */
          nsize = i - start + 1;
          U[(n - 1) + (size_t)(n - 1) * ldu] = copysign(1.0, d[n - 1]);
          VT[(n - 1) + (size_t)(n - 1) * ldvt] = 1.0;
          d[n - 1] = fabs(d[n - 1]);
        }
        nx_c_status st = la_sd0(nsize, 0, d + start, e + start,
                               &U[start + (size_t)start * ldu], ldu,
                               &VT[start + (size_t)start * ldvt], ldvt,
                               idxq + start, work, iwk, gemm);
        if (st != NX_C_OK) return st;
        start = i + 1;
      }
    }
    for (int i = 0; i < n; i++) d[i] *= orgnrm;
  }
  /* descending selection sort, swapping U columns and VT rows along */
  for (int i = 0; i < n - 1; i++) {
    int kk = i;
    double p = d[i];
    for (int j = i + 1; j < n; j++)
      if (d[j] > p) {
        kk = j;
        p = d[j];
      }
    if (kk != i) {
      d[kk] = d[i];
      d[i] = p;
      for (int r = 0; r < n; r++) {
        double t = U[r + (size_t)i * ldu];
        U[r + (size_t)i * ldu] = U[r + (size_t)kk * ldu];
        U[r + (size_t)kk * ldu] = t;
      }
      for (int c = 0; c < n; c++) {
        double t = VT[i + (size_t)c * ldvt];
        VT[i + (size_t)c * ldvt] = VT[kk + (size_t)c * ldvt];
        VT[kk + (size_t)c * ldvt] = t;
      }
    }
  }
  return NX_C_OK;
}

/* Apply the D&C factors to the assembled transforms (the BLAS-3 replacement
   for both la_bdsvd's accumulation and la_formp): U1[:, :pc] <- U1[:, :pc]·U_s
   and V1 <- V_sᴴ·Q_Vᴴ. Q_Vᴴ is never formed by la_formp here — dorgbr('P')'s
   identity: Q_Vᴴ = [[1, 0], [0, Qpᴴ]] where Qp is the compact-WY QR form-Q of
   the stored right-reflector tails shifted up one column (P row i, cols
   i+2.. -> panel column i, rows i+1..; tau = taup, since the stored rows are
   already the conjugated apply-form vectors and form-Q's ∏(I - τ v vᴴ) is
   exactly (∏ G_iᴴ)ᴴ). U_s/V_sᴴ arrive double column-major from la_sd_dc and
   are lifted to the compute type in `lift` (element order preserved, so lift
   stays column-major). AR reuses the driver's unpack buffer wa (mp² <= m·n);
   Prod is the pr×pc product / conj-transpose temp (the driver's gPm). */
#define LA_GEN_SDDC(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)             \
  static nx_c_status la_sd_apply_##sfx(                                         \
      void *vU1, int64_t pr, int64_t ldu1, void *vV1, int64_t pc,              \
      const void *vP, int64_t ldp, void *vtaup, const double *Us,              \
      const double *VTs, void *vlift, void *vAR, void *vProd, void *qV,        \
      void *qVc, void *qT, void *qW, void *qP, void *gemm) {                   \
    T *U1 = (T *)vU1;                                                          \
    T *V1 = (T *)vV1;                                                          \
    const T *P = (const T *)vP;                                                \
    T *lift = (T *)vlift;                                                      \
    T *AR = (T *)vAR;                                                          \
    T *Prod = (T *)vProd;                                                      \
    nx_c_status st;                                                             \
    for (int64_t i = 0; i < pc * pc; i++)                                      \
      lift[i] = LA_MK_##sfx((R)Us[i], (R)0);                                   \
    st = nx_c_gemm2d_ct_ws(DT, pr, pc, pc, (const char *)U1, ldu1, 1,           \
                          (const char *)lift, 1, pc, (char *)Prod, pc, 1,      \
                          (char *)gemm);                                       \
    if (st != NX_C_OK) return st;                                               \
    for (int64_t r = 0; r < pr; r++)                                           \
      for (int64_t c = 0; c < pc; c++) U1[r * ldu1 + c] = Prod[r * pc + c];    \
    for (int64_t i = 0; i < pc * pc; i++)                                      \
      lift[i] = LA_MK_##sfx((R)VTs[i], (R)0);                                  \
    int64_t mp = pc - 1;                                                       \
    if (mp > 0) {                                                              \
      for (int64_t c = 0; c < mp; c++)                                         \
        for (int64_t r = c + 1; r < mp; r++)                                   \
          AR[r * mp + c] = P[c * ldp + (r + 1)];                               \
      nx_c_la_qrq_##sfx(AR, mp, mp, mp, vtaup, V1, mp, mp, qV, qVc, qT, qW, qP, \
                       gemm);                                                  \
      for (int64_t r = 0; r < mp; r++)                                         \
        for (int64_t c = 0; c < mp; c++) Prod[r * mp + c] = CONJ(V1[c * mp + r]); \
    }                                                                          \
    for (int64_t r = 0; r < pc; r++) V1[r * pc] = lift[r];                     \
    if (mp > 0) {                                                              \
      st = nx_c_gemm2d_ct_ws(DT, pc, mp, mp, (const char *)(lift + pc), 1, pc,  \
                            (const char *)Prod, mp, 1, (char *)(V1 + 1), pc, 1, \
                            (char *)gemm);                                     \
      if (st != NX_C_OK) return st;                                             \
    }                                                                          \
    return NX_C_OK;                                                             \
  }
#define LA_EXPAND_SDDC(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)          \
  LA_GEN_SDDC(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_SDDC)
LA_TRAITS_double(LA_EXPAND_SDDC)
LA_TRAITS_c32(LA_EXPAND_SDDC)
LA_TRAITS_c64(LA_EXPAND_SDDC)
#undef LA_EXPAND_SDDC

typedef nx_c_status (*la_sd_apply_fn)(void *, int64_t, int64_t, void *, int64_t,
                                     const void *, int64_t, void *,
                                     const double *, const double *, void *,
                                     void *, void *, void *, void *, void *,
                                     void *, void *, void *);
static const la_sd_apply_fn la_sd_apply[LA_NCOMPUTE] = {
    [LA_F32] = la_sd_apply_f32, [LA_F64] = la_sd_apply_f64,
    [LA_C32] = la_sd_apply_c32, [LA_C64] = la_sd_apply_c64};

static const la_compute_desc la_desc[LA_NCOMPUTE] = {
    [LA_F32] = {.csize = sizeof(float), .gemm_dt = NX_C_DTYPE_f32,
                .qrq = nx_c_la_qrq_f32, .gebrd = la_gebrd_f32,
                .formp = la_formp_f32, .bdsvd = la_bdsvd_f32,
                .cpc = la_cpc_f32, .cpct = la_cpct_f32},
    [LA_F64] = {.csize = sizeof(double), .gemm_dt = NX_C_DTYPE_f64,
                .qrq = nx_c_la_qrq_f64, .gebrd = la_gebrd_f64,
                .formp = la_formp_f64, .bdsvd = la_bdsvd_f64,
                .cpc = la_cpc_f64, .cpct = la_cpct_f64},
    [LA_C32] = {.csize = sizeof(nx_c_complex32), .gemm_dt = NX_C_DTYPE_c32,
                .qrq = nx_c_la_qrq_c32, .gebrd = la_gebrd_c32,
                .formp = la_formp_c32, .bdsvd = la_bdsvd_c32,
                .cpc = la_cpc_c32, .cpct = la_cpct_c32},
    [LA_C64] = {.csize = sizeof(nx_c_complex64), .gemm_dt = NX_C_DTYPE_c64,
                .qrq = nx_c_la_qrq_c64, .gebrd = la_gebrd_c64,
                .formp = la_formp_c64, .bdsvd = la_bdsvd_c64,
                .cpc = la_cpc_c64, .cpct = la_cpct_c64},
};

/* ── SVD driver: batched, pooled (Golub–Kahan + Demmel–Kahan) ─────────────
   in is [batch, m, n]. Outputs: U [batch, m, ncu], S [batch, k] float64
   descending, Vᴴ [batch, nrv, n] (ncu/nrv = m/n for full_matrices, else k). Each
   worker unpacks A, builds the tall/square working matrix P (A when m>=n, else
   Aᴴ), bidiagonalizes, runs the bidiagonal SVD, and assembles U/S/Vᴴ (swapping
   and conjugate-transposing U/V when it bidiagonalized Aᴴ). Non-convergence in
   the bidiagonal QR → LA_ERR_NO_CONVERGE. */
typedef struct {
  const nx_c_ndarray *in;
  const nx_c_ndarray *u;
  const nx_c_ndarray *s;
  const nx_c_ndarray *vt;
  nx_c_dtype dt;
  la_compute lc;
  int is_double;
  int trans;
  int64_t m, n, k, pr, pc, ncu_p, ncu, nrv, esz;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *in_bs;
  const int64_t *u_bs;
  const int64_t *s_bs;
  const int64_t *vt_bs;
  int64_t in_rs, in_cs, u_rs, u_cs, s_cs, vt_rs, vt_cs;
  char *scratch;
  int64_t stride, off_wa, off_P, off_U1, off_V1, off_Uo, off_Vo, off_tauq,
      off_taup, off_d, off_e, off_qV, off_qVc, off_qT, off_qW, off_qP, off_qg,
      off_gX, off_gY, off_gMc, off_gPm, off_gg;
  /* divide-and-conquer scratch (touched only when use_dc, i.e. pc > SMLSIZ):
     widened double d/e, the double column-major U_s/V_sᴴ factors, the dlasd
     workspace + iwork, the compute-typed lift buffer, and the GEMM scratch
     sized for the back-multiplies. */
  int use_dc;
  int64_t off_sdd, off_sde, off_sdu, off_sdvt, off_sdw, off_sdi, off_sdl,
      off_sdg;
  nx_c_status *werr;
} la_svd_ctx;

static void la_svd_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  la_svd_ctx *x = (la_svd_ctx *)vctx;
  const la_compute_desc *cd = &la_desc[x->lc];
  const la_move_desc *mv = &la_move[x->dt];
  char *base = x->scratch + (int64_t)worker * x->stride;
  void *wa = base + x->off_wa;
  void *P = base + x->off_P;
  void *U1 = base + x->off_U1;
  void *V1 = base + x->off_V1;
  void *Uo = base + x->off_Uo;
  void *Vo = base + x->off_Vo;
  void *tauq = base + x->off_tauq;
  void *taup = base + x->off_taup;
  void *d = base + x->off_d;
  void *e = base + x->off_e;
  void *qV = base + x->off_qV;
  void *qVc = base + x->off_qVc;
  void *qT = base + x->off_qT;
  void *qW = base + x->off_qW;
  void *qP = base + x->off_qP;
  void *qg = base + x->off_qg;
  void *gX = base + x->off_gX;
  void *gY = base + x->off_gY;
  void *gMc = base + x->off_gMc;
  void *gPm = base + x->off_gPm;
  void *gg = base + x->off_gg;
  int64_t m = x->m, n = x->n, k = x->k, pr = x->pr, pc = x->pc;
  int64_t ncu_p = x->ncu_p, ncu = x->ncu, nrv = x->nrv;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *inb;
    const char *sb;
    la_batch_base(bt, x->batch_nd, x->bshape, x->in_bs, x->in->offset, x->esz,
                  (const char *)x->in->data, &inb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->s_bs, x->s->offset,
                  (int64_t)sizeof(double), (const char *)x->s->data, &sb);
    mv->unpack(inb, x->in_rs, x->in_cs, m, n, wa, n);
    if (!x->trans)
      cd->cpc(wa, n, P, pc, m, n);
    else
      cd->cpct(wa, n, P, pc, n, m);
    cd->gebrd(P, pr, pc, pc, d, e, tauq, taup, gX, gY, gMc, gPm, gg);
    cd->qrq(P, pr, pc, pc, tauq, U1, ncu_p, ncu_p, qV, qVc, qT, qW, qP, qg);
    double *sd = (double *)sb;
    if (x->use_dc) {
      /* divide-and-conquer bidiagonal SVD (double core), then the BLAS-3
         back-multiply into U1/V1 — replaces la_formp + la_bdsvd. */
      double *dd = (double *)(base + x->off_sdd);
      double *de = (double *)(base + x->off_sde);
      double *Us = (double *)(base + x->off_sdu);
      double *VTs = (double *)(base + x->off_sdvt);
      double *sdw = (double *)(base + x->off_sdw);
      int *sdi = (int *)(base + x->off_sdi);
      void *sdl = base + x->off_sdl;
      void *sdg = base + x->off_sdg;
      if (x->is_double) {
        const double *dv = (const double *)d, *ev = (const double *)e;
        for (int64_t i = 0; i < pc; i++) dd[i] = dv[i];
        for (int64_t i = 0; i + 1 < pc; i++) de[i] = ev[i];
      } else {
        const float *dv = (const float *)d, *ev = (const float *)e;
        for (int64_t i = 0; i < pc; i++) dd[i] = (double)dv[i];
        for (int64_t i = 0; i + 1 < pc; i++) de[i] = (double)ev[i];
      }
      nx_c_status st =
          la_sd_dc((int)pc, dd, de, Us, (int)pc, VTs, (int)pc, sdw, sdi, sdg);
      if (st == NX_C_OK)
        st = la_sd_apply[x->lc](U1, pr, ncu_p, V1, pc, P, pc, taup, Us, VTs,
                                sdl, wa, gPm, qV, qVc, qT, qW, qP, sdg);
      if (st != NX_C_OK) {
        if (x->werr[worker] == NX_C_OK) x->werr[worker] = st;
        continue;
      }
      for (int64_t i = 0; i < k; i++) sd[i * x->s_cs] = dd[i];
    } else {
      cd->formp(P, pc, pc, taup, V1, pc);
      nx_c_status st = cd->bdsvd(d, e, U1, pr, ncu_p, V1, pc, pc);
      if (st != NX_C_OK) {
        if (x->werr[worker] == NX_C_OK) x->werr[worker] = st;
        continue;
      }
      if (x->is_double) {
        const double *dv = (const double *)d;
        for (int64_t i = 0; i < k; i++) sd[i * x->s_cs] = dv[i];
      } else {
        const float *dv = (const float *)d;
        for (int64_t i = 0; i < k; i++) sd[i * x->s_cs] = (double)dv[i];
      }
    }
    const char *ub;
    const char *vtb;
    la_batch_base(bt, x->batch_nd, x->bshape, x->u_bs, x->u->offset, x->esz,
                  (const char *)x->u->data, &ub);
    la_batch_base(bt, x->batch_nd, x->bshape, x->vt_bs, x->vt->offset, x->esz,
                  (const char *)x->vt->data, &vtb);
    if (!x->trans) {
      mv->packfull(U1, ncu_p, m, ncu, (char *)ub, x->u_rs, x->u_cs);
      mv->packfull(V1, pc, nrv, n, (char *)vtb, x->vt_rs, x->vt_cs);
    } else {
      cd->cpct(V1, pc, Uo, ncu, m, ncu);
      cd->cpct(U1, ncu_p, Vo, n, nrv, n);
      mv->packfull(Uo, ncu, m, ncu, (char *)ub, x->u_rs, x->u_cs);
      mv->packfull(Vo, n, nrv, n, (char *)vtb, x->vt_rs, x->vt_cs);
    }
  }
}

static nx_c_status nx_c_svd_run(const nx_c_ndarray *in, const nx_c_ndarray *u,
                              const nx_c_ndarray *s, const nx_c_ndarray *vt,
                              nx_c_dtype dt) {
  if (in->ndim < 2 || u->ndim != in->ndim || vt->ndim != in->ndim ||
      s->ndim != in->ndim - 1)
    return LA_ERR_SHAPE_LA;
  int64_t m = in->shape[in->ndim - 2];
  int64_t n = in->shape[in->ndim - 1];
  int64_t k = m < n ? m : n;
  int64_t ncu = u->shape[u->ndim - 1];
  int64_t nrv = vt->shape[vt->ndim - 2];
  if (u->shape[u->ndim - 2] != m || vt->shape[vt->ndim - 1] != n ||
      s->shape[s->ndim - 1] != k || (ncu != k && ncu != m) ||
      (nrv != k && nrv != n))
    return LA_ERR_SHAPE_LA;
  la_compute lc = la_compute_of(dt);
  if (lc == LA_NCOMPUTE) return LA_ERR_NOT_FLOAT;
  const la_compute_desc *cd = &la_desc[lc];
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = in->ndim - 2;
  int64_t bshape[NX_C_MAX_NDIM], in_bs[NX_C_MAX_NDIM], u_bs[NX_C_MAX_NDIM],
      s_bs[NX_C_MAX_NDIM], vt_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (u->shape[i] != in->shape[i] || vt->shape[i] != in->shape[i] ||
        s->shape[i] != in->shape[i])
      return LA_ERR_SHAPE_LA;
    bshape[i] = in->shape[i];
    in_bs[i] = in->strides[i];
    u_bs[i] = u->strides[i];
    s_bs[i] = s->strides[i];
    vt_bs[i] = vt->strides[i];
    nbatch *= bshape[i];
  }
  if (k == 0 || nbatch == 0) return NX_C_OK;

  int trans = m < n;
  int64_t pr = m >= n ? m : n;
  int64_t pc = k;
  int full = (ncu > k) || (nrv > k);
  int64_t ncu_p = full ? pr : pc;

  int64_t bytes = nbatch * pr * pc * esz;
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nbatch, pr * pc * pc, bytes);
  if (nth > nbatch) nth = (int)nbatch;
  if (nth > LA_MAX_WORKERS) nth = LA_MAX_WORKERS;
  if (nth < 1) nth = 1;

  int64_t rsize = (lc == LA_F64 || lc == LA_C64) ? 8 : 4;
#define LA_ALN(bytes_) (((bytes_) + 63) & ~(int64_t)63)
  int64_t a_wa = LA_ALN(m * n * cd->csize);
  int64_t a_P = LA_ALN(pr * pc * cd->csize);
  int64_t a_U1 = LA_ALN(pr * ncu_p * cd->csize);
  int64_t a_V1 = LA_ALN(pc * pc * cd->csize);
  int64_t a_Uo = LA_ALN(m * ncu * cd->csize);
  int64_t a_Vo = LA_ALN(nrv * n * cd->csize);
  int64_t a_tau = LA_ALN(pc * cd->csize);
  int64_t a_r = LA_ALN(pc * rsize);
  /* Compact-WY scratch for the blocked cd->qrq forming U (m=pr, cols=ncu_p). */
  int64_t a_qV = LA_ALN(pr * LA_QR_NB * cd->csize);
  int64_t a_qT = LA_ALN((int64_t)LA_QR_NB * LA_QR_NB * cd->csize);
  int64_t a_qW = LA_ALN((int64_t)LA_QR_NB * ncu_p * cd->csize);
  int64_t a_qP = LA_ALN(pr * ncu_p * cd->csize);
  int64_t qg1 = nx_c_gemm2d_ct_scratch(cd->gemm_dt, LA_QR_NB, ncu_p, pr);
  int64_t qg2 = nx_c_gemm2d_ct_scratch(cd->gemm_dt, pr, ncu_p, LA_QR_NB);
  int64_t a_qg = LA_ALN(qg1 > qg2 ? qg1 : qg2);
  /* Blocked cd->gebrd (dlabrd): X (pr×NB), Y (pc×NB), the conj materialization
     panel Mc (max(pr,pc)×NB), the trailing GEMM product Pm (pr×pc), and the GEMM
     panels. Unused below LA_SVD_NB. */
  int64_t a_gX = LA_ALN(pr * (int64_t)LA_SVD_NB * cd->csize);
  int64_t a_gY = LA_ALN(pc * (int64_t)LA_SVD_NB * cd->csize);
  int64_t a_gMc = LA_ALN((pr > pc ? pr : pc) * (int64_t)LA_SVD_NB * cd->csize);
  int64_t a_gPm = LA_ALN(pr * pc * cd->csize);
  int64_t a_gg = LA_ALN(nx_c_gemm2d_ct_scratch(cd->gemm_dt, pr, pc, LA_SVD_NB));
  /* D&C scratch (0 below the SMLSIZ crossover): widened double d/e, the double
     column-major U_s/V_sᴴ, the dlasd workspace (4pc²+8pc+8 doubles: z + dsigma
     + U2 + VT2 + Q + the beta=1 accumulation temp) + iwork (5pc: idxq + 4pc),
     the compute-typed lift buffer, and GEMM scratch covering the core's f64
     merges, the two compute-typed back-multiplies, and la_sd_apply's qrq
     internals (its AR panel and product temp reuse wa and gPm). */
  int use_dc = pc > LA_SD_SMLSIZ;
  int64_t a_sdd = use_dc ? LA_ALN(pc * (int64_t)sizeof(double)) : 0;
  int64_t a_sdu = use_dc ? LA_ALN(pc * pc * (int64_t)sizeof(double)) : 0;
  int64_t a_sdw =
      use_dc ? LA_ALN((4 * pc * pc + 8 * pc + 8) * (int64_t)sizeof(double)) : 0;
  int64_t a_sdi = use_dc ? LA_ALN(5 * pc * (int64_t)sizeof(int)) : 0;
  int64_t a_sdl = use_dc ? LA_ALN(pc * pc * cd->csize) : 0;
  int64_t a_sdg = 0;
  if (use_dc) {
    int64_t sg[5];
    sg[0] = nx_c_gemm2d_ct_scratch(NX_C_DTYPE_f64, pc, pc, pc);
    sg[1] = nx_c_gemm2d_ct_scratch(cd->gemm_dt, pr, pc, pc);
    sg[2] = nx_c_gemm2d_ct_scratch(cd->gemm_dt, pc, pc - 1, pc - 1);
    sg[3] = nx_c_gemm2d_ct_scratch(cd->gemm_dt, LA_QR_NB, pc - 1, pc - 1);
    sg[4] = nx_c_gemm2d_ct_scratch(cd->gemm_dt, pc - 1, pc - 1, LA_QR_NB);
    for (int i = 0; i < 5; i++)
      if (sg[i] > a_sdg) a_sdg = sg[i];
    a_sdg = LA_ALN(a_sdg);
  }
  int64_t off_wa = 0, off_P = off_wa + a_wa, off_U1 = off_P + a_P;
  int64_t off_V1 = off_U1 + a_U1, off_Uo = off_V1 + a_V1, off_Vo = off_Uo + a_Uo;
  int64_t off_tauq = off_Vo + a_Vo, off_taup = off_tauq + a_tau;
  int64_t off_d = off_taup + a_tau, off_e = off_d + a_r;
  int64_t off_qV = off_e + a_r, off_qVc = off_qV + a_qV, off_qT = off_qVc + a_qV;
  int64_t off_qW = off_qT + a_qT, off_qP = off_qW + a_qW, off_qg = off_qP + a_qP;
  int64_t off_gX = off_qg + a_qg, off_gY = off_gX + a_gX,
          off_gMc = off_gY + a_gY;
  int64_t off_gPm = off_gMc + a_gMc, off_gg = off_gPm + a_gPm;
  int64_t off_sdd = off_gg + a_gg, off_sde = off_sdd + a_sdd,
          off_sdu = off_sde + a_sdd, off_sdvt = off_sdu + a_sdu;
  int64_t off_sdw = off_sdvt + a_sdu, off_sdi = off_sdw + a_sdw,
          off_sdl = off_sdi + a_sdi, off_sdg = off_sdl + a_sdl;
  int64_t stride = off_sdg + a_sdg;
#undef LA_ALN
  char *scratch = aligned_alloc(64, (size_t)stride * nth);
  if (!scratch) return NX_C_ERR_ALLOC;

  nx_c_status werr[LA_MAX_WORKERS];
  for (int i = 0; i < nth; i++) werr[i] = NX_C_OK;

  la_svd_ctx x;
  x.in = in;
  x.u = u;
  x.s = s;
  x.vt = vt;
  x.dt = dt;
  x.lc = lc;
  x.is_double = (lc == LA_F64 || lc == LA_C64);
  x.trans = trans;
  x.m = m;
  x.n = n;
  x.k = k;
  x.pr = pr;
  x.pc = pc;
  x.ncu_p = ncu_p;
  x.ncu = ncu;
  x.nrv = nrv;
  x.esz = esz;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.in_bs = in_bs;
  x.u_bs = u_bs;
  x.s_bs = s_bs;
  x.vt_bs = vt_bs;
  x.in_rs = in->strides[in->ndim - 2];
  x.in_cs = in->strides[in->ndim - 1];
  x.u_rs = u->strides[u->ndim - 2];
  x.u_cs = u->strides[u->ndim - 1];
  x.s_cs = s->strides[s->ndim - 1];
  x.vt_rs = vt->strides[vt->ndim - 2];
  x.vt_cs = vt->strides[vt->ndim - 1];
  x.scratch = scratch;
  x.stride = stride;
  x.off_wa = off_wa;
  x.off_P = off_P;
  x.off_U1 = off_U1;
  x.off_V1 = off_V1;
  x.off_Uo = off_Uo;
  x.off_Vo = off_Vo;
  x.off_tauq = off_tauq;
  x.off_taup = off_taup;
  x.off_d = off_d;
  x.off_e = off_e;
  x.off_qV = off_qV;
  x.off_qVc = off_qVc;
  x.off_qT = off_qT;
  x.off_qW = off_qW;
  x.off_qP = off_qP;
  x.off_qg = off_qg;
  x.off_gX = off_gX;
  x.off_gY = off_gY;
  x.off_gMc = off_gMc;
  x.off_gPm = off_gPm;
  x.off_gg = off_gg;
  x.use_dc = use_dc;
  x.off_sdd = off_sdd;
  x.off_sde = off_sde;
  x.off_sdu = off_sdu;
  x.off_sdvt = off_sdvt;
  x.off_sdw = off_sdw;
  x.off_sdi = off_sdi;
  x.off_sdl = off_sdl;
  x.off_sdg = off_sdg;
  x.werr = werr;

  nx_c_parallel_for(nth, nbatch, bytes, la_svd_body, &x, scratch);

  nx_c_status err = NX_C_OK;
  for (int i = 0; i < nth; i++)
    if (werr[i] != NX_C_OK) {
      err = werr[i];
      break;
    }
  return err;
}

/* U, S (float64), Vᴴ are allocated by the binding to the thin/full shapes from
   full_matrices; the flag is encoded in those shapes so it needs no separate
   argument here. */
CAMLprim value caml_nx_c_svd(value vu, value vs, value vvt, value vin) {
  CAMLparam4(vu, vs, vvt, vin);
  nx_c_ndarray in, u, s, vt;
  nx_c_status st = nx_c_ndarray_of_value(vin, &in);
  if (st == NX_C_OK) st = nx_c_ndarray_of_value(vu, &u);
  if (st == NX_C_OK) st = nx_c_ndarray_of_value(vs, &s);
  if (st == NX_C_OK) st = nx_c_ndarray_of_value(vvt, &vt);
  if (st != NX_C_OK) la_raise("svd", st);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt == NX_C_DTYPE_COUNT) la_raise("svd", NX_C_ERR_BAD_KIND);
  st = nx_c_svd_run(&in, &u, &s, &vt, dt);
  if (st != NX_C_OK) la_raise("svd", st);
  CAMLreturn(Val_unit);
}
