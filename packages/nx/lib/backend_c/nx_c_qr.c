/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_qr.c — Householder QR (compact-WY blocked), linalg tier 1. The panel,
   block-T, V-builder, and form-Q are exported (nx_c_la_* in nx_c_linalg.h) for
   SVD's U formation and the D&C eigensolvers. Shared machinery in nx_c_linalg.h. */

#include "nx_c_linalg.h"

/* Column-chunk width for the unblocked reflector apply (panel and form-Q). The
   apply is reordered so its inner loop runs stride-1 over columns of the
   row-major buffer (clang then vectorizes it); the per-column accumulators w[]
   live on the stack, chunked to this width so an arbitrarily wide col_lim/nq
   still needs no heap. Each column sums in the original row order, so the
   reorder is bit-exact. */
#define LA_QR_CB 64

/* ── Householder QR: A = Q·R, compact-WY blocked ──────────────────────────

   Column j's reflector H_j = I - tau_j v_j v_jᴴ (v_j[j]=1, tail stored below the
   diagonal, tau in `tau`) zeros the sub-diagonal; the upper trapezoid becomes R.
   The pivot sign follows LAPACK (beta = -sign(re(alpha))·‖x‖), so R's diagonal is
   real for any reflected column. A column already zero below the diagonal takes no
   reflector (tau=0, R[j][j]=alpha).

   la_qr_panel is the unblocked kernel: reflectors for columns [j0,j0+jb), each
   applied to columns [., col_lim). la_qr (the driver-facing factor) runs it whole
   below LA_QR_NB (col_lim=n — the exact pre-blocking kernel, so small-matrix
   numerics are unchanged), and above it factors LA_QR_NB-wide panels whose
   trailing update is a compact-WY block reflector: la_larft forms the jb×jb T,
   la_buildv materializes the panel V (unit diagonal explicit, so the GEMM needs
   no implicit-diagonal handling and no conjugation — its conjugate is materialized
   too), and the trailing C := (I - V Tᴴ Vᴴ) C is two GEMMs (W = Vᴴ C, then
   C -= V (Tᴴ W)) through nx_c_gemm2d_ct_ws. la_qrq forms Q the same way: block
   reflectors (I - V T Vᴴ, T not Tᴴ) applied to the identity in reverse panel
   order, matching the unblocked la_qrq_unb's H_j. Real types drop the conjugation
   (CONJ, the imaginary text), so one body serves real and complex. V, Vc, T, W, P,
   and the GEMM panels are caller per-worker scratch. */
#define LA_GEN_QR(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)               \
  void nx_c_la_qr_panel_##sfx(void *vA, int64_t m, int64_t lda, int64_t j0,  \
                                int64_t jb, void *vtau, int64_t col_lim) {     \
    T *A = (T *)vA;                                                           \
    T *tau = (T *)vtau;                                                       \
    for (int64_t c = 0; c < jb; c++) {                                        \
      int64_t j = j0 + c;                                                     \
      R xnorm2 = (R)0;                                                        \
      for (int64_t i = j + 1; i < m; i++) xnorm2 += NORM2(A[i * lda + j]);    \
      T alpha = A[j * lda + j];                                               \
      R alphr = REAL(alpha);                                                  \
      if (xnorm2 == (R)0) {                                                   \
        tau[j] = (T)0;                                                        \
        continue;                                                            \
      }                                                                       \
      R anorm = SQRT(NORM2(alpha) + xnorm2);                                  \
      R beta = alphr >= (R)0 ? -anorm : anorm;                               \
      tau[j] = LA_MK_##sfx((beta - alphr) / beta, -LA_IMAG_##sfx(alpha) / beta); \
      T scal = alpha - LA_MK_##sfx(beta, (R)0);                              \
      for (int64_t i = j + 1; i < m; i++) A[i * lda + j] = A[i * lda + j] / scal; \
      A[j * lda + j] = LA_MK_##sfx(beta, (R)0);                              \
      /* factor applies H_jᴴ (conj tau) so A = Q·R with Q = ∏ H_j; form_q       \
         below uses H_j. For real, conj is the identity. Columns are chunked    \
         with the inner loop stride-1 over columns (row-major) to vectorize. */ \
      T tauc = CONJ(tau[j]);                                                  \
      for (int64_t cc0 = j + 1; cc0 < col_lim; cc0 += LA_QR_CB) {            \
        int64_t ccN = cc0 + LA_QR_CB < col_lim ? cc0 + LA_QR_CB : col_lim;   \
        int64_t cw = ccN - cc0;                                              \
        T w[LA_QR_CB];                                                        \
        for (int64_t q = 0; q < cw; q++) w[q] = A[j * lda + cc0 + q];        \
        for (int64_t i = j + 1; i < m; i++) {                               \
          T vi = CONJ(A[i * lda + j]);                                       \
          const T *Ai = &A[i * lda + cc0];                                   \
          for (int64_t q = 0; q < cw; q++) w[q] += vi * Ai[q];              \
        }                                                                    \
        for (int64_t q = 0; q < cw; q++) {                                  \
          w[q] = tauc * w[q];                                                \
          A[j * lda + cc0 + q] -= w[q];                                      \
        }                                                                    \
        for (int64_t i = j + 1; i < m; i++) {                              \
          T vi = A[i * lda + j];                                             \
          T *Ai = &A[i * lda + cc0];                                         \
          for (int64_t q = 0; q < cw; q++) Ai[q] -= w[q] * vi;             \
        }                                                                    \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  void nx_c_la_larft_##sfx(const void *vA, int64_t m, int64_t lda,          \
                             int64_t j0, int64_t jb, const void *vtau,        \
                             void *vT, int64_t ldt) {                         \
    const T *A = (const T *)vA;                                              \
    const T *tau = (const T *)vtau;                                          \
    T *Tm = (T *)vT;                                                         \
    for (int64_t c = 0; c < jb; c++) {                                       \
      T tc = tau[j0 + c];                                                    \
      if (tc == (T)0) {                                                      \
        for (int64_t i = 0; i <= c; i++) Tm[i * ldt + c] = (T)0;             \
        continue;                                                           \
      }                                                                      \
      for (int64_t i = 0; i < c; i++) {                                      \
        T s = (T)0;                                                          \
        for (int64_t r = c; r < m - j0; r++) {                              \
          T vri = A[(j0 + r) * lda + (j0 + i)];                             \
          T vrc = (r == c) ? (T)1 : A[(j0 + r) * lda + (j0 + c)];           \
          s += CONJ(vri) * vrc;                                             \
        }                                                                    \
        Tm[i * ldt + c] = -tc * s;                                          \
      }                                                                      \
      for (int64_t i = 0; i < c; i++) {                                      \
        T acc = (T)0;                                                        \
        for (int64_t l = i; l < c; l++)                                     \
          acc += Tm[i * ldt + l] * Tm[l * ldt + c];                         \
        Tm[i * ldt + c] = acc;                                              \
      }                                                                      \
      Tm[c * ldt + c] = tc;                                                  \
    }                                                                        \
  }                                                                          \
  void nx_c_la_buildv_##sfx(const void *vA, int64_t m, int64_t lda,        \
                              int64_t j0, int64_t jb, void *vV, void *vVc) {  \
    const T *A = (const T *)vA;                                             \
    T *V = (T *)vV;                                                         \
    T *Vc = (T *)vVc;                                                       \
    int64_t mp = m - j0;                                                    \
    for (int64_t r = 0; r < mp; r++)                                        \
      for (int64_t l = 0; l < jb; l++) {                                    \
        T v = (r < l) ? (T)0                                                \
                      : (r == l ? (T)1 : A[(j0 + r) * lda + (j0 + l)]);     \
        V[r * jb + l] = v;                                                  \
        Vc[r * jb + l] = CONJ(v);                                          \
      }                                                                      \
  }                                                                          \
  static void la_qrq_unb_##sfx(void *vA, int64_t m, int64_t n, int64_t lda,   \
                               void *vtau, void *vQ, int64_t nq,             \
                               int64_t ldq) {                                \
    T *A = (T *)vA;                                                         \
    T *tau = (T *)vtau;                                                     \
    T *Q = (T *)vQ;                                                         \
    int64_t k = m < n ? m : n;                                             \
    for (int64_t i = 0; i < m; i++)                                        \
      for (int64_t c = 0; c < nq; c++) Q[i * ldq + c] = i == c ? (T)1 : (T)0; \
    for (int64_t jj = 0; jj < k; jj++) {                                   \
      int64_t j = k - 1 - jj;                                              \
      for (int64_t cc0 = 0; cc0 < nq; cc0 += LA_QR_CB) {                  \
        int64_t ccN = cc0 + LA_QR_CB < nq ? cc0 + LA_QR_CB : nq;          \
        int64_t cw = ccN - cc0;                                           \
        T w[LA_QR_CB];                                                     \
        for (int64_t q = 0; q < cw; q++) w[q] = Q[j * ldq + cc0 + q];     \
        for (int64_t i = j + 1; i < m; i++) {                            \
          T vi = CONJ(A[i * lda + j]);                                    \
          const T *Qi = &Q[i * ldq + cc0];                                \
          for (int64_t q = 0; q < cw; q++) w[q] += vi * Qi[q];           \
        }                                                                 \
        for (int64_t q = 0; q < cw; q++) {                               \
          w[q] = tau[j] * w[q];                                           \
          Q[j * ldq + cc0 + q] -= w[q];                                   \
        }                                                                 \
        for (int64_t i = j + 1; i < m; i++) {                           \
          T vi = A[i * lda + j];                                          \
          T *Qi = &Q[i * ldq + cc0];                                      \
          for (int64_t q = 0; q < cw; q++) Qi[q] -= w[q] * vi;          \
        }                                                                 \
      }                                                                    \
    }                                                                      \
  }                                                                         \
  static void la_qr_##sfx(void *vA, int64_t m, int64_t n, int64_t lda,       \
                          void *vtau, void *vV, void *vVc, void *vT,        \
                          void *vW, void *vP, void *vg) {                   \
    T *A = (T *)vA;                                                         \
    int64_t k = m < n ? m : n;                                             \
    if (k <= LA_QR_UNB) {                                                    \
      nx_c_la_qr_panel_##sfx(vA, m, lda, 0, k, vtau, n);                        \
      return;                                                              \
    }                                                                       \
    T *Vm = (T *)vV;                                                        \
    T *Vc = (T *)vVc;                                                       \
    T *Tm = (T *)vT;                                                        \
    T *W = (T *)vW;                                                         \
    T *P = (T *)vP;                                                         \
    for (int64_t j0 = 0; j0 < k; j0 += LA_QR_NB) {                          \
      int64_t jb = k - j0 < LA_QR_NB ? k - j0 : LA_QR_NB;                   \
      nx_c_la_qr_panel_##sfx(vA, m, lda, j0, jb, vtau, j0 + jb);                 \
      int64_t nt = n - (j0 + jb);                                           \
      if (nt <= 0) continue;                                                \
      int64_t mp = m - j0;                                                  \
      nx_c_la_buildv_##sfx(vA, m, lda, j0, jb, Vm, Vc);                         \
      nx_c_la_larft_##sfx(vA, m, lda, j0, jb, vtau, Tm, jb);                    \
      nx_c_gemm2d_ct_ws(DT, jb, nt, mp, (const char *)Vc, 1, jb,           \
                       (const char *)&A[j0 * lda + (j0 + jb)], lda, 1,     \
                       (char *)W, nt, 1, (char *)vg);                      \
      for (int64_t a = jb - 1; a >= 0; a--)                                \
        for (int64_t q = 0; q < nt; q++) {                                 \
          T acc = (T)0;                                                    \
          for (int64_t l = 0; l <= a; l++)                                \
            acc += CONJ(Tm[l * jb + a]) * W[l * nt + q];                   \
          W[a * nt + q] = acc;                                             \
        }                                                                   \
      nx_c_gemm2d_ct_ws(DT, mp, nt, jb, (const char *)Vm, jb, 1,           \
                       (const char *)W, nt, 1, (char *)P, nt, 1,           \
                       (char *)vg);                                        \
      for (int64_t r = 0; r < mp; r++)                                     \
        for (int64_t q = 0; q < nt; q++)                                   \
          A[(j0 + r) * lda + (j0 + jb + q)] -= P[r * nt + q];              \
    }                                                                       \
  }                                                                         \
  void nx_c_la_qrq_##sfx(void *vA, int64_t m, int64_t n, int64_t lda,      \
                           void *vtau, void *vQ, int64_t nq, int64_t ldq,    \
                           void *vV, void *vVc, void *vT, void *vW,         \
                           void *vP, void *vg) {                            \
    T *Q = (T *)vQ;                                                         \
    int64_t k = m < n ? m : n;                                             \
    if (k <= LA_QR_UNB) {                                                    \
      la_qrq_unb_##sfx(vA, m, n, lda, vtau, vQ, nq, ldq);                  \
      return;                                                              \
    }                                                                       \
    T *Vm = (T *)vV;                                                        \
    T *Vc = (T *)vVc;                                                       \
    T *Tm = (T *)vT;                                                        \
    T *W = (T *)vW;                                                         \
    T *P = (T *)vP;                                                         \
    for (int64_t i = 0; i < m; i++)                                        \
      for (int64_t c = 0; c < nq; c++) Q[i * ldq + c] = i == c ? (T)1 : (T)0; \
    int64_t nblk = (k + LA_QR_NB - 1) / LA_QR_NB;                          \
    for (int64_t bi = nblk - 1; bi >= 0; bi--) {                           \
      int64_t j0 = bi * LA_QR_NB;                                          \
      int64_t jb = k - j0 < LA_QR_NB ? k - j0 : LA_QR_NB;                  \
      int64_t mp = m - j0;                                                 \
      nx_c_la_buildv_##sfx(vA, m, lda, j0, jb, Vm, Vc);                        \
      nx_c_la_larft_##sfx(vA, m, lda, j0, jb, vtau, Tm, jb);                   \
      nx_c_gemm2d_ct_ws(DT, jb, nq, mp, (const char *)Vc, 1, jb,          \
                       (const char *)&Q[j0 * ldq], ldq, 1, (char *)W, nq, \
                       1, (char *)vg);                                     \
      for (int64_t a = 0; a < jb; a++)                                     \
        for (int64_t q = 0; q < nq; q++) {                                \
          T acc = (T)0;                                                   \
          for (int64_t l = a; l < jb; l++)                               \
            acc += Tm[a * jb + l] * W[l * nq + q];                        \
          W[a * nq + q] = acc;                                            \
        }                                                                  \
      nx_c_gemm2d_ct_ws(DT, mp, nq, jb, (const char *)Vm, jb, 1,          \
                       (const char *)W, nq, 1, (char *)P, nq, 1,          \
                       (char *)vg);                                       \
      for (int64_t r = 0; r < mp; r++)                                    \
        for (int64_t q = 0; q < nq; q++)                                  \
          Q[(j0 + r) * ldq + q] -= P[r * nq + q];                        \
    }                                                                      \
  }
#define LA_EXPAND_QR(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)            \
  LA_GEN_QR(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_QR)
LA_TRAITS_double(LA_EXPAND_QR)
LA_TRAITS_c32(LA_EXPAND_QR)
LA_TRAITS_c64(LA_EXPAND_QR)
#undef LA_EXPAND_QR

static const la_compute_desc la_desc[LA_NCOMPUTE] = {
    [LA_F32] = {.csize = sizeof(float), .gemm_dt = NX_C_DTYPE_f32,
                .qr = la_qr_f32, .qrq = nx_c_la_qrq_f32},
    [LA_F64] = {.csize = sizeof(double), .gemm_dt = NX_C_DTYPE_f64,
                .qr = la_qr_f64, .qrq = nx_c_la_qrq_f64},
    [LA_C32] = {.csize = sizeof(nx_c_complex32), .gemm_dt = NX_C_DTYPE_c32,
                .qr = la_qr_c32, .qrq = nx_c_la_qrq_c32},
    [LA_C64] = {.csize = sizeof(nx_c_complex64), .gemm_dt = NX_C_DTYPE_c64,
                .qr = la_qr_c64, .qrq = nx_c_la_qrq_c64},
};

/* ── QR driver: batched, pooled ──────────────────────────────────────────
   in is [batch, m, n]. reduced ⇒ Q is [batch, m, k], R is [batch, k, n]
   (k = min(m,n)); full ⇒ Q is [batch, m, m], R is [batch, m, n]. Each worker
   unpacks A, factors (Householder), forms Q, packs R (upper trapezoid) and Q.
   QR is unconditionally defined, so there is no failure status here. */
typedef struct {
  const nx_c_ndarray *in;
  const nx_c_ndarray *q;
  const nx_c_ndarray *r;
  nx_c_dtype dt;
  la_compute lc;
  int64_t m, n, k, nq, esz;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *in_bs;
  const int64_t *q_bs;
  const int64_t *r_bs;
  int64_t in_rs, in_cs, q_rs, q_cs, r_rs, r_cs;
  char *scratch; /* nthreads * stride; blocked-QR WY buffers + GEMM panels */
  int64_t stride, off_q, off_tau, off_V, off_Vc, off_T, off_W, off_P, off_g;
} la_qr_ctx;

static void la_qr_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  la_qr_ctx *x = (la_qr_ctx *)vctx;
  const la_compute_desc *cd = &la_desc[x->lc];
  const la_move_desc *mv = &la_move[x->dt];
  char *base = x->scratch + (int64_t)worker * x->stride;
  void *work = base;
  void *qbuf = base + x->off_q;
  void *tau = base + x->off_tau;
  void *V = base + x->off_V;
  void *Vc = base + x->off_Vc;
  void *T = base + x->off_T;
  void *W = base + x->off_W;
  void *P = base + x->off_P;
  void *g = base + x->off_g;
  int64_t m = x->m, n = x->n, nq = x->nq;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *inb;
    const char *qb;
    const char *rb;
    la_batch_base(bt, x->batch_nd, x->bshape, x->in_bs, x->in->offset, x->esz,
                  (const char *)x->in->data, &inb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->q_bs, x->q->offset, x->esz,
                  (const char *)x->q->data, &qb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->r_bs, x->r->offset, x->esz,
                  (const char *)x->r->data, &rb);
    mv->unpack(inb, x->in_rs, x->in_cs, m, n, work, n);
    cd->qr(work, m, n, n, tau, V, Vc, T, W, P, g);
    cd->qrq(work, m, n, n, tau, qbuf, nq, nq, V, Vc, T, W, P, g);
    mv->packR(work, n, nq, n, (char *)rb, x->r_rs, x->r_cs);
    mv->packfull(qbuf, nq, m, nq, (char *)qb, x->q_rs, x->q_cs);
  }
}

static nx_c_status nx_c_qr_run(const nx_c_ndarray *in, const nx_c_ndarray *q,
                             const nx_c_ndarray *r, nx_c_dtype dt, int reduced) {
  if (in->ndim < 2 || q->ndim != in->ndim || r->ndim != in->ndim)
    return LA_ERR_SHAPE_LA;
  int64_t m = in->shape[in->ndim - 2];
  int64_t n = in->shape[in->ndim - 1];
  int64_t k = m < n ? m : n;
  int64_t nq = reduced ? k : m; /* Q columns == R rows */
  if (q->shape[q->ndim - 2] != m || q->shape[q->ndim - 1] != nq)
    return LA_ERR_SHAPE_LA;
  if (r->shape[r->ndim - 2] != nq || r->shape[r->ndim - 1] != n)
    return LA_ERR_SHAPE_LA;
  la_compute lc = la_compute_of(dt);
  if (lc == LA_NCOMPUTE) return LA_ERR_NOT_FLOAT;
  const la_compute_desc *cd = &la_desc[lc];
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = in->ndim - 2;
  int64_t bshape[NX_C_MAX_NDIM], in_bs[NX_C_MAX_NDIM], q_bs[NX_C_MAX_NDIM],
      r_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (q->shape[i] != in->shape[i] || r->shape[i] != in->shape[i])
      return LA_ERR_SHAPE_LA;
    bshape[i] = in->shape[i];
    in_bs[i] = in->strides[i];
    q_bs[i] = q->strides[i];
    r_bs[i] = r->strides[i];
    nbatch *= bshape[i];
  }
  if (m == 0 || n == 0 || nbatch == 0) return NX_C_OK;

  int64_t bytes = nbatch * ((m * n) + (m * nq)) * esz;
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nbatch, m * n * k, bytes);
  if (nth > nbatch) nth = (int)nbatch;
  if (nth < 1) nth = 1;

  /* Per-worker: working A (m×n), Q accumulator (m×nq), tau (k), then the
     compact-WY buffers (panel V and its conjugate m×NB, block factor T NB×NB, the
     Vᴴ·C / Vᴴ·Q product W NB×maxcol, the V·W product P m×maxcol) and the GEMM
     panels. The WY/GEMM slots are unused below the LA_QR_NB crossover but always
     allocated (sized for the max of the factor and Q-formation GEMMs). */
  int64_t maxcol = n > nq ? n : nq;
#define LA_ALN(b) (((b) + 63) & ~(int64_t)63)
  int64_t a_work = LA_ALN(m * n * cd->csize);
  int64_t a_q = LA_ALN(m * nq * cd->csize);
  int64_t a_tau = LA_ALN(k * cd->csize);
  int64_t a_V = LA_ALN(m * LA_QR_NB * cd->csize);
  int64_t a_T = LA_ALN((int64_t)LA_QR_NB * LA_QR_NB * cd->csize);
  int64_t a_W = LA_ALN((int64_t)LA_QR_NB * maxcol * cd->csize);
  int64_t a_P = LA_ALN(m * maxcol * cd->csize);
  int64_t g1 = nx_c_gemm2d_ct_scratch(cd->gemm_dt, LA_QR_NB, maxcol, m);
  int64_t g2 = nx_c_gemm2d_ct_scratch(cd->gemm_dt, m, maxcol, LA_QR_NB);
  int64_t a_g = LA_ALN(g1 > g2 ? g1 : g2);
  int64_t off_q = a_work, off_tau = off_q + a_q, off_V = off_tau + a_tau;
  int64_t off_Vc = off_V + a_V, off_T = off_Vc + a_V, off_W = off_T + a_T;
  int64_t off_P = off_W + a_W, off_g = off_P + a_P;
  int64_t stride = off_g + a_g;
#undef LA_ALN
  char *scratch = aligned_alloc(64, (size_t)stride * nth);
  if (!scratch) return NX_C_ERR_ALLOC;

  la_qr_ctx x;
  x.in = in;
  x.q = q;
  x.r = r;
  x.dt = dt;
  x.lc = lc;
  x.m = m;
  x.n = n;
  x.k = k;
  x.nq = nq;
  x.esz = esz;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.in_bs = in_bs;
  x.q_bs = q_bs;
  x.r_bs = r_bs;
  x.in_rs = in->strides[in->ndim - 2];
  x.in_cs = in->strides[in->ndim - 1];
  x.q_rs = q->strides[q->ndim - 2];
  x.q_cs = q->strides[q->ndim - 1];
  x.r_rs = r->strides[r->ndim - 2];
  x.r_cs = r->strides[r->ndim - 1];
  x.scratch = scratch;
  x.stride = stride;
  x.off_q = off_q;
  x.off_tau = off_tau;
  x.off_V = off_V;
  x.off_Vc = off_Vc;
  x.off_T = off_T;
  x.off_W = off_W;
  x.off_P = off_P;
  x.off_g = off_g;

  nx_c_parallel_for(nth, nbatch, bytes, la_qr_body, &x, scratch);
  return NX_C_OK;
}

/* Q and R are allocated by the binding to the reduced/full shapes. */
CAMLprim value caml_nx_c_qr(value vq, value vr, value vin, value vreduced) {
  CAMLparam4(vq, vr, vin, vreduced);
  nx_c_ndarray in, q, r;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vq, &q);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vr, &r);
  if (s != NX_C_OK) la_raise("qr", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt == NX_C_DTYPE_COUNT) la_raise("qr", NX_C_ERR_BAD_KIND);
  s = nx_c_qr_run(&in, &q, &r, dt, Bool_val(vreduced));
  if (s != NX_C_OK) la_raise("qr", s);
  CAMLreturn(Val_unit);
}
