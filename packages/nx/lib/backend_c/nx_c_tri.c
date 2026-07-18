/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_tri.c — Cholesky factorization and triangular solve (linalg tier 1).
   Shared machinery in nx_c_linalg.h. */

#include "nx_c_linalg.h"

/* ── Cholesky (lower): A = L·Lᴴ, right-looking, blocked ───────────────────

   In-place on a contiguous row-major n×n compute buffer (leading dim lda). On
   success the lower triangle holds L; the strict upper triangle is scratch (the
   pack-out zeroes it, or mirrors for ~upper). Non-PD (a non-positive pivot,
   NaN/inf included via the !(d>0) test) returns LA_ERR_NOT_PD.

   Per block column j (width jb): the trailing update from earlier panels has
   already been applied (right-looking), so
     1. factor the jb×jb diagonal block unblocked (Cholesky within the block);
     2. solve the panel below: L21 = A21·(L11ᴴ)^{-1} (forward substitution by
        column of L11, unblocked — jb is small);
     3. trailing update A22 -= L21·L21ᴴ through the GEMM.
   bscratch holds the (n-j-jb)×jb conj copy of L21, pscratch the GEMM product,
   and gscratch the GEMM's own packing panels (nx_c_gemm2d_ct_ws) — all three the
   caller's per-worker buffers, so the trailing update allocates nothing. */
#define LA_GEN_CHOL(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)             \
  static nx_c_status la_chol_##sfx(void *vA, int64_t n, int64_t lda, void *vb,  \
                                  void *vp, void *vg) {                        \
    T *A = (T *)vA;                                                           \
    T *bscratch = (T *)vb;                                                    \
    T *pscratch = (T *)vp;                                                    \
    for (int64_t j = 0; j < n; j += LA_NB) {                                   \
      int64_t jb = n - j < LA_NB ? n - j : LA_NB;                              \
      /* 1. diagonal block */                                                  \
      for (int64_t c = 0; c < jb; c++) {                                       \
        int64_t jj = j + c;                                                    \
        R d = REAL(A[jj * lda + jj]);                                          \
        for (int64_t kk = j; kk < jj; kk++) d -= NORM2(A[jj * lda + kk]);      \
        if (!(d > (R)0)) return LA_ERR_NOT_PD;                                 \
        R ljj = SQRT(d);                                                       \
        A[jj * lda + jj] = FROMR(ljj);                                         \
        for (int64_t ii = jj + 1; ii < j + jb; ii++) {                        \
          T s = A[ii * lda + jj];                                              \
          for (int64_t kk = j; kk < jj; kk++)                                  \
            s -= A[ii * lda + kk] * CONJ(A[jj * lda + kk]);                    \
          A[ii * lda + jj] = s / ljj;                                          \
        }                                                                      \
      }                                                                        \
      int64_t m2 = n - (j + jb);                                               \
      if (m2 <= 0) continue;                                                   \
      /* 2. panel below the diagonal block: forward-solve by column */         \
      for (int64_t ii = j + jb; ii < n; ii++)                                  \
        for (int64_t c = 0; c < jb; c++) {                                     \
          int64_t jj = j + c;                                                  \
          T s = A[ii * lda + jj];                                              \
          for (int64_t kk = j; kk < jj; kk++)                                  \
            s -= A[ii * lda + kk] * CONJ(A[jj * lda + kk]);                    \
          A[ii * lda + jj] = s / REAL(A[jj * lda + jj]);                       \
        }                                                                      \
      /* 3. trailing update A22 -= L21·L21ᴴ via GEMM (into pscratch). The B     \
         operand is L21ᴴ = conj(L21)ᵀ: materialize conj(L21) in bscratch and    \
         present it transposed (strides swapped); for real types conj is the    \
         identity but the copy keeps one code path. */                         \
      T *L21 = &A[(j + jb) * lda + j];                                         \
      for (int64_t ii = 0; ii < m2; ii++)                                      \
        for (int64_t c = 0; c < jb; c++)                                       \
          bscratch[ii * jb + c] = CONJ(L21[ii * lda + c]);                     \
      nx_c_status gs = nx_c_gemm2d_ct_ws(                                       \
          DT, m2, m2, jb, (const char *)L21, lda, 1, (const char *)bscratch,   \
          1, jb, (char *)pscratch, m2, 1, (char *)vg);                         \
      if (gs != NX_C_OK) return gs;                                             \
      for (int64_t ii = 0; ii < m2; ii++)                                      \
        for (int64_t c = 0; c < m2; c++)                                       \
          A[(j + jb + ii) * lda + (j + jb + c)] -= pscratch[ii * m2 + c];      \
    }                                                                          \
    return NX_C_OK;                                                            \
  }

#define LA_EXPAND_CHOL(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)          \
  LA_GEN_CHOL(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_CHOL)
LA_TRAITS_double(LA_EXPAND_CHOL)
LA_TRAITS_c32(LA_EXPAND_CHOL)
LA_TRAITS_c64(LA_EXPAND_CHOL)
#undef LA_EXPAND_CHOL

/* ── Triangular solve: A·X = B or Aᴴ·X = B, A triangular ──────────────────

   X arrives pre-loaded with B (n×nrhs) and is solved in place. Substitution runs
   forward (rows 0..n-1) iff (upper == transpose) and backward otherwise; the
   coefficient and pivot are conjugated in the transpose case, so `transpose`
   means the CONJUGATE transpose for complex (Aᴴ·X = B) and plain transpose
   for real (conj is the identity). A zero pivot
   (and only when the diagonal is not assumed unit) is a singular matrix, raised
   rather than inf-poisoned.

   la_trsm_unb is the unblocked BLAS-2 substitution; la_trsm blocks it (dtrsm
   structure) once n exceeds LA_TRSM_NB: the effective operator M = op(A) is
   partitioned into LA_TRSM_NB diagonal blocks, each solved by la_trsm_unb, with
   the coupling to the not-yet-solved blocks applied as one GEMM per diagonal
   block. G materializes the off-diagonal M-block (conjugated for the transpose
   case, since the GEMM does not conjugate), P holds the product, gscratch the
   GEMM panels — all caller per-worker buffers. Below the crossover la_trsm is the
   unblocked kernel verbatim (the small-n path), so its numerics are unchanged. */
#define LA_GEN_TRSM(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)             \
  static nx_c_status la_trsm_unb_##sfx(void *vA, void *vX, int64_t n,           \
                                      int64_t lda, int64_t nrhs, int64_t ldx,  \
                                      int upper, int transpose, int unit) {    \
    T *A = (T *)vA;                                                           \
    T *X = (T *)vX;                                                           \
    int forward = (upper == transpose);                                       \
    for (int64_t ii = 0; ii < n; ii++) {                                      \
      int64_t i = forward ? ii : n - 1 - ii;                                  \
      T diag = transpose ? CONJ(A[i * lda + i]) : A[i * lda + i];             \
      /* Exact zero pivot raises; a NaN pivot is NOT caught here and propagates \
         NaN into x (matches LAPACK xTRTRS, which flags only exact singularity). \
         Deliberately asymmetric with cholesky's !(d>0), which DOES catch NaN. */ \
      if (!unit && diag == (T)0) return LA_ERR_SINGULAR;                      \
      for (int64_t j = 0; j < nrhs; j++) {                                    \
        T s = X[i * ldx + j];                                                 \
        if (forward)                                                          \
          for (int64_t k = 0; k < i; k++) {                                   \
            T c = transpose ? CONJ(A[k * lda + i]) : A[i * lda + k];          \
            s -= c * X[k * ldx + j];                                          \
          }                                                                    \
        else                                                                  \
          for (int64_t k = i + 1; k < n; k++) {                              \
            T c = transpose ? CONJ(A[k * lda + i]) : A[i * lda + k];          \
            s -= c * X[k * ldx + j];                                          \
          }                                                                    \
        X[i * ldx + j] = unit ? s : s / diag;                                \
      }                                                                        \
    }                                                                          \
    return NX_C_OK;                                                            \
  }                                                                            \
  static nx_c_status la_trsm_##sfx(void *vA, void *vX, int64_t n, int64_t lda,  \
                                  int64_t nrhs, int64_t ldx, int upper,        \
                                  int transpose, int unit, void *vG, void *vP, \
                                  void *vg) {                                  \
    if (n <= LA_TRSM_NB)                                                       \
      return la_trsm_unb_##sfx(vA, vX, n, lda, nrhs, ldx, upper, transpose,    \
                               unit);                                          \
    T *A = (T *)vA;                                                           \
    T *X = (T *)vX;                                                           \
    T *G = (T *)vG;                                                           \
    T *P = (T *)vP;                                                           \
    int forward = (upper == transpose);                                       \
    int64_t nblk = (n + LA_TRSM_NB - 1) / LA_TRSM_NB;                          \
    for (int64_t bi = 0; bi < nblk; bi++) {                                   \
      int64_t p0 = forward ? bi * LA_TRSM_NB                                   \
                           : (nblk - 1 - bi) * LA_TRSM_NB;                     \
      int64_t pb = n - p0 < LA_TRSM_NB ? n - p0 : LA_TRSM_NB;                  \
      nx_c_status s = la_trsm_unb_##sfx(&A[p0 * lda + p0], &X[p0 * ldx], pb,    \
                                       lda, nrhs, ldx, upper, transpose, unit);\
      if (s != NX_C_OK) return s;                                              \
      int64_t r0 = forward ? p0 + pb : 0;                                     \
      int64_t m2 = forward ? n - (p0 + pb) : p0;                              \
      if (m2 <= 0) continue;                                                   \
      /* G[a][b] = M[r0+a][p0+b], M = op(A): the block coupling the just-solved \
         X[p0:p0+pb] to the untouched rows [r0, r0+m2). */                     \
      for (int64_t a = 0; a < m2; a++)                                        \
        for (int64_t b = 0; b < pb; b++)                                      \
          G[a * pb + b] = transpose ? CONJ(A[(p0 + b) * lda + (r0 + a)])       \
                                    : A[(r0 + a) * lda + (p0 + b)];           \
      nx_c_status gs = nx_c_gemm2d_ct_ws(                                       \
          DT, m2, nrhs, pb, (const char *)G, pb, 1,                           \
          (const char *)&X[p0 * ldx], ldx, 1, (char *)P, nrhs, 1,             \
          (char *)vg);                                                        \
      if (gs != NX_C_OK) return gs;                                            \
      for (int64_t a = 0; a < m2; a++)                                        \
        for (int64_t j = 0; j < nrhs; j++)                                    \
          X[(r0 + a) * ldx + j] -= P[a * nrhs + j];                          \
    }                                                                          \
    return NX_C_OK;                                                            \
  }
#define LA_EXPAND_TRSM(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)          \
  LA_GEN_TRSM(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_TRSM)
LA_TRAITS_double(LA_EXPAND_TRSM)
LA_TRAITS_c32(LA_EXPAND_TRSM)
LA_TRAITS_c64(LA_EXPAND_TRSM)
#undef LA_EXPAND_TRSM

static const la_compute_desc la_desc[LA_NCOMPUTE] = {
    [LA_F32] = {.csize = sizeof(float), .gemm_dt = NX_C_DTYPE_f32,
                .chol = la_chol_f32, .trsm = la_trsm_f32},
    [LA_F64] = {.csize = sizeof(double), .gemm_dt = NX_C_DTYPE_f64,
                .chol = la_chol_f64, .trsm = la_trsm_f64},
    [LA_C32] = {.csize = sizeof(nx_c_complex32), .gemm_dt = NX_C_DTYPE_c32,
                .chol = la_chol_c32, .trsm = la_trsm_c32},
    [LA_C64] = {.csize = sizeof(nx_c_complex64), .gemm_dt = NX_C_DTYPE_c64,
                .chol = la_chol_c64, .trsm = la_trsm_c64},
};

/* ── Cholesky driver: batched, pooled ─────────────────────────────────────

   in/out are same-dtype, shape [batch..., n, n] (out contiguous, allocated by
   the binding). One batch matrix per job; each worker owns three compute-typed
   scratch buffers (the working matrix, the conj panel, the GEMM product), sized
   n×n each — allocated nthreads× under the lock. A non-PD matrix sets a shared
   error flag (first writer wins); the driver reports it after the region. */
typedef struct {
  const nx_c_ndarray *in;
  const nx_c_ndarray *out;
  nx_c_dtype dt;
  la_compute lc;
  int64_t n;
  int64_t esz;
  int upper;
  int inplace; /* factor directly into out (dt == compute type, out contiguous) */
  int batch_nd;
  const int64_t *bshape;
  const int64_t *in_bstride;  /* elements */
  const int64_t *out_bstride; /* elements */
  int64_t in_rs, in_cs, out_rs, out_cs; /* matrix strides, elements */
  char *work;                 /* nthreads * work_slot bytes (0 when inplace) */
  char *bscr;                 /* nthreads * panel_slot */
  char *pscr;                 /* nthreads * prod_slot */
  char *gscr;                 /* nthreads * gemm_slot (nx_c_gemm2d_ct_ws panels) */
  int64_t work_slot, bscr_slot, pscr_slot, gscr_slot;
  nx_c_status *werr; /* one slot per worker — no cross-thread write race */
} la_chol_ctx;

/* Cholesky computes L (lower). For ~upper the frontend wants U with A = Uᴴ·U;
   U = Lᴴ, i.e. the conjugate transpose of L. We factor lower, then the pack
   writes the upper triangle as the conjugate transpose — handled by asking
   packtri for `upper` and transposing+conjugating there. Simplest correct route:
   factor lower into `work`, and if upper is requested, conj-transpose `work` into
   its own upper triangle before packing. Done per-compute-type. */
#define LA_GEN_L2U(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)              \
  static void la_l2u_##sfx(void *vwork, int64_t n, int64_t lda) {              \
    T *w = (T *)vwork;                                                        \
    for (int64_t i = 0; i < n; i++)                                           \
      for (int64_t k = 0; k <= i; k++)                                        \
        w[k * lda + i] = CONJ(w[i * lda + k]);                                \
  }
#define LA_EXPAND_L2U(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)           \
  LA_GEN_L2U(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)
LA_TRAITS_float(LA_EXPAND_L2U)
LA_TRAITS_double(LA_EXPAND_L2U)
LA_TRAITS_c32(LA_EXPAND_L2U)
LA_TRAITS_c64(LA_EXPAND_L2U)
#undef LA_EXPAND_L2U

typedef void (*la_l2u_fn)(void *, int64_t, int64_t);
static const la_l2u_fn la_l2u[LA_NCOMPUTE] = {
    [LA_F32] = la_l2u_f32,
    [LA_F64] = la_l2u_f64,
    [LA_C32] = la_l2u_c32,
    [LA_C64] = la_l2u_c64,
};

/* One body for all four (upper × inplace) combinations. Out-of-line: unpack A
   into `work`, factor, (L->U for upper), packtri into out. Inplace (dt == compute
   type and out contiguous): unpack A straight into `out`, factor there, (L->U for
   upper), then zero the opposite triangle — a zero element is all-zero bytes for
   every native-compute type, so the strict triangle is a contiguous per-row memset
   rather than a second n×n copy. Inplace drops the `work` buffer and its alloc
   (which, at n<=LA_NB where there is also no trailing-update scratch, is the whole
   per-call heap request). */
static void la_chol_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  la_chol_ctx *x = (la_chol_ctx *)vctx;
  const la_compute_desc *cd = &la_desc[x->lc];
  const la_move_desc *mv = &la_move[x->dt];
  la_l2u_fn l2u = x->upper ? la_l2u[x->lc] : NULL;
  void *work = x->work_slot ? x->work + (int64_t)worker * x->work_slot : NULL;
  void *bscr = x->bscr_slot ? x->bscr + (int64_t)worker * x->bscr_slot : NULL;
  void *pscr = x->pscr_slot ? x->pscr + (int64_t)worker * x->pscr_slot : NULL;
  void *gscr = x->gscr_slot ? x->gscr + (int64_t)worker * x->gscr_slot : NULL;
  int64_t n = x->n, esz = x->esz;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *inb;
    const char *outb;
    la_batch_base(bt, x->batch_nd, x->bshape, x->in_bstride, x->in->offset, esz,
                  (const char *)x->in->data, &inb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->out_bstride, x->out->offset,
                  esz, (const char *)x->out->data, &outb);
    void *cbuf = x->inplace ? (void *)outb : work;
    int64_t lda = x->inplace ? x->out_rs : n;
    mv->unpack(inb, x->in_rs, x->in_cs, n, n, cbuf, lda);
    nx_c_status s = cd->chol(cbuf, n, lda, bscr, pscr, gscr);
    if (s != NX_C_OK) {
      if (x->werr[worker] == NX_C_OK) x->werr[worker] = s;
      continue;
    }
    if (l2u) l2u(cbuf, n, lda);
    if (x->inplace) {
      char *o = (char *)outb;
      int64_t rs = x->out_rs;
      if (x->upper)
        for (int64_t i = 1; i < n; i++)
          memset(o + i * rs * esz, 0, (size_t)i * esz);
      else
        for (int64_t i = 0; i < n - 1; i++)
          memset(o + (i * rs + i + 1) * esz, 0, (size_t)(n - 1 - i) * esz);
    } else {
      mv->packtri(cbuf, n, n, (char *)outb, x->out_rs, x->out_cs, x->upper);
    }
  }
}

static nx_c_status nx_c_cholesky_run(const nx_c_ndarray *in, const nx_c_ndarray *out,
                                   nx_c_dtype dt, int upper) {
  if (in->ndim < 2 || out->ndim != in->ndim) return LA_ERR_SHAPE_LA;
  int64_t n = in->shape[in->ndim - 1];
  if (in->shape[in->ndim - 2] != n) return LA_ERR_NOT_SQUARE;
  if (out->shape[out->ndim - 1] != n || out->shape[out->ndim - 2] != n)
    return LA_ERR_SHAPE_LA;

  la_compute lc = la_compute_of(dt);
  if (lc == LA_NCOMPUTE) return LA_ERR_NOT_FLOAT;
  const la_compute_desc *cd = &la_desc[lc];
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = in->ndim - 2;
  int64_t bshape[NX_C_MAX_NDIM], in_bs[NX_C_MAX_NDIM], out_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (out->shape[i] != in->shape[i]) return LA_ERR_SHAPE_LA;
    bshape[i] = in->shape[i];
    in_bs[i] = in->strides[i];
    out_bs[i] = out->strides[i];
    nbatch *= bshape[i];
  }
  if (n == 0 || nbatch == 0) return NX_C_OK;

  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nbatch, n * n * n, nbatch * n * n * esz);
  if (nth > nbatch) nth = (int)nbatch;
  if (nth > LA_MAX_WORKERS) nth = LA_MAX_WORKERS;
  if (nth < 1) nth = 1;

  /* When the storage dtype IS the compute type (f32/f64/c32/c64, not a converted
     f16/bf16/fp8) and out is contiguous with unit column stride, factor straight
     into out: no separate `work` buffer, and the pack-out collapses to zeroing one
     triangle in place. */
  int inplace = (esz == cd->csize) && (out->strides[out->ndim - 1] == 1);

  /* One block for all four per-worker scratches (working matrix, conj panel,
     GEMM product, GEMM packing panels), handed to the primitive as free_on_exit.
     Only `work` needs n×n (and only out-of-line); the trailing-update scratch is
     sized to the largest trailing block — the first one, (n-LA_NB)² fed by a
     k=LA_NB product. At or below the block size there is no trailing update, so the
     conj panel, product, and GEMM panels collapse to zero: an n<=LA_NB inplace call
     needs no scratch at all and skips the heap request entirely. */
#define LA_ALN(b) (((b) + 63) & ~(int64_t)63)
  int64_t tmax = n - LA_NB;
  if (tmax < 0) tmax = 0;
  int64_t work_slot = inplace ? 0 : LA_ALN(n * n * cd->csize);
  int64_t bscr_slot = LA_ALN(tmax * LA_NB * cd->csize);
  int64_t pscr_slot = LA_ALN(tmax * tmax * cd->csize);
  int64_t gslot = LA_ALN(nx_c_gemm2d_ct_scratch(cd->gemm_dt, tmax, tmax, LA_NB));
#undef LA_ALN
  int64_t stride = work_slot + bscr_slot + pscr_slot + gslot;
  char *scratch = NULL;
  if (stride > 0) {
    scratch = aligned_alloc(64, (size_t)stride * nth);
    if (!scratch) return NX_C_ERR_ALLOC;
  }

  nx_c_status werr[LA_MAX_WORKERS];
  for (int w = 0; w < nth; w++) werr[w] = NX_C_OK;

  la_chol_ctx x;
  x.in = in;
  x.out = out;
  x.dt = dt;
  x.lc = lc;
  x.n = n;
  x.esz = esz;
  x.upper = upper;
  x.inplace = inplace;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.in_bstride = in_bs;
  x.out_bstride = out_bs;
  x.in_rs = in->strides[in->ndim - 2];
  x.in_cs = in->strides[in->ndim - 1];
  x.out_rs = out->strides[out->ndim - 2];
  x.out_cs = out->strides[out->ndim - 1];
  x.work = scratch;
  x.bscr = scratch + (size_t)work_slot * nth;
  x.pscr = scratch + (size_t)(work_slot + bscr_slot) * nth;
  x.gscr = scratch + (size_t)(work_slot + bscr_slot + pscr_slot) * nth;
  x.work_slot = work_slot;
  x.bscr_slot = bscr_slot;
  x.pscr_slot = pscr_slot;
  x.gscr_slot = gslot;
  x.werr = werr;

  int64_t bytes = nbatch * n * n * esz;
  nx_c_parallel_for(nth, nbatch, bytes, la_chol_body, &x, scratch);

  nx_c_status err = NX_C_OK;
  for (int w = 0; w < nth; w++)
    if (werr[w] != NX_C_OK) {
      err = werr[w];
      break;
    }
  return err;
}

/* ── Triangular solve driver: batched, pooled ────────────────────────────
   a is [batch, n, n] triangular, b is [batch, n, nrhs], out (same shape as b) is
   the solution. Each worker unpacks A and B into its own scratch, solves in
   place, packs the full result. Singular (zero pivot, non-unit diagonal) is
   reported per worker and raised. */
typedef struct {
  const nx_c_ndarray *a;
  const nx_c_ndarray *b;
  const nx_c_ndarray *out;
  nx_c_dtype dt;
  la_compute lc;
  int64_t n, nrhs, esz;
  int upper, transpose, unit;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *a_bs;
  const int64_t *b_bs;
  const int64_t *o_bs;
  int64_t a_rs, a_cs, b_rs, b_cs, o_rs, o_cs;
  char *awork, *xwork, *gwork, *pwork, *gemm;
  int64_t awork_slot, xwork_slot, gwork_slot, pwork_slot, gemm_slot;
  nx_c_status *werr;
} la_trsm_ctx;

static void la_trsm_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  la_trsm_ctx *x = (la_trsm_ctx *)vctx;
  const la_compute_desc *cd = &la_desc[x->lc];
  const la_move_desc *mv = &la_move[x->dt];
  void *aw = x->awork + (int64_t)worker * x->awork_slot;
  void *xw = x->xwork + (int64_t)worker * x->xwork_slot;
  void *gw = x->gwork + (int64_t)worker * x->gwork_slot;
  void *pw = x->pwork + (int64_t)worker * x->pwork_slot;
  void *gm = x->gemm + (int64_t)worker * x->gemm_slot;
  int64_t n = x->n, nrhs = x->nrhs;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *ab;
    const char *bb;
    const char *ob;
    la_batch_base(bt, x->batch_nd, x->bshape, x->a_bs, x->a->offset, x->esz,
                  (const char *)x->a->data, &ab);
    la_batch_base(bt, x->batch_nd, x->bshape, x->b_bs, x->b->offset, x->esz,
                  (const char *)x->b->data, &bb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->o_bs, x->out->offset, x->esz,
                  (const char *)x->out->data, &ob);
    mv->unpack(ab, x->a_rs, x->a_cs, n, n, aw, n);
    mv->unpack(bb, x->b_rs, x->b_cs, n, nrhs, xw, nrhs);
    nx_c_status s = cd->trsm(aw, xw, n, n, nrhs, nrhs, x->upper, x->transpose,
                            x->unit, gw, pw, gm);
    if (s != NX_C_OK) {
      if (x->werr[worker] == NX_C_OK) x->werr[worker] = s;
      continue;
    }
    mv->packfull(xw, nrhs, n, nrhs, (char *)ob, x->o_rs, x->o_cs);
  }
}

static nx_c_status nx_c_trsm_run(const nx_c_ndarray *a, const nx_c_ndarray *b,
                               const nx_c_ndarray *out, nx_c_dtype dt, int upper,
                               int transpose, int unit) {
  if (a->ndim < 2 || b->ndim != a->ndim || out->ndim != b->ndim)
    return LA_ERR_SHAPE_LA;
  int64_t n = a->shape[a->ndim - 1];
  if (a->shape[a->ndim - 2] != n) return LA_ERR_NOT_SQUARE;
  int64_t nrhs = b->shape[b->ndim - 1];
  if (b->shape[b->ndim - 2] != n) return LA_ERR_SHAPE_LA;
  if (out->shape[out->ndim - 2] != n || out->shape[out->ndim - 1] != nrhs)
    return LA_ERR_SHAPE_LA;
  la_compute lc = la_compute_of(dt);
  if (lc == LA_NCOMPUTE) return LA_ERR_NOT_FLOAT;
  const la_compute_desc *cd = &la_desc[lc];
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = a->ndim - 2;
  int64_t bshape[NX_C_MAX_NDIM], a_bs[NX_C_MAX_NDIM], b_bs[NX_C_MAX_NDIM],
      o_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (b->shape[i] != a->shape[i] || out->shape[i] != a->shape[i])
      return LA_ERR_SHAPE_LA;
    bshape[i] = a->shape[i];
    a_bs[i] = a->strides[i];
    b_bs[i] = b->strides[i];
    o_bs[i] = out->strides[i];
    nbatch *= bshape[i];
  }
  if (n == 0 || nrhs == 0 || nbatch == 0) return NX_C_OK;

  int64_t bytes = nbatch * ((n * n) + (n * nrhs)) * esz;
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nbatch, n * n * nrhs, bytes);
  if (nth > nbatch) nth = (int)nbatch;
  if (nth > LA_MAX_WORKERS) nth = LA_MAX_WORKERS;
  if (nth < 1) nth = 1;

  /* Per-worker scratches (all free_on_exit as one block): A copy, X = B copy
     solved in place, the off-diagonal M-block G (m2 <= n rows, LA_TRSM_NB cols),
     the GEMM product P (m2 x nrhs), and the GEMM packing panels. G/P/gemm stay 0
     below the crossover (la_trsm runs unblocked). */
  int64_t aw_slot = ((n * n * cd->csize) + 63) & ~(int64_t)63;
  int64_t xw_slot = ((n * nrhs * cd->csize) + 63) & ~(int64_t)63;
  int64_t gw_slot = ((n * LA_TRSM_NB * cd->csize) + 63) & ~(int64_t)63;
  int64_t pw_slot = ((n * nrhs * cd->csize) + 63) & ~(int64_t)63;
  int64_t gm_slot = nx_c_gemm2d_ct_scratch(cd->gemm_dt, n, nrhs, LA_TRSM_NB);
  gm_slot = (gm_slot + 63) & ~(int64_t)63;
  char *scratch = aligned_alloc(
      64, (size_t)(aw_slot + xw_slot + gw_slot + pw_slot + gm_slot) * nth);
  if (!scratch) return NX_C_ERR_ALLOC;
  char *aw = scratch;
  char *xw = scratch + (size_t)aw_slot * nth;
  char *gw = scratch + (size_t)(aw_slot + xw_slot) * nth;
  char *pw = scratch + (size_t)(aw_slot + xw_slot + gw_slot) * nth;
  char *gm = scratch + (size_t)(aw_slot + xw_slot + gw_slot + pw_slot) * nth;

  nx_c_status werr[LA_MAX_WORKERS];
  for (int w = 0; w < nth; w++) werr[w] = NX_C_OK;

  la_trsm_ctx x;
  x.a = a;
  x.b = b;
  x.out = out;
  x.dt = dt;
  x.lc = lc;
  x.n = n;
  x.nrhs = nrhs;
  x.esz = esz;
  x.upper = upper;
  x.transpose = transpose;
  x.unit = unit;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.a_bs = a_bs;
  x.b_bs = b_bs;
  x.o_bs = o_bs;
  x.a_rs = a->strides[a->ndim - 2];
  x.a_cs = a->strides[a->ndim - 1];
  x.b_rs = b->strides[b->ndim - 2];
  x.b_cs = b->strides[b->ndim - 1];
  x.o_rs = out->strides[out->ndim - 2];
  x.o_cs = out->strides[out->ndim - 1];
  x.awork = aw;
  x.xwork = xw;
  x.gwork = gw;
  x.pwork = pw;
  x.gemm = gm;
  x.awork_slot = aw_slot;
  x.xwork_slot = xw_slot;
  x.gwork_slot = gw_slot;
  x.pwork_slot = pw_slot;
  x.gemm_slot = gm_slot;
  x.werr = werr;

  nx_c_parallel_for(nth, nbatch, bytes, la_trsm_body, &x, scratch);

  nx_c_status err = NX_C_OK;
  for (int w = 0; w < nth; w++)
    if (werr[w] != NX_C_OK) {
      err = werr[w];
      break;
    }
  return err;
}

CAMLprim value caml_nx_c_cholesky(value vout, value vin, value vupper) {
  CAMLparam3(vout, vin, vupper);
  nx_c_ndarray in, out;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) la_raise("cholesky", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  if (dt == NX_C_DTYPE_COUNT) la_raise("cholesky", NX_C_ERR_BAD_KIND);
  s = nx_c_cholesky_run(&in, &out, dt, Bool_val(vupper));
  if (s != NX_C_OK) la_raise("cholesky", s);
  CAMLreturn(Val_unit);
}

/* vflags packs the three booleans (bit 0 upper, bit 1 transpose, bit 2 unit
   diagonal) into one int so the stub stays at four arguments — no bytecode
   wrapper. out has b's shape; the binding allocates it. */
CAMLprim value caml_nx_c_triangular_solve(value vout, value va, value vb,
                                         value vflags) {
  CAMLparam4(vout, va, vb, vflags);
  nx_c_ndarray a, b, out;
  nx_c_status s = nx_c_ndarray_of_value(va, &a);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vb, &b);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vout, &out);
  if (s != NX_C_OK) la_raise("triangular_solve", s);
  nx_c_dtype dt = nx_c_dtype_of_value(va);
  if (dt == NX_C_DTYPE_COUNT) la_raise("triangular_solve", NX_C_ERR_BAD_KIND);
  int flags = Int_val(vflags);
  s = nx_c_trsm_run(&a, &b, &out, dt, flags & 1, (flags >> 1) & 1,
                   (flags >> 2) & 1);
  if (s != NX_C_OK) la_raise("triangular_solve", s);
  CAMLreturn(Val_unit);
}
