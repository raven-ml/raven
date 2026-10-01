/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_lu.c — LU factorization with partial pivoting, linalg tier 1. Shared
   machinery in nx_c_linalg.h. */

#include "nx_c_linalg.h"

/* Pivot magnitude, LAPACK's: |x| for real, |re| + |im| for complex. */
#define LU_MAG_f32(x) fabsf(x)
#define LU_MAG_f64(x) fabs(x)
#define LU_MAG_c32(x) (fabsf(crealf(x)) + fabsf(cimagf(x)))
#define LU_MAG_c64(x) (fabs(creal(x)) + fabs(cimag(x)))

/* ── Unblocked right-looking LU (LAPACK getf2) ─────────────────────────────

   Column j: the row of largest magnitude at or below the diagonal (the first
   on a tie) is swapped into row j across the whole row, the column tail below
   a nonzero pivot is divided by it (L's column), and the trailing submatrix
   takes the rank-1 update A[i][c] -= L[i][j] U[j][c]. A zero pivot leaves its
   column unscaled; the update then subtracts zeros. The inner update runs
   stride-1 over the row-major buffer. perm starts as the identity and follows
   every interchange. */
#define LA_GEN_LU(sfx, T, R, DT, CONJ, NORM2, REAL, FROMR, SQRT)              \
  static void la_lu_##sfx(void *vA, int64_t m, int64_t n, int64_t lda,         \
                          int64_t *piv, int64_t *perm) {                       \
    T *A = (T *)vA;                                                           \
    int64_t k = m < n ? m : n;                                                 \
    for (int64_t i = 0; i < m; i++) perm[i] = i;                               \
    for (int64_t j = 0; j < k; j++) {                                          \
      int64_t p = j;                                                           \
      R best = LU_MAG_##sfx(A[j * lda + j]);                                   \
      for (int64_t i = j + 1; i < m; i++) {                                    \
        R v = LU_MAG_##sfx(A[i * lda + j]);                                    \
        if (v > best) {                                                        \
          best = v;                                                            \
          p = i;                                                               \
        }                                                                      \
      }                                                                        \
      piv[j] = p;                                                              \
      if (p != j) {                                                            \
        T *rj = A + j * lda;                                                   \
        T *rp = A + p * lda;                                                   \
        for (int64_t c = 0; c < n; c++) {                                      \
          T t = rj[c];                                                         \
          rj[c] = rp[c];                                                       \
          rp[c] = t;                                                           \
        }                                                                      \
        int64_t q = perm[j];                                                   \
        perm[j] = perm[p];                                                     \
        perm[p] = q;                                                           \
      }                                                                        \
      T pivot = A[j * lda + j];                                                \
      if (pivot != (T)0)                                                       \
        for (int64_t i = j + 1; i < m; i++) A[i * lda + j] /= pivot;          \
      const T *urow = A + j * lda;                                             \
      for (int64_t i = j + 1; i < m; i++) {                                    \
        T *row = A + i * lda;                                                  \
        T l = row[j];                                                          \
        for (int64_t c = j + 1; c < n; c++) row[c] -= l * urow[c];             \
      }                                                                        \
    }                                                                          \
  }
LA_TRAITS_float(LA_GEN_LU)
LA_TRAITS_double(LA_GEN_LU)
LA_TRAITS_c32(LA_GEN_LU)
LA_TRAITS_c64(LA_GEN_LU)
#undef LA_GEN_LU

typedef void (*la_lu_fn)(void *A, int64_t m, int64_t n, int64_t lda,
                         int64_t *piv, int64_t *perm);

static const la_lu_fn la_lu[LA_NCOMPUTE] = {
    [LA_F32] = la_lu_f32,
    [LA_F64] = la_lu_f64,
    [LA_C32] = la_lu_c32,
    [LA_C64] = la_lu_c64,
};

static const int64_t la_csize[LA_NCOMPUTE] = {
    [LA_F32] = sizeof(float),
    [LA_F64] = sizeof(double),
    [LA_C32] = sizeof(nx_c_complex32),
    [LA_C64] = sizeof(nx_c_complex64),
};

/* ── LU driver: batched, pooled ──────────────────────────────────────────
   in and lu are [batch, m, n], piv [batch, k], perm [batch, m]. Each worker
   unpacks A, factors it in place, and packs the factors, the interchanges and
   the row order. LU is defined for every input, so there is no failure
   status. */
typedef struct {
  const nx_c_ndarray *in;
  const nx_c_ndarray *lu;
  const nx_c_ndarray *piv;
  const nx_c_ndarray *perm;
  nx_c_dtype dt;
  la_compute lc;
  int64_t m, n, k, esz;
  int batch_nd;
  const int64_t *bshape;
  const int64_t *in_bs;
  const int64_t *lu_bs;
  const int64_t *piv_bs;
  const int64_t *perm_bs;
  int64_t in_rs, in_cs, lu_rs, lu_cs, piv_s, perm_s;
  char *scratch; /* nthreads * stride: working A, then piv and perm */
  int64_t stride, off_piv, off_perm;
} la_lu_ctx;

static void la_lu_store(const int64_t *src, int64_t len, const char *dst,
                        int64_t stride) {
  int64_t *d = (int64_t *)dst;
  for (int64_t i = 0; i < len; i++) d[i * stride] = src[i];
}

static void la_lu_body(int64_t lo, int64_t hi, int worker, void *vctx) {
  la_lu_ctx *x = (la_lu_ctx *)vctx;
  const la_move_desc *mv = &la_move[x->dt];
  char *base = x->scratch + (int64_t)worker * x->stride;
  int64_t *piv = (int64_t *)(base + x->off_piv);
  int64_t *perm = (int64_t *)(base + x->off_perm);
  int64_t m = x->m, n = x->n;
  for (int64_t bt = lo; bt < hi; bt++) {
    const char *inb, *lub, *pivb, *permb;
    la_batch_base(bt, x->batch_nd, x->bshape, x->in_bs, x->in->offset, x->esz,
                  (const char *)x->in->data, &inb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->lu_bs, x->lu->offset, x->esz,
                  (const char *)x->lu->data, &lub);
    la_batch_base(bt, x->batch_nd, x->bshape, x->piv_bs, x->piv->offset,
                  (int64_t)sizeof(int64_t), (const char *)x->piv->data, &pivb);
    la_batch_base(bt, x->batch_nd, x->bshape, x->perm_bs, x->perm->offset,
                  (int64_t)sizeof(int64_t), (const char *)x->perm->data,
                  &permb);
    mv->unpack(inb, x->in_rs, x->in_cs, m, n, base, n);
    la_lu[x->lc](base, m, n, n, piv, perm);
    mv->packfull(base, n, m, n, (char *)lub, x->lu_rs, x->lu_cs);
    la_lu_store(piv, x->k, pivb, x->piv_s);
    la_lu_store(perm, m, permb, x->perm_s);
  }
}

static nx_c_status nx_c_lu_run(const nx_c_ndarray *in, const nx_c_ndarray *lu,
                               const nx_c_ndarray *piv,
                               const nx_c_ndarray *perm, nx_c_dtype dt) {
  int nd = in->ndim;
  if (nd < 2 || lu->ndim != nd || piv->ndim != nd - 1 || perm->ndim != nd - 1)
    return LA_ERR_SHAPE_LA;
  int64_t m = in->shape[nd - 2];
  int64_t n = in->shape[nd - 1];
  int64_t k = m < n ? m : n;
  if (lu->shape[nd - 2] != m || lu->shape[nd - 1] != n ||
      piv->shape[nd - 2] != k || perm->shape[nd - 2] != m)
    return LA_ERR_SHAPE_LA;
  la_compute lc = la_compute_of(dt);
  if (lc == LA_NCOMPUTE) return LA_ERR_NOT_FLOAT;
  int64_t esz = nx_c_elem_size(dt);

  int batch_nd = nd - 2;
  int64_t bshape[NX_C_MAX_NDIM], in_bs[NX_C_MAX_NDIM], lu_bs[NX_C_MAX_NDIM],
      piv_bs[NX_C_MAX_NDIM], perm_bs[NX_C_MAX_NDIM];
  int64_t nbatch = 1;
  for (int i = 0; i < batch_nd; i++) {
    if (lu->shape[i] != in->shape[i] || piv->shape[i] != in->shape[i] ||
        perm->shape[i] != in->shape[i])
      return LA_ERR_SHAPE_LA;
    bshape[i] = in->shape[i];
    in_bs[i] = in->strides[i];
    lu_bs[i] = lu->strides[i];
    piv_bs[i] = piv->strides[i];
    perm_bs[i] = perm->strides[i];
    nbatch *= bshape[i];
  }
  if (nbatch == 0 || m == 0) return NX_C_OK;

  int64_t bytes = nbatch * (2 * m * n * esz + (k + m) * 8);
  int nth = nx_c_threads_for(NX_C_COST_HEAVY, nbatch, m * n * k, bytes);
  if (nth > nbatch) nth = (int)nbatch;
  if (nth < 1) nth = 1;

#define LA_ALN(b) (((b) + 63) & ~(int64_t)63)
  int64_t off_piv = LA_ALN(m * n * la_csize[lc]);
  int64_t off_perm = off_piv + LA_ALN(k * (int64_t)sizeof(int64_t));
  int64_t stride = off_perm + LA_ALN(m * (int64_t)sizeof(int64_t));
#undef LA_ALN
  char *scratch = nx_c_aligned_alloc((size_t)stride * nth);
  if (!scratch) return NX_C_ERR_ALLOC;

  la_lu_ctx x;
  x.in = in;
  x.lu = lu;
  x.piv = piv;
  x.perm = perm;
  x.dt = dt;
  x.lc = lc;
  x.m = m;
  x.n = n;
  x.k = k;
  x.esz = esz;
  x.batch_nd = batch_nd;
  x.bshape = bshape;
  x.in_bs = in_bs;
  x.lu_bs = lu_bs;
  x.piv_bs = piv_bs;
  x.perm_bs = perm_bs;
  x.in_rs = in->strides[nd - 2];
  x.in_cs = in->strides[nd - 1];
  x.lu_rs = lu->strides[nd - 2];
  x.lu_cs = lu->strides[nd - 1];
  x.piv_s = piv->strides[nd - 2];
  x.perm_s = perm->strides[nd - 2];
  x.scratch = scratch;
  x.stride = stride;
  x.off_piv = off_piv;
  x.off_perm = off_perm;

  nx_c_parallel_for(nth, nbatch, bytes, la_lu_body, &x, scratch);
  return NX_C_OK;
}

/* The factors, pivots and permutation are allocated by the binding. */
CAMLprim value caml_nx_c_lu(value vlu, value vpiv, value vperm, value vin) {
  CAMLparam4(vlu, vpiv, vperm, vin);
  nx_c_ndarray in, lu, piv, perm;
  nx_c_status s = nx_c_ndarray_of_value(vin, &in);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vlu, &lu);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vpiv, &piv);
  if (s == NX_C_OK) s = nx_c_ndarray_of_value(vperm, &perm);
  if (s != NX_C_OK) la_raise("lu", s);
  nx_c_dtype dt = nx_c_dtype_of_value(vin);
  s = nx_c_lu_run(&in, &lu, &piv, &perm, dt);
  if (s != NX_C_OK) la_raise("lu", s);
  CAMLreturn(Val_unit);
}
