/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_linalg.h — shared machinery for the owned dense linear algebra family:
   compute-type traits, failure statuses, block-size crossovers, the batched
   dispatch descriptor + strided<->contiguous unpack/pack, batch addressing,
   and the cross-family QR entry points SVD's U formation consumes. The four
   family translation units (nx_c_tri.c cholesky+triangular-solve, nx_c_qr.c QR,
   nx_c_eigh.c eigh, nx_c_svd.c SVD) each include this once. Code-only header;
   the static tables/helpers it defines are instantiated per translation unit. */

#ifndef NX_C_LINALG_H
#define NX_C_LINALG_H

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h" /* pool + status + nx_c_raise; nx_c.h comes with it */
#include "nx_c_matmul.h" /* nx_c_gemm2d_ct — trailing updates */

/* Tier-1 failure statuses (static strings; the stub maps them to exceptions). */
static const char LA_ERR_NOT_PD[] = "matrix is not positive definite";
static const char LA_ERR_SINGULAR[] = "triangular matrix is singular";
static const char LA_ERR_NOT_FLOAT[] =
    "linalg requires a float or complex dtype";
static const char LA_ERR_NOT_SQUARE[] = "matrix must be square";
static const char LA_ERR_SHAPE_LA[] = "operand shapes are incompatible";
static const char LA_ERR_NO_CONVERGE[] =
    "eigenvalue iteration did not converge";

/* Block size for the right-looking factorizations: the diagonal panel is factored
   unblocked, the trailing (n-j-nb) submatrix updated through the GEMM. Chosen so
   the panel is L1-resident; the trailing GEMM is where the flops (and the speed)
   live. One named constant per op — each is the crossover below which the op runs
   the unblocked BLAS-2 kernel (which is also the panel/diagonal-block kernel above
   it) and above which the trailing update goes through nx_c_gemm2d_ct_ws. Tuned
   empirically; the GEMM dominates well before these, so the exact value is not
   delicate. */
#define LA_NB 64      /* cholesky */
#define LA_TRSM_NB 64 /* triangular solve */
#define LA_QR_NB 32   /* compact-WY QR blocked panel width */
#define LA_QR_UNB 128 /* QR unblocked crossover (below this, no block reflector):
                         the reflector apply is vectorized (stride-1 over columns of
                         the row-major panel), so the pure BLAS-2 path — which has no
                         per-panel GEMM setup and no tiny-GEMM overhead — beats the
                         block reflector until the trailing GEMM's BLAS-3 reuse takes
                         over. Probed f64: unblocked vs blocked was 9.7/2.8 at n=64
                         and 10.4/6.8 at n=128 (unblocked wins), 7.6/9.7 at n=256
                         (blocked wins) — so the crossover sits well above the panel
                         width, unlike the pre-vectorization tie at n=64. */
#define LA_EIGH_NB 32 /* latrd tridiagonalization panel width */
#define LA_SVD_NB 32  /* labrd bidiagonalization panel width */

/* Per-worker error slots live on the driver stack (no heap, so leak-safe across
   the pool's lock re-acquire, and race-free since each worker writes its own
   slot). MUST be >= the engine pool cap NX_C_MAX_THREADS (nx_c_engine.c), since a
   worker index can reach nthreads-1; every driver also clamps nth to it. A
   _Static_assert would pin this, but NX_C_MAX_THREADS is private to nx_c_engine.c —
   both are literal 64, and the engine clamps threads to physical cores anyway
   (Apple Silicon <= 64), so the bound holds with margin. Bump both together. */
#define LA_MAX_WORKERS 64

/* ── Compute-type traits ──────────────────────────────────────────────────
   The four linalg compute types and, per type, the real scalar R, the GEMM
   dtype tag, and the arithmetic a factorization needs: conjugate, |x|^2 as R,
   real part as R, lift R->T, and real sqrt. Real types make conj/real identities
   so one kernel body serves all four. */
#define LA_TRAITS_float(M)                                                     \
  M(f32, float, float, NX_C_DTYPE_f32, LA_ID, LA_SQ_R, LA_ID, LA_ID, sqrtf)
#define LA_TRAITS_double(M)                                                    \
  M(f64, double, double, NX_C_DTYPE_f64, LA_ID, LA_SQ_R, LA_ID, LA_ID, sqrt)
#define LA_TRAITS_c32(M)                                                       \
  M(c32, nx_c_complex32, float, NX_C_DTYPE_c32, conjf, LA_CNORM2f, crealf,       \
    LA_CFROM, sqrtf)
#define LA_TRAITS_c64(M)                                                       \
  M(c64, nx_c_complex64, double, NX_C_DTYPE_c64, conj, LA_CNORM2, creal,         \
    LA_CFROM, sqrt)

#define LA_ID(x) (x)
#define LA_SQ_R(x) ((x) * (x))
#define LA_CFROM(r) (r) /* real -> complex is an implicit conversion */
#define LA_CNORM2f(x) (crealf((x) * conjf(x)))
#define LA_CNORM2(x) (creal((x) * conj(x)))

/* Imaginary part and (real,imag)->T construction, per dtype suffix. For the real
   types the imaginary text is dropped and construction is the real part, so one
   QR body serves all four; keyed by suffix so the traits macro's arity is
   unchanged. */
#define LA_IMAG_f32(x) 0.0f
#define LA_IMAG_f64(x) 0.0
#define LA_IMAG_c32(x) cimagf(x)
#define LA_IMAG_c64(x) cimag(x)
#define LA_MK_f32(re, im) (re)
#define LA_MK_f64(re, im) (re)
#define LA_MK_c32(re, im) CMPLXF((re), (im))
#define LA_MK_c64(re, im) CMPLX((re), (im))

/* Real-scalar |.| and hypot for the eigensolver's shift math, keyed by suffix
   (R is float for f32/c32, double for f64/c64). */
#define LA_ABS_f32(x) fabsf(x)
#define LA_ABS_f64(x) fabs(x)
#define LA_ABS_c32(x) fabsf(x)
#define LA_ABS_c64(x) fabs(x)
#define LA_HYP_f32(a, b) hypotf((a), (b))
#define LA_HYP_f64(a, b) hypot((a), (b))
#define LA_HYP_c32(a, b) hypotf((a), (b))
#define LA_HYP_c64(a, b) hypot((a), (b))

/* ── Compute-type descriptor and storage->compute mapping ─────────────────

   la_compute is the factorization compute type; every storage dtype maps to one
   (f16/bf16/fp8/f32 -> F32, f64 -> F64, c32/c64 -> their own). csize is the
   compute element size; the *_gemm tag drives nx_c_gemm2d_ct. The cholesky slot
   is the type-erased kernel (T* -> void*). */
typedef enum { LA_F32, LA_F64, LA_C32, LA_C64, LA_NCOMPUTE } la_compute;

typedef nx_c_status (*la_chol_fn)(void *A, int64_t n, int64_t lda, void *bscr,
                                 void *pscr, void *gscr);
typedef nx_c_status (*la_trsm_fn)(void *A, void *X, int64_t n, int64_t lda,
                                 int64_t nrhs, int64_t ldx, int upper,
                                 int transpose, int unit, void *G, void *P,
                                 void *gscr);
typedef void (*la_qr_fn)(void *A, int64_t m, int64_t n, int64_t lda, void *tau,
                         void *V, void *Vc, void *T, void *W, void *P,
                         void *gscr);
typedef void (*la_qrq_fn)(void *A, int64_t m, int64_t n, int64_t lda, void *tau,
                          void *Q, int64_t nq, int64_t ldq, void *V, void *Vc,
                          void *T, void *W, void *P, void *gscr);
typedef void (*la_tridiag_fn)(void *A, int64_t n, int64_t lda, void *d, void *e,
                              void *tau, void *wv, void *W, void *Wc, void *P,
                              void *gscr);
typedef void (*la_orgtr_fn)(void *A, int64_t n, int64_t lda, void *tau, void *Z,
                            int64_t ldz);
typedef nx_c_status (*la_tql2_fn)(void *d, void *e, void *Z, int64_t n,
                                 int64_t ldz, int want_vec);
typedef void (*la_eigsort_fn)(void *d, void *Z, int64_t n, int64_t ldz,
                              int want_vec);
typedef void (*la_gebrd_fn)(void *P, int64_t pr, int64_t pc, int64_t ld,
                            void *d, void *e, void *tauq, void *taup, void *X,
                            void *Y, void *Mc, void *Pm, void *gscr);
typedef void (*la_formp_fn)(void *P, int64_t pc, int64_t ld, void *taup,
                            void *VT, int64_t ldvt);
typedef nx_c_status (*la_bdsvd_fn)(void *d, void *e, void *U, int64_t pr,
                                  int64_t ldu, void *VT, int64_t ldvt,
                                  int64_t pc);
typedef void (*la_cpc_fn)(const void *src, int64_t sld, void *dst, int64_t dld,
                          int64_t rows, int64_t cols);

typedef struct {
  int64_t csize;
  nx_c_dtype gemm_dt;
  la_chol_fn chol;
  la_trsm_fn trsm;
  la_qr_fn qr;
  la_qrq_fn qrq;
  la_tridiag_fn tridiag;
  la_orgtr_fn orgtr;
  la_tql2_fn tql2;
  la_eigsort_fn eigsort;
  la_gebrd_fn gebrd;
  la_formp_fn formp;
  la_bdsvd_fn bdsvd;
  la_cpc_fn cpc;  /* plain copy */
  la_cpc_fn cpct; /* conjugate-transpose copy */
} la_compute_desc;

/* Each family .c defines its OWN la_desc[LA_NCOMPUTE] as a per-TU copy of this
   struct, designated-initializer-filled with only that family's kernel slots
   (csize + gemm_dt plus, e.g., .chol/.trsm in nx_c_tri.c). Every other
   function-pointer slot is NULL BY DESIGN — a family's driver never
   dereferences a slot it did not fill. SVD's .qrq points at the cross-family
   nx_c_la_qrq_* entry declared at the end of this header. */

/* Storage dtype -> compute type, or LA_NCOMPUTE for the unsupported (integer,
   bool, packed) dtypes the frontend already screens out. */
static la_compute la_compute_of(nx_c_dtype dt) {
  switch (dt) {
    case NX_C_DTYPE_f16:
    case NX_C_DTYPE_bf16:
    case NX_C_DTYPE_f8e4m3:
    case NX_C_DTYPE_f8e5m2:
    case NX_C_DTYPE_f32:
      return LA_F32;
    case NX_C_DTYPE_f64:
      return LA_F64;
    case NX_C_DTYPE_c32:
      return LA_C32;
    case NX_C_DTYPE_c64:
      return LA_C64;
    default:
      return LA_NCOMPUTE;
  }
}

/* ── Unpack / pack: strided storage <-> contiguous compute buffer ─────────

   Generated per storage dtype (all compute dtypes; the non-float ones are never
   dispatched here but cost nothing). unpack reads element (i,k) at
   base + (i*rs + k*cs)*esz through nx_c_ld (converting to the compute type) into a
   dense n×lda buffer. pack_tri writes the lower (or upper) triangle back through
   nx_c_st and zeroes the other triangle — the cholesky output shape. */
#define LA_GEN_MOVE(sfx, kind, storage, compute, ld, st, cat)                  \
  static void la_unpack_##sfx(const char *src, int64_t rs, int64_t cs,         \
                              int64_t rows, int64_t cols, void *vdst,          \
                              int64_t ld_) {                                   \
    compute *dst = (compute *)vdst;                                           \
    int64_t esz = (int64_t)sizeof(storage);                                    \
    for (int64_t i = 0; i < rows; i++)                                         \
      for (int64_t k = 0; k < cols; k++)                                       \
        dst[i * ld_ + k] = nx_c_ld_##sfx(src + (i * rs + k * cs) * esz);        \
  }                                                                            \
  static void la_packtri_##sfx(const void *vsrc, int64_t ld_, int64_t n,       \
                               char *dst, int64_t rs, int64_t cs, int upper) { \
    const compute *src = (const compute *)vsrc;                               \
    int64_t esz = (int64_t)sizeof(storage);                                    \
    for (int64_t i = 0; i < n; i++)                                            \
      for (int64_t k = 0; k < n; k++) {                                        \
        compute v = (upper ? (k >= i) : (k <= i)) ? src[i * ld_ + k]           \
                                                  : (compute)0;               \
        nx_c_st_##sfx(dst + (i * rs + k * cs) * esz, v);                        \
      }                                                                        \
  }                                                                            \
  static void la_packfull_##sfx(const void *vsrc, int64_t ld_, int64_t rows,   \
                                int64_t cols, char *dst, int64_t rs,           \
                                int64_t cs) {                                  \
    const compute *src = (const compute *)vsrc;                               \
    int64_t esz = (int64_t)sizeof(storage);                                    \
    for (int64_t i = 0; i < rows; i++)                                         \
      for (int64_t k = 0; k < cols; k++)                                       \
        nx_c_st_##sfx(dst + (i * rs + k * cs) * esz, src[i * ld_ + k]);         \
  }                                                                            \
  static void la_packR_##sfx(const void *vsrc, int64_t ld_, int64_t rows,      \
                             int64_t cols, char *dst, int64_t rs, int64_t cs) { \
    const compute *src = (const compute *)vsrc;                               \
    int64_t esz = (int64_t)sizeof(storage);                                    \
    for (int64_t i = 0; i < rows; i++)                                         \
      for (int64_t k = 0; k < cols; k++) {                                     \
        compute v = k >= i ? src[i * ld_ + k] : (compute)0;                   \
        nx_c_st_##sfx(dst + (i * rs + k * cs) * esz, v);                        \
      }                                                                        \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(LA_GEN_MOVE)
#undef LA_GEN_MOVE

typedef void (*la_unpack_fn)(const char *, int64_t, int64_t, int64_t, int64_t,
                             void *, int64_t);
typedef void (*la_packtri_fn)(const void *, int64_t, int64_t, char *, int64_t,
                              int64_t, int);
typedef void (*la_packfull_fn)(const void *, int64_t, int64_t, int64_t, char *,
                               int64_t, int64_t);

typedef struct {
  la_unpack_fn unpack;
  la_packtri_fn packtri;
  la_packfull_fn packfull;
  la_packfull_fn packR; /* upper trapezoid; same signature as packfull */
} la_move_desc;

static const la_move_desc la_move[NX_C_DTYPE_COUNT] = {
#define LA_MOVE_ROW(sfx, kind, storage, compute, ld, st, cat)                  \
  [NX_C_DTYPE_##sfx] = {la_unpack_##sfx, la_packtri_##sfx, la_packfull_##sfx,   \
                       la_packR_##sfx},
    NX_C_FOR_EACH_COMPUTE_DTYPE(LA_MOVE_ROW)
#undef LA_MOVE_ROW
};

static void la_batch_base(int64_t bt, int nd, const int64_t *bshape,
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

/* ── Cross-family QR entry points (defined in nx_c_qr.c) ────────────────────
   SVD forms its left factor U through the QR compact-WY form-Q (nx_c_la_qrq);
   the panel factorization, its block T (nx_c_la_larft), and the V-panel builder
   (nx_c_la_buildv) are exported alongside it for the divide-and-conquer
   eigensolvers. There is no standalone block-reflector apply — it is inlined in
   the QR factor and form-Q. These are the only linalg symbols with external
   linkage, hence the nx_c_ prefix (the every-global-is-nx_c_* invariant). */
#define LA_QR_EXPORT(sfx)                                                      \
  void nx_c_la_qr_panel_##sfx(void *A, int64_t m, int64_t lda, int64_t j0,      \
                             int64_t jb, void *tau, int64_t col_lim);          \
  void nx_c_la_larft_##sfx(const void *A, int64_t m, int64_t lda, int64_t j0,   \
                          int64_t jb, const void *tau, void *T, int64_t ldt);  \
  void nx_c_la_buildv_##sfx(const void *A, int64_t m, int64_t lda, int64_t j0,  \
                           int64_t jb, void *V, void *Vc);                     \
  void nx_c_la_qrq_##sfx(void *A, int64_t m, int64_t n, int64_t lda, void *tau, \
                        void *Q, int64_t nq, int64_t ldq, void *V, void *Vc,   \
                        void *T, void *W, void *P, void *gscr);
LA_QR_EXPORT(f32)
LA_QR_EXPORT(f64)
LA_QR_EXPORT(c32)
LA_QR_EXPORT(c64)
#undef LA_QR_EXPORT

static NX_C_NORETURN void la_raise(const char *op, nx_c_status s) {
  /* Shape/dtype preconditions are the caller's bad argument (Invalid_argument);
     a non-PD matrix or a singular solve is a runtime Failure — the interface
     documents cholesky raising Failure when not positive definite. */
  if (strcmp(s, LA_ERR_NOT_FLOAT) == 0 || strcmp(s, LA_ERR_NOT_SQUARE) == 0 ||
      strcmp(s, LA_ERR_SHAPE_LA) == 0)
    nx_c_raise_invalid(op, s);
  nx_c_raise(op, s);
}

#endif /* NX_C_LINALG_H */
