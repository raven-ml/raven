/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The spectral routines' steps: eigh, svd and eig, batched, at float32,
   float64, complex64 and complex128 (kernels.h, Spectral routines).
   spectral.ml plans them, with the contraction's products (contract.hip)
   between them.

   sp_jacobi.h: both extents at most 32, a matrix in LDS: two-sided
     cyclic Jacobi (eigh), one-sided (svd).
   sp_tridiag.h, sp_stedc.h: eigh past it, xHETRD's blocked
     tridiagonalisation, then divide and conquer (xSTEDC, leaves of 32 by
     the Jacobi kernel) or, for values alone, bisection.
   sp_bd.h: svd past it, xGEBD2's bidiagonalisation, then bisection on
     the Golub-Kahan matrix for values alone, or QR sweeps (xBDSQR)
     recorded by one work-item a matrix and applied 8 a pass.
   sp_eig.h: eig, one group a matrix: balancing, Hessenberg reduction,
     Francis QR (xLAHQR) and back substitution (xTREVC3).

   Each entry reads its step's parameters from words [0, NX_SP_FMT) in
   order: an address, an integer, or a size. */

#include "sp_common.h"

namespace nx_spectral {

#include "sp_jacobi.h"
#undef NT
#undef MAX_SWEEPS
#undef ROW

#include "sp_tridiag.h"
#undef NB

#include "sp_stedc.h"
#undef LEAF
#undef Q
#undef DL

#include "sp_bd.h"
#undef S
#undef G
#undef ARGS

#include "sp_eig.h"
#undef KEXSH

/* x[z] = the r x c identity's first rows and columns: an svd's unitary
   factors where k = 0. */
template <class R> D void eye(R *x, int64_t nz, int r, int c, int P) {
  const int64_t per = (int64_t)r * c;
  const bool small = is_small(nz * per);
  for (int64_t q = (int64_t)bid_x() * nthr() + tid(); q < nz * per;
       q += (int64_t)ngx() * nthr()) {
    int64_t z = q, k = split(z, per, small);
    const int64_t col = split(k, c, small);
    x[q * P] = (R)(k == col);
    if (P == 2) x[q * P + 1] = 0;
  }
}

/* A step of parts alone: its real type, and its element's parts. */
#define PARTS typename real_of<T>::R
#define NPARTS (real_of<T>::cplx ? 2 : 1)

/* A word as the parameter it stands for. */
struct word {
  int64_t v;
  template <class P> D operator P *() const { return (P *)(uintptr_t)v; }
  D operator int() const { return (int)v; }
  D operator long() const { return (long)v; }
  D operator unsigned long() const { return (unsigned long)v; }
};

#define W(k) word{p[k]}

template <class T> D void sp_eigh_jacobi(const int64_t *p) {
  eigh_jacobi<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_svd_jacobi(const int64_t *p) {
  svd_jacobi<T>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), W(8));
}

template <class T> D void sp_td_stage(const int64_t *p) {
  td_stage<PARTS>(W(0), W(1), W(2), W(3), NPARTS);
}

template <class T> D void sp_td_step(const int64_t *p) {
  td_step<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_td_hemv(const int64_t *p) {
  td_hemv<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_td_diag(const int64_t *p) {
  td_diag<PARTS>(W(0), W(1), W(2), NPARTS);
}

template <class T> D void sp_td_vcopy(const int64_t *p) {
  td_vcopy<PARTS>(W(0), W(1), W(2), W(3), W(4), NPARTS);
}

template <class T> D void sp_larft(const int64_t *p) {
  larft<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_td_zinit(const int64_t *p) {
  td_zinit<PARTS>(W(0), W(1), W(2), W(3), W(4), NPARTS);
}

template <class T> D void sp_td_out(const int64_t *p) {
  td_out<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), W(8), W(9),
                NPARTS);
}

template <class R> D void sp_dc_leaves(const int64_t *p) {
  dc_leaves<R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7));
}

template <class R> D void sp_dc_scatter(const int64_t *p) {
  dc_scatter<R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6));
}

template <class R> D void sp_dc_deflate(const int64_t *p) {
  dc_deflate<R>(W(0), W(1), W(2), W(3));
}

template <class R> D void sp_dc_secular(const int64_t *p) {
  dc_secular<R>(W(0), W(1), W(2), W(3), W(4));
}

template <class R> D void sp_dc_product(const int64_t *p) {
  dc_product<R>(W(0), W(1), W(2), W(3));
}

template <class R> D void sp_dc_sort(const int64_t *p) {
  dc_sort<R>(W(0), W(1), W(2), W(3));
}

template <class R> D void sp_tri_bisect(const int64_t *p) {
  tri_bisect<R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7));
}

template <class T> D void sp_bd_stage(const int64_t *p) {
  bd_stage<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), NPARTS);
}

template <class T> D void sp_bd_left(const int64_t *p) {
  bd_left<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_wh(const int64_t *p) {
  bd_wh<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_updl(const int64_t *p) {
  bd_updl<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_right(const int64_t *p) {
  bd_right<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_wn(const int64_t *p) {
  bd_wn<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_updr(const int64_t *p) {
  bd_updr<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_tgk(const int64_t *p) {
  bd_tgk<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), (int)sizeof(T));
}

template <class T> D void sp_bd_values(const int64_t *p) {
  bd_values<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6),
                   (int)sizeof(T));
}

template <class T> D void sp_bd_ginit(const int64_t *p) {
  bd_ginit<PARTS>(W(0), W(1), W(2), W(3), W(4), (int)sizeof(T));
}

template <class T> D void sp_bd_gen(const int64_t *p) {
  bd_gen<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_bd_apply(const int64_t *p) {
  bd_apply<PARTS>(W(0), W(1), W(2), W(3), W(4), (int)sizeof(T));
}

template <class T> D void sp_bd_xinit(const int64_t *p) {
  bd_xinit<PARTS>(W(0), W(1), W(2), W(3), W(4), NPARTS);
}

template <class T> D void sp_bd_vcopy(const int64_t *p) {
  bd_vcopy<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), NPARTS);
}

template <class T> D void sp_bd_rank(const int64_t *p) {
  bd_rank<PARTS>(W(0), W(1), W(2), W(3), W(4), (int)sizeof(T));
}

template <class T> D void sp_bd_out(const int64_t *p) {
  bd_out<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), W(8), W(9),
                W(10), W(11), NPARTS);
}

template <class T> D void sp_eig_schur(const int64_t *p) {
  eig_schur<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_eig_norms(const int64_t *p) {
  eig_norms<T>(W(0), W(1), W(2));
}

template <class T> D void sp_eig_vectors(const int64_t *p) {
  eig_vectors<T>(W(0), W(1), W(2));
}

template <class T> D void sp_eig_finish(const int64_t *p) {
  eig_finish<PARTS>(W(0), W(1), W(2), W(3), W(4), real_of<T>::cplx);
}

template <class T> D void sp_eye(const int64_t *p) {
  eye<PARTS>(W(0), W(1), W(2), W(3), NPARTS);
}

#undef W
#undef PARTS
#undef NPARTS

} /* namespace nx_spectral */

/* Entries */

#define SP_T_f32 float
#define SP_T_f64 double
#define SP_T_c64 cf
#define SP_T_c128 cd

/* A kernel of one format, of [threads] work-items a group. */
#define SPECTRAL(name, threads, words, entry, fmt)                            \
  KERNEL SIZED(threads) void name(Words<words>) {                             \
    stage(sp_words, words, threads);                                          \
    nx_spectral::sp_##entry<SP_T_##fmt>(staged());                            \
  }

/* A kernel of every format, or every real one, by the word NX_SP_FMT. */
#define SPECTRAL_ANY(name, threads, words, entry)                             \
  KERNEL SIZED(threads) void name(Words<words>) {                             \
    stage(sp_words, words, threads);                                          \
    const int64_t *p = staged();                                              \
    switch (p[NX_SP_FMT]) {                                                   \
    case NX_SP_F32: nx_spectral::sp_##entry<SP_T_f32>(p); break;              \
    case NX_SP_F64: nx_spectral::sp_##entry<SP_T_f64>(p); break;              \
    case NX_SP_C64: nx_spectral::sp_##entry<SP_T_c64>(p); break;              \
    default: nx_spectral::sp_##entry<SP_T_c128>(p);                           \
    }                                                                         \
  }

#define SPECTRAL_REAL(name, threads, words, entry)                            \
  KERNEL SIZED(threads) void name(Words<words>) {                             \
    stage(sp_words, words, threads);                                          \
    const int64_t *p = staged();                                              \
    if (p[NX_SP_FMT] == NX_SP_F32)                                            \
      nx_spectral::sp_##entry<SP_T_f32>(p);                                   \
    else                                                                      \
      nx_spectral::sp_##entry<SP_T_f64>(p);                                   \
  }

#define EACH(name, threads, words)                                            \
  SPECTRAL(sp_##name##_f32, threads, words, name, f32)                        \
  SPECTRAL(sp_##name##_f64, threads, words, name, f64)                        \
  SPECTRAL(sp_##name##_c64, threads, words, name, c64)                        \
  SPECTRAL(sp_##name##_c128, threads, words, name, c128)

#define EACH_REAL(name, threads, words)                                       \
  SPECTRAL(sp_##name##_f32, threads, words, name, f32)                        \
  SPECTRAL(sp_##name##_f64, threads, words, name, f64)

EACH(eigh_jacobi, NX_SP_THREADS, NX_SP_PLAIN)
EACH(svd_jacobi, NX_SP_THREADS, NX_SP_PLAIN)
SPECTRAL_ANY(sp_td_stage, NX_SP_THREADS, NX_SP_PLAIN, td_stage)
EACH(td_step, NX_SP_THREADS, NX_SP_PLAIN)
SPECTRAL_ANY(sp_td_hemv, NX_SP_THREADS, NX_SP_PLAIN, td_hemv)
SPECTRAL_ANY(sp_td_diag, NX_SP_THREADS, NX_SP_PLAIN, td_diag)
SPECTRAL_ANY(sp_td_vcopy, NX_SP_THREADS, NX_SP_PLAIN, td_vcopy)
SPECTRAL_ANY(sp_larft, NX_SP_WAVE, NX_SP_PLAIN, larft)
SPECTRAL_ANY(sp_td_zinit, NX_SP_THREADS, NX_SP_PLAIN, td_zinit)
SPECTRAL_ANY(sp_td_out, NX_SP_THREADS, NX_SP_PLAIN, td_out)
SPECTRAL_REAL(sp_dc_leaves, NX_SP_THREADS, NX_SP_PLAIN, dc_leaves)
SPECTRAL_REAL(sp_dc_scatter, NX_SP_THREADS, NX_SP_PLAIN, dc_scatter)
EACH_REAL(dc_deflate, NX_SP_THREADS, NX_SP_PLAIN)
EACH_REAL(dc_secular, NX_SP_HALF, NX_SP_PLAIN)
SPECTRAL_REAL(sp_dc_product, NX_SP_THREADS, NX_SP_PLAIN, dc_product)
SPECTRAL_REAL(sp_dc_sort, NX_SP_THREADS, NX_SP_PLAIN, dc_sort)
SPECTRAL_REAL(sp_tri_bisect, NX_SP_HALF, NX_SP_PLAIN, tri_bisect)
SPECTRAL_ANY(sp_bd_stage, NX_SP_THREADS, NX_SP_WORDS, bd_stage)
EACH(bd_left, NX_SP_THREADS, NX_SP_WORDS)
SPECTRAL_ANY(sp_bd_wh, NX_SP_THREADS, NX_SP_WORDS, bd_wh)
SPECTRAL_ANY(sp_bd_updl, NX_SP_THREADS, NX_SP_WORDS, bd_updl)
EACH(bd_right, NX_SP_THREADS, NX_SP_WORDS)
SPECTRAL_ANY(sp_bd_wn, NX_SP_HALF, NX_SP_WORDS, bd_wn)
SPECTRAL_ANY(sp_bd_updr, NX_SP_THREADS, NX_SP_WORDS, bd_updr)
SPECTRAL_ANY(sp_bd_tgk, NX_SP_THREADS, NX_SP_WORDS, bd_tgk)
SPECTRAL_ANY(sp_bd_values, NX_SP_THREADS, NX_SP_WORDS, bd_values)
SPECTRAL_ANY(sp_bd_ginit, NX_SP_THREADS, NX_SP_WORDS, bd_ginit)
EACH(bd_gen, NX_SP_WAVE, NX_SP_WORDS)
SPECTRAL_ANY(sp_bd_apply, NX_SP_HALF, NX_SP_WORDS, bd_apply)
SPECTRAL_ANY(sp_bd_xinit, NX_SP_THREADS, NX_SP_WORDS, bd_xinit)
SPECTRAL_ANY(sp_bd_vcopy, NX_SP_THREADS, NX_SP_WORDS, bd_vcopy)
SPECTRAL_ANY(sp_bd_rank, NX_SP_THREADS, NX_SP_WORDS, bd_rank)
SPECTRAL_ANY(sp_bd_out, NX_SP_THREADS, NX_SP_WORDS, bd_out)
SPECTRAL(sp_eig_schur_f32, NX_SP_HALF, NX_SP_PLAIN, eig_schur, f32)
SPECTRAL(sp_eig_schur_f64, NX_SP_EIG_DOUBLE, NX_SP_PLAIN, eig_schur, f64)
SPECTRAL(sp_eig_schur_c64, NX_SP_HALF, NX_SP_PLAIN, eig_schur, c64)
SPECTRAL(sp_eig_schur_c128, NX_SP_EIG_DOUBLE, NX_SP_PLAIN, eig_schur, c128)
SPECTRAL_ANY(sp_eig_norms, NX_SP_HALF, NX_SP_PLAIN, eig_norms)
EACH(eig_vectors, NX_SP_HALF, NX_SP_PLAIN)
SPECTRAL_ANY(sp_eig_finish, NX_SP_HALF, NX_SP_PLAIN, eig_finish)
SPECTRAL_ANY(sp_eye, NX_SP_THREADS, NX_SP_PLAIN, eye)

