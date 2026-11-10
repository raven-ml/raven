/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The spectral routines' steps: eigh, svd and eig, batched, at float32,
   float64, complex64 and complex128 (kernels.h, Spectral routines).
   spectral.ml plans them, with the contraction's FMA kernels for their
   products.

   sp_jacobi.cuh: both extents at most 32, a matrix in shared memory:
     two-sided cyclic Jacobi (eigh), one-sided (svd).
   sp_tridiag.cuh, sp_stedc.cuh: eigh past it, xHETRD's blocked
     tridiagonalisation, then divide and conquer (xSTEDC, leaves of 32 by
     the Jacobi kernel) or, for values alone, bisection.
   sp_bd.cuh: svd past it, xGEBD2's bidiagonalisation, then bisection on
     the Golub-Kahan matrix for values alone, or QR sweeps (xBDSQR)
     recorded by one thread a matrix and applied 8 a pass.
   sp_eig.cuh: eig, one block a matrix: balancing, Hessenberg reduction,
     Francis QR (xLAHQR) and back substitution (xTREVC3).

   Each entry reads its function's parameters from the words of an
   sp_args, in order: an address, an integer, or a size. */

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace nx_spectral {

#include "sp_jacobi.cuh"
#undef NT
#undef MAX_SWEEPS

#include "sp_tridiag.cuh"
#undef NB

#include "sp_stedc.cuh"
#undef LEAF
#undef Q
#undef DL

#include "sp_bd.cuh"
#undef S
#undef G
#undef ARGS

#include "sp_eig.cuh"
#undef KEXSH

/* x[z] = the r x c identity's first rows and columns: an svd's unitary
   factors where k = 0. */
template <class R> __device__ __noinline__ void eye(R *x, int64_t nz, int r, int c, int P) {
  const int64_t per = (int64_t)r * c;
  for (int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; q < nz * per;
       q += (int64_t)gridDim.x * blockDim.x) {
    const int64_t k = q % per;
    x[q * P] = (R)(k / c == k % c);
    if (P == 2) x[q * P + 1] = 0;
  }
}

/* A step of parts alone: its real type, and its element's parts. */
#define PARTS typename real_of<T>::R
#define NPARTS (real_of<T>::cplx ? 2 : 1)

/* A word as the parameter it stands for. */
struct word {
  uint64_t v;
  template <class P> D operator P *() const { return (P *)v; }
  D operator int() const { return (int)v; }
  D operator long() const { return (long)v; }
  D operator unsigned long() const { return (unsigned long)v; }
};

#define W(k) word{p.w[k]}

template <class T> D void sp_eigh_jacobi(const sp_args &p) {
  eigh_jacobi<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_svd_jacobi(const sp_args &p) {
  svd_jacobi<T>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), W(8));
}

template <class T> D void sp_td_stage(const sp_args &p) {
  td_stage<PARTS>(W(0), W(1), W(2), W(3), NPARTS);
}

template <class T> D void sp_td_step(const sp_args &p) {
  td_step<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_td_hemv(const sp_args &p) {
  td_hemv<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_td_diag(const sp_args &p) {
  td_diag<PARTS>(W(0), W(1), W(2), NPARTS);
}

template <class T> D void sp_td_vcopy(const sp_args &p) {
  td_vcopy<PARTS>(W(0), W(1), W(2), W(3), W(4), NPARTS);
}

template <class T> D void sp_td_larft(const sp_args &p) {
  td_larft<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_td_zinit(const sp_args &p) {
  td_zinit<PARTS>(W(0), W(1), W(2), W(3), W(4), NPARTS);
}

template <class T> D void sp_td_out(const sp_args &p) {
  td_out<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), W(8), W(9), NPARTS);
}

template <class R> D void sp_dc_leaves(const sp_args &p) {
  dc_leaves<R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7));
}

template <class R> D void sp_dc_scatter(const sp_args &p) {
  dc_scatter<R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6));
}

template <class R> D void sp_dc_deflate(const sp_args &p) {
  dc_deflate<R>(W(0), W(1), W(2), W(3));
}

template <class R> D void sp_dc_secular(const sp_args &p) {
  dc_secular<R>(W(0), W(1), W(2), W(3), W(4));
}

template <class R> D void sp_dc_product(const sp_args &p) {
  dc_product<R>(W(0), W(1), W(2), W(3));
}

template <class R> D void sp_dc_sort(const sp_args &p) {
  dc_sort<R>(W(0), W(1), W(2), W(3));
}

template <class R> D void sp_tri_bisect(const sp_args &p) {
  tri_bisect<R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7));
}

template <class T> D void sp_bd_stage(const sp_args &p) {
  bd_stage<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), NPARTS);
}

template <class T> D void sp_bd_left(const sp_args &p) {
  bd_left<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_wh(const sp_args &p) {
  bd_wh<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_updl(const sp_args &p) {
  bd_updl<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_right(const sp_args &p) {
  bd_right<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_wn(const sp_args &p) {
  bd_wn<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_updr(const sp_args &p) {
  bd_updr<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_bd_tgk(const sp_args &p) {
  bd_tgk<typename real_of<T>::R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), (int)sizeof(T));
}

template <class T> D void sp_bd_values(const sp_args &p) {
  bd_values<typename real_of<T>::R>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), (int)sizeof(T));
}

template <class T> D void sp_bd_ginit(const sp_args &p) {
  bd_ginit<typename real_of<T>::R>(W(0), W(1), W(2), W(3), W(4), (int)sizeof(T));
}

template <class T> D void sp_bd_gen(const sp_args &p) {
  bd_gen<T>(W(0), W(1), W(2), W(3), W(4));
}

template <class T> D void sp_bd_apply(const sp_args &p) {
  bd_apply<typename real_of<T>::R>(W(0), W(1), W(2), W(3), W(4), (int)sizeof(T));
}

template <class T> D void sp_bd_xinit(const sp_args &p) {
  bd_xinit<PARTS>(W(0), W(1), W(2), W(3), W(4), NPARTS);
}

template <class T> D void sp_bd_vcopy(const sp_args &p) {
  bd_vcopy<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), NPARTS);
}

template <class T> D void sp_bd_larft(const sp_args &p) {
  bd_larft<T>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7));
}

template <class T> D void sp_bd_rank(const sp_args &p) {
  bd_rank<typename real_of<T>::R>(W(0), W(1), W(2), W(3), W(4), (int)sizeof(T));
}

template <class T> D void sp_bd_out(const sp_args &p) {
  bd_out<PARTS>(W(0), W(1), W(2), W(3), W(4), W(5), W(6), W(7), W(8), W(9), W(10), W(11), NPARTS);
}

template <class T> D void sp_eig_schur(const sp_args &p) {
  eig_schur<T>(W(0), W(1), W(2), W(3), W(4), W(5));
}

template <class T> D void sp_eig_norms(const sp_args &p) {
  eig_norms<T>(W(0), W(1), W(2));
}

template <class T> D void sp_eig_vectors(const sp_args &p) {
  eig_vectors<T>(W(0), W(1), W(2));
}

template <class T> D void sp_eig_finish(const sp_args &p) {
  eig_finish<PARTS>(W(0), W(1), W(2), W(3), W(4), real_of<T>::cplx);
}

template <class T> D void sp_eye(const sp_args &p) {
  eye<PARTS>(W(0), W(1), W(2), W(3), NPARTS);
}

#undef W
#undef PARTS
#undef NPARTS

} /* namespace nx_spectral */

#undef D
#undef M

/* Entries */

#define SP_T_f32 float
#define SP_T_f64 double
#define SP_T_c64 nx_spectral::cf
#define SP_T_c128 nx_spectral::cd

#define SPECTRAL(name, entry, fmt)                                            \
  extern "C" __global__ void __launch_bounds__(256)                           \
      name(const __grid_constant__ sp_args p) {                               \
    nx_spectral::sp_##entry<SP_T_##fmt>(p);                                   \
  }

/* One entry for every format, or every real one, by the word NX_SP_FMT. */
#define SPECTRAL_ANY(name, entry)                                             \
  extern "C" __global__ void __launch_bounds__(256)                           \
      name(const __grid_constant__ sp_args p) {                               \
    switch (p.w[NX_SP_FMT]) {                                                 \
    case NX_FMT_F32: nx_spectral::sp_##entry<SP_T_f32>(p); break;             \
    case NX_FMT_F64: nx_spectral::sp_##entry<SP_T_f64>(p); break;             \
    case NX_FMT_C64: nx_spectral::sp_##entry<SP_T_c64>(p); break;             \
    default: nx_spectral::sp_##entry<SP_T_c128>(p);                           \
    }                                                                         \
  }

#define SPECTRAL_REAL(name, entry)                                            \
  extern "C" __global__ void __launch_bounds__(256)                           \
      name(const __grid_constant__ sp_args p) {                               \
    if (p.w[NX_SP_FMT] == NX_FMT_F32) nx_spectral::sp_##entry<SP_T_f32>(p);   \
    else nx_spectral::sp_##entry<SP_T_f64>(p);                                \
  }
