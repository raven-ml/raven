/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* matmul: the products of the matrices of two operands, as nx.cpu computes
   them (cpu/nx_c_matmul.c). MATMUL(float32) defines the kernel of float32.

   Each output element is +0 plus the products along k, in increasing k, at
   the dtype's compute type: floats narrower than binary32 widen to binary32
   and round once at the store, floats fuse each product into the running sum,
   and integers multiply and add in 64 bits, so they wrap. An output whose
   products sum to exactly zero, or that has none, is +0.

   A workgroup computes a tile of TM x TN outputs of one matrix of the batch,
   its threads TM / 16 x TN / 16 outputs each, walking k by TK through tiles
   of both operands in the workgroup's memory. The grid's x covers the
   columns, its y the rows and its z the batch.

   - [s (dst, a, b, mmeta)]: [dst] C-contiguous, [a]'s element (i, k) of
     matrix z at [a_off + place_a(z) + i * a_rs + k * a_cs], [b]'s alike,
     where place_a(z) is the batch index z's place in [bext] at [astr]. */

#include "common.h"

enum { TM = 64, TN = 64, TK = 16, SIDE = 16, RM = TM / SIDE, RN = TN / SIDE };

struct mmeta {
  i64 m, n, k, brank;
  i64 a_off, a_rs, a_cs, b_off, b_rs, b_cs;
  i64 bext[MAX_RANK], astr[MAX_RANK], bstr[MAX_RANK];
};

/* Where batch index [z] lies in extents [ext] at strides [str]. */
DEVICE i64 place(i64 z, i64 rank, const i64 *ext, const i64 *str) {
  i64 o = 0;
  for (i64 d = rank - 1; d >= 0; d--) {
    o += (z % ext[d]) * str[d];
    z /= ext[d];
  }
  return o;
}

/* [acc] plus the product of [a] and [b]. */
DEVICE float madd(float a, float b, float acc) {
  return __builtin_fmaf(a, b, acc);
}
DEVICE double madd(double a, double b, double acc) {
  return __builtin_fma(a, b, acc);
}
DEVICE int64_t madd(int64_t a, int64_t b, int64_t acc) {
  return (int64_t)((uint64_t)a * (uint64_t)b + (uint64_t)acc);
}
DEVICE uint64_t madd(uint64_t a, uint64_t b, uint64_t acc) {
  return a * b + acc;
}

template <class D>
DEVICE void product(typename D::storage *dst, const typename D::storage *a,
                    const typename D::storage *b, const mmeta &m) {
  typedef typename D::compute T;
  /* The operands' tiles, k-major; [as]'s rows are padded by one so that the
     threads storing a column of it reach distinct banks. */
  __attribute__((shared)) static T as[TK][TM + 1];
  __attribute__((shared)) static T bs[TK][TN];
  int t = __builtin_amdgcn_workitem_id_x();
  int tx = t % SIDE, ty = t / SIDE;
  i64 i0 = (i64)__builtin_amdgcn_workgroup_id_y() * TM;
  i64 j0 = (i64)__builtin_amdgcn_workgroup_id_x() * TN;
  i64 z = __builtin_amdgcn_workgroup_id_z();
  const typename D::storage *az = a + m.a_off + place(z, m.brank, m.bext, m.astr);
  const typename D::storage *bz = b + m.b_off + place(z, m.brank, m.bext, m.bstr);
  T acc[RM][RN];
  for (int r = 0; r < RM; r++)
    for (int c = 0; c < RN; c++) acc[r][c] = 0;
  for (i64 k0 = 0; k0 < m.k; k0 += TK) {
    /* Elements past an edge load as 0: past k, both operands' are 0, so their
       products are 0. */
    for (int e = t; e < TM * TK; e += THREADS) {
      int row = e / TK, kk = e % TK;
      i64 i = i0 + row, k = k0 + kk;
      as[kk][row] = i < m.m && k < m.k ? D::load(az[i * m.a_rs + k * m.a_cs]) : 0;
    }
    for (int e = t; e < TK * TN; e += THREADS) {
      int kk = e / TN, col = e % TN;
      i64 k = k0 + kk, j = j0 + col;
      bs[kk][col] = k < m.k && j < m.n ? D::load(bz[k * m.b_rs + j * m.b_cs]) : 0;
    }
    barrier();
    for (int kk = 0; kk < TK; kk++) {
      T x[RM], y[RN];
      for (int r = 0; r < RM; r++) x[r] = as[kk][ty + SIDE * r];
      for (int c = 0; c < RN; c++) y[c] = bs[kk][tx + SIDE * c];
      for (int r = 0; r < RM; r++)
        for (int c = 0; c < RN; c++) acc[r][c] = madd(x[r], y[c], acc[r][c]);
    }
    barrier();
  }
  typename D::storage *dz = dst + z * m.m * m.n;
  for (int r = 0; r < RM; r++) {
    i64 i = i0 + ty + SIDE * r;
    if (i >= m.m) continue;
    for (int c = 0; c < RN; c++) {
      i64 j = j0 + tx + SIDE * c;
      if (j < m.n) dz[i * m.n + j] = store<D>(acc[r][c]);
    }
  }
}

#define MATMUL(D)                                                              \
  KERNEL void NAME(s)(D::storage *dst, const D::storage *a,                    \
                      const D::storage *b, mmeta m) {                          \
    product<D>(dst, a, b, m);                                                  \
  }
