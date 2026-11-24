/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The spectral routines' steps: eigh, svd and eig, batched, at float32
   and complex64 (kernels.h, Spectral routines). The host plans them,
   with the contraction's products between them.

   sp_jacobi.h: both extents at most 32, a matrix in threadgroup memory:
     two-sided cyclic Jacobi (eigh), one-sided (svd).
   sp_tridiag.h, sp_stedc.h: eigh past it, xHETRD's blocked
     tridiagonalisation, then divide and conquer (xSTEDC, leaves of 32 by
     the Jacobi kernel) or, for values alone, bisection.
   sp_bd.h: svd past it, bidiagonalisation (xGEBD2), then bisection on
     the Golub-Kahan tridiagonal or, with vectors, implicit QR (xBDSQR).
   sp_eig.h: eig, balancing, Hessenberg, Schur by QR (xLAHQR) and back
     substitution (xTREVC3).

   Each entry reads its step's parameters from its words, a struct at
   buffer 0 whose fields kernels.h's enumerations index.

   This follows nx.amd's spectral.hip. */

#include "sp_bd.h"
#include "sp_eig.h"
#include "sp_jacobi.h"
#include "sp_stedc.h"
#include "sp_tridiag.h"

/* Words */

struct sp_eigh_jacobi_words {
  device const uchar *a;
  device float *w;
  device uchar *v;
  long nb, n, per, sweeps, first, count;
};
NX_FIELD(sp_eigh_jacobi_words, a, NX_SP_EIGH_JACOBI_A);
NX_FIELD(sp_eigh_jacobi_words, w, NX_SP_EIGH_JACOBI_W);
NX_FIELD(sp_eigh_jacobi_words, v, NX_SP_EIGH_JACOBI_V);
NX_FIELD(sp_eigh_jacobi_words, nb, NX_SP_EIGH_JACOBI_NB);
NX_FIELD(sp_eigh_jacobi_words, n, NX_SP_EIGH_JACOBI_N);
NX_FIELD(sp_eigh_jacobi_words, per, NX_SP_EIGH_JACOBI_PER);
NX_FIELD(sp_eigh_jacobi_words, sweeps, NX_SP_EIGH_JACOBI_SWEEPS);
NX_FIELD(sp_eigh_jacobi_words, first, NX_SP_EIGH_JACOBI_FIRST);
NX_FIELD(sp_eigh_jacobi_words, count, NX_SP_EIGH_JACOBI_COUNT);
NX_SIZE(sp_eigh_jacobi_words, NX_SP_EIGH_JACOBI_WORDS);

struct sp_svd_jacobi_words {
  device const uchar *a;
  device float *s;
  device uchar *u, *vh;
  long nb, m, n, uc, vr, per, span, sweeps, first, count;
};
NX_FIELD(sp_svd_jacobi_words, a, NX_SP_SVD_JACOBI_A);
NX_FIELD(sp_svd_jacobi_words, s, NX_SP_SVD_JACOBI_S);
NX_FIELD(sp_svd_jacobi_words, u, NX_SP_SVD_JACOBI_U);
NX_FIELD(sp_svd_jacobi_words, vh, NX_SP_SVD_JACOBI_VH);
NX_FIELD(sp_svd_jacobi_words, nb, NX_SP_SVD_JACOBI_NB);
NX_FIELD(sp_svd_jacobi_words, m, NX_SP_SVD_JACOBI_M);
NX_FIELD(sp_svd_jacobi_words, n, NX_SP_SVD_JACOBI_N);
NX_FIELD(sp_svd_jacobi_words, uc, NX_SP_SVD_JACOBI_UC);
NX_FIELD(sp_svd_jacobi_words, vr, NX_SP_SVD_JACOBI_VR);
NX_FIELD(sp_svd_jacobi_words, per, NX_SP_SVD_JACOBI_PER);
NX_FIELD(sp_svd_jacobi_words, span, NX_SP_SVD_JACOBI_SPAN);
NX_FIELD(sp_svd_jacobi_words, sweeps, NX_SP_SVD_JACOBI_SWEEPS);
NX_FIELD(sp_svd_jacobi_words, first, NX_SP_SVD_JACOBI_FIRST);
NX_FIELD(sp_svd_jacobi_words, count, NX_SP_SVD_JACOBI_COUNT);
NX_SIZE(sp_svd_jacobi_words, NX_SP_SVD_JACOBI_WORDS);

struct sp_td_stage_words {
  device const float *src;
  device uchar *ws;
  long n, parts, per, phase, du, dn, first, count;
};
NX_FIELD(sp_td_stage_words, src, NX_SP_TD_STAGE_SRC);
NX_FIELD(sp_td_stage_words, ws, NX_SP_TD_STAGE_WS);
NX_FIELD(sp_td_stage_words, n, NX_SP_TD_STAGE_N);
NX_FIELD(sp_td_stage_words, parts, NX_SP_TD_STAGE_PARTS);
NX_FIELD(sp_td_stage_words, per, NX_SP_TD_STAGE_PER);
NX_FIELD(sp_td_stage_words, phase, NX_SP_TD_STAGE_PHASE);
NX_FIELD(sp_td_stage_words, du, NX_SP_TD_STAGE_DU);
NX_FIELD(sp_td_stage_words, dn, NX_SP_TD_STAGE_DN);
NX_FIELD(sp_td_stage_words, first, NX_SP_TD_STAGE_FIRST);
NX_FIELD(sp_td_stage_words, count, NX_SP_TD_STAGE_COUNT);
NX_SIZE(sp_td_stage_words, NX_SP_TD_STAGE_WORDS);

struct sp_td_step_words {
  device uchar *ws;
  long n, j0, nbk, i, per, first, count;
};
NX_FIELD(sp_td_step_words, ws, NX_SP_TD_STEP_WS);
NX_FIELD(sp_td_step_words, n, NX_SP_TD_STEP_N);
NX_FIELD(sp_td_step_words, j0, NX_SP_TD_STEP_J0);
NX_FIELD(sp_td_step_words, nbk, NX_SP_TD_STEP_NBK);
NX_FIELD(sp_td_step_words, i, NX_SP_TD_STEP_I);
NX_FIELD(sp_td_step_words, per, NX_SP_TD_STEP_PER);
NX_FIELD(sp_td_step_words, first, NX_SP_TD_STEP_FIRST);
NX_FIELD(sp_td_step_words, count, NX_SP_TD_STEP_COUNT);
NX_SIZE(sp_td_step_words, NX_SP_TD_STEP_WORDS);

struct sp_td_hemv_words {
  device uchar *ws;
  long n, c, i, parts, per, first, count;
};
NX_FIELD(sp_td_hemv_words, ws, NX_SP_TD_HEMV_WS);
NX_FIELD(sp_td_hemv_words, n, NX_SP_TD_HEMV_N);
NX_FIELD(sp_td_hemv_words, c, NX_SP_TD_HEMV_C);
NX_FIELD(sp_td_hemv_words, i, NX_SP_TD_HEMV_I);
NX_FIELD(sp_td_hemv_words, parts, NX_SP_TD_HEMV_PARTS);
NX_FIELD(sp_td_hemv_words, per, NX_SP_TD_HEMV_PER);
NX_FIELD(sp_td_hemv_words, first, NX_SP_TD_HEMV_FIRST);
NX_FIELD(sp_td_hemv_words, count, NX_SP_TD_HEMV_COUNT);
NX_SIZE(sp_td_hemv_words, NX_SP_TD_HEMV_WORDS);

struct sp_td_diag_words {
  device uchar *ws;
  long n, parts, per, first, count;
};
NX_FIELD(sp_td_diag_words, ws, NX_SP_TD_DIAG_WS);
NX_FIELD(sp_td_diag_words, n, NX_SP_TD_DIAG_N);
NX_FIELD(sp_td_diag_words, parts, NX_SP_TD_DIAG_PARTS);
NX_FIELD(sp_td_diag_words, per, NX_SP_TD_DIAG_PER);
NX_FIELD(sp_td_diag_words, first, NX_SP_TD_DIAG_FIRST);
NX_FIELD(sp_td_diag_words, count, NX_SP_TD_DIAG_COUNT);
NX_SIZE(sp_td_diag_words, NX_SP_TD_DIAG_WORDS);

struct sp_td_vcopy_words {
  device uchar *ws;
  long n, j0, nbk, parts, per, first, count;
};
NX_FIELD(sp_td_vcopy_words, ws, NX_SP_TD_VCOPY_WS);
NX_FIELD(sp_td_vcopy_words, n, NX_SP_TD_VCOPY_N);
NX_FIELD(sp_td_vcopy_words, j0, NX_SP_TD_VCOPY_J0);
NX_FIELD(sp_td_vcopy_words, nbk, NX_SP_TD_VCOPY_NBK);
NX_FIELD(sp_td_vcopy_words, parts, NX_SP_TD_VCOPY_PARTS);
NX_FIELD(sp_td_vcopy_words, per, NX_SP_TD_VCOPY_PER);
NX_FIELD(sp_td_vcopy_words, first, NX_SP_TD_VCOPY_FIRST);
NX_FIELD(sp_td_vcopy_words, count, NX_SP_TD_VCOPY_COUNT);
NX_SIZE(sp_td_vcopy_words, NX_SP_TD_VCOPY_WORDS);

struct sp_td_zinit_words {
  device const float *q;
  device float *z;
  long qs, n, parts, first, count;
};
NX_FIELD(sp_td_zinit_words, q, NX_SP_TD_ZINIT_Q);
NX_FIELD(sp_td_zinit_words, z, NX_SP_TD_ZINIT_Z);
NX_FIELD(sp_td_zinit_words, qs, NX_SP_TD_ZINIT_QS);
NX_FIELD(sp_td_zinit_words, n, NX_SP_TD_ZINIT_N);
NX_FIELD(sp_td_zinit_words, parts, NX_SP_TD_ZINIT_PARTS);
NX_FIELD(sp_td_zinit_words, first, NX_SP_TD_ZINIT_FIRST);
NX_FIELD(sp_td_zinit_words, count, NX_SP_TD_ZINIT_COUNT);
NX_SIZE(sp_td_zinit_words, NX_SP_TD_ZINIT_WORDS);

struct sp_td_out_words {
  device uchar *ws;
  device const float *d;
  device const int *dcx;
  device const float *z;
  device float *w, *v;
  long ds, es, n, parts, per, first, count;
};
NX_FIELD(sp_td_out_words, ws, NX_SP_TD_OUT_WS);
NX_FIELD(sp_td_out_words, d, NX_SP_TD_OUT_D);
NX_FIELD(sp_td_out_words, dcx, NX_SP_TD_OUT_DCX);
NX_FIELD(sp_td_out_words, z, NX_SP_TD_OUT_Z);
NX_FIELD(sp_td_out_words, w, NX_SP_TD_OUT_W);
NX_FIELD(sp_td_out_words, v, NX_SP_TD_OUT_V);
NX_FIELD(sp_td_out_words, ds, NX_SP_TD_OUT_DS);
NX_FIELD(sp_td_out_words, es, NX_SP_TD_OUT_ES);
NX_FIELD(sp_td_out_words, n, NX_SP_TD_OUT_N);
NX_FIELD(sp_td_out_words, parts, NX_SP_TD_OUT_PARTS);
NX_FIELD(sp_td_out_words, per, NX_SP_TD_OUT_PER);
NX_FIELD(sp_td_out_words, first, NX_SP_TD_OUT_FIRST);
NX_FIELD(sp_td_out_words, count, NX_SP_TD_OUT_COUNT);
NX_SIZE(sp_td_out_words, NX_SP_TD_OUT_WORDS);

struct sp_dc_leaves_words {
  device uchar *ws;
  device const float *d, *e;
  device float *full, *tail;
  long stride, n, per, first, count;
};
NX_FIELD(sp_dc_leaves_words, ws, NX_SP_DC_LEAVES_WS);
NX_FIELD(sp_dc_leaves_words, d, NX_SP_DC_LEAVES_D);
NX_FIELD(sp_dc_leaves_words, e, NX_SP_DC_LEAVES_E);
NX_FIELD(sp_dc_leaves_words, full, NX_SP_DC_LEAVES_FULL);
NX_FIELD(sp_dc_leaves_words, tail, NX_SP_DC_LEAVES_TAIL);
NX_FIELD(sp_dc_leaves_words, stride, NX_SP_DC_LEAVES_STRIDE);
NX_FIELD(sp_dc_leaves_words, n, NX_SP_DC_LEAVES_N);
NX_FIELD(sp_dc_leaves_words, per, NX_SP_DC_LEAVES_PER);
NX_FIELD(sp_dc_leaves_words, first, NX_SP_DC_LEAVES_FIRST);
NX_FIELD(sp_dc_leaves_words, count, NX_SP_DC_LEAVES_COUNT);
NX_SIZE(sp_dc_leaves_words, NX_SP_DC_LEAVES_WORDS);

struct sp_dc_scatter_words {
  device uchar *ws;
  device const float *wf, *vf, *wt, *vt;
  long n, per, first, count;
};
NX_FIELD(sp_dc_scatter_words, ws, NX_SP_DC_SCATTER_WS);
NX_FIELD(sp_dc_scatter_words, wf, NX_SP_DC_SCATTER_WF);
NX_FIELD(sp_dc_scatter_words, vf, NX_SP_DC_SCATTER_VF);
NX_FIELD(sp_dc_scatter_words, wt, NX_SP_DC_SCATTER_WT);
NX_FIELD(sp_dc_scatter_words, vt, NX_SP_DC_SCATTER_VT);
NX_FIELD(sp_dc_scatter_words, n, NX_SP_DC_SCATTER_N);
NX_FIELD(sp_dc_scatter_words, per, NX_SP_DC_SCATTER_PER);
NX_FIELD(sp_dc_scatter_words, first, NX_SP_DC_SCATTER_FIRST);
NX_FIELD(sp_dc_scatter_words, count, NX_SP_DC_SCATTER_COUNT);
NX_SIZE(sp_dc_scatter_words, NX_SP_DC_SCATTER_WORDS);

struct sp_merge_words {
  device uchar *ws;
  long n, lev, per, tx, ty, du, dr, first, count;
};
NX_FIELD(sp_merge_words, ws, NX_SP_MERGE_WS);
NX_FIELD(sp_merge_words, n, NX_SP_MERGE_N);
NX_FIELD(sp_merge_words, lev, NX_SP_MERGE_LEV);
NX_FIELD(sp_merge_words, per, NX_SP_MERGE_PER);
NX_FIELD(sp_merge_words, tx, NX_SP_MERGE_TX);
NX_FIELD(sp_merge_words, ty, NX_SP_MERGE_TY);
NX_FIELD(sp_merge_words, du, NX_SP_MERGE_DU);
NX_FIELD(sp_merge_words, dr, NX_SP_MERGE_DR);
NX_FIELD(sp_merge_words, first, NX_SP_MERGE_FIRST);
NX_FIELD(sp_merge_words, count, NX_SP_MERGE_COUNT);
NX_SIZE(sp_merge_words, NX_SP_MERGE_WORDS);

struct sp_dc_deflate_words {
  device uchar *ws;
  long n, lev, per, q, du, first, count;
};
NX_FIELD(sp_dc_deflate_words, ws, NX_SP_DC_DEFLATE_WS);
NX_FIELD(sp_dc_deflate_words, n, NX_SP_DC_DEFLATE_N);
NX_FIELD(sp_dc_deflate_words, lev, NX_SP_DC_DEFLATE_LEV);
NX_FIELD(sp_dc_deflate_words, per, NX_SP_DC_DEFLATE_PER);
NX_FIELD(sp_dc_deflate_words, q, NX_SP_DC_DEFLATE_Q);
NX_FIELD(sp_dc_deflate_words, du, NX_SP_DC_DEFLATE_DU);
NX_FIELD(sp_dc_deflate_words, first, NX_SP_DC_DEFLATE_FIRST);
NX_FIELD(sp_dc_deflate_words, count, NX_SP_DC_DEFLATE_COUNT);
NX_SIZE(sp_dc_deflate_words, NX_SP_DC_DEFLATE_WORDS);

struct sp_dc_secular_words {
  device uchar *ws;
  long n, lev, phase, steps, per, q, start, first, count;
};
NX_FIELD(sp_dc_secular_words, ws, NX_SP_DC_SECULAR_WS);
NX_FIELD(sp_dc_secular_words, n, NX_SP_DC_SECULAR_N);
NX_FIELD(sp_dc_secular_words, lev, NX_SP_DC_SECULAR_LEV);
NX_FIELD(sp_dc_secular_words, phase, NX_SP_DC_SECULAR_PHASE);
NX_FIELD(sp_dc_secular_words, steps, NX_SP_DC_SECULAR_STEPS);
NX_FIELD(sp_dc_secular_words, per, NX_SP_DC_SECULAR_PER);
NX_FIELD(sp_dc_secular_words, q, NX_SP_DC_SECULAR_Q);
NX_FIELD(sp_dc_secular_words, start, NX_SP_DC_SECULAR_START);
NX_FIELD(sp_dc_secular_words, first, NX_SP_DC_SECULAR_FIRST);
NX_FIELD(sp_dc_secular_words, count, NX_SP_DC_SECULAR_COUNT);
NX_SIZE(sp_dc_secular_words, NX_SP_DC_SECULAR_WORDS);

struct sp_larft_words {
  device uchar *tm;
  device const uchar *g, *tau;
  long nbk, parts, per, first, count;
};
NX_FIELD(sp_larft_words, tm, NX_SP_LARFT_TM);
NX_FIELD(sp_larft_words, g, NX_SP_LARFT_G);
NX_FIELD(sp_larft_words, tau, NX_SP_LARFT_TAU);
NX_FIELD(sp_larft_words, nbk, NX_SP_LARFT_NBK);
NX_FIELD(sp_larft_words, parts, NX_SP_LARFT_PARTS);
NX_FIELD(sp_larft_words, per, NX_SP_LARFT_PER);
NX_FIELD(sp_larft_words, first, NX_SP_LARFT_FIRST);
NX_FIELD(sp_larft_words, count, NX_SP_LARFT_COUNT);
NX_SIZE(sp_larft_words, NX_SP_LARFT_WORDS);

struct sp_tri_bounds_words {
  device const float *d, *e;
  device float *bd;
  long stride, n, first, count;
};
NX_FIELD(sp_tri_bounds_words, d, NX_SP_TRI_BOUNDS_D);
NX_FIELD(sp_tri_bounds_words, e, NX_SP_TRI_BOUNDS_E);
NX_FIELD(sp_tri_bounds_words, bd, NX_SP_TRI_BOUNDS_BD);
NX_FIELD(sp_tri_bounds_words, stride, NX_SP_TRI_BOUNDS_STRIDE);
NX_FIELD(sp_tri_bounds_words, n, NX_SP_TRI_BOUNDS_N);
NX_FIELD(sp_tri_bounds_words, first, NX_SP_TRI_BOUNDS_FIRST);
NX_FIELD(sp_tri_bounds_words, count, NX_SP_TRI_BOUNDS_COUNT);
NX_SIZE(sp_tri_bounds_words, NX_SP_TRI_BOUNDS_WORDS);

struct sp_tri_bisect_words {
  device const float *d, *e, *bd;
  device int *st;
  device float *w;
  long stride, n, i0, cnt, q, start, first, count;
};
NX_FIELD(sp_tri_bisect_words, d, NX_SP_TRI_BISECT_D);
NX_FIELD(sp_tri_bisect_words, e, NX_SP_TRI_BISECT_E);
NX_FIELD(sp_tri_bisect_words, bd, NX_SP_TRI_BISECT_BD);
NX_FIELD(sp_tri_bisect_words, st, NX_SP_TRI_BISECT_ST);
NX_FIELD(sp_tri_bisect_words, w, NX_SP_TRI_BISECT_W);
NX_FIELD(sp_tri_bisect_words, stride, NX_SP_TRI_BISECT_STRIDE);
NX_FIELD(sp_tri_bisect_words, n, NX_SP_TRI_BISECT_N);
NX_FIELD(sp_tri_bisect_words, i0, NX_SP_TRI_BISECT_I0);
NX_FIELD(sp_tri_bisect_words, cnt, NX_SP_TRI_BISECT_CNT);
NX_FIELD(sp_tri_bisect_words, q, NX_SP_TRI_BISECT_Q);
NX_FIELD(sp_tri_bisect_words, start, NX_SP_TRI_BISECT_START);
NX_FIELD(sp_tri_bisect_words, first, NX_SP_TRI_BISECT_FIRST);
NX_FIELD(sp_tri_bisect_words, count, NX_SP_TRI_BISECT_COUNT);
NX_SIZE(sp_tri_bisect_words, NX_SP_TRI_BISECT_WORDS);

struct sp_eye_words {
  device float *x;
  long nz, r, c, parts;
};
NX_FIELD(sp_eye_words, x, NX_SP_EYE_X);
NX_FIELD(sp_eye_words, nz, NX_SP_EYE_NZ);
NX_FIELD(sp_eye_words, r, NX_SP_EYE_R);
NX_FIELD(sp_eye_words, c, NX_SP_EYE_C);
NX_FIELD(sp_eye_words, parts, NX_SP_EYE_PARTS);
NX_SIZE(sp_eye_words, NX_SP_EYE_WORDS);

NX_FIELD(bd_layout, per, NX_SP_BDL_PER);
NX_FIELD(bd_layout, g, NX_SP_BDL_G);
NX_FIELD(bd_layout, x, NX_SP_BDL_X);
NX_FIELD(bd_layout, y, NX_SP_BDL_Y);
NX_FIELD(bd_layout, vc, NX_SP_BDL_VC);
NX_FIELD(bd_layout, wv, NX_SP_BDL_WV);
NX_FIELD(bd_layout, tauq, NX_SP_BDL_TAUQ);
NX_FIELD(bd_layout, taup, NX_SP_BDL_TAUP);
NX_FIELD(bd_layout, d, NX_SP_BDL_D);
NX_FIELD(bd_layout, e, NX_SP_BDL_E);
NX_FIELD(bd_layout, ub, NX_SP_BDL_UB);
NX_FIELD(bd_layout, vb, NX_SP_BDL_VB);
NX_FIELD(bd_layout, rec, NX_SP_BDL_REC);
NX_FIELD(bd_layout, sc, NX_SP_BDL_SC);
NX_FIELD(bd_layout, sw, NX_SP_BDL_SW);
NX_FIELD(bd_layout, st, NX_SP_BDL_ST);
NX_FIELD(bd_layout, cap, NX_SP_BDL_CAP);
NX_FIELD(bd_layout, pt, NX_SP_BDL_PT);
NX_SIZE(bd_layout, NX_SP_BDL_WORDS);

struct sp_bd_stage_words {
  device const float *a;
  device uchar *ws;
  long M, K, rows, cols, parts, phase, du, dm;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_stage_words, a, NX_SP_BD_STAGE_A);
NX_FIELD(sp_bd_stage_words, ws, NX_SP_BD_STAGE_WS);
NX_FIELD(sp_bd_stage_words, M, NX_SP_BD_STAGE_M);
NX_FIELD(sp_bd_stage_words, K, NX_SP_BD_STAGE_K);
NX_FIELD(sp_bd_stage_words, rows, NX_SP_BD_STAGE_ROWS);
NX_FIELD(sp_bd_stage_words, cols, NX_SP_BD_STAGE_COLS);
NX_FIELD(sp_bd_stage_words, parts, NX_SP_BD_STAGE_PARTS);
NX_FIELD(sp_bd_stage_words, phase, NX_SP_BD_STAGE_PHASE);
NX_FIELD(sp_bd_stage_words, du, NX_SP_BD_STAGE_DU);
NX_FIELD(sp_bd_stage_words, dm, NX_SP_BD_STAGE_DM);
NX_FIELD(sp_bd_stage_words, l, NX_SP_BD_STAGE_LAYOUT);
NX_FIELD(sp_bd_stage_words, first, NX_SP_BD_STAGE_FIRST);
NX_FIELD(sp_bd_stage_words, count, NX_SP_BD_STAGE_COUNT);
NX_SIZE(sp_bd_stage_words, NX_SP_BD_STAGE_WORDS);

struct sp_bd_col_words {
  device uchar *ws;
  long M, K, c, parts;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_col_words, ws, NX_SP_BD_COL_WS);
NX_FIELD(sp_bd_col_words, M, NX_SP_BD_COL_M);
NX_FIELD(sp_bd_col_words, K, NX_SP_BD_COL_K);
NX_FIELD(sp_bd_col_words, c, NX_SP_BD_COL_C);
NX_FIELD(sp_bd_col_words, parts, NX_SP_BD_COL_PARTS);
NX_FIELD(sp_bd_col_words, l, NX_SP_BD_COL_LAYOUT);
NX_FIELD(sp_bd_col_words, first, NX_SP_BD_COL_FIRST);
NX_FIELD(sp_bd_col_words, count, NX_SP_BD_COL_COUNT);
NX_SIZE(sp_bd_col_words, NX_SP_BD_COL_WORDS);

struct sp_bd_chunk_words {
  device uchar *ws;
  long M, K, c, parts, phase, chunks, dchunks, dcols;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_chunk_words, ws, NX_SP_BD_CHUNK_WS);
NX_FIELD(sp_bd_chunk_words, M, NX_SP_BD_CHUNK_M);
NX_FIELD(sp_bd_chunk_words, K, NX_SP_BD_CHUNK_K);
NX_FIELD(sp_bd_chunk_words, c, NX_SP_BD_CHUNK_C);
NX_FIELD(sp_bd_chunk_words, parts, NX_SP_BD_CHUNK_PARTS);
NX_FIELD(sp_bd_chunk_words, phase, NX_SP_BD_CHUNK_PHASE);
NX_FIELD(sp_bd_chunk_words, chunks, NX_SP_BD_CHUNK_CHUNKS);
NX_FIELD(sp_bd_chunk_words, dchunks, NX_SP_BD_CHUNK_DCHUNKS);
NX_FIELD(sp_bd_chunk_words, dcols, NX_SP_BD_CHUNK_DCOLS);
NX_FIELD(sp_bd_chunk_words, l, NX_SP_BD_CHUNK_LAYOUT);
NX_FIELD(sp_bd_chunk_words, first, NX_SP_BD_CHUNK_FIRST);
NX_FIELD(sp_bd_chunk_words, count, NX_SP_BD_CHUNK_COUNT);
NX_SIZE(sp_bd_chunk_words, NX_SP_BD_CHUNK_WORDS);

struct sp_bd_tgk_words {
  device uchar *ws;
  device float *dt, *et;
  long K;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_tgk_words, ws, NX_SP_BD_TGK_WS);
NX_FIELD(sp_bd_tgk_words, dt, NX_SP_BD_TGK_DT);
NX_FIELD(sp_bd_tgk_words, et, NX_SP_BD_TGK_ET);
NX_FIELD(sp_bd_tgk_words, K, NX_SP_BD_TGK_K);
NX_FIELD(sp_bd_tgk_words, l, NX_SP_BD_TGK_LAYOUT);
NX_FIELD(sp_bd_tgk_words, first, NX_SP_BD_TGK_FIRST);
NX_FIELD(sp_bd_tgk_words, count, NX_SP_BD_TGK_COUNT);
NX_SIZE(sp_bd_tgk_words, NX_SP_BD_TGK_WORDS);

struct sp_bd_values_words {
  device uchar *ws;
  device const float *lam;
  device float *s;
  long K;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_values_words, ws, NX_SP_BD_VALUES_WS);
NX_FIELD(sp_bd_values_words, lam, NX_SP_BD_VALUES_LAM);
NX_FIELD(sp_bd_values_words, s, NX_SP_BD_VALUES_S);
NX_FIELD(sp_bd_values_words, K, NX_SP_BD_VALUES_K);
NX_FIELD(sp_bd_values_words, l, NX_SP_BD_VALUES_LAYOUT);
NX_FIELD(sp_bd_values_words, first, NX_SP_BD_VALUES_FIRST);
NX_FIELD(sp_bd_values_words, count, NX_SP_BD_VALUES_COUNT);
NX_SIZE(sp_bd_values_words, NX_SP_BD_VALUES_WORDS);

struct sp_bd_words {
  device uchar *ws;
  long K, rotations;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_words, ws, NX_SP_BD_WS);
NX_FIELD(sp_bd_words, K, NX_SP_BD_K);
NX_FIELD(sp_bd_words, rotations, NX_SP_BD_ROTATIONS);
NX_FIELD(sp_bd_words, l, NX_SP_BD_LAYOUT);
NX_FIELD(sp_bd_words, first, NX_SP_BD_FIRST);
NX_FIELD(sp_bd_words, count, NX_SP_BD_COUNT);
NX_SIZE(sp_bd_words, NX_SP_BD_WORDS);

struct sp_bd_xinit_words {
  device uchar *ws;
  long M, K, gc, parts;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_xinit_words, ws, NX_SP_BD_XINIT_WS);
NX_FIELD(sp_bd_xinit_words, M, NX_SP_BD_XINIT_M);
NX_FIELD(sp_bd_xinit_words, K, NX_SP_BD_XINIT_K);
NX_FIELD(sp_bd_xinit_words, gc, NX_SP_BD_XINIT_GC);
NX_FIELD(sp_bd_xinit_words, parts, NX_SP_BD_XINIT_PARTS);
NX_FIELD(sp_bd_xinit_words, l, NX_SP_BD_XINIT_LAYOUT);
NX_FIELD(sp_bd_xinit_words, first, NX_SP_BD_XINIT_FIRST);
NX_FIELD(sp_bd_xinit_words, count, NX_SP_BD_XINIT_COUNT);
NX_SIZE(sp_bd_xinit_words, NX_SP_BD_XINIT_WORDS);

struct sp_bd_vcopy_words {
  device uchar *ws;
  long M, K, j0, nbk, side, parts;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_vcopy_words, ws, NX_SP_BD_VCOPY_WS);
NX_FIELD(sp_bd_vcopy_words, M, NX_SP_BD_VCOPY_M);
NX_FIELD(sp_bd_vcopy_words, K, NX_SP_BD_VCOPY_K);
NX_FIELD(sp_bd_vcopy_words, j0, NX_SP_BD_VCOPY_J0);
NX_FIELD(sp_bd_vcopy_words, nbk, NX_SP_BD_VCOPY_NBK);
NX_FIELD(sp_bd_vcopy_words, side, NX_SP_BD_VCOPY_SIDE);
NX_FIELD(sp_bd_vcopy_words, parts, NX_SP_BD_VCOPY_PARTS);
NX_FIELD(sp_bd_vcopy_words, l, NX_SP_BD_VCOPY_LAYOUT);
NX_FIELD(sp_bd_vcopy_words, first, NX_SP_BD_VCOPY_FIRST);
NX_FIELD(sp_bd_vcopy_words, count, NX_SP_BD_VCOPY_COUNT);
NX_SIZE(sp_bd_vcopy_words, NX_SP_BD_VCOPY_WORDS);

struct sp_bd_out_words {
  device uchar *ws;
  device float *s, *u, *vh;
  long M, K, rows, cols, uc, vr, parts;
  bd_layout l;
  long first, count;
};
NX_FIELD(sp_bd_out_words, ws, NX_SP_BD_OUT_WS);
NX_FIELD(sp_bd_out_words, s, NX_SP_BD_OUT_S);
NX_FIELD(sp_bd_out_words, u, NX_SP_BD_OUT_U);
NX_FIELD(sp_bd_out_words, vh, NX_SP_BD_OUT_VH);
NX_FIELD(sp_bd_out_words, M, NX_SP_BD_OUT_M);
NX_FIELD(sp_bd_out_words, K, NX_SP_BD_OUT_K);
NX_FIELD(sp_bd_out_words, rows, NX_SP_BD_OUT_ROWS);
NX_FIELD(sp_bd_out_words, cols, NX_SP_BD_OUT_COLS);
NX_FIELD(sp_bd_out_words, uc, NX_SP_BD_OUT_UC);
NX_FIELD(sp_bd_out_words, vr, NX_SP_BD_OUT_VR);
NX_FIELD(sp_bd_out_words, parts, NX_SP_BD_OUT_PARTS);
NX_FIELD(sp_bd_out_words, l, NX_SP_BD_OUT_LAYOUT);
NX_FIELD(sp_bd_out_words, first, NX_SP_BD_OUT_FIRST);
NX_FIELD(sp_bd_out_words, count, NX_SP_BD_OUT_COUNT);
NX_SIZE(sp_bd_out_words, NX_SP_BD_OUT_WORDS);

struct sp_eig_words {
  device const uchar *a;
  device float *w;
  device const float *zx;
  device float *v;
  device uchar *ws;
  long n, vectors, from, to, itq, q, parts, per, passes, first, count;
};
NX_FIELD(sp_eig_words, a, NX_SP_EIG_A);
NX_FIELD(sp_eig_words, w, NX_SP_EIG_W);
NX_FIELD(sp_eig_words, zx, NX_SP_EIG_ZX);
NX_FIELD(sp_eig_words, v, NX_SP_EIG_V);
NX_FIELD(sp_eig_words, ws, NX_SP_EIG_WS);
NX_FIELD(sp_eig_words, n, NX_SP_EIG_N);
NX_FIELD(sp_eig_words, vectors, NX_SP_EIG_VECTORS);
NX_FIELD(sp_eig_words, from, NX_SP_EIG_FROM);
NX_FIELD(sp_eig_words, to, NX_SP_EIG_TO);
NX_FIELD(sp_eig_words, itq, NX_SP_EIG_ITQ);
NX_FIELD(sp_eig_words, q, NX_SP_EIG_Q);
NX_FIELD(sp_eig_words, parts, NX_SP_EIG_PARTS);
NX_FIELD(sp_eig_words, per, NX_SP_EIG_PER);
NX_FIELD(sp_eig_words, passes, NX_SP_EIG_PASSES);
NX_FIELD(sp_eig_words, first, NX_SP_EIG_FIRST);
NX_FIELD(sp_eig_words, count, NX_SP_EIG_COUNT);
NX_SIZE(sp_eig_words, NX_SP_EIG_WORDS);

/* Entries */

/* A kernel's arguments: its words, its threadgroup memory, and its
   group's place in a one-dimensional grid. */
#define SP_ARGS(words)                                                        \
  constant words &w [[buffer(0)]], threadgroup uchar *lds [[threadgroup(0)]], \
      uint gx [[threadgroup_position_in_grid]],                               \
      uint ngx [[threadgroups_per_grid]],                                     \
      uint item [[thread_index_in_threadgroup]],                              \
      uint nthr [[threads_per_threadgroup]]

/* The group's place: its meeting's word is the first of its memory. */
#define SP_GRP sp::grp{int(item), int(nthr), (threadgroup int *)lds}

#define SP_UNITS long(gx), long(ngx)

NX_KERNEL(NX_SP_THREADS)
void sp_eigh_jacobi_f32(SP_ARGS(sp_eigh_jacobi_words)) {
  sp::eigh_jacobi<float>(SP_GRP, SP_UNITS, lds, (device const float *)w.a, w.w,
                         (device float *)w.v, w.nb, int(w.n), int(w.per),
                         int(w.sweeps), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS)
void sp_eigh_jacobi_c64(SP_ARGS(sp_eigh_jacobi_words)) {
  sp::eigh_jacobi<sp::cf>(SP_GRP, SP_UNITS, lds, (device const sp::cf *)w.a,
                          w.w, (device sp::cf *)w.v, w.nb, int(w.n),
                          int(w.per), int(w.sweeps), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS)
void sp_svd_jacobi_f32(SP_ARGS(sp_svd_jacobi_words)) {
  sp::svd_jacobi<float>(SP_GRP, SP_UNITS, lds, (device const float *)w.a, w.s,
                        (device float *)w.u, (device float *)w.vh, w.nb,
                        int(w.m), int(w.n), int(w.uc), int(w.vr), int(w.per),
                        ulong(w.span), int(w.sweeps), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS)
void sp_svd_jacobi_c64(SP_ARGS(sp_svd_jacobi_words)) {
  sp::svd_jacobi<sp::cf>(SP_GRP, SP_UNITS, lds, (device const sp::cf *)w.a,
                         w.s, (device sp::cf *)w.u, (device sp::cf *)w.vh,
                         w.nb, int(w.m), int(w.n), int(w.uc), int(w.vr),
                         int(w.per), ulong(w.span), int(w.sweeps), w.first,
                         w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_stage(SP_ARGS(sp_td_stage_words)) {
  sp::td_stage(SP_GRP, SP_UNITS, w.src, w.ws, int(w.n), int(w.parts),
               ulong(w.per), int(w.phase), ulong(w.du), ulong(w.dn), w.first,
               w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_step_f32(SP_ARGS(sp_td_step_words)) {
  sp::td_step<float>(SP_GRP, SP_UNITS, lds, w.ws, int(w.n), int(w.j0),
                     int(w.nbk), int(w.i), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_step_c64(SP_ARGS(sp_td_step_words)) {
  sp::td_step<sp::cf>(SP_GRP, SP_UNITS, lds, w.ws, int(w.n), int(w.j0),
                      int(w.nbk), int(w.i), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_hemv(SP_ARGS(sp_td_hemv_words)) {
  if (w.parts == 1)
    sp::td_hemv<float>(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.c), int(w.i),
                       ulong(w.per), w.first, w.count);
  else
    sp::td_hemv<sp::cf>(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.c), int(w.i),
                        ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_diag(SP_ARGS(sp_td_diag_words)) {
  sp::td_diag(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.parts), ulong(w.per),
              w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_vcopy(SP_ARGS(sp_td_vcopy_words)) {
  sp::td_vcopy(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.j0), int(w.nbk),
               int(w.parts), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_zinit(SP_ARGS(sp_td_zinit_words)) {
  sp::td_zinit(SP_GRP, SP_UNITS, w.q, ulong(w.qs), w.z, int(w.n),
               int(w.parts), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_td_out(SP_ARGS(sp_td_out_words)) {
  sp::td_out(SP_GRP, SP_UNITS, w.ws, w.d, ulong(w.ds), w.dcx, ulong(w.es), w.z,
             w.w, w.v, int(w.n), int(w.parts), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_leaves(SP_ARGS(sp_dc_leaves_words)) {
  sp::dc_leaves(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws, w.d, w.e,
                ulong(w.stride), w.full, w.tail, int(w.n), ulong(w.per),
                w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_scatter(SP_ARGS(sp_dc_scatter_words)) {
  sp::dc_scatter(SP_GRP, SP_UNITS, w.ws, w.wf, w.vf, w.wt, w.vt, int(w.n),
                 ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_prepare(SP_ARGS(sp_merge_words)) {
  sp::dc_prepare(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
                 ulong(w.du), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_deflate(SP_ARGS(sp_dc_deflate_words)) {
  sp::dc_deflate(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
                 int(w.q), ulong(w.du), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_rotate(SP_ARGS(sp_merge_words)) {
  sp::dc_rotate(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
                w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_rank(SP_ARGS(sp_merge_words)) {
  sp::dc_rank(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
              w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_gather(SP_ARGS(sp_merge_words)) {
  sp::dc_gather(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
                ulong(w.du), ulong(w.dr), w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_dc_secular(SP_ARGS(sp_dc_secular_words)) {
  sp::dc_secular(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), int(w.phase),
                 int(w.steps), ulong(w.per), int(w.q), w.start != 0, w.first,
                 w.count);
}

NX_KERNEL(NX_SP_THREADS)
void sp_dc_product(constant sp_merge_words &w [[buffer(0)]],
                   threadgroup uchar *lds [[threadgroup(0)]],
                   uint3 gid [[threadgroup_position_in_grid]],
                   uint3 ngrid [[threadgroups_per_grid]],
                   uint item [[thread_index_in_threadgroup]],
                   uint3 tpg [[threads_per_threadgroup]]) {
  const uint nthr = tpg.x;
  sp::dc_product(SP_GRP, gid, ngrid, (threadgroup float *)lds, w.ws, int(w.n),
                 int(w.lev), ulong(w.per), int(w.tx), int(w.ty), w.first,
                 w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_sort(SP_ARGS(sp_merge_words)) {
  sp::dc_sort(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
              ulong(w.du), w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_dc_place(SP_ARGS(sp_merge_words)) {
  sp::dc_place(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.lev), ulong(w.per),
               ulong(w.du), ulong(w.dr), w.first, w.count);
}

NX_KERNEL(NX_SP_WAVE) void sp_larft(SP_ARGS(sp_larft_words)) {
  if (w.parts == 1)
    sp::larft<float>(SP_GRP, SP_UNITS, w.tm, w.g, w.tau, w.first, w.count,
                     int(w.nbk), ulong(w.per));
  else
    sp::larft<sp::cf>(SP_GRP, SP_UNITS, w.tm, w.g, w.tau, w.first, w.count,
                      int(w.nbk), ulong(w.per));
}

NX_KERNEL(NX_SP_HALF) void sp_tri_bounds(SP_ARGS(sp_tri_bounds_words)) {
  sp::tri_bounds(SP_GRP, SP_UNITS, w.d, w.e, ulong(w.stride), w.bd, int(w.n),
                 w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_tri_bisect(SP_ARGS(sp_tri_bisect_words)) {
  sp::tri_bisect(SP_GRP, SP_UNITS, w.d, w.e, ulong(w.stride), w.bd, w.st, w.w,
                 int(w.n), int(w.i0), int(w.cnt), int(w.q), w.start != 0,
                 w.first, w.count);
}

/* svd's bidiagonal steps */

/* x[z] = the r x c identity's first rows and columns: an svd's unitary
   factors where k = 0. */
NX_KERNEL(NX_SP_THREADS) void sp_eye(SP_ARGS(sp_eye_words)) {
  const long per = w.r * w.c, all = w.nz * per;
  const bool small = sp::is_small(all);
  const long nt = long(nthr);
  for (long q = long(gx) * nt + item; q < all; q += long(ngx) * nt) {
    long z = q, k = sp::split(z, per, small);
    const long col = sp::split(k, w.c, small);
    w.x[q * w.parts] = float(k == col);
    if (w.parts == 2) w.x[q * w.parts + 1] = 0;
  }
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_stage(SP_ARGS(sp_bd_stage_words)) {
  sp::bd_stage(SP_GRP, SP_UNITS, w.a, w.ws, w.M, int(w.K), w.rows, w.cols,
               int(w.parts), int(w.phase), ulong(w.du), ulong(w.dm), w.l,
               w.first, w.count);
}

#define SP_COL w.ws, w.M, int(w.K), int(w.c), w.l, w.first, w.count

NX_KERNEL(NX_SP_THREADS) void sp_bd_left_f32(SP_ARGS(sp_bd_col_words)) {
  sp::bd_left<float>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws, w.M,
                     int(w.c), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_left_c64(SP_ARGS(sp_bd_col_words)) {
  sp::bd_left<sp::cf>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws, w.M,
                      int(w.c), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_wh(SP_ARGS(sp_bd_col_words)) {
  if (w.parts == 1)
    sp::bd_wh<float>(SP_GRP, SP_UNITS, (threadgroup float *)lds, SP_COL);
  else
    sp::bd_wh<sp::cf>(SP_GRP, SP_UNITS, (threadgroup float *)lds, SP_COL);
}

/* The steps of a column past a chunk, at the format [parts] names. */
#define SP_BOTH(f, ...)                                                       \
  if (w.parts == 1)                                                           \
    sp::f<float>(__VA_ARGS__);                                                \
  else                                                                        \
    sp::f<sp::cf>(__VA_ARGS__)
#define SP_SH (threadgroup float *)lds

NX_KERNEL(NX_SP_THREADS) void sp_bd_lsum(SP_ARGS(sp_bd_chunk_words)) {
  SP_BOTH(bd_lsum, SP_GRP, SP_UNITS, SP_SH, w.ws, w.M, int(w.c), int(w.phase),
          w.chunks, ulong(w.dchunks), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_lfin(SP_ARGS(sp_bd_chunk_words)) {
  SP_BOTH(bd_lfin, SP_GRP, SP_UNITS, SP_SH, w.ws, w.M, int(w.c), int(w.phase),
          w.chunks, w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_lscale(SP_ARGS(sp_bd_chunk_words)) {
  SP_BOTH(bd_lscale, SP_GRP, SP_UNITS, w.ws, w.M, int(w.c), w.chunks,
          ulong(w.dchunks), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_whc(SP_ARGS(sp_bd_chunk_words)) {
  SP_BOTH(bd_whc, SP_GRP, SP_UNITS, SP_SH, w.ws, w.M, int(w.K), int(w.c),
          w.chunks, ulong(w.dchunks), ulong(w.dcols), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_whfin(SP_ARGS(sp_bd_chunk_words)) {
  SP_BOTH(bd_whfin, SP_GRP, SP_UNITS, SP_SH, w.ws, int(w.K), int(w.c),
          w.chunks, ulong(w.dcols), w.l, w.first, w.count);
}

#undef SP_SH
#undef SP_BOTH

NX_KERNEL(NX_SP_THREADS) void sp_bd_updl(SP_ARGS(sp_bd_col_words)) {
  if (w.parts == 1)
    sp::bd_updl<float>(SP_GRP, SP_UNITS, SP_COL);
  else
    sp::bd_updl<sp::cf>(SP_GRP, SP_UNITS, SP_COL);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_right_f32(SP_ARGS(sp_bd_col_words)) {
  sp::bd_right<float>(SP_GRP, SP_UNITS, (threadgroup float *)lds, SP_COL);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_right_c64(SP_ARGS(sp_bd_col_words)) {
  sp::bd_right<sp::cf>(SP_GRP, SP_UNITS, (threadgroup float *)lds, SP_COL);
}

NX_KERNEL(NX_SP_HALF) void sp_bd_wn(SP_ARGS(sp_bd_col_words)) {
  if (w.parts == 1)
    sp::bd_wn<float>(SP_GRP, SP_UNITS, SP_COL);
  else
    sp::bd_wn<sp::cf>(SP_GRP, SP_UNITS, SP_COL);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_updr(SP_ARGS(sp_bd_col_words)) {
  if (w.parts == 1)
    sp::bd_updr<float>(SP_GRP, SP_UNITS, SP_COL);
  else
    sp::bd_updr<sp::cf>(SP_GRP, SP_UNITS, SP_COL);
}

#undef SP_COL

NX_KERNEL(NX_SP_THREADS) void sp_bd_tgk(SP_ARGS(sp_bd_tgk_words)) {
  sp::bd_tgk(SP_GRP, SP_UNITS, w.ws, int(w.K), w.dt, w.et, w.l, w.first,
             w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_values(SP_ARGS(sp_bd_values_words)) {
  sp::bd_values(SP_GRP, SP_UNITS, w.ws, int(w.K), w.lam, w.s, w.l, w.first,
                w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_ginit(SP_ARGS(sp_bd_words)) {
  sp::bd_ginit(SP_GRP, SP_UNITS, w.ws, int(w.K), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_WAVE) void sp_bd_gen(SP_ARGS(sp_bd_words)) {
  sp::bd_gen(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws, int(w.K),
             int(w.rotations), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_bd_apply(SP_ARGS(sp_bd_words)) {
  sp::bd_apply(SP_GRP, SP_UNITS, w.ws, int(w.K), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_xinit(SP_ARGS(sp_bd_xinit_words)) {
  sp::bd_xinit(SP_GRP, SP_UNITS, w.ws, w.M, int(w.K), int(w.gc), int(w.parts),
               w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_vcopy(SP_ARGS(sp_bd_vcopy_words)) {
  sp::bd_vcopy(SP_GRP, SP_UNITS, w.ws, w.M, int(w.K), int(w.j0), int(w.nbk),
               int(w.side), int(w.parts), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_rank(SP_ARGS(sp_bd_words)) {
  sp::bd_rank(SP_GRP, SP_UNITS, w.ws, int(w.K), w.l, w.first, w.count);
}

NX_KERNEL(NX_SP_THREADS) void sp_bd_out(SP_ARGS(sp_bd_out_words)) {
  sp::bd_out(SP_GRP, SP_UNITS, w.ws, w.M, int(w.K), w.s, w.u, w.vh, w.rows,
             w.cols, int(w.uc), int(w.vr), int(w.parts), w.l, w.first,
             w.count);
}

/* eig's steps */

/* Balancing's steps. in, big and scale take a float of threadgroup
   memory a thread; iso and rows a float, then an int. */
#define SP_EIG_BAL(T, sfx)                                                    \
  NX_KERNEL(NX_SP_HALF) void sp_eig_in_##sfx(SP_ARGS(sp_eig_words)) {        \
    sp::eig_in<T>(SP_GRP, SP_UNITS, (threadgroup float *)lds,                 \
                  (device const T *)w.a, w.ws, int(w.n), ulong(w.per),        \
                  w.first, w.count);                                          \
  }                                                                           \
  NX_KERNEL(NX_SP_HALF) void sp_eig_count_##sfx(SP_ARGS(sp_eig_words)) {     \
    sp::eig_count<T>(SP_GRP, SP_UNITS, w.ws, int(w.n), ulong(w.per),          \
                     w.first, w.count);                                       \
  }                                                                           \
  NX_KERNEL(NX_SP_HALF) void sp_eig_colcnt_##sfx(SP_ARGS(sp_eig_words)) {    \
    sp::eig_colcnt<T>(SP_GRP, SP_UNITS, w.ws, int(w.n), ulong(w.per),         \
                      w.first, w.count);                                      \
  }                                                                           \
  NX_KERNEL(NX_SP_HALF) void sp_eig_iso_##sfx(SP_ARGS(sp_eig_words)) {       \
    sp::eig_iso<T>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws,          \
                   int(w.n), ulong(w.per), int(w.from), int(w.q), w.first,    \
                   w.count);                                                  \
  }                                                                           \
  NX_KERNEL(NX_SP_HALF) void sp_eig_rows_##sfx(SP_ARGS(sp_eig_words)) {      \
    sp::eig_rows<T>(SP_GRP, SP_UNITS, (threadgroup float *)lds,               \
                    (threadgroup int *)(lds + 4 * NX_SP_HALF), w.ws,          \
                    int(w.n), ulong(w.per), int(w.passes), int(w.q),          \
                    w.first, w.count);                                        \
  }                                                                           \
  NX_KERNEL(NX_SP_HALF) void sp_eig_big_##sfx(SP_ARGS(sp_eig_words)) {       \
    sp::eig_big<T>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws,          \
                   int(w.n), ulong(w.per), w.first, w.count);                 \
  }                                                                           \
  NX_KERNEL(NX_SP_HALF) void sp_eig_scale_##sfx(SP_ARGS(sp_eig_words)) {     \
    sp::eig_scale<T>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws,        \
                     int(w.vectors), int(w.n), ulong(w.per), w.first,         \
                     w.count);                                                \
  }

SP_EIG_BAL(float, f32)
SP_EIG_BAL(sp::cf, c64)

#undef SP_EIG_BAL

NX_KERNEL(NX_SP_HALF) void sp_eig_hess_f32(SP_ARGS(sp_eig_words)) {
  sp::eig_hess<float>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws,
                      int(w.n), int(w.from), int(w.to), ulong(w.per), w.first,
                      w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_hess_c64(SP_ARGS(sp_eig_words)) {
  sp::eig_hess<sp::cf>(SP_GRP, SP_UNITS, (threadgroup float *)lds, w.ws,
                       int(w.n), int(w.from), int(w.to), ulong(w.per),
                       w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_orghr_f32(SP_ARGS(sp_eig_words)) {
  sp::eig_orghr<float>(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.from),
                       int(w.to), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_orghr_c64(SP_ARGS(sp_eig_words)) {
  sp::eig_orghr<sp::cf>(SP_GRP, SP_UNITS, w.ws, int(w.n), int(w.from),
                        int(w.to), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_qr_f32(SP_ARGS(sp_eig_words)) {
  sp::eig_qr<float>(SP_GRP, SP_UNITS, w.ws, int(w.vectors), int(w.n),
                    int(w.itq), int(w.q), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_qr_c64(SP_ARGS(sp_eig_words)) {
  sp::eig_qr<sp::cf>(SP_GRP, SP_UNITS, w.ws, int(w.vectors), int(w.n),
                     int(w.itq), int(w.q), ulong(w.per), w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_w_f32(SP_ARGS(sp_eig_words)) {
  sp::eig_w<float>(SP_GRP, SP_UNITS, w.w, w.ws, int(w.n), ulong(w.per),
                   w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_w_c64(SP_ARGS(sp_eig_words)) {
  sp::eig_w<sp::cf>(SP_GRP, SP_UNITS, w.w, w.ws, int(w.n), ulong(w.per),
                    w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_norms_f32(SP_ARGS(sp_eig_words)) {
  sp::eig_norms<float>(SP_GRP, SP_UNITS, w.ws, int(w.n), ulong(w.per),
                       w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_norms_c64(SP_ARGS(sp_eig_words)) {
  sp::eig_norms<sp::cf>(SP_GRP, SP_UNITS, w.ws, int(w.n), ulong(w.per),
                        w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_vectors_f32(SP_ARGS(sp_eig_words)) {
  sp::eig_vectors<float>(SP_GRP, SP_UNITS, w.ws, int(w.n), ulong(w.per),
                         w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_vectors_c64(SP_ARGS(sp_eig_words)) {
  sp::eig_vectors<sp::cf>(SP_GRP, SP_UNITS, w.ws, int(w.n), ulong(w.per),
                          w.first, w.count);
}

NX_KERNEL(NX_SP_HALF) void sp_eig_finish(SP_ARGS(sp_eig_words)) {
  sp::eig_finish(SP_GRP, SP_UNITS, w.zx, w.v, w.ws, int(w.n), w.parts == 2,
                 ulong(w.per), w.first, w.count);
}
