/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// QR decomposition implementations
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <complex.h>
#include <float.h>
#include <lapacke.h>

#include "nx_c_shared.h"

// Machine epsilon for float32 and float64
#define NX_EPS32 FLT_EPSILON
#define NX_EPS64 DBL_EPSILON

// Helper functions for packing/unpacking matrices
static void nx_pack_f32(float* dst, const float* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_f32(float* dst, const float* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void nx_pack_f64(double* dst, const double* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_f64(double* dst, const double* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void nx_pack_c32(complex32* dst, const complex32* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_c32(complex32* dst, const complex32* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void nx_pack_c64(complex64* dst, const complex64* src, int m, int n,
                        int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = src[i * stride_row + j * stride_col];
    }
  }
}

static void nx_unpack_c64(complex64* dst, const complex64* src, int m, int n,
                          int stride_row, int stride_col) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * stride_row + j * stride_col] = src[i * n + j];
    }
  }
}

static void qr_decompose_float32(float* a, float* q, float* r, int m, int n,
                                 int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  const int minmn = m < n ? m : n;
  const int lda = k > n ? k : n;  // Leading dimension must be >= max(k, n)

  // LAPACK destroys the input matrix, so we need to make a copy with proper size
  float* a_copy = (float*)calloc(m * lda, sizeof(float));
  if (!a_copy) return;

  // Copy input matrix to a_copy (only the m×n part)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a_copy[i * lda + j] = a[i * n + j];
    }
  }

  // Allocate workspace for Householder reflectors
  float* tau = (float*)malloc(minmn * sizeof(float));
  if (!tau) {
    free(a_copy);
    return;
  }

  // Step 1: QR factorization using Householder reflectors
  lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Extract R from the upper triangular part of a_copy
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      r[i * n + j] = (i <= j) ? a_copy[i * lda + j] : 0.0f;
    }
  }

  // Step 2: Generate Q from the Householder reflectors
  info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, k, minmn, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Copy Q to the output (only the first k columns)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      q[i * k + j] = a_copy[i * lda + j];
    }
  }

  free(a_copy);
  free(tau);
}

static void qr_decompose_float64(double* a, double* q, double* r, int m, int n,
                                 int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  const int minmn = m < n ? m : n;
  const int lda = k > n ? k : n;  // Leading dimension must be >= max(k, n)

  // LAPACK destroys the input matrix, so we need to make a copy with proper size
  double* a_copy = (double*)calloc(m * lda, sizeof(double));
  if (!a_copy) return;

  // Copy input matrix to a_copy (only the m×n part)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a_copy[i * lda + j] = a[i * n + j];
    }
  }

  // Allocate workspace for Householder reflectors
  double* tau = (double*)malloc(minmn * sizeof(double));
  if (!tau) {
    free(a_copy);
    return;
  }

  // Step 1: QR factorization using Householder reflectors
  lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Extract R from the upper triangular part of a_copy
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      r[i * n + j] = (i <= j) ? a_copy[i * lda + j] : 0.0;
    }
  }

  // Step 2: Generate Q from the Householder reflectors
  info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, k, minmn, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Copy Q to the output (only the first k columns)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      q[i * k + j] = a_copy[i * lda + j];
    }
  }

  free(a_copy);
  free(tau);
}

static void qr_decompose_complex32(complex32* a, complex32* q, complex32* r,
                                   int m, int n, int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  const int minmn = m < n ? m : n;
  const int lda = k > n ? k : n;  // Leading dimension must be >= max(k, n)

  // LAPACK destroys the input matrix, so we need to make a copy with proper size
  complex32* a_copy = (complex32*)calloc(m * lda, sizeof(complex32));
  if (!a_copy) return;

  // Copy input matrix to a_copy (only the m×n part)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a_copy[i * lda + j] = a[i * n + j];
    }
  }

  // Allocate workspace for Householder reflectors
  complex32* tau = (complex32*)malloc(minmn * sizeof(complex32));
  if (!tau) {
    free(a_copy);
    return;
  }

  // Step 1: QR factorization using Householder reflectors
  lapack_int info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Extract R from the upper triangular part of a_copy
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      r[i * n + j] = (i <= j) ? a_copy[i * lda + j] : 0.0f + 0.0f * I;
    }
  }

  // Step 2: Generate Q from the Householder reflectors
  info = LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, k, minmn, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Copy Q to the output (only the first k columns)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      q[i * k + j] = a_copy[i * lda + j];
    }
  }

  free(a_copy);
  free(tau);
}

static void qr_decompose_complex64(complex64* a, complex64* q, complex64* r,
                                   int m, int n, int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  const int minmn = m < n ? m : n;
  const int lda = k > n ? k : n;  // Leading dimension must be >= max(k, n)

  // LAPACK destroys the input matrix, so we need to make a copy with proper size
  complex64* a_copy = (complex64*)calloc(m * lda, sizeof(complex64));
  if (!a_copy) return;

  // Copy input matrix to a_copy (only the m×n part)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a_copy[i * lda + j] = a[i * n + j];
    }
  }

  // Allocate workspace for Householder reflectors
  complex64* tau = (complex64*)malloc(minmn * sizeof(complex64));
  if (!tau) {
    free(a_copy);
    return;
  }

  // Step 1: QR factorization using Householder reflectors
  lapack_int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Extract R from the upper triangular part of a_copy
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      r[i * n + j] = (i <= j) ? a_copy[i * lda + j] : 0.0 + 0.0 * I;
    }
  }

  // Step 2: Generate Q from the Householder reflectors
  info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, k, minmn, a_copy, lda, tau);
  if (info != 0) {
    free(a_copy);
    free(tau);
    return;
  }

  // Copy Q to the output (only the first k columns)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      q[i * k + j] = a_copy[i * lda + j];
    }
  }

  free(a_copy);
  free(tau);
}

static int qr_decompose_float16(uint16_t* a, uint16_t* q, uint16_t* r, int m,
                                int n, int reduced) {
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int k = reduced ? (m < n ? m : n) : m;
  float* q_float = (float*)malloc(m * k * sizeof(float));
  float* r_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !q_float || !r_float) {
    free(a_float);
    free(q_float);
    free(r_float);
    return -1;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = half_to_float(a[i]);
  qr_decompose_float32(a_float, q_float, r_float, m, n, reduced);
  for (int i = 0; i < m * k; i++) q[i] = float_to_half(q_float[i]);
  for (int i = 0; i < m * n; i++) r[i] = float_to_half(r_float[i]);
  free(a_float);
  free(q_float);
  free(r_float);
  return 0;
}

static int qr_decompose_bfloat16(caml_ba_bfloat16* a, caml_ba_bfloat16* q,
                                 caml_ba_bfloat16* r, int m, int n,
                                 int reduced) {
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int k = reduced ? (m < n ? m : n) : m;
  float* q_float = (float*)malloc(m * k * sizeof(float));
  float* r_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !q_float || !r_float) {
    free(a_float);
    free(q_float);
    free(r_float);
    return -1;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = bfloat16_to_float(a[i]);
  qr_decompose_float32(a_float, q_float, r_float, m, n, reduced);
  for (int i = 0; i < m * k; i++) q[i] = float_to_bfloat16(q_float[i]);
  for (int i = 0; i < m * n; i++) r[i] = float_to_bfloat16(r_float[i]);
  free(a_float);
  free(q_float);
  free(r_float);
  return 0;
}

static int qr_decompose_f8e4m3(caml_ba_fp8_e4m3* a, caml_ba_fp8_e4m3* q,
                               caml_ba_fp8_e4m3* r, int m, int n, int reduced) {
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int k = reduced ? (m < n ? m : n) : m;
  float* q_float = (float*)malloc(m * k * sizeof(float));
  float* r_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !q_float || !r_float) {
    free(a_float);
    free(q_float);
    free(r_float);
    return -1;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = fp8_e4m3_to_float(a[i]);
  qr_decompose_float32(a_float, q_float, r_float, m, n, reduced);
  for (int i = 0; i < m * k; i++) q[i] = float_to_fp8_e4m3(q_float[i]);
  for (int i = 0; i < m * n; i++) r[i] = float_to_fp8_e4m3(r_float[i]);
  free(a_float);
  free(q_float);
  free(r_float);
  return 0;
}

static int qr_decompose_f8e5m2(caml_ba_fp8_e5m2* a, caml_ba_fp8_e5m2* q,
                               caml_ba_fp8_e5m2* r, int m, int n, int reduced) {
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int k = reduced ? (m < n ? m : n) : m;
  float* q_float = (float*)malloc(m * k * sizeof(float));
  float* r_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !q_float || !r_float) {
    free(a_float);
    free(q_float);
    free(r_float);
    return -1;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = fp8_e5m2_to_float(a[i]);
  qr_decompose_float32(a_float, q_float, r_float, m, n, reduced);
  for (int i = 0; i < m * k; i++) q[i] = float_to_fp8_e5m2(q_float[i]);
  for (int i = 0; i < m * n; i++) r[i] = float_to_fp8_e5m2(r_float[i]);
  free(a_float);
  free(q_float);
  free(r_float);
  return 0;
}


CAMLprim value caml_nx_op_qr(value v_in, value v_q, value v_r,
                             value v_reduced) {
  CAMLparam4(v_in, v_q, v_r, v_reduced);
  int reduced = Int_val(v_reduced);
  ndarray_t in = extract_ndarray(v_in);
  ndarray_t q_nd = extract_ndarray(v_q);
  ndarray_t r_nd = extract_ndarray(v_r);
  struct caml_ba_array* ba_in = Caml_ba_array_val(Field(v_in, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_q = Caml_ba_array_val(Field(v_q, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_r = Caml_ba_array_val(Field(v_r, FFI_TENSOR_DATA));
  int kind = nx_ba_get_kind(ba_in);
  if (in.ndim < 2) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&q_nd);
    cleanup_ndarray(&r_nd);
    caml_failwith("qr: input must have at least 2 dimensions");
  }
  int m = in.shape[in.ndim - 2];
  int n = in.shape[in.ndim - 1];
  int k = reduced ? (m < n ? m : n) : m;
  int rows_r = reduced ? k : m;
  int batch_size = 1;
  for (int i = 0; i < in.ndim - 2; i++) {
    batch_size *= in.shape[i];
  }
  int s_in_row = in.strides[in.ndim - 2];
  int s_in_col = in.strides[in.ndim - 1];
  int s_q_row = q_nd.strides[q_nd.ndim - 2];
  int s_q_col = q_nd.strides[q_nd.ndim - 1];
  int s_r_row = r_nd.strides[r_nd.ndim - 2];
  int s_r_col = r_nd.strides[r_nd.ndim - 1];
  caml_enter_blocking_section();
  for (int b = 0; b < batch_size; b++) {
    size_t off_in = in.offset;
    size_t off_q = q_nd.offset;
    size_t off_r = r_nd.offset;
    if (in.ndim > 2) {
      int remaining = b;
      for (int i = in.ndim - 3; i >= 0; i--) {
        int coord = remaining % in.shape[i];
        remaining /= in.shape[i];
        off_in += coord * in.strides[i];
        off_q += coord * q_nd.strides[i];
        off_r += coord * r_nd.strides[i];
      }
    }
    int status = 0;
    switch (kind) {
      case CAML_BA_FLOAT32: {
        float* base_in = (float*)ba_in->data + off_in;
        float* base_q = (float*)ba_q->data + off_q;
        float* base_r = (float*)ba_r->data + off_r;
        float* A = (float*)malloc((size_t)m * n * sizeof(float));
        float* Q = (float*)malloc((size_t)m * k * sizeof(float));
        float* R = (float*)malloc((size_t)m * n * sizeof(float));
        nx_pack_f32(A, base_in, m, n, s_in_row, s_in_col);
        qr_decompose_float32(A, Q, R, m, n, reduced);
        nx_unpack_f32(base_q, Q, m, k, s_q_row, s_q_col);
        nx_unpack_f32(base_r, R, rows_r, n, s_r_row, s_r_col);
        free(A);
        free(Q);
        free(R);
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_in = (double*)ba_in->data + off_in;
        double* base_q = (double*)ba_q->data + off_q;
        double* base_r = (double*)ba_r->data + off_r;
        double* A = (double*)malloc((size_t)m * n * sizeof(double));
        double* Q = (double*)malloc((size_t)m * k * sizeof(double));
        double* R = (double*)malloc((size_t)m * n * sizeof(double));
        nx_pack_f64(A, base_in, m, n, s_in_row, s_in_col);
        qr_decompose_float64(A, Q, R, m, n, reduced);
        nx_unpack_f64(base_q, Q, m, k, s_q_row, s_q_col);
        nx_unpack_f64(base_r, R, rows_r, n, s_r_row, s_r_col);
        free(A);
        free(Q);
        free(R);
        break;
      }
      case CAML_BA_COMPLEX32: {
        complex32* base_in = (complex32*)ba_in->data + off_in;
        complex32* base_q = (complex32*)ba_q->data + off_q;
        complex32* base_r = (complex32*)ba_r->data + off_r;
        complex32* A = (complex32*)malloc((size_t)m * n * sizeof(complex32));
        complex32* Q = (complex32*)malloc((size_t)m * k * sizeof(complex32));
        complex32* R = (complex32*)malloc((size_t)m * n * sizeof(complex32));
        nx_pack_c32(A, base_in, m, n, s_in_row, s_in_col);
        qr_decompose_complex32(A, Q, R, m, n, reduced);
        nx_unpack_c32(base_q, Q, m, k, s_q_row, s_q_col);
        nx_unpack_c32(base_r, R, rows_r, n, s_r_row, s_r_col);
        free(A);
        free(Q);
        free(R);
        break;
      }
      case CAML_BA_COMPLEX64: {
        complex64* base_in = (complex64*)ba_in->data + off_in;
        complex64* base_q = (complex64*)ba_q->data + off_q;
        complex64* base_r = (complex64*)ba_r->data + off_r;
        complex64* A = (complex64*)malloc((size_t)m * n * sizeof(complex64));
        complex64* Q = (complex64*)malloc((size_t)m * k * sizeof(complex64));
        complex64* R = (complex64*)malloc((size_t)m * n * sizeof(complex64));
        nx_pack_c64(A, base_in, m, n, s_in_row, s_in_col);
        qr_decompose_complex64(A, Q, R, m, n, reduced);
        nx_unpack_c64(base_q, Q, m, k, s_q_row, s_q_col);
        nx_unpack_c64(base_r, R, rows_r, n, s_r_row, s_r_col);
        free(A);
        free(Q);
        free(R);
        break;
      }
      case CAML_BA_FLOAT16: {
        uint16_t* base_in = (uint16_t*)ba_in->data + off_in;
        uint16_t* base_q = (uint16_t*)ba_q->data + off_q;
        uint16_t* base_r = (uint16_t*)ba_r->data + off_r;
        uint16_t* A = (uint16_t*)malloc((size_t)m * n * sizeof(uint16_t));
        uint16_t* Q = (uint16_t*)malloc((size_t)m * k * sizeof(uint16_t));
        uint16_t* R = (uint16_t*)malloc((size_t)m * n * sizeof(uint16_t));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = qr_decompose_float16(A, Q, R, m, n, reduced);
        if (status == 0) {
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
              base_q[i * s_q_row + j * s_q_col] = Q[i * k + j];
            }
          }
          for (int i = 0; i < rows_r; i++) {
            for (int j = 0; j < n; j++) {
              base_r[i * s_r_row + j * s_r_col] = R[i * n + j];
            }
          }
        }
        free(A);
        free(Q);
        free(R);
        break;
      }
      case NX_BA_BFLOAT16: {
        caml_ba_bfloat16* base_in = (caml_ba_bfloat16*)ba_in->data + off_in;
        caml_ba_bfloat16* base_q = (caml_ba_bfloat16*)ba_q->data + off_q;
        caml_ba_bfloat16* base_r = (caml_ba_bfloat16*)ba_r->data + off_r;
        caml_ba_bfloat16* A =
            (caml_ba_bfloat16*)malloc((size_t)m * n * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* Q =
            (caml_ba_bfloat16*)malloc((size_t)m * k * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* R =
            (caml_ba_bfloat16*)malloc((size_t)m * n * sizeof(caml_ba_bfloat16));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = qr_decompose_bfloat16(A, Q, R, m, n, reduced);
        if (status == 0) {
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
              base_q[i * s_q_row + j * s_q_col] = Q[i * k + j];
            }
          }
          for (int i = 0; i < rows_r; i++) {
            for (int j = 0; j < n; j++) {
              base_r[i * s_r_row + j * s_r_col] = R[i * n + j];
            }
          }
        }
        free(A);
        free(Q);
        free(R);
        break;
      }
      case NX_BA_FP8_E4M3: {
        caml_ba_fp8_e4m3* base_in = (caml_ba_fp8_e4m3*)ba_in->data + off_in;
        caml_ba_fp8_e4m3* base_q = (caml_ba_fp8_e4m3*)ba_q->data + off_q;
        caml_ba_fp8_e4m3* base_r = (caml_ba_fp8_e4m3*)ba_r->data + off_r;
        caml_ba_fp8_e4m3* A =
            (caml_ba_fp8_e4m3*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* Q =
            (caml_ba_fp8_e4m3*)malloc((size_t)m * k * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* R =
            (caml_ba_fp8_e4m3*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e4m3));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = qr_decompose_f8e4m3(A, Q, R, m, n, reduced);
        if (status == 0) {
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
              base_q[i * s_q_row + j * s_q_col] = Q[i * k + j];
            }
          }
          for (int i = 0; i < rows_r; i++) {
            for (int j = 0; j < n; j++) {
              base_r[i * s_r_row + j * s_r_col] = R[i * n + j];
            }
          }
        }
        free(A);
        free(Q);
        free(R);
        break;
      }
      case NX_BA_FP8_E5M2: {
        caml_ba_fp8_e5m2* base_in = (caml_ba_fp8_e5m2*)ba_in->data + off_in;
        caml_ba_fp8_e5m2* base_q = (caml_ba_fp8_e5m2*)ba_q->data + off_q;
        caml_ba_fp8_e5m2* base_r = (caml_ba_fp8_e5m2*)ba_r->data + off_r;
        caml_ba_fp8_e5m2* A =
            (caml_ba_fp8_e5m2*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* Q =
            (caml_ba_fp8_e5m2*)malloc((size_t)m * k * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* R =
            (caml_ba_fp8_e5m2*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e5m2));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = qr_decompose_f8e5m2(A, Q, R, m, n, reduced);
        if (status == 0) {
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
              base_q[i * s_q_row + j * s_q_col] = Q[i * k + j];
            }
          }
          for (int i = 0; i < rows_r; i++) {
            for (int j = 0; j < n; j++) {
              base_r[i * s_r_row + j * s_r_col] = R[i * n + j];
            }
          }
        }
        free(A);
        free(Q);
        free(R);
        break;
      }
      default:
        caml_leave_blocking_section();
        cleanup_ndarray(&in);
        cleanup_ndarray(&q_nd);
        cleanup_ndarray(&r_nd);
        caml_failwith("qr: unsupported dtype");
    }
    if (status != 0) {
      caml_leave_blocking_section();
      cleanup_ndarray(&in);
      cleanup_ndarray(&q_nd);
      cleanup_ndarray(&r_nd);
      caml_failwith("qr: decomposition failed");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&in);
  cleanup_ndarray(&q_nd);
  cleanup_ndarray(&r_nd);
  CAMLreturn(Val_unit);
}
