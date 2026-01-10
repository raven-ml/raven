/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// Eigenvalue decomposition implementations
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

// Math helper functions
static inline float sign_float32(float x) {
  return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

static inline double sign_float64(double x) {
  return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
}

static inline float hypot_float32(float x, float y) {
  return hypotf(x, y);
}

static inline double hypot_float64(double x, double y) {
  return hypot(x, y);
}

// Givens rotation computation
static void givens_float32(float f, float g, float *c, float *s) {
  if (g == 0.0f) {
    *c = 1.0f;
    *s = 0.0f;
  } else if (fabsf(g) > fabsf(f)) {
    float t = f / g;
    float tt = hypotf(1.0f, t);
    *c = 1.0f / tt;
    *s = t * (*c);
  } else {
    float t = g / f;
    float tt = hypotf(1.0f, t);
    *s = 1.0f / tt;
    *c = t * (*s);
  }
}

static void givens_float64(double f, double g, double *c, double *s) {
  if (g == 0.0) {
    *c = 1.0;
    *s = 0.0;
  } else if (fabs(g) > fabs(f)) {
    double t = f / g;
    double tt = hypot(1.0, t);
    *c = 1.0 / tt;
    *s = t * (*c);
  } else {
    double t = g / f;
    double tt = hypot(1.0, t);
    *s = 1.0 / tt;
    *c = t * (*s);
  }
}

// Apply Givens rotation from the right
static void apply_givens_right_float32(float *a, int m, int n, int i, int j,
                                       float c, float s) {
  for (int k = 0; k < m; k++) {
    float temp = c * a[k * n + i] + s * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

static void apply_givens_right_float64(double *a, int m, int n, int i, int j,
                                       double c, double s) {
  for (int k = 0; k < m; k++) {
    double temp = c * a[k * n + i] + s * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

// Apply Givens rotation from the left
static void apply_givens_left_float32(float *a, int m, int n, int i, int j,
                                      float c, float s) {
  for (int k = 0; k < n; k++) {
    float temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -s * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

static void apply_givens_left_float64(double *a, int m, int n, int i, int j,
                                      double c, double s) {
  for (int k = 0; k < n; k++) {
    double temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -s * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

// Eigenvalue decomposition helpers
static void tridiagonalize_float32(float* a, float* q, float* diag,
                                   float* offdiag, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      q[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
  }
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n - 2; k++) {
    float norm2 = 0.0f;
    for (int i = k + 1; i < n; i++) {
      norm2 += a[i * n + k] * a[i * n + k];
    }
    if (norm2 <= 0.0f) continue;
    float norm = sqrtf(norm2);
    float sign = sign_float32(a[(k + 1) * n + k]);
    float alpha = -sign * norm;
    float beta = 1.0f / (alpha * (a[(k + 1) * n + k] / norm));
    float* v = (float*)calloc(n, sizeof(float));
    if (!v) continue;
    for (int i = k + 1; i < n; i++) v[i] = a[i * n + k] / alpha;
    v[k + 1] -= 1.0f;
    for (int j = k + 1; j < n; j++) {
      float gamma = 0.0f;
      for (int i = k + 1; i < n; i++) gamma += v[i] * a[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) a[i * n + j] -= gamma * v[i];
      gamma = 0.0f;
      for (int i = k + 1; i < n; i++) gamma += v[i] * a[j * n + i];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) a[j * n + i] -= gamma * v[i];
    }
    for (int j = 0; j < n; j++) {
      float gamma = 0.0f;
      for (int i = k + 1; i < n; i++) gamma += v[i] * q[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) q[i * n + j] -= gamma * v[i];
    }
    free(v);
  }
  for (int i = 0; i < n; i++) diag[i] = a[i * n + i];
  for (int i = 0; i < n - 1; i++) offdiag[i] = a[i * n + (i + 1)];
}

static void tridiagonalize_float64(double* a, double* q, double* diag,
                                   double* offdiag, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      q[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n - 2; k++) {
    double norm2 = 0.0;
    for (int i = k + 1; i < n; i++) {
      norm2 += a[i * n + k] * a[i * n + k];
    }
    if (norm2 <= 0.0) continue;
    double norm = sqrt(norm2);
    double sign = sign_float64(a[(k + 1) * n + k]);
    double alpha = -sign * norm;
    double beta = 1.0 / (alpha * (a[(k + 1) * n + k] / norm));
    double* v = (double*)calloc(n, sizeof(double));
    if (!v) continue;
    for (int i = k + 1; i < n; i++) v[i] = a[i * n + k] / alpha;
    v[k + 1] -= 1.0;
    for (int j = k + 1; j < n; j++) {
      double gamma = 0.0;
      for (int i = k + 1; i < n; i++) gamma += v[i] * a[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) a[i * n + j] -= gamma * v[i];
      gamma = 0.0;
      for (int i = k + 1; i < n; i++) gamma += v[i] * a[j * n + i];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) a[j * n + i] -= gamma * v[i];
    }
    for (int j = 0; j < n; j++) {
      double gamma = 0.0;
      for (int i = k + 1; i < n; i++) gamma += v[i] * q[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) q[i * n + j] -= gamma * v[i];
    }
    free(v);
  }
  for (int i = 0; i < n; i++) diag[i] = a[i * n + i];
  for (int i = 0; i < n - 1; i++) offdiag[i] = a[i * n + (i + 1)];
}

static void qr_iteration_tridiag_float32(float* diag, float* offdiag, float* q,
                                         int n) {
  const float tol = NX_EPS32 * n;
  const int max_iter = 30 * n;
  int iter = 0;
  while (iter++ < max_iter) {
    int converged = 1;
    for (int i = 0; i < n - 1; i++) {
      if (fabsf(offdiag[i]) > tol * (fabsf(diag[i]) + fabsf(diag[i + 1]))) {
        converged = 0;
        break;
      } else {
        offdiag[i] = 0.0f;
      }
    }
    if (converged) break;
    int q_pos = n - 1;
    while (q_pos > 0 && offdiag[q_pos - 1] == 0.0f) q_pos--;
    if (q_pos == 0) continue;
    int p_pos = q_pos - 1;
    while (p_pos > 0 && offdiag[p_pos - 1] != 0.0f) p_pos--;
    float d = (diag[q_pos - 1] - diag[q_pos]) / 2.0f;
    float shift =
        diag[q_pos] -
        offdiag[q_pos - 1] * offdiag[q_pos - 1] /
            (d + sign_float32(d) * hypot_float32(d, offdiag[q_pos - 1]));
    float f = diag[p_pos] - shift;
    float g = offdiag[p_pos];
    for (int k = p_pos; k < q_pos; k++) {
      float c, s;
      givens_float32(f, g, &c, &s);
      if (k > p_pos) offdiag[k - 1] = hypot_float32(f, g);
      f = c * diag[k] + s * offdiag[k];
      offdiag[k] = -s * diag[k] + c * offdiag[k];
      g = s * diag[k + 1];
      diag[k + 1] = c * diag[k + 1];
      apply_givens_right_float32(q, n, n, k, k + 1, c, s);
      givens_float32(f, g, &c, &s);
      diag[k] = hypot_float32(f, g);
      f = c * offdiag[k] + s * diag[k + 1];
      diag[k + 1] = -s * offdiag[k] + c * diag[k + 1];
      if (k < q_pos - 1) {
        g = s * offdiag[k + 1];
        offdiag[k + 1] = c * offdiag[k + 1];
      }
      apply_givens_left_float32(q, n, n, k, k + 1, c, s);
    }
  }
}

static void qr_iteration_tridiag_float64(double* diag, double* offdiag,
                                         double* q, int n) {
  const double tol = NX_EPS64 * n;
  const int max_iter = 30 * n;
  int iter = 0;
  while (iter++ < max_iter) {
    int converged = 1;
    for (int i = 0; i < n - 1; i++) {
      if (fabs(offdiag[i]) > tol * (fabs(diag[i]) + fabs(diag[i + 1]))) {
        converged = 0;
        break;
      } else {
        offdiag[i] = 0.0;
      }
    }
    if (converged) break;
    int q_pos = n - 1;
    while (q_pos > 0 && offdiag[q_pos - 1] == 0.0) q_pos--;
    if (q_pos == 0) continue;
    int p_pos = q_pos - 1;
    while (p_pos > 0 && offdiag[p_pos - 1] != 0.0) p_pos--;
    double d = (diag[q_pos - 1] - diag[q_pos]) / 2.0;
    double shift =
        diag[q_pos] -
        offdiag[q_pos - 1] * offdiag[q_pos - 1] /
            (d + sign_float64(d) * hypot_float64(d, offdiag[q_pos - 1]));
    double f = diag[p_pos] - shift;
    double g = offdiag[p_pos];
    for (int k = p_pos; k < q_pos; k++) {
      double c, s;
      givens_float64(f, g, &c, &s);
      if (k > p_pos) offdiag[k - 1] = hypot_float64(f, g);
      f = c * diag[k] + s * offdiag[k];
      offdiag[k] = -s * diag[k] + c * offdiag[k];
      g = s * diag[k + 1];
      diag[k + 1] = c * diag[k + 1];
      apply_givens_right_float64(q, n, n, k, k + 1, c, s);
      givens_float64(f, g, &c, &s);
      diag[k] = hypot_float64(f, g);
      f = c * offdiag[k] + s * diag[k + 1];
      diag[k + 1] = -s * offdiag[k] + c * diag[k + 1];
      if (k < q_pos - 1) {
        g = s * offdiag[k + 1];
        offdiag[k + 1] = c * offdiag[k + 1];
      }
      apply_givens_left_float64(q, n, n, k, k + 1, c, s);
    }
  }
}

// Forward declarations
static void eigh_float32(float* a, float* eigvals, float* eigvecs, int n);
static void eigh_float64(double* a, double* eigvals, double* eigvecs, int n);

// General eigenvalue decomposition for float32
static void eig_float32(float* a, complex32* eigvals, complex32* eigvecs, int n) {
   // Create a copy of the input matrix since LAPACK overwrites it
   float* a_copy = (float*)malloc(n * n * sizeof(float));
   if (!a_copy) return;
   memcpy(a_copy, a, n * n * sizeof(float));

   // Allocate workspace for eigenvalues
   float* wr = (float*)malloc(n * sizeof(float));
   float* wi = (float*)malloc(n * sizeof(float));
   if (!wr || !wi) {
     free(a_copy);
     free(wr);
     free(wi);
     return;
   }

   // Allocate workspace for eigenvectors if requested
   float* vr = NULL;
   if (eigvecs) {
     vr = (float*)malloc(n * n * sizeof(float));
     if (!vr) {
       free(a_copy);
       free(wr);
       free(wi);
       return;
     }
   }

   // Call LAPACK general eigenvalue decomposition
   // LAPACK_ROW_MAJOR: row-major storage (matches our layout)
   // 'N': don't compute left eigenvectors
   // 'V': compute right eigenvectors if requested
   int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR,
                            'N', eigvecs ? 'V' : 'N',
                            n, a_copy, n,
                            wr, wi, NULL, n,
                            vr, n);

   if (info == 0) {
     // Convert real/imaginary eigenvalues to complex format
     for (int i = 0; i < n; i++) {
       eigvals[i] = wr[i] + wi[i] * I;
     }

     // Convert eigenvectors to complex format if requested
     if (eigvecs) {
       for (int i = 0; i < n * n; i++) {
         eigvecs[i] = vr[i] + 0.0f * I;
       }
     }
   }

   free(a_copy);
   free(wr);
   free(wi);
   free(vr);
}

// General eigenvalue decomposition for float64
static void eig_float64(double* a, complex64* eigvals, complex64* eigvecs, int n) {
   // Create a copy of the input matrix since LAPACK overwrites it
   double* a_copy = (double*)malloc(n * n * sizeof(double));
   if (!a_copy) return;
   memcpy(a_copy, a, n * n * sizeof(double));

   // Allocate workspace for eigenvalues
   double* wr = (double*)malloc(n * sizeof(double));
   double* wi = (double*)malloc(n * sizeof(double));
   if (!wr || !wi) {
     free(a_copy);
     free(wr);
     free(wi);
     return;
   }

   // Allocate workspace for eigenvectors if requested
   double* vr = NULL;
   if (eigvecs) {
     vr = (double*)malloc(n * n * sizeof(double));
     if (!vr) {
       free(a_copy);
       free(wr);
       free(wi);
       return;
     }
   }

   // Call LAPACK general eigenvalue decomposition
   // LAPACK_ROW_MAJOR: row-major storage (matches our layout)
   // 'N': don't compute left eigenvectors
   // 'V': compute right eigenvectors if requested
   int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,
                            'N', eigvecs ? 'V' : 'N',
                            n, a_copy, n,
                            wr, wi, NULL, n,
                            vr, n);

   if (info == 0) {
     // Convert real/imaginary eigenvalues to complex format
     for (int i = 0; i < n; i++) {
       eigvals[i] = wr[i] + wi[i] * I;
     }

     // Convert eigenvectors to complex format if requested
     if (eigvecs) {
       for (int i = 0; i < n * n; i++) {
         eigvecs[i] = vr[i] + 0.0 * I;
       }
     }
   }

   free(a_copy);
   free(wr);
   free(wi);
   free(vr);
}

static void eigh_float32(float* a, float* eigvals, float* eigvecs, int n) {
   // Create a copy of the input matrix since LAPACK overwrites it
   float* a_copy = (float*)malloc(n * n * sizeof(float));
   if (!a_copy) return;
   memcpy(a_copy, a, n * n * sizeof(float));

   // Call LAPACK symmetric eigenvalue decomposition
   // LAPACK_ROW_MAJOR: row-major storage
   // 'V': compute eigenvectors, 'N': eigenvalues only
   // 'L': lower triangular (arbitrary choice)
   int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR,
                            eigvecs ? 'V' : 'N', 'L',
                            n, a_copy, n, eigvals);

   if (info == 0 && eigvecs) {
     // Copy eigenvectors to output
     memcpy(eigvecs, a_copy, n * n * sizeof(float));
   }

   free(a_copy);
}

static void eigh_float64(double* a, double* eigvals, double* eigvecs, int n) {
   // Create a copy of the input matrix since LAPACK overwrites it
   double* a_copy = (double*)malloc(n * n * sizeof(double));
   if (!a_copy) return;
   memcpy(a_copy, a, n * n * sizeof(double));

   // Call LAPACK symmetric eigenvalue decomposition
   // LAPACK_ROW_MAJOR: row-major storage
   // 'V': compute eigenvectors, 'N': eigenvalues only
   // 'L': lower triangular (arbitrary choice)
   int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR,
                            eigvecs ? 'V' : 'N', 'L',
                            n, a_copy, n, eigvals);

   if (info == 0 && eigvecs) {
     // Copy eigenvectors to output
     memcpy(eigvecs, a_copy, n * n * sizeof(double));
   }

   free(a_copy);
}

static void eigh_complex32(complex32* a, float* eigvals, complex32* eigvecs,
                           int n) {
   // Create a copy of the input matrix since LAPACK overwrites it
   complex32* a_copy = (complex32*)malloc(n * n * sizeof(complex32));
   if (!a_copy) return;
   memcpy(a_copy, a, n * n * sizeof(complex32));

   // Call LAPACK Hermitian eigenvalue decomposition
   // LAPACK_ROW_MAJOR: row-major storage
   // 'V': compute eigenvectors, 'N': eigenvalues only
   // 'L': lower triangular (arbitrary choice)
   int info = LAPACKE_cheev(LAPACK_ROW_MAJOR,
                            eigvecs ? 'V' : 'N', 'L',
                            n, a_copy, n, eigvals);

   if (info == 0 && eigvecs) {
     // Copy eigenvectors to output
     memcpy(eigvecs, a_copy, n * n * sizeof(complex32));
   }

   free(a_copy);
}

static void eigh_complex64(complex64* a, double* eigvals, complex64* eigvecs,
                           int n) {
   // Create a copy of the input matrix since LAPACK overwrites it
   complex64* a_copy = (complex64*)malloc(n * n * sizeof(complex64));
   if (!a_copy) return;
   memcpy(a_copy, a, n * n * sizeof(complex64));

   // Call LAPACK Hermitian eigenvalue decomposition
   // LAPACK_ROW_MAJOR: row-major storage
   // 'V': compute eigenvectors, 'N': eigenvalues only
   // 'L': lower triangular (arbitrary choice)
   int info = LAPACKE_zheev(LAPACK_ROW_MAJOR,
                            eigvecs ? 'V' : 'N', 'L',
                            n, a_copy, n, eigvals);

   if (info == 0 && eigvecs) {
     // Copy eigenvectors to output
     memcpy(eigvecs, a_copy, n * n * sizeof(complex64));
   }

   free(a_copy);
}

static void eigh_float16(uint16_t* a, uint16_t* eigvals, uint16_t* eigvecs,
                         int n) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  float* eigvals_float = (float*)malloc(n * sizeof(float));
  float* eigvecs_float = eigvecs ? (float*)malloc(n * n * sizeof(float)) : NULL;
  if (!a_float || !eigvals_float || (eigvecs && !eigvecs_float)) {
    free(a_float);
    free(eigvals_float);
    free(eigvecs_float);
    return;
  }
  for (int i = 0; i < n * n; i++) a_float[i] = half_to_float(a[i]);
  eigh_float32(a_float, eigvals_float, eigvecs_float, n);
  for (int i = 0; i < n; i++) eigvals[i] = float_to_half(eigvals_float[i]);
  if (eigvecs) {
    for (int i = 0; i < n * n; i++)
      eigvecs[i] = float_to_half(eigvecs_float[i]);
    free(eigvecs_float);
  }
  free(a_float);
  free(eigvals_float);
}

static void eigh_bfloat16(caml_ba_bfloat16* a, caml_ba_bfloat16* eigvals,
                          caml_ba_bfloat16* eigvecs, int n) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  float* eigvals_float = (float*)malloc(n * sizeof(float));
  float* eigvecs_float = eigvecs ? (float*)malloc(n * n * sizeof(float)) : NULL;
  if (!a_float || !eigvals_float || (eigvecs && !eigvecs_float)) {
    free(a_float);
    free(eigvals_float);
    free(eigvecs_float);
    return;
  }
  for (int i = 0; i < n * n; i++) a_float[i] = bfloat16_to_float(a[i]);
  eigh_float32(a_float, eigvals_float, eigvecs_float, n);
  for (int i = 0; i < n; i++) eigvals[i] = float_to_bfloat16(eigvals_float[i]);
  if (eigvecs) {
    for (int i = 0; i < n * n; i++)
      eigvecs[i] = float_to_bfloat16(eigvecs_float[i]);
    free(eigvecs_float);
  }
  free(a_float);
  free(eigvals_float);
}

static void eigh_f8e4m3(caml_ba_fp8_e4m3* a, caml_ba_fp8_e4m3* eigvals,
                        caml_ba_fp8_e4m3* eigvecs, int n) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  float* eigvals_float = (float*)malloc(n * sizeof(float));
  float* eigvecs_float = eigvecs ? (float*)malloc(n * n * sizeof(float)) : NULL;
  if (!a_float || !eigvals_float || (eigvecs && !eigvecs_float)) {
    free(a_float);
    free(eigvals_float);
    free(eigvecs_float);
    return;
  }
  for (int i = 0; i < n * n; i++) a_float[i] = fp8_e4m3_to_float(a[i]);
  eigh_float32(a_float, eigvals_float, eigvecs_float, n);
  for (int i = 0; i < n; i++) eigvals[i] = float_to_fp8_e4m3(eigvals_float[i]);
  if (eigvecs) {
    for (int i = 0; i < n * n; i++)
      eigvecs[i] = float_to_fp8_e4m3(eigvecs_float[i]);
    free(eigvecs_float);
  }
  free(a_float);
  free(eigvals_float);
}

static void eigh_f8e5m2(caml_ba_fp8_e5m2* a, caml_ba_fp8_e5m2* eigvals,
                        caml_ba_fp8_e5m2* eigvecs, int n) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  float* eigvals_float = (float*)malloc(n * sizeof(float));
  float* eigvecs_float = eigvecs ? (float*)malloc(n * n * sizeof(float)) : NULL;
  if (!a_float || !eigvals_float || (eigvecs && !eigvecs_float)) {
    free(a_float);
    free(eigvals_float);
    free(eigvecs_float);
    return;
  }
  for (int i = 0; i < n * n; i++) a_float[i] = fp8_e5m2_to_float(a[i]);
  eigh_float32(a_float, eigvals_float, eigvecs_float, n);
  for (int i = 0; i < n; i++) eigvals[i] = float_to_fp8_e5m2(eigvals_float[i]);
  if (eigvecs) {
    for (int i = 0; i < n * n; i++)
      eigvecs[i] = float_to_fp8_e5m2(eigvecs_float[i]);
    free(eigvecs_float);
  }
  free(a_float);
  free(eigvals_float);
}

CAMLprim value caml_nx_op_eig(value v_in, value v_vals, value v_vecs,
                              value v_symmetric, value v_compute_vectors) {
  CAMLparam5(v_in, v_vals, v_vecs, v_symmetric, v_compute_vectors);
  int symmetric = Int_val(v_symmetric);
  int compute_vectors = Int_val(v_compute_vectors);
  ndarray_t in = extract_ndarray(v_in);
  ndarray_t vals = extract_ndarray(v_vals);
  ndarray_t vecs = extract_ndarray(v_vecs);
  struct caml_ba_array* ba_in = Caml_ba_array_val(Field(v_in, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_vals =
      Caml_ba_array_val(Field(v_vals, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_vecs =
      Caml_ba_array_val(Field(v_vecs, FFI_TENSOR_DATA));
  int kind = nx_ba_get_kind(ba_in);
  if (in.ndim < 2) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&vals);
    cleanup_ndarray(&vecs);
    caml_failwith("eig: input must have at least 2 dimensions");
  }
  int n = in.shape[in.ndim - 1];
  if (in.shape[in.ndim - 2] != n) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&vals);
    cleanup_ndarray(&vecs);
    caml_failwith("eig: input must be square matrix");
  }
  // General eigenvalue decomposition is now supported
  int batch_size = 1;
  for (int i = 0; i < in.ndim - 2; i++) {
    batch_size *= in.shape[i];
  }
  int s_in_row = in.strides[in.ndim - 2];
  int s_in_col = in.strides[in.ndim - 1];
  int s_vals_stride = vals.strides[vals.ndim - 1];
  int s_vecs_row = compute_vectors ? vecs.strides[vecs.ndim - 2] : 0;
  int s_vecs_col = compute_vectors ? vecs.strides[vecs.ndim - 1] : 0;
  caml_enter_blocking_section();
  for (int b = 0; b < batch_size; b++) {
    size_t off_in = in.offset;
    size_t off_vals = vals.offset;
    size_t off_vecs = compute_vectors ? vecs.offset : 0;
    if (in.ndim > 2) {
      int remaining = b;
      for (int i = in.ndim - 3; i >= 0; i--) {
        int coord = remaining % in.shape[i];
        remaining /= in.shape[i];
        off_in += coord * in.strides[i];
        off_vals += coord * vals.strides[i];
        if (compute_vectors) off_vecs += coord * vecs.strides[i];
      }
    }
    switch (kind) {
      case CAML_BA_FLOAT32: {
        float* base_in = (float*)ba_in->data + off_in;
        if (symmetric) {
          double* base_vals = (double*)ba_vals->data + off_vals;  // Eigenvalues are always float64
          float* base_vecs =
              compute_vectors ? (float*)ba_vecs->data + off_vecs : NULL;
          float* A = (float*)malloc((size_t)n * n * sizeof(float));
          float* temp_vals = (float*)malloc(n * sizeof(float));
          float* temp_vecs = compute_vectors ? (float*)malloc(n * n * sizeof(float)) : NULL;
          if (!A || !temp_vals || (compute_vectors && !temp_vecs)) {
            free(A);
            free(temp_vals);
            free(temp_vecs);
            continue;
          }
          nx_pack_f32(A, base_in, n, n, s_in_row, s_in_col);
          eigh_float32(A, temp_vals, temp_vecs, n);
          // Convert eigenvalues from float32 to float64
          for (int i = 0; i < n; i++) {
            base_vals[i * s_vals_stride] = (double)temp_vals[i];
          }
          if (compute_vectors) {
            nx_unpack_f32(base_vecs, temp_vecs, n, n, s_vecs_row, s_vecs_col);
          }
          free(A);
          free(temp_vals);
          free(temp_vecs);
        } else {
          // General eigenvalue decomposition - output is complex64
          complex64* base_vals = (complex64*)ba_vals->data + off_vals;
          complex64* base_vecs =
              compute_vectors ? (complex64*)ba_vecs->data + off_vecs : NULL;
          float* A = (float*)malloc((size_t)n * n * sizeof(float));
          if (!A) continue;
          nx_pack_f32(A, base_in, n, n, s_in_row, s_in_col);
          // Allocate temporary buffers for complex32 results from LAPACK

          complex32* temp_vals = (complex32*)malloc(n * sizeof(complex32));
          complex32* temp_vecs = compute_vectors ?
              (complex32*)malloc(n * n * sizeof(complex32)) : NULL;
          if (!temp_vals || (compute_vectors && !temp_vecs)) {
            free(A);
            free(temp_vals);
            free(temp_vecs);
            continue;
          }

          eig_float32(A, temp_vals, temp_vecs, n);

          // Convert eigenvalues from complex32 to complex64
          for (int i = 0; i < n; i++) {
            base_vals[i * s_vals_stride] = (double)crealf(temp_vals[i]) + (double)cimagf(temp_vals[i]) * I;
          }

          if (compute_vectors) {
            // Unpack and convert complex eigenvectors from complex32 to complex64
            for (int i = 0; i < n; i++) {
              for (int j = 0; j < n; j++) {
                base_vecs[i * s_vecs_row + j * s_vecs_col] =
                    (double)crealf(temp_vecs[i * n + j]) + (double)cimagf(temp_vecs[i * n + j]) * I;
              }
            }
            free(temp_vecs);
          }
          free(temp_vals);
          free(A);
        }
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_in = (double*)ba_in->data + off_in;
        if (symmetric) {
          double* base_vals = (double*)ba_vals->data + off_vals;
          double* base_vecs =
              compute_vectors ? (double*)ba_vecs->data + off_vecs : NULL;
          double* A = (double*)malloc((size_t)n * n * sizeof(double));
          if (!A) continue;
          nx_pack_f64(A, base_in, n, n, s_in_row, s_in_col);
          eigh_float64(A, base_vals, base_vecs, n);
          if (compute_vectors) {
            nx_unpack_f64(base_vecs, base_vecs, n, n, s_vecs_row, s_vecs_col);
          }
          free(A);
        } else {
          // General eigenvalue decomposition - output is complex
          complex64* base_vals = (complex64*)ba_vals->data + off_vals;
          complex64* base_vecs =
              compute_vectors ? (complex64*)ba_vecs->data + off_vecs : NULL;
          double* A = (double*)malloc((size_t)n * n * sizeof(double));
          if (!A) continue;
          nx_pack_f64(A, base_in, n, n, s_in_row, s_in_col);
          // Allocate temporary buffers for complex results
          complex64* temp_vals = (complex64*)malloc(n * sizeof(complex64));
          complex64* temp_vecs = compute_vectors ?
              (complex64*)malloc(n * n * sizeof(complex64)) : NULL;
          if (!temp_vals || (compute_vectors && !temp_vecs)) {
            free(A);
            free(temp_vals);
            free(temp_vecs);
            continue;
          }

          eig_float64(A, temp_vals, temp_vecs, n);

          // Copy eigenvalues to output with proper striding
          for (int i = 0; i < n; i++) {
            base_vals[i * s_vals_stride] = temp_vals[i];
          }

          if (compute_vectors) {
            // Unpack complex eigenvectors with proper striding
            for (int i = 0; i < n; i++) {
              for (int j = 0; j < n; j++) {
                base_vecs[i * s_vecs_row + j * s_vecs_col] = temp_vecs[i * n + j];
              }
            }
            free(temp_vecs);
          }
          free(temp_vals);
          free(A);
        }
        break;
      }
      case CAML_BA_COMPLEX32: {
        complex32* base_in = (complex32*)ba_in->data + off_in;
        float* base_vals = (float*)ba_vals->data + off_vals;
        complex32* base_vecs =
            compute_vectors ? (complex32*)ba_vecs->data + off_vecs : NULL;
        complex32* A = (complex32*)malloc((size_t)n * n * sizeof(complex32));
        if (!A) continue;
        nx_pack_c32(A, base_in, n, n, s_in_row, s_in_col);
        eigh_complex32(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          nx_unpack_c32(base_vecs, base_vecs, n, n, s_vecs_row, s_vecs_col);
        }
        free(A);
        break;
      }
      case CAML_BA_COMPLEX64: {
        complex64* base_in = (complex64*)ba_in->data + off_in;
        double* base_vals = (double*)ba_vals->data + off_vals;
        complex64* base_vecs =
            compute_vectors ? (complex64*)ba_vecs->data + off_vecs : NULL;
        complex64* A = (complex64*)malloc((size_t)n * n * sizeof(complex64));
        if (!A) continue;
        nx_pack_c64(A, base_in, n, n, s_in_row, s_in_col);
        eigh_complex64(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          nx_unpack_c64(base_vecs, base_vecs, n, n, s_vecs_row, s_vecs_col);
        }
        free(A);
        break;
      }
      case CAML_BA_FLOAT16: {
        uint16_t* base_in = (uint16_t*)ba_in->data + off_in;
        uint16_t* base_vals = (uint16_t*)ba_vals->data + off_vals;
        uint16_t* base_vecs =
            compute_vectors ? (uint16_t*)ba_vecs->data + off_vecs : NULL;
        uint16_t* A = (uint16_t*)malloc((size_t)n * n * sizeof(uint16_t));
        if (!A) continue;
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        eigh_float16(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_vecs[i * s_vecs_row + j * s_vecs_col] = base_vecs[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_BFLOAT16: {
        caml_ba_bfloat16* base_in = (caml_ba_bfloat16*)ba_in->data + off_in;
        caml_ba_bfloat16* base_vals =
            (caml_ba_bfloat16*)ba_vals->data + off_vals;
        caml_ba_bfloat16* base_vecs =
            compute_vectors ? (caml_ba_bfloat16*)ba_vecs->data + off_vecs
                            : NULL;
        caml_ba_bfloat16* A =
            (caml_ba_bfloat16*)malloc((size_t)n * n * sizeof(caml_ba_bfloat16));
        if (!A) continue;
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        eigh_bfloat16(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_vecs[i * s_vecs_row + j * s_vecs_col] = base_vecs[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_FP8_E4M3: {
        caml_ba_fp8_e4m3* base_in = (caml_ba_fp8_e4m3*)ba_in->data + off_in;
        caml_ba_fp8_e4m3* base_vals =
            (caml_ba_fp8_e4m3*)ba_vals->data + off_vals;
        caml_ba_fp8_e4m3* base_vecs =
            compute_vectors ? (caml_ba_fp8_e4m3*)ba_vecs->data + off_vecs
                            : NULL;
        caml_ba_fp8_e4m3* A =
            (caml_ba_fp8_e4m3*)malloc((size_t)n * n * sizeof(caml_ba_fp8_e4m3));
        if (!A) continue;
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        eigh_f8e4m3(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_vecs[i * s_vecs_row + j * s_vecs_col] = base_vecs[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_FP8_E5M2: {
        caml_ba_fp8_e5m2* base_in = (caml_ba_fp8_e5m2*)ba_in->data + off_in;
        caml_ba_fp8_e5m2* base_vals =
            (caml_ba_fp8_e5m2*)ba_vals->data + off_vals;
        caml_ba_fp8_e5m2* base_vecs =
            compute_vectors ? (caml_ba_fp8_e5m2*)ba_vecs->data + off_vecs
                            : NULL;
        caml_ba_fp8_e5m2* A =
            (caml_ba_fp8_e5m2*)malloc((size_t)n * n * sizeof(caml_ba_fp8_e5m2));
        if (!A) continue;
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        eigh_f8e5m2(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_vecs[i * s_vecs_row + j * s_vecs_col] = base_vecs[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      default:
        caml_leave_blocking_section();
        cleanup_ndarray(&in);
        cleanup_ndarray(&vals);
        cleanup_ndarray(&vecs);
        caml_failwith("eig: unsupported dtype");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&in);
  cleanup_ndarray(&vals);
  cleanup_ndarray(&vecs);
  CAMLreturn(Val_unit);
}
