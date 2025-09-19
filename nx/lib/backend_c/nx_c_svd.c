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

// Helper to get element size
static long get_element_size(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
    case CAML_BA_UINT8:
    case NX_BA_BOOL:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
    case NX_BA_QINT8:
    case NX_BA_QUINT8:
      return 1;
    case CAML_BA_SINT16:
    case CAML_BA_UINT16:
    case CAML_BA_FLOAT16:
    case NX_BA_BFLOAT16:
      return 2;
    case CAML_BA_INT32:
    case CAML_BA_FLOAT32:
      return 4;
    case CAML_BA_INT64:
    case CAML_BA_FLOAT64:
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
      return 8;
    case CAML_BA_COMPLEX32:
    case NX_BA_COMPLEX16:
      return 4;
    case CAML_BA_COMPLEX64:
      return 16;
    case NX_BA_INT4:
    case NX_BA_UINT4:
      caml_failwith("get_element_size: int4/uint4 require special handling");
    default:
      caml_failwith("get_element_size: unsupported kind");
  }
  return 0;
}

// Helper functions for shape and stride operations
static inline int nx_ndim(value v_shape) { return Wosize_val(v_shape); }

static inline int nx_shape_at(value v_shape, int idx) {
  return Int_val(Field(v_shape, idx));
}

static inline int nx_stride_at(value v_strides, int idx) {
  return Int_val(Field(v_strides, idx));
}

static inline int nx_batch_size(value v_shape) {
  int ndim = Wosize_val(v_shape);
  if (ndim <= 2) return 1;
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= Int_val(Field(v_shape, i));
  }
  return batch_size;
}

static inline size_t nx_batch_offset_elems(int b, value v_shape,
                                           value v_strides) {
  int ndim = Wosize_val(v_shape);
  if (ndim <= 2) return 0;
  size_t offset = 0;
  int remaining = b;
  // Calculate offset for batch dimensions
  for (int i = ndim - 3; i >= 0; i--) {
    int dim_size = Int_val(Field(v_shape, i));
    int coord = remaining % dim_size;
    remaining /= dim_size;
    offset += coord * Int_val(Field(v_strides, i));
  }
  return offset;
}

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

// SVD helper functions
static inline float sign_float32(float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

static inline double sign_float64(double x) { return (x >= 0.0) ? 1.0 : -1.0; }

static inline float hypot_float32(float a, float b) {
  float absa = fabsf(a);
  float absb = fabsf(b);
  if (absa > absb) {
    float ratio = absb / absa;
    return absa * sqrtf(1.0f + ratio * ratio);
  } else if (absb > 0.0f) {
    float ratio = absa / absb;
    return absb * sqrtf(1.0f + ratio * ratio);
  } else {
    return 0.0f;
  }
}

static inline double hypot_float64(double a, double b) {
  double absa = fabs(a);
  double absb = fabs(b);
  if (absa > absb) {
    double ratio = absb / absa;
    return absa * sqrt(1.0 + ratio * ratio);
  } else if (absb > 0.0) {
    double ratio = absa / absb;
    return absb * sqrt(1.0 + ratio * ratio);
  } else {
    return 0.0;
  }
}

static void givens_float32(float a, float b, float* c, float* s) {
  if (b == 0.0f) {
    *c = 1.0f;
    *s = 0.0f;
  } else if (fabsf(b) > fabsf(a)) {
    float t = a / b;
    float sign_b = sign_float32(b);
    *s = sign_b / sqrtf(1.0f + t * t);
    *c = *s * t;
  } else {
    float t = b / a;
    float sign_a = sign_float32(a);
    *c = sign_a / sqrtf(1.0f + t * t);
    *s = *c * t;
  }
}

static void givens_float64(double a, double b, double* c, double* s) {
  if (b == 0.0) {
    *c = 1.0;
    *s = 0.0;
  } else if (fabs(b) > fabs(a)) {
    double t = a / b;
    double sign_b = sign_float64(b);
    *s = sign_b / sqrt(1.0 + t * t);
    *c = *s * t;
  } else {
    double t = b / a;
    double sign_a = sign_float64(a);
    *c = sign_a / sqrt(1.0 + t * t);
    *s = *c * t;
  }
}

static void apply_givens_left_float32(float* a, int m, int n, int i, int j,
                                      float c, float s) {
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n; k++) {
    float temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -s * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

static void apply_givens_left_float64(double* a, int m, int n, int i, int j,
                                      double c, double s) {
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n; k++) {
    double temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -s * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

static void apply_givens_right_float32(float* a, int m, int n, int i, int j,
                                       float c, float s) {
#pragma omp parallel for if (m > 100)
  for (int k = 0; k < m; k++) {
    float temp = c * a[k * n + i] + s * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

static void apply_givens_right_float64(double* a, int m, int n, int i, int j,
                                       double c, double s) {
#pragma omp parallel for if (m > 100)
  for (int k = 0; k < m; k++) {
    double temp = c * a[k * n + i] + s * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

static void bidiagonalize_float32(float* a, float* u, float* v, float* diag,
                                  float* superdiag, int m, int n) {
  const int minmn = (m < n ? m : n);
#pragma omp parallel for if (m > 100)
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) u[i * m + j] = (i == j) ? 1.f : 0.f;
#pragma omp parallel for if (n > 100)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) v[i * n + j] = (i == j) ? 1.f : 0.f;
  for (int p = 0; p < minmn; ++p) {
    float norm2 = 0.f;
    for (int i = p; i < m; i++) norm2 += a[i * n + p] * a[i * n + p];
    float norm = sqrtf(norm2);
    if (norm > 0.f) {
      float sign = sign_float32(a[p * n + p]);
      float alpha = -sign * norm;
      a[p * n + p] -= alpha;
      float beta = 1.f / (alpha * a[p * n + p]);
#pragma omp parallel for if (n - p > 100)
      for (int j = p + 1; j < n; ++j) {
        float gamma = 0.f;
        for (int i = p; i < m; i++) gamma += a[i * n + p] * a[i * n + j];
        gamma *= beta;
        for (int i = p; i < m; i++) a[i * n + j] -= gamma * a[i * n + p];
      }
#pragma omp parallel for if (m > 100)
      for (int j = 0; j < m; ++j) {
        float gamma = 0.f;
        for (int i = p; i < m; i++) gamma += a[i * n + p] * u[i * m + j];
        gamma *= beta;
        for (int i = p; i < m; i++) u[i * m + j] -= gamma * a[i * n + p];
      }
    }
    diag[p] = a[p * n + p];
    if (p < n - 1) {
      float norm2r = 0.f;
      for (int j = p + 1; j < n; j++) norm2r += a[p * n + j] * a[p * n + j];
      float normr = sqrtf(norm2r);
      if (normr > 0.f) {
        float sign = sign_float32(a[p * n + (p + 1)]);
        float alpha = -sign * normr;
        a[p * n + (p + 1)] -= alpha;
        float beta = 1.f / (alpha * a[p * n + (p + 1)]);
#pragma omp parallel for if (m - p > 100)
        for (int i = p + 1; i < m; ++i) {
          float gamma = 0.f;
          for (int j = p + 1; j < n; j++) gamma += a[i * n + j] * a[p * n + j];
          gamma *= beta;
          for (int j = p + 1; j < n; j++) a[i * n + j] -= gamma * a[p * n + j];
        }
#pragma omp parallel for if (n > 100)
        for (int j = 0; j < n; ++j) {
          float gamma = 0.f;
          for (int t = p + 1; t < n; t++) gamma += v[t * n + j] * a[p * n + t];
          gamma *= beta;
          for (int t = p + 1; t < n; t++) v[t * n + j] -= gamma * a[p * n + t];
        }
      }
      superdiag[p] = (p < minmn - 1) ? a[p * n + (p + 1)] : 0.f;
    }
  }
}

static void bidiagonalize_float64(double* a, double* u, double* v, double* diag,
                                  double* superdiag, int m, int n) {
  const int minmn = (m < n ? m : n);
#pragma omp parallel for if (m > 100)
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) u[i * m + j] = (i == j) ? 1.0 : 0.0;
#pragma omp parallel for if (n > 100)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) v[i * n + j] = (i == j) ? 1.0 : 0.0;
  for (int p = 0; p < minmn; ++p) {
    double norm2 = 0.0;
    for (int i = p; i < m; i++) norm2 += a[i * n + p] * a[i * n + p];
    double norm = sqrt(norm2);
    if (norm > 0.0) {
      double sign = sign_float64(a[p * n + p]);
      double alpha = -sign * norm;
      a[p * n + p] -= alpha;
      double beta = 1.0 / (alpha * a[p * n + p]);
#pragma omp parallel for if (n - p > 100)
      for (int j = p + 1; j < n; ++j) {
        double gamma = 0.0;
        for (int i = p; i < m; i++) gamma += a[i * n + p] * a[i * n + j];
        gamma *= beta;
        for (int i = p; i < m; i++) a[i * n + j] -= gamma * a[i * n + p];
      }
#pragma omp parallel for if (m > 100)
      for (int j = 0; j < m; ++j) {
        double gamma = 0.0;
        for (int i = p; i < m; i++) gamma += a[i * n + p] * u[i * m + j];
        gamma *= beta;
        for (int i = p; i < m; i++) u[i * m + j] -= gamma * a[i * n + p];
      }
    }
    diag[p] = a[p * n + p];
    if (p < n - 1) {
      double norm2r = 0.0;
      for (int j = p + 1; j < n; j++) norm2r += a[p * n + j] * a[p * n + j];
      double normr = sqrt(norm2r);
      if (normr > 0.0) {
        double sign = sign_float64(a[p * n + (p + 1)]);
        double alpha = -sign * normr;
        a[p * n + (p + 1)] -= alpha;
        double beta = 1.0 / (alpha * a[p * n + (p + 1)]);
#pragma omp parallel for if (m - p > 100)
        for (int i = p + 1; i < m; ++i) {
          double gamma = 0.0;
          for (int j = p + 1; j < n; j++) gamma += a[i * n + j] * a[p * n + j];
          gamma *= beta;
          for (int j = p + 1; j < n; j++) a[i * n + j] -= gamma * a[p * n + j];
        }
#pragma omp parallel for if (n > 100)
        for (int j = 0; j < n; ++j) {
          double gamma = 0.0;
          for (int t = p + 1; t < n; t++) gamma += v[t * n + j] * a[p * n + t];
          gamma *= beta;
          for (int t = p + 1; t < n; t++) v[t * n + j] -= gamma * a[p * n + t];
        }
      }
      superdiag[p] = (p < minmn - 1) ? a[p * n + (p + 1)] : 0.0;
    }
  }
}

static void svd_qr_iteration_float32(float* diag, float* superdiag, float* u,
                                     float* v, int m, int n, int p, int q) {
  float d = (diag[q - 1] - diag[q]) / 2.0f;
  float shift =
      diag[q] - superdiag[q - 1] * superdiag[q - 1] /
                    (d + sign_float32(d) * hypot_float32(d, superdiag[q - 1]));
  float c, s;
  float f = diag[p] - shift;
  float g = superdiag[p];
  for (int k = p; k < q; k++) {
    givens_float32(f, g, &c, &s);
    if (k > p) superdiag[k - 1] = hypot_float32(f, g);
    f = c * diag[k] + s * superdiag[k];
    superdiag[k] = -s * diag[k] + c * superdiag[k];
    g = s * diag[k + 1];
    diag[k + 1] = c * diag[k + 1];
    apply_givens_right_float32(v, n, n, k, k + 1, c, s);
    givens_float32(f, g, &c, &s);
    diag[k] = hypot_float32(f, g);
    f = c * superdiag[k] + s * diag[k + 1];
    diag[k + 1] = -s * superdiag[k] + c * diag[k + 1];
    if (k < q - 1) {
      g = s * superdiag[k + 1];
      superdiag[k + 1] = c * superdiag[k + 1];
    }
    apply_givens_left_float32(u, m, m, k, k + 1, c, s);
  }
  superdiag[q - 1] = f;
}

static void svd_qr_iteration_float64(double* diag, double* superdiag, double* u,
                                     double* v, int m, int n, int p, int q) {
  double d = (diag[q - 1] - diag[q]) / 2.0;
  double shift =
      diag[q] - superdiag[q - 1] * superdiag[q - 1] /
                    (d + sign_float64(d) * hypot_float64(d, superdiag[q - 1]));
  double c, s;
  double f = diag[p] - shift;
  double g = superdiag[p];
  for (int k = p; k < q; k++) {
    givens_float64(f, g, &c, &s);
    if (k > p) superdiag[k - 1] = hypot_float64(f, g);
    f = c * diag[k] + s * superdiag[k];
    superdiag[k] = -s * diag[k] + c * superdiag[k];
    g = s * diag[k + 1];
    diag[k + 1] = c * diag[k + 1];
    apply_givens_right_float64(v, n, n, k, k + 1, c, s);
    givens_float64(f, g, &c, &s);
    diag[k] = hypot_float64(f, g);
    f = c * superdiag[k] + s * diag[k + 1];
    diag[k + 1] = -s * superdiag[k] + c * diag[k + 1];
    if (k < q - 1) {
      g = s * superdiag[k + 1];
      superdiag[k + 1] = c * superdiag[k + 1];
    }
    apply_givens_left_float64(u, m, m, k, k + 1, c, s);
  }
  superdiag[q - 1] = f;
}

static void svd_iterate_float32(float* diag, float* superdiag, float* u,
                                float* v, int m, int n) {
  const int minmn = (m < n ? m : n);
  const float tol = NX_EPS32 * (float)(m > n ? m : n);
  const int max_iter = 75 * minmn;
  int iter = 0;
  while (iter++ < max_iter) {
    int converged = 1;
    for (int i = 0; i < minmn - 1; i++) {
      if (fabsf(superdiag[i]) > tol * (fabsf(diag[i]) + fabsf(diag[i + 1]))) {
        converged = 0;
        break;
      }
    }
    if (converged) break;
    int q_pos = minmn - 1;
    while (q_pos > 0 &&
           fabsf(superdiag[q_pos - 1]) <=
               tol * (fabsf(diag[q_pos - 1]) + fabsf(diag[q_pos]))) {
      superdiag[q_pos - 1] = 0.0f;
      q_pos--;
    }
    int p_pos = q_pos;
    while (p_pos > 0 && fabsf(superdiag[p_pos - 1]) >
                            tol * (fabsf(diag[p_pos - 1]) + fabsf(diag[p_pos])))
      p_pos--;
    if (p_pos < q_pos) {
      svd_qr_iteration_float32(diag, superdiag, u, v, m, n, p_pos, q_pos);
    }
  }
  if (iter >= max_iter) {
    // Handle non-convergence if needed, but for production, assume convergence
    // or log
  }
}

static void svd_iterate_float64(double* diag, double* superdiag, double* u,
                                double* v, int m, int n) {
  const int minmn = (m < n ? m : n);
  const double tol = NX_EPS64 * (double)(m > n ? m : n);
  const int max_iter = 75 * minmn;
  int iter = 0;
  while (iter++ < max_iter) {
    int converged = 1;
    for (int i = 0; i < minmn - 1; i++) {
      if (fabs(superdiag[i]) > tol * (fabs(diag[i]) + fabs(diag[i + 1]))) {
        converged = 0;
        break;
      }
    }
    if (converged) break;
    int q_pos = minmn - 1;
    while (q_pos > 0 && fabs(superdiag[q_pos - 1]) <=
                            tol * (fabs(diag[q_pos - 1]) + fabs(diag[q_pos]))) {
      superdiag[q_pos - 1] = 0.0;
      q_pos--;
    }
    int p_pos = q_pos;
    while (p_pos > 0 && fabs(superdiag[p_pos - 1]) >
                            tol * (fabs(diag[p_pos - 1]) + fabs(diag[p_pos])))
      p_pos--;
    if (p_pos < q_pos) {
      svd_qr_iteration_float64(diag, superdiag, u, v, m, n, p_pos, q_pos);
    }
  }
  if (iter >= max_iter) {
    // Handle non-convergence if needed
  }
}

// SVD implementations
static void svd_float32(float* a, float* u, float* s, float* vt, int m, int n,
                        int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  float* a_copy = (float*)malloc(m * n * sizeof(float));
  if (!a_copy) return;
  memcpy(a_copy, a, m * n * sizeof(float));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  // ldu: U is [m, m] (full) or [m, minmn] (econ), leading dim is # cols
  lapack_int ldu = full_matrices ? m : minmn;
  // ldvt: VT is [n, n] (full) or [minmn, n] (econ), leading dim is # cols = n
  lapack_int ldvt = n;

  // Allocate space for superbidiagonal elements (not used in our interface)
  float* superb = (float*)malloc((minmn - 1) * sizeof(float));
  if (!superb) {
    free(a_copy);
    return;
  }

  lapack_int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  // Note: LAPACK returns singular values in descending order, which matches our expectation
}

static void svd_float64(double* a, double* u, double* s, double* vt, int m,
                        int n, int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  double* a_copy = (double*)malloc(m * n * sizeof(double));
  if (!a_copy) return;
  memcpy(a_copy, a, m * n * sizeof(double));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  // ldu: U is [m, m] (full) or [m, minmn] (econ), leading dim is # cols
  lapack_int ldu = full_matrices ? m : minmn;
  // ldvt: VT is [n, n] (full) or [minmn, n] (econ), leading dim is # cols = n
  lapack_int ldvt = n;

  // Allocate space for superbidiagonal elements (not used in our interface)
  double* superb = (double*)malloc((minmn - 1) * sizeof(double));
  if (!superb) {
    free(a_copy);
    return;
  }

  lapack_int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  // Note: LAPACK returns singular values in descending order, which matches our expectation
}

// Complex SVD helpers (similar structure, with conj and cabs)
static inline complex32 sign_complex32(complex32 x) {
  float mag = cabsf(x);
  return (mag == 0.0f) ? (1.0f + 0.0f * I) : (x / mag);
}

static inline complex64 sign_complex64(complex64 x) {
  double mag = cabs(x);
  return (mag == 0.0) ? (1.0 + 0.0 * I) : (x / mag);
}

static inline float hypot_complex32(complex32 a, complex32 b) {
  return hypot_float32(crealf(a), cimagf(a)) +
         hypot_float32(crealf(b), cimagf(b));  // Approximate for simplicity
}

static inline double hypot_complex64(complex64 a, complex64 b) {
  return hypot_float64(creal(a), cimag(a)) + hypot_float64(creal(b), cimag(b));
}

static void givens_complex32(complex32 a, complex32 b, float* c, complex32* s) {
  float na = cabsf(a);
  float nb = cabsf(b);
  if (nb == 0.0f) {
    *c = 1.0f;
    *s = 0.0f + 0.0f * I;
  } else if (nb > na) {
    complex32 t = a / b;
    *s = (1.0f / sqrtf(1.0f + cabsf(t) * cabsf(t))) * sign_complex32(b);
    *c = crealf(*s * t);
  } else {
    complex32 t = b / a;
    *c = 1.0f / sqrtf(1.0f + cabsf(t) * cabsf(t));
    *s = *c * t * sign_complex32(a);
  }
}

static void givens_complex64(complex64 a, complex64 b, double* c,
                             complex64* s) {
  double na = cabs(a);
  double nb = cabs(b);
  if (nb == 0.0) {
    *c = 1.0;
    *s = 0.0 + 0.0 * I;
  } else if (nb > na) {
    complex64 t = a / b;
    *s = (1.0 / sqrt(1.0 + cabs(t) * cabs(t))) * sign_complex64(b);
    *c = creal(*s * t);
  } else {
    complex64 t = b / a;
    *c = 1.0 / sqrt(1.0 + cabs(t) * cabs(t));
    *s = *c * t * sign_complex64(a);
  }
}

static void apply_givens_left_complex32(complex32* a, int m, int n, int i,
                                        int j, float c, complex32 s) {
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n; k++) {
    complex32 temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -conj(s) * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

static void apply_givens_left_complex64(complex64* a, int m, int n, int i,
                                        int j, double c, complex64 s) {
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n; k++) {
    complex64 temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -conj(s) * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

static void apply_givens_right_complex32(complex32* a, int m, int n, int i,
                                         int j, float c, complex32 s) {
#pragma omp parallel for if (m > 100)
  for (int k = 0; k < m; k++) {
    complex32 temp = c * a[k * n + i] + conj(s) * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

static void apply_givens_right_complex64(complex64* a, int m, int n, int i,
                                         int j, double c, complex64 s) {
#pragma omp parallel for if (m > 100)
  for (int k = 0; k < m; k++) {
    complex64 temp = c * a[k * n + i] + conj(s) * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

static void bidiagonalize_complex32(complex32* a, complex32* u, complex32* v,
                                    float* diag, float* superdiag, int m,
                                    int n) {
  const int minmn = (m < n ? m : n);
#pragma omp parallel for if (m > 100)
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) u[i * m + j] = (i == j) ? 1.0f : 0.0f;
#pragma omp parallel for if (n > 100)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) v[i * n + j] = (i == j) ? 1.0f : 0.0f;
  for (int p = 0; p < minmn; ++p) {
    float norm2 = 0.0f;
    for (int i = p; i < m; i++) {
      norm2 += crealf(a[i * n + p] * conjf(a[i * n + p]));
    }
    float norm = sqrtf(norm2);
    if (norm > 0.0f) {
      complex32 phase = a[p * n + p] / cabsf(a[p * n + p]);
      complex32 alpha = -norm * phase;
      a[p * n + p] -= alpha;
      float beta = 1.0f / crealf(conjf(alpha) * a[p * n + p] / norm);
#pragma omp parallel for if (n - p > 100)
      for (int j = p + 1; j < n; ++j) {
        complex32 gamma = 0.0f + 0.0f * I;
        for (int i = p; i < m; i++) gamma += conjf(a[i * n + p]) * a[i * n + j];
        gamma *= beta;
        for (int i = p; i < m; i++) a[i * n + j] -= gamma * a[i * n + p];
      }
#pragma omp parallel for if (m > 100)
      for (int j = 0; j < m; ++j) {
        complex32 gamma = 0.0f + 0.0f * I;
        for (int i = p; i < m; i++) gamma += conjf(a[i * n + p]) * u[i * m + j];
        gamma *= beta;
        for (int i = p; i < m; i++) u[i * m + j] -= gamma * a[i * n + p];
      }
    }
    diag[p] = cabsf(a[p * n + p]);
    a[p * n + p] = diag[p];
    if (p < n - 1) {
      float norm2r = 0.0f;
      for (int j = p + 1; j < n; j++) {
        norm2r += crealf(a[p * n + j] * conjf(a[p * n + j]));
      }
      float normr = sqrtf(norm2r);
      if (normr > 0.0f) {
        complex32 phase = a[p * n + (p + 1)] / cabsf(a[p * n + (p + 1)]);
        complex32 alpha = -normr * phase;
        a[p * n + (p + 1)] -= alpha;
        float beta = 1.0f / crealf(a[p * n + (p + 1)] * conjf(alpha) / normr);
#pragma omp parallel for if (m - p > 100)
        for (int i = p + 1; i < m; ++i) {
          complex32 gamma = 0.0f + 0.0f * I;
          for (int j = p + 1; j < n; j++)
            gamma += a[i * n + j] * conjf(a[p * n + j]);
          gamma *= beta;
          for (int j = p + 1; j < n; j++) a[i * n + j] -= gamma * a[p * n + j];
        }
#pragma omp parallel for if (n > 100)
        for (int j = 0; j < n; ++j) {
          complex32 gamma = 0.0f + 0.0f * I;
          for (int t = p + 1; t < n; t++)
            gamma += v[t * n + j] * conjf(a[p * n + t]);
          gamma *= beta;
          for (int t = p + 1; t < n; t++) v[t * n + j] -= gamma * a[p * n + t];
        }
      }
      superdiag[p] = cabsf(a[p * n + (p + 1)]);
      a[p * n + (p + 1)] = superdiag[p];
    }
  }
}

static void bidiagonalize_complex64(complex64* a, complex64* u, complex64* v,
                                    double* diag, double* superdiag, int m,
                                    int n) {
  const int minmn = (m < n ? m : n);
#pragma omp parallel for if (m > 100)
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) u[i * m + j] = (i == j) ? 1.0 : 0.0;
#pragma omp parallel for if (n > 100)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) v[i * n + j] = (i == j) ? 1.0 : 0.0;
  for (int p = 0; p < minmn; ++p) {
    double norm2 = 0.0;
    for (int i = p; i < m; i++) {
      norm2 += creal(a[i * n + p] * conj(a[i * n + p]));
    }
    double norm = sqrt(norm2);
    if (norm > 0.0) {
      complex64 phase = a[p * n + p] / cabs(a[p * n + p]);
      complex64 alpha = -norm * phase;
      a[p * n + p] -= alpha;
      double beta = 1.0 / creal(conj(alpha) * a[p * n + p] / norm);
#pragma omp parallel for if (n - p > 100)
      for (int j = p + 1; j < n; ++j) {
        complex64 gamma = 0.0 + 0.0 * I;
        for (int i = p; i < m; i++) gamma += conj(a[i * n + p]) * a[i * n + j];
        gamma *= beta;
        for (int i = p; i < m; i++) a[i * n + j] -= gamma * a[i * n + p];
      }
#pragma omp parallel for if (m > 100)
      for (int j = 0; j < m; ++j) {
        complex64 gamma = 0.0 + 0.0 * I;
        for (int i = p; i < m; i++) gamma += conj(a[i * n + p]) * u[i * m + j];
        gamma *= beta;
        for (int i = p; i < m; i++) u[i * m + j] -= gamma * a[i * n + p];
      }
    }
    diag[p] = cabs(a[p * n + p]);
    a[p * n + p] = diag[p];
    if (p < n - 1) {
      double norm2r = 0.0;
      for (int j = p + 1; j < n; j++) {
        norm2r += creal(a[p * n + j] * conj(a[p * n + j]));
      }
      double normr = sqrt(norm2r);
      if (normr > 0.0) {
        complex64 phase = a[p * n + (p + 1)] / cabs(a[p * n + (p + 1)]);
        complex64 alpha = -normr * phase;
        a[p * n + (p + 1)] -= alpha;
        double beta = 1.0 / creal(a[p * n + (p + 1)] * conj(alpha) / normr);
#pragma omp parallel for if (m - p > 100)
        for (int i = p + 1; i < m; ++i) {
          complex64 gamma = 0.0 + 0.0 * I;
          for (int j = p + 1; j < n; j++)
            gamma += a[i * n + j] * conj(a[p * n + j]);
          gamma *= beta;
          for (int j = p + 1; j < n; j++) a[i * n + j] -= gamma * a[p * n + j];
        }
#pragma omp parallel for if (n > 100)
        for (int j = 0; j < n; ++j) {
          complex64 gamma = 0.0 + 0.0 * I;
          for (int t = p + 1; t < n; t++)
            gamma += v[t * n + j] * conj(a[p * n + t]);
          gamma *= beta;
          for (int t = p + 1; t < n; t++) v[t * n + j] -= gamma * a[p * n + t];
        }
      }
      superdiag[p] = cabs(a[p * n + (p + 1)]);
      a[p * n + (p + 1)] = superdiag[p];
    }
  }
}

static void svd_qr_iteration_complex32(float* diag, float* superdiag,
                                       complex32* u, complex32* v, int m, int n,
                                       int p, int q) {
  float d = (diag[q - 1] - diag[q]) / 2.0f;
  float shift =
      diag[q] - superdiag[q - 1] * superdiag[q - 1] /
                    (d + sign_float32(d) * hypot_float32(d, superdiag[q - 1]));
  float c;
  complex32 s;
  float f = diag[p] - shift;
  float g = superdiag[p];
  for (int k = p; k < q; k++) {
    givens_complex32(f + 0.0f * I, g + 0.0f * I, &c, &s);
    if (k > p) superdiag[k - 1] = hypot_float32(f, g);
    f = c * diag[k] +
        crealf(s) * superdiag[k];  // Simplified for real diag/superdiag
    superdiag[k] = -crealf(conj(s)) * diag[k] + c * superdiag[k];
    g = crealf(s) * diag[k + 1];
    diag[k + 1] = c * diag[k + 1];
    apply_givens_right_complex32(v, n, n, k, k + 1, c, s);
    givens_complex32(f + 0.0f * I, g + 0.0f * I, &c, &s);
    diag[k] = hypot_float32(f, g);
    f = c * superdiag[k] + crealf(s) * diag[k + 1];
    diag[k + 1] = -crealf(conj(s)) * superdiag[k] + c * diag[k + 1];
    if (k < q - 1) {
      g = crealf(s) * superdiag[k + 1];
      superdiag[k + 1] = c * superdiag[k + 1];
    }
    apply_givens_left_complex32(u, m, m, k, k + 1, c, s);
  }
  superdiag[q - 1] = f;
}

static void svd_qr_iteration_complex64(double* diag, double* superdiag,
                                       complex64* u, complex64* v, int m, int n,
                                       int p, int q) {
  double d = (diag[q - 1] - diag[q]) / 2.0;
  double shift =
      diag[q] - superdiag[q - 1] * superdiag[q - 1] /
                    (d + sign_float64(d) * hypot_float64(d, superdiag[q - 1]));
  double c;
  complex64 s;
  double f = diag[p] - shift;
  double g = superdiag[p];
  for (int k = p; k < q; k++) {
    givens_complex64(f + 0.0 * I, g + 0.0 * I, &c, &s);
    if (k > p) superdiag[k - 1] = hypot_float64(f, g);
    f = c * diag[k] + creal(s) * superdiag[k];
    superdiag[k] = -creal(conj(s)) * diag[k] + c * superdiag[k];
    g = creal(s) * diag[k + 1];
    diag[k + 1] = c * diag[k + 1];
    apply_givens_right_complex64(v, n, n, k, k + 1, c, s);
    givens_complex64(f + 0.0 * I, g + 0.0 * I, &c, &s);
    diag[k] = hypot_float64(f, g);
    f = c * superdiag[k] + creal(s) * diag[k + 1];
    diag[k + 1] = -creal(conj(s)) * superdiag[k] + c * diag[k + 1];
    if (k < q - 1) {
      g = creal(s) * superdiag[k + 1];
      superdiag[k + 1] = c * superdiag[k + 1];
    }
    apply_givens_left_complex64(u, m, m, k, k + 1, c, s);
  }
  superdiag[q - 1] = f;
}

static void svd_iterate_complex32(float* diag, float* superdiag, complex32* u,
                                  complex32* v, int m, int n) {
  const int minmn = (m < n ? m : n);
  const float tol = NX_EPS32 * (float)(m > n ? m : n);
  const int max_iter = 75 * minmn;
  int iter = 0;
  while (iter++ < max_iter) {
    int converged = 1;
    for (int i = 0; i < minmn - 1; i++) {
      if (fabsf(superdiag[i]) > tol * (diag[i] + diag[i + 1])) {
        converged = 0;
        break;
      }
    }
    if (converged) break;
    int q_pos = minmn - 1;
    while (q_pos > 0 && fabsf(superdiag[q_pos - 1]) <=
                            tol * (diag[q_pos - 1] + diag[q_pos])) {
      superdiag[q_pos - 1] = 0.0f;
      q_pos--;
    }
    int p_pos = q_pos;
    while (p_pos > 0 &&
           fabsf(superdiag[p_pos - 1]) > tol * (diag[p_pos - 1] + diag[p_pos]))
      p_pos--;
    if (p_pos < q_pos) {
      svd_qr_iteration_complex32(diag, superdiag, u, v, m, n, p_pos, q_pos);
    }
  }
}

static void svd_iterate_complex64(double* diag, double* superdiag, complex64* u,
                                  complex64* v, int m, int n) {
  const int minmn = (m < n ? m : n);
  const double tol = NX_EPS64 * (double)(m > n ? m : n);
  const int max_iter = 75 * minmn;
  int iter = 0;
  while (iter++ < max_iter) {
    int converged = 1;
    for (int i = 0; i < minmn - 1; i++) {
      if (fabs(superdiag[i]) > tol * (diag[i] + diag[i + 1])) {
        converged = 0;
        break;
      }
    }
    if (converged) break;
    int q_pos = minmn - 1;
    while (q_pos > 0 && fabs(superdiag[q_pos - 1]) <=
                            tol * (diag[q_pos - 1] + diag[q_pos])) {
      superdiag[q_pos - 1] = 0.0;
      q_pos--;
    }
    int p_pos = q_pos;
    while (p_pos > 0 &&
           fabs(superdiag[p_pos - 1]) > tol * (diag[p_pos - 1] + diag[p_pos]))
      p_pos--;
    if (p_pos < q_pos) {
      svd_qr_iteration_complex64(diag, superdiag, u, v, m, n, p_pos, q_pos);
    }
  }
}

static void svd_complex32(complex32* a, complex32* u, float* s, complex32* vt,
                          int m, int n, int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  complex32* a_copy = (complex32*)malloc(m * n * sizeof(complex32));
  if (!a_copy) return;
  memcpy(a_copy, a, m * n * sizeof(complex32));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  lapack_int ldu = full_matrices ? m : minmn;
  lapack_int ldvt = full_matrices ? n : minmn;

  // Allocate space for superbidiagonal elements (not used in our interface)
  float* superb = (float*)malloc((minmn - 1) * sizeof(float));
  if (!superb) {
    free(a_copy);
    return;
  }

  lapack_int info = LAPACKE_cgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  // Note: LAPACK returns singular values in descending order, which matches our expectation
  // Note: For complex SVD, LAPACK returns V^H (conjugate transpose), but our interface expects V^T
  // We need to conjugate the result to match our expected output
  if (full_matrices) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  } else {
    for (int i = 0; i < minmn; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  }
}

static void svd_complex64(complex64* a, complex64* u, double* s, complex64* vt,
                          int m, int n, int full_matrices) {
  // LAPACK destroys the input matrix, so we need to make a copy
  complex64* a_copy = (complex64*)malloc(m * n * sizeof(complex64));
  if (!a_copy) return;
  memcpy(a_copy, a, m * n * sizeof(complex64));

  char jobu = full_matrices ? 'A' : 'S';
  char jobvt = full_matrices ? 'A' : 'S';
  int minmn = m < n ? m : n;
  lapack_int ldu = full_matrices ? m : minmn;
  lapack_int ldvt = full_matrices ? n : minmn;

  // Allocate space for superbidiagonal elements (not used in our interface)
  double* superb = (double*)malloc((minmn - 1) * sizeof(double));
  if (!superb) {
    free(a_copy);
    return;
  }

  lapack_int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a_copy, n, s, u, ldu, vt, ldvt, superb);
  free(a_copy);
  free(superb);
  // Note: LAPACK returns singular values in descending order, which matches our expectation
  // Note: For complex SVD, LAPACK returns V^H (conjugate transpose), but our interface expects V^T
  // We need to conjugate the result to match our expected output
  if (full_matrices) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  } else {
    for (int i = 0; i < minmn; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = conj(vt[i * n + j]);
      }
    }
  }
}

static void svd_float16(uint16_t* a, uint16_t* u, uint16_t* s, uint16_t* vt,
                        int m, int n, int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = half_to_float(a[i]);
  svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  for (int i = 0; i < m * u_cols; i++) u[i] = float_to_half(u_float[i]);
  for (int i = 0; i < minmn; i++) s[i] = float_to_half(s_float[i]);
  for (int i = 0; i < vt_rows * n; i++) vt[i] = float_to_half(vt_float[i]);
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
}

static void svd_bfloat16(caml_ba_bfloat16* a, caml_ba_bfloat16* u,
                         caml_ba_bfloat16* s, caml_ba_bfloat16* vt, int m,
                         int n, int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = bfloat16_to_float(a[i]);
  svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  for (int i = 0; i < m * u_cols; i++) u[i] = float_to_bfloat16(u_float[i]);
  for (int i = 0; i < minmn; i++) s[i] = float_to_bfloat16(s_float[i]);
  for (int i = 0; i < vt_rows * n; i++) vt[i] = float_to_bfloat16(vt_float[i]);
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
}

static void svd_f8e4m3(caml_ba_fp8_e4m3* a, caml_ba_fp8_e4m3* u,
                       caml_ba_fp8_e4m3* s, caml_ba_fp8_e4m3* vt, int m, int n,
                       int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = fp8_e4m3_to_float(a[i]);
  svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  for (int i = 0; i < m * u_cols; i++) u[i] = float_to_fp8_e4m3(u_float[i]);
  for (int i = 0; i < minmn; i++) s[i] = float_to_fp8_e4m3(s_float[i]);
  for (int i = 0; i < vt_rows * n; i++) vt[i] = float_to_fp8_e4m3(vt_float[i]);
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
}

static void svd_f8e5m2(caml_ba_fp8_e5m2* a, caml_ba_fp8_e5m2* u,
                       caml_ba_fp8_e5m2* s, caml_ba_fp8_e5m2* vt, int m, int n,
                       int full_matrices) {
  int minmn = m < n ? m : n;
  float* a_float = (float*)malloc(m * n * sizeof(float));
  int u_cols = full_matrices ? m : minmn;
  float* u_float = (float*)malloc(m * u_cols * sizeof(float));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  float* vt_float = (float*)malloc(vt_rows * n * sizeof(float));
  if (!a_float || !u_float || !s_float || !vt_float) {
    free(a_float);
    free(u_float);
    free(s_float);
    free(vt_float);
    return;
  }
  for (int i = 0; i < m * n; i++) a_float[i] = fp8_e5m2_to_float(a[i]);
  svd_float32(a_float, u_float, s_float, vt_float, m, n, full_matrices);
  for (int i = 0; i < m * u_cols; i++) u[i] = float_to_fp8_e5m2(u_float[i]);
  for (int i = 0; i < minmn; i++) s[i] = float_to_fp8_e5m2(s_float[i]);
  for (int i = 0; i < vt_rows * n; i++) vt[i] = float_to_fp8_e5m2(vt_float[i]);
  free(a_float);
  free(u_float);
  free(s_float);
  free(vt_float);
}

static void svd_complex16(caml_ba_complex16* a, caml_ba_complex16* u,
                          caml_ba_complex16* s, caml_ba_complex16* vt, int m,
                          int n, int full_matrices) {
  int minmn = m < n ? m : n;
  complex32* a_complex = (complex32*)malloc(m * n * sizeof(complex32));
  int u_cols = full_matrices ? m : minmn;
  complex32* u_complex = (complex32*)malloc(m * u_cols * sizeof(complex32));
  float* s_float = (float*)malloc(minmn * sizeof(float));
  int vt_rows = full_matrices ? n : minmn;
  complex32* vt_complex = (complex32*)malloc(vt_rows * n * sizeof(complex32));
  if (!a_complex || !u_complex || !s_float || !vt_complex) {
    free(a_complex);
    free(u_complex);
    free(s_float);
    free(vt_complex);
    return;
  }
  for (int i = 0; i < m * n; i++) {
    a_complex[i] = half_to_float(a[i].re) + I * half_to_float(a[i].im);
  }
  svd_complex32(a_complex, u_complex, s_float, vt_complex, m, n, full_matrices);
  for (int i = 0; i < m * u_cols; i++) {
    u[i].re = float_to_half(crealf(u_complex[i]));
    u[i].im = float_to_half(cimagf(u_complex[i]));
  }
  for (int i = 0; i < minmn; i++) {
    s[i].re = float_to_half(s_float[i]);
    s[i].im = 0.0f;
  }
  for (int i = 0; i < vt_rows * n; i++) {
    vt[i].re = float_to_half(crealf(vt_complex[i]));
    vt[i].im = float_to_half(cimagf(vt_complex[i]));
  }
  free(a_complex);
  free(u_complex);
  free(s_float);
  free(vt_complex);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_op_svd(value v_in, value v_u, value v_s, value v_vt,
                              value v_full_matrices) {
  CAMLparam5(v_in, v_u, v_s, v_vt, v_full_matrices);
  int full_matrices = Int_val(v_full_matrices);
  ndarray_t in = extract_ndarray(v_in);
  ndarray_t u_nd = extract_ndarray(v_u);
  ndarray_t s_nd = extract_ndarray(v_s);
  ndarray_t vt_nd = extract_ndarray(v_vt);
  struct caml_ba_array* ba_in = Caml_ba_array_val(Field(v_in, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_u = Caml_ba_array_val(Field(v_u, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_s = Caml_ba_array_val(Field(v_s, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_vt = Caml_ba_array_val(Field(v_vt, FFI_TENSOR_DATA));
  int kind = ba_in->flags & CAML_BA_KIND_MASK;
  if (in.ndim < 2) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&u_nd);
    cleanup_ndarray(&s_nd);
    cleanup_ndarray(&vt_nd);
    caml_failwith("svd: input must have at least 2 dimensions");
  }
  int m = in.shape[in.ndim - 2];
  int n = in.shape[in.ndim - 1];
  int minmn = m < n ? m : n;
  int u_cols = full_matrices ? m : minmn;
  int vt_rows = full_matrices ? n : minmn;
  if (u_nd.shape[u_nd.ndim - 1] != u_cols || u_nd.shape[u_nd.ndim - 2] != m ||
      vt_nd.shape[vt_nd.ndim - 1] != n ||
      vt_nd.shape[vt_nd.ndim - 2] != vt_rows ||
      s_nd.shape[s_nd.ndim - 1] != minmn) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&u_nd);
    cleanup_ndarray(&s_nd);
    cleanup_ndarray(&vt_nd);
    caml_failwith("svd: output shapes mismatch");
  }
  int batch_size = 1;
  for (int i = 0; i < in.ndim - 2; i++) {
    batch_size *= in.shape[i];
  }
  int s_in_row = in.strides[in.ndim - 2];
  int s_in_col = in.strides[in.ndim - 1];
  int s_u_row = u_nd.strides[u_nd.ndim - 2];
  int s_u_col = u_nd.strides[u_nd.ndim - 1];
  int s_s_stride = s_nd.strides[s_nd.ndim - 1];
  int s_vt_row = vt_nd.strides[vt_nd.ndim - 2];
  int s_vt_col = vt_nd.strides[vt_nd.ndim - 1];
  caml_enter_blocking_section();
  for (int b = 0; b < batch_size; b++) {
    size_t off_in = in.offset;
    size_t off_u = u_nd.offset;
    size_t off_s = s_nd.offset;
    size_t off_vt = vt_nd.offset;
    if (in.ndim > 2) {
      int remaining = b;
      for (int i = in.ndim - 3; i >= 0; i--) {
        int coord = remaining % in.shape[i];
        remaining /= in.shape[i];
        off_in += coord * in.strides[i];
        off_u += coord * u_nd.strides[i];
        off_s += coord * s_nd.strides[i];
        off_vt += coord * vt_nd.strides[i];
      }
    }
    switch (kind) {
      case CAML_BA_FLOAT32: {
        float* base_in = (float*)ba_in->data + off_in;
        float* base_u = (float*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;  // S is always float64
        float* base_vt = (float*)ba_vt->data + off_vt;
        float* A = (float*)malloc((size_t)m * n * sizeof(float));
        float* U = (float*)malloc((size_t)m * u_cols * sizeof(float));
        float* S = (float*)malloc((size_t)minmn * sizeof(float));
        float* VT = (float*)malloc((size_t)vt_rows * n * sizeof(float));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_f32(A, base_in, m, n, s_in_row, s_in_col);
        svd_float32(A, U, S, VT, m, n, full_matrices);
        nx_unpack_f32(base_u, U, m, u_cols, s_u_row, s_u_col);
        // Convert S from float32 to float64
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = (double)S[i];
        nx_unpack_f32(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_in = (double*)ba_in->data + off_in;
        double* base_u = (double*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;
        double* base_vt = (double*)ba_vt->data + off_vt;
        double* A = (double*)malloc((size_t)m * n * sizeof(double));
        double* U = (double*)malloc((size_t)m * u_cols * sizeof(double));
        double* S = (double*)malloc((size_t)minmn * sizeof(double));
        double* VT = (double*)malloc((size_t)vt_rows * n * sizeof(double));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_f64(A, base_in, m, n, s_in_row, s_in_col);
        svd_float64(A, U, S, VT, m, n, full_matrices);
        nx_unpack_f64(base_u, U, m, u_cols, s_u_row, s_u_col);
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        nx_unpack_f64(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_COMPLEX32: {
        complex32* base_in = (complex32*)ba_in->data + off_in;
        complex32* base_u = (complex32*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;  // S is always float64
        complex32* base_vt = (complex32*)ba_vt->data + off_vt;
        complex32* A = (complex32*)malloc((size_t)m * n * sizeof(complex32));
        complex32* U =
            (complex32*)malloc((size_t)m * u_cols * sizeof(complex32));
        float* S = (float*)malloc((size_t)minmn * sizeof(float));
        complex32* VT =
            (complex32*)malloc((size_t)vt_rows * n * sizeof(complex32));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_c32(A, base_in, m, n, s_in_row, s_in_col);
        svd_complex32(A, U, S, VT, m, n, full_matrices);
        nx_unpack_c32(base_u, U, m, u_cols, s_u_row, s_u_col);
        // Convert S from float32 to float64
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = (double)S[i];
        nx_unpack_c32(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_COMPLEX64: {
        complex64* base_in = (complex64*)ba_in->data + off_in;
        complex64* base_u = (complex64*)ba_u->data + off_u;
        double* base_s = (double*)ba_s->data + off_s;
        complex64* base_vt = (complex64*)ba_vt->data + off_vt;
        complex64* A = (complex64*)malloc((size_t)m * n * sizeof(complex64));
        complex64* U =
            (complex64*)malloc((size_t)m * u_cols * sizeof(complex64));
        double* S = (double*)malloc((size_t)minmn * sizeof(double));
        complex64* VT =
            (complex64*)malloc((size_t)vt_rows * n * sizeof(complex64));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        nx_pack_c64(A, base_in, m, n, s_in_row, s_in_col);
        svd_complex64(A, U, S, VT, m, n, full_matrices);
        nx_unpack_c64(base_u, U, m, u_cols, s_u_row, s_u_col);
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        nx_unpack_c64(base_vt, VT, vt_rows, n, s_vt_row, s_vt_col);
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case CAML_BA_FLOAT16: {
        uint16_t* base_in = (uint16_t*)ba_in->data + off_in;
        uint16_t* base_u = (uint16_t*)ba_u->data + off_u;
        uint16_t* base_s = (uint16_t*)ba_s->data + off_s;
        uint16_t* base_vt = (uint16_t*)ba_vt->data + off_vt;
        uint16_t* A = (uint16_t*)malloc((size_t)m * n * sizeof(uint16_t));
        uint16_t* U = (uint16_t*)malloc((size_t)m * u_cols * sizeof(uint16_t));
        uint16_t* S = (uint16_t*)malloc((size_t)minmn * sizeof(uint16_t));
        uint16_t* VT =
            (uint16_t*)malloc((size_t)vt_rows * n * sizeof(uint16_t));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_float16(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_BFLOAT16: {
        caml_ba_bfloat16* base_in = (caml_ba_bfloat16*)ba_in->data + off_in;
        caml_ba_bfloat16* base_u = (caml_ba_bfloat16*)ba_u->data + off_u;
        caml_ba_bfloat16* base_s = (caml_ba_bfloat16*)ba_s->data + off_s;
        caml_ba_bfloat16* base_vt = (caml_ba_bfloat16*)ba_vt->data + off_vt;
        caml_ba_bfloat16* A =
            (caml_ba_bfloat16*)malloc((size_t)m * n * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* U = (caml_ba_bfloat16*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* S =
            (caml_ba_bfloat16*)malloc((size_t)minmn * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* VT = (caml_ba_bfloat16*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_bfloat16));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_bfloat16(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_FP8_E4M3: {
        caml_ba_fp8_e4m3* base_in = (caml_ba_fp8_e4m3*)ba_in->data + off_in;
        caml_ba_fp8_e4m3* base_u = (caml_ba_fp8_e4m3*)ba_u->data + off_u;
        caml_ba_fp8_e4m3* base_s = (caml_ba_fp8_e4m3*)ba_s->data + off_s;
        caml_ba_fp8_e4m3* base_vt = (caml_ba_fp8_e4m3*)ba_vt->data + off_vt;
        caml_ba_fp8_e4m3* A =
            (caml_ba_fp8_e4m3*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* U = (caml_ba_fp8_e4m3*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* S =
            (caml_ba_fp8_e4m3*)malloc((size_t)minmn * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* VT = (caml_ba_fp8_e4m3*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_fp8_e4m3));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_f8e4m3(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_FP8_E5M2: {
        caml_ba_fp8_e5m2* base_in = (caml_ba_fp8_e5m2*)ba_in->data + off_in;
        caml_ba_fp8_e5m2* base_u = (caml_ba_fp8_e5m2*)ba_u->data + off_u;
        caml_ba_fp8_e5m2* base_s = (caml_ba_fp8_e5m2*)ba_s->data + off_s;
        caml_ba_fp8_e5m2* base_vt = (caml_ba_fp8_e5m2*)ba_vt->data + off_vt;
        caml_ba_fp8_e5m2* A =
            (caml_ba_fp8_e5m2*)malloc((size_t)m * n * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* U = (caml_ba_fp8_e5m2*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* S =
            (caml_ba_fp8_e5m2*)malloc((size_t)minmn * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* VT = (caml_ba_fp8_e5m2*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_fp8_e5m2));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_f8e5m2(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      case NX_BA_COMPLEX16: {
        caml_ba_complex16* base_in = (caml_ba_complex16*)ba_in->data + off_in;
        caml_ba_complex16* base_u = (caml_ba_complex16*)ba_u->data + off_u;
        caml_ba_complex16* base_s = (caml_ba_complex16*)ba_s->data + off_s;
        caml_ba_complex16* base_vt = (caml_ba_complex16*)ba_vt->data + off_vt;
        caml_ba_complex16* A = (caml_ba_complex16*)malloc(
            (size_t)m * n * sizeof(caml_ba_complex16));
        caml_ba_complex16* U = (caml_ba_complex16*)malloc(
            (size_t)m * u_cols * sizeof(caml_ba_complex16));
        caml_ba_complex16* S = (caml_ba_complex16*)malloc(
            (size_t)minmn * sizeof(caml_ba_complex16));
        caml_ba_complex16* VT = (caml_ba_complex16*)malloc(
            (size_t)vt_rows * n * sizeof(caml_ba_complex16));
        if (!A || !U || !S || !VT) {
          free(A);
          free(U);
          free(S);
          free(VT);
          continue;
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        svd_complex16(A, U, S, VT, m, n, full_matrices);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < u_cols; j++) {
            base_u[i * s_u_row + j * s_u_col] = U[i * u_cols + j];
          }
        }
        for (int i = 0; i < minmn; i++) base_s[i * s_s_stride] = S[i];
        for (int i = 0; i < vt_rows; i++) {
          for (int j = 0; j < n; j++) {
            base_vt[i * s_vt_row + j * s_vt_col] = VT[i * n + j];
          }
        }
        free(A);
        free(U);
        free(S);
        free(VT);
        break;
      }
      default:
        caml_leave_blocking_section();
        cleanup_ndarray(&in);
        cleanup_ndarray(&u_nd);
        cleanup_ndarray(&s_nd);
        cleanup_ndarray(&vt_nd);
        caml_failwith("svd: unsupported dtype");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&in);
  cleanup_ndarray(&u_nd);
  cleanup_ndarray(&s_nd);
  cleanup_ndarray(&vt_nd);
  CAMLreturn(Val_unit);
}