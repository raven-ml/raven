#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <complex.h>
#include <float.h>

#include "nx_c_shared.h"

// Machine epsilon for float32 and float64
#define NX_EPS32 FLT_EPSILON
#define NX_EPS64 DBL_EPSILON

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

// Cholesky decomposition implementations
static int cholesky_float32(float* a, int n, int upper) {
  float tol = NX_EPS32 * n;
  if (upper) {
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        sum += a[j * n + k] * a[j * n + k];
      }
      float diag = a[k * n + k] - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrtf(fmaxf(diag, 0.0f));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        float sum_i = 0.0f;
        for (int j = 0; j < k; j++) {
          sum_i += a[j * n + i] * a[j * n + k];
        }
        a[k * n + i] = (a[k * n + i] - sum_i) / a[k * n + k];
      }
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0f;
      }
    }
  } else {
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        sum += a[k * n + j] * a[k * n + j];
      }
      float diag = a[k * n + k] - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrtf(fmaxf(diag, 0.0f));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        float sum_i = 0.0f;
        for (int j = 0; j < k; j++) {
          sum_i += a[i * n + j] * a[k * n + j];
        }
        a[i * n + k] = (a[i * n + k] - sum_i) / a[k * n + k];
      }
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0f;
      }
    }
  }
  return 0;
}

static int cholesky_float64(double* a, int n, int upper) {
  double tol = NX_EPS64 * n;
  if (upper) {
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        sum += a[j * n + k] * a[j * n + k];
      }
      double diag = a[k * n + k] - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrt(fmax(diag, 0.0));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        double sum_i = 0.0;
        for (int j = 0; j < k; j++) {
          sum_i += a[j * n + i] * a[j * n + k];
        }
        a[k * n + i] = (a[k * n + i] - sum_i) / a[k * n + k];
      }
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0;
      }
    }
  } else {
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        sum += a[k * n + j] * a[k * n + j];
      }
      double diag = a[k * n + k] - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrt(fmax(diag, 0.0));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        double sum_i = 0.0;
        for (int j = 0; j < k; j++) {
          sum_i += a[i * n + j] * a[k * n + j];
        }
        a[i * n + k] = (a[i * n + k] - sum_i) / a[k * n + k];
      }
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0;
      }
    }
  }
  return 0;
}

static int cholesky_complex32(complex32* a, int n, int upper) {
  float tol = NX_EPS32 * n;
  if (upper) {
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        complex32 val = a[j * n + k];
        sum += crealf(val) * crealf(val) + cimagf(val) * cimagf(val);
      }
      float diag = crealf(a[k * n + k]) - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrtf(fmaxf(diag, 0.0f));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        complex32 sum_c = 0.0f + 0.0f * I;
        for (int j = 0; j < k; j++) {
          sum_c += conjf(a[j * n + k]) * a[j * n + i];
        }
        a[k * n + i] = (a[k * n + i] - sum_c) / a[k * n + k];
      }
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0f + 0.0f * I;
      }
    }
  } else {
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        complex32 val = a[k * n + j];
        sum += crealf(val) * crealf(val) + cimagf(val) * cimagf(val);
      }
      float diag = crealf(a[k * n + k]) - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrtf(fmaxf(diag, 0.0f));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        complex32 sum_c = 0.0f + 0.0f * I;
        for (int j = 0; j < k; j++) {
          sum_c += a[i * n + j] * conjf(a[k * n + j]);
        }
        a[i * n + k] = (a[i * n + k] - sum_c) / a[k * n + k];
      }
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0f + 0.0f * I;
      }
    }
  }
  return 0;
}

static int cholesky_complex64(complex64* a, int n, int upper) {
  double tol = NX_EPS64 * n;
  if (upper) {
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        complex64 val = a[j * n + k];
        sum += creal(val) * creal(val) + cimag(val) * cimag(val);
      }
      double diag = creal(a[k * n + k]) - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrt(fmax(diag, 0.0));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        complex64 sum_c = 0.0 + 0.0 * I;
        for (int j = 0; j < k; j++) {
          sum_c += conj(a[j * n + k]) * a[j * n + i];
        }
        a[k * n + i] = (a[k * n + i] - sum_c) / a[k * n + k];
      }
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0 + 0.0 * I;
      }
    }
  } else {
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        complex64 val = a[k * n + j];
        sum += creal(val) * creal(val) + cimag(val) * cimag(val);
      }
      double diag = creal(a[k * n + k]) - sum;
      if (diag < -tol) return -1;
      a[k * n + k] = sqrt(fmax(diag, 0.0));
#pragma omp parallel for if (n - k > 100)
      for (int i = k + 1; i < n; i++) {
        complex64 sum_c = 0.0 + 0.0 * I;
        for (int j = 0; j < k; j++) {
          sum_c += a[i * n + j] * conj(a[k * n + j]);
        }
        a[i * n + k] = (a[i * n + k] - sum_c) / a[k * n + k];
      }
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0 + 0.0 * I;
      }
    }
  }
  return 0;
}

static int cholesky_float16(uint16_t* a, int n, int upper) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  if (!a_float) return -1;
  for (int i = 0; i < n * n; i++) {
    a_float[i] = half_to_float(a[i]);
  }
  int status = cholesky_float32(a_float, n, upper);
  if (status == 0) {
    for (int i = 0; i < n * n; i++) {
      a[i] = float_to_half(a_float[i]);
    }
  }
  free(a_float);
  return status;
}

static int cholesky_bfloat16(caml_ba_bfloat16* a, int n, int upper) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  if (!a_float) return -1;
  for (int i = 0; i < n * n; i++) {
    a_float[i] = bfloat16_to_float(a[i]);
  }
  int status = cholesky_float32(a_float, n, upper);
  if (status == 0) {
    for (int i = 0; i < n * n; i++) {
      a[i] = float_to_bfloat16(a_float[i]);
    }
  }
  free(a_float);
  return status;
}

static int cholesky_f8e4m3(caml_ba_fp8_e4m3* a, int n, int upper) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  if (!a_float) return -1;
  for (int i = 0; i < n * n; i++) {
    a_float[i] = fp8_e4m3_to_float(a[i]);
  }
  int status = cholesky_float32(a_float, n, upper);
  if (status == 0) {
    for (int i = 0; i < n * n; i++) {
      a[i] = float_to_fp8_e4m3(a_float[i]);
    }
  }
  free(a_float);
  return status;
}

static int cholesky_f8e5m2(caml_ba_fp8_e5m2* a, int n, int upper) {
  float* a_float = (float*)malloc(n * n * sizeof(float));
  if (!a_float) return -1;
  for (int i = 0; i < n * n; i++) {
    a_float[i] = fp8_e5m2_to_float(a[i]);
  }
  int status = cholesky_float32(a_float, n, upper);
  if (status == 0) {
    for (int i = 0; i < n * n; i++) {
      a[i] = float_to_fp8_e5m2(a_float[i]);
    }
  }
  free(a_float);
  return status;
}

static int cholesky_complex16(caml_ba_complex16* a, int n, int upper) {
  complex32* a_complex = (complex32*)malloc(n * n * sizeof(complex32));
  if (!a_complex) return -1;
  for (int i = 0; i < n * n; i++) {
    a_complex[i] = half_to_float(a[i].re) + I * half_to_float(a[i].im);
  }
  int status = cholesky_complex32(a_complex, n, upper);
  if (status == 0) {
    for (int i = 0; i < n * n; i++) {
      a[i].re = float_to_half(crealf(a_complex[i]));
      a[i].im = float_to_half(cimagf(a_complex[i]));
    }
  }
  free(a_complex);
  return status;
}

// Triangular solve implementations
static void triangular_solve_float32(const float* a, const float* b, float* x,
                                     int m, int n, int upper, int transpose,
                                     int unit_diag) {
  float tol = NX_EPS32 * m;
  memcpy(x, b, m * n * sizeof(float));
  if (!transpose) {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          float sum = 0.0f;
          for (int k = i + 1; k < m; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0f) ? 0.0f : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0.0f;
          for (int k = 0; k < i; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0f) ? 0.0f : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    }
  } else {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0.0f;
          for (int k = 0; k < i; k++) {
            sum += a[k * m + i] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0f) ? 0.0f : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          float sum = 0.0f;
          for (int k = i + 1; k < m; k++) {
            sum += a[k * m + i] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0f) ? 0.0f : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    }
  }
}

static void triangular_solve_float64(const double* a, const double* b,
                                     double* x, int m, int n, int upper,
                                     int transpose, int unit_diag) {
  double tol = NX_EPS64 * m;
  memcpy(x, b, m * n * sizeof(double));
  if (!transpose) {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          double sum = 0.0;
          for (int k = i + 1; k < m; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabs(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0) ? 0.0 : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          double sum = 0.0;
          for (int k = 0; k < i; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabs(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0) ? 0.0 : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    }
  } else {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          double sum = 0.0;
          for (int k = 0; k < i; k++) {
            sum += a[k * m + i] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabs(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0) ? 0.0 : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          double sum = 0.0;
          for (int k = i + 1; k < m; k++) {
            sum += a[k * m + i] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (fabs(a[i * m + i]) < tol) {
              x[i * n + j] = (x[i * n + j] == 0.0) ? 0.0 : INFINITY;
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    }
  }
}

static void triangular_solve_complex32(const complex32* a, const complex32* b,
                                       complex32* x, int m, int n, int upper,
                                       int transpose, int unit_diag) {
  float tol = NX_EPS32 * m;
  memcpy(x, b, m * n * sizeof(complex32));
  if (!transpose) {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          complex32 sum = 0.0f + 0.0f * I;
          for (int k = i + 1; k < m; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (cabsf(x[i * n + j]) == 0.0f)
                                 ? (0.0f + 0.0f * I)
                                 : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          complex32 sum = 0.0f + 0.0f * I;
          for (int k = 0; k < i; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (cabsf(x[i * n + j]) == 0.0f)
                                 ? (0.0f + 0.0f * I)
                                 : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    }
  } else {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          complex32 sum = 0.0f + 0.0f * I;
          for (int k = 0; k < i; k++) {
            sum += conjf(a[k * m + i]) * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (cabsf(x[i * n + j]) == 0.0f)
                                 ? (0.0f + 0.0f * I)
                                 : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= conjf(a[i * m + i]);
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          complex32 sum = 0.0f + 0.0f * I;
          for (int k = i + 1; k < m; k++) {
            sum += conjf(a[k * m + i]) * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabsf(a[i * m + i]) < tol) {
              x[i * n + j] = (cabsf(x[i * n + j]) == 0.0f)
                                 ? (0.0f + 0.0f * I)
                                 : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= conjf(a[i * m + i]);
            }
          }
        }
      }
    }
  }
}

static void triangular_solve_complex64(const complex64* a, const complex64* b,
                                       complex64* x, int m, int n, int upper,
                                       int transpose, int unit_diag) {
  double tol = NX_EPS64 * m;
  memcpy(x, b, m * n * sizeof(complex64));
  if (!transpose) {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          complex64 sum = 0.0 + 0.0 * I;
          for (int k = i + 1; k < m; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabs(a[i * m + i]) < tol) {
              x[i * n + j] = (cabs(x[i * n + j]) == 0.0) ? (0.0 + 0.0 * I)
                                                         : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          complex64 sum = 0.0 + 0.0 * I;
          for (int k = 0; k < i; k++) {
            sum += a[i * m + k] * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabs(a[i * m + i]) < tol) {
              x[i * n + j] = (cabs(x[i * n + j]) == 0.0) ? (0.0 + 0.0 * I)
                                                         : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= a[i * m + i];
            }
          }
        }
      }
    }
  } else {
    if (upper) {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          complex64 sum = 0.0 + 0.0 * I;
          for (int k = 0; k < i; k++) {
            sum += conj(a[k * m + i]) * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabs(a[i * m + i]) < tol) {
              x[i * n + j] = (cabs(x[i * n + j]) == 0.0) ? (0.0 + 0.0 * I)
                                                         : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= conj(a[i * m + i]);
            }
          }
        }
      }
    } else {
#pragma omp parallel for if (n > 100)
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          complex64 sum = 0.0 + 0.0 * I;
          for (int k = i + 1; k < m; k++) {
            sum += conj(a[k * m + i]) * x[k * n + j];
          }
          x[i * n + j] -= sum;
          if (!unit_diag) {
            if (cabs(a[i * m + i]) < tol) {
              x[i * n + j] = (cabs(x[i * n + j]) == 0.0) ? (0.0 + 0.0 * I)
                                                         : (INFINITY + NAN * I);
            } else {
              x[i * n + j] /= conj(a[i * m + i]);
            }
          }
        }
      }
    }
  }
}

static int triangular_solve_float16(const uint16_t* a, const uint16_t* b,
                                    uint16_t* x, int m, int n, int upper,
                                    int transpose, int unit_diag) {
  float* a_float = (float*)malloc(m * m * sizeof(float));
  float* b_float = (float*)malloc(m * n * sizeof(float));
  float* x_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !b_float || !x_float) {
    free(a_float);
    free(b_float);
    free(x_float);
    return -1;
  }
  for (int i = 0; i < m * m; i++) a_float[i] = half_to_float(a[i]);
  for (int i = 0; i < m * n; i++) b_float[i] = half_to_float(b[i]);
  triangular_solve_float32(a_float, b_float, x_float, m, n, upper, transpose,
                           unit_diag);
  for (int i = 0; i < m * n; i++) x[i] = float_to_half(x_float[i]);
  free(a_float);
  free(b_float);
  free(x_float);
  return 0;
}

static int triangular_solve_bfloat16(const caml_ba_bfloat16* a,
                                     const caml_ba_bfloat16* b,
                                     caml_ba_bfloat16* x, int m, int n,
                                     int upper, int transpose, int unit_diag) {
  float* a_float = (float*)malloc(m * m * sizeof(float));
  float* b_float = (float*)malloc(m * n * sizeof(float));
  float* x_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !b_float || !x_float) {
    free(a_float);
    free(b_float);
    free(x_float);
    return -1;
  }
  for (int i = 0; i < m * m; i++) a_float[i] = bfloat16_to_float(a[i]);
  for (int i = 0; i < m * n; i++) b_float[i] = bfloat16_to_float(b[i]);
  triangular_solve_float32(a_float, b_float, x_float, m, n, upper, transpose,
                           unit_diag);
  for (int i = 0; i < m * n; i++) x[i] = float_to_bfloat16(x_float[i]);
  free(a_float);
  free(b_float);
  free(x_float);
  return 0;
}

static int triangular_solve_f8e4m3(const caml_ba_fp8_e4m3* a,
                                   const caml_ba_fp8_e4m3* b,
                                   caml_ba_fp8_e4m3* x, int m, int n, int upper,
                                   int transpose, int unit_diag) {
  float* a_float = (float*)malloc(m * m * sizeof(float));
  float* b_float = (float*)malloc(m * n * sizeof(float));
  float* x_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !b_float || !x_float) {
    free(a_float);
    free(b_float);
    free(x_float);
    return -1;
  }
  for (int i = 0; i < m * m; i++) a_float[i] = fp8_e4m3_to_float(a[i]);
  for (int i = 0; i < m * n; i++) b_float[i] = fp8_e4m3_to_float(b[i]);
  triangular_solve_float32(a_float, b_float, x_float, m, n, upper, transpose,
                           unit_diag);
  for (int i = 0; i < m * n; i++) x[i] = float_to_fp8_e4m3(x_float[i]);
  free(a_float);
  free(b_float);
  free(x_float);
  return 0;
}

static int triangular_solve_f8e5m2(const caml_ba_fp8_e5m2* a,
                                   const caml_ba_fp8_e5m2* b,
                                   caml_ba_fp8_e5m2* x, int m, int n, int upper,
                                   int transpose, int unit_diag) {
  float* a_float = (float*)malloc(m * m * sizeof(float));
  float* b_float = (float*)malloc(m * n * sizeof(float));
  float* x_float = (float*)malloc(m * n * sizeof(float));
  if (!a_float || !b_float || !x_float) {
    free(a_float);
    free(b_float);
    free(x_float);
    return -1;
  }
  for (int i = 0; i < m * m; i++) a_float[i] = fp8_e5m2_to_float(a[i]);
  for (int i = 0; i < m * n; i++) b_float[i] = fp8_e5m2_to_float(b[i]);
  triangular_solve_float32(a_float, b_float, x_float, m, n, upper, transpose,
                           unit_diag);
  for (int i = 0; i < m * n; i++) x[i] = float_to_fp8_e5m2(x_float[i]);
  free(a_float);
  free(b_float);
  free(x_float);
  return 0;
}

static int triangular_solve_complex16(const caml_ba_complex16* a,
                                      const caml_ba_complex16* b,
                                      caml_ba_complex16* x, int m, int n,
                                      int upper, int transpose, int unit_diag) {
  complex32* a_complex = (complex32*)malloc(m * m * sizeof(complex32));
  complex32* b_complex = (complex32*)malloc(m * n * sizeof(complex32));
  complex32* x_complex = (complex32*)malloc(m * n * sizeof(complex32));
  if (!a_complex || !b_complex || !x_complex) {
    free(a_complex);
    free(b_complex);
    free(x_complex);
    return -1;
  }
  for (int i = 0; i < m * m; i++) {
    a_complex[i] = half_to_float(a[i].re) + I * half_to_float(a[i].im);
  }
  for (int i = 0; i < m * n; i++) {
    b_complex[i] = half_to_float(b[i].re) + I * half_to_float(b[i].im);
  }
  triangular_solve_complex32(a_complex, b_complex, x_complex, m, n, upper,
                             transpose, unit_diag);
  for (int i = 0; i < m * n; i++) {
    x[i].re = float_to_half(crealf(x_complex[i]));
    x[i].im = float_to_half(cimagf(x_complex[i]));
  }
  free(a_complex);
  free(b_complex);
  free(x_complex);
  return 0;
}

// QR decomposition implementations
static void qr_decompose_float32(float* a, float* q, float* r, int m, int n,
                                 int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j) q[i * k + j] = (i == j) ? 1.f : 0.f;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) r[i * n + j] = a[i * n + j];
  const int lim = (m < n ? m : n);
#pragma omp parallel for if (lim > 100)
  for (int j = 0; j < lim; ++j) {
    float norm2 = 0.f;
    for (int i = j; i < m; ++i) norm2 += r[i * n + j] * r[i * n + j];
    float norm = sqrtf(norm2);
    if (norm == 0.f) continue;
    float sign = (r[j * n + j] >= 0.f) ? 1.f : -1.f;
    float u1 = r[j * n + j] + sign * norm;
    float* v = (float*)calloc((size_t)m, sizeof(float));
    if (!v) continue;
    v[j] = 1.f;
    float tail2 = 0.f;
    for (int i = j + 1; i < m; ++i) {
      float vi = r[i * n + j] / u1;
      v[i] = vi;
      tail2 += vi * vi;
    }
    float beta = 2.f / (1.f + tail2);
    for (int col = j; col < n; ++col) {
      float dot = 0.f;
      for (int i = j; i < m; ++i) dot += v[i] * r[i * n + col];
      dot *= beta;
      for (int i = j; i < m; ++i) r[i * n + col] -= dot * v[i];
    }
    for (int col = 0; col < k; ++col) {
      float dot = 0.f;
      for (int i = j; i < m; ++i) dot += v[i] * q[i * k + col];
      dot *= beta;
      for (int i = j; i < m; ++i) q[i * k + col] -= dot * v[i];
    }
    free(v);
  }
  for (int i = 1; i < m; ++i)
    for (int j = 0; j < i && j < n; ++j) r[i * n + j] = 0.f;
}

static void qr_decompose_float64(double* a, double* q, double* r, int m, int n,
                                 int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j) q[i * k + j] = (i == j) ? 1.0 : 0.0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) r[i * n + j] = a[i * n + j];
  const int lim = (m < n ? m : n);
#pragma omp parallel for if (lim > 100)
  for (int j = 0; j < lim; ++j) {
    double norm2 = 0.0;
    for (int i = j; i < m; ++i) norm2 += r[i * n + j] * r[i * n + j];
    double norm = sqrt(norm2);
    if (norm == 0.0) continue;
    double sign = (r[j * n + j] >= 0.0) ? 1.0 : -1.0;
    double u1 = r[j * n + j] + sign * norm;
    double* v = (double*)calloc((size_t)m, sizeof(double));
    if (!v) continue;
    v[j] = 1.0;
    double tail2 = 0.0;
    for (int i = j + 1; i < m; ++i) {
      double vi = r[i * n + j] / u1;
      v[i] = vi;
      tail2 += vi * vi;
    }
    double beta = 2.0 / (1.0 + tail2);
    for (int col = j; col < n; ++col) {
      double dot = 0.0;
      for (int i = j; i < m; ++i) dot += v[i] * r[i * n + col];
      dot *= beta;
      for (int i = j; i < m; ++i) r[i * n + col] -= dot * v[i];
    }
    for (int col = 0; col < k; ++col) {
      double dot = 0.0;
      for (int i = j; i < m; ++i) dot += v[i] * q[i * k + col];
      dot *= beta;
      for (int i = j; i < m; ++i) q[i * k + col] -= dot * v[i];
    }
    free(v);
  }
  for (int i = 1; i < m; ++i)
    for (int j = 0; j < i && j < n; ++j) r[i * n + j] = 0.0;
}

static void qr_decompose_complex32(complex32* a, complex32* q, complex32* r,
                                   int m, int n, int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j) q[i * k + j] = (i == j) ? 1.0f : 0.0f;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) r[i * n + j] = a[i * n + j];
  const int lim = (m < n ? m : n);
#pragma omp parallel for if (lim > 100)
  for (int j = 0; j < lim; ++j) {
    float norm2 = 0.0f;
    for (int i = j; i < m; ++i) {
      norm2 += crealf(r[i * n + j]) * crealf(r[i * n + j]) +
               cimagf(r[i * n + j]) * cimagf(r[i * n + j]);
    }
    float norm = sqrtf(norm2);
    if (norm == 0.0f) continue;
    float sign = (crealf(r[j * n + j]) >= 0.0f) ? 1.0f : -1.0f;
    complex32 u1 = r[j * n + j] + sign * norm;
    complex32* v = (complex32*)calloc((size_t)m, sizeof(complex32));
    if (!v) continue;
    v[j] = 1.0f;
    float tail2 = 0.0f;
    for (int i = j + 1; i < m; ++i) {
      complex32 vi = r[i * n + j] / u1;
      v[i] = vi;
      tail2 += crealf(vi) * crealf(vi) + cimagf(vi) * cimagf(vi);
    }
    float beta = 2.0f / (1.0f + tail2);
    for (int col = j; col < n; ++col) {
      complex32 dot = 0.0f + 0.0f * I;
      for (int i = j; i < m; ++i) dot += conjf(v[i]) * r[i * n + col];
      dot *= beta;
      for (int i = j; i < m; ++i) r[i * n + col] -= dot * v[i];
    }
    for (int col = 0; col < k; ++col) {
      complex32 dot = 0.0f + 0.0f * I;
      for (int i = j; i < m; ++i) dot += conjf(v[i]) * q[i * k + col];
      dot *= beta;
      for (int i = j; i < m; ++i) q[i * k + col] -= dot * v[i];
    }
    free(v);
  }
  for (int i = 1; i < m; ++i)
    for (int j = 0; j < i && j < n; ++j) r[i * n + j] = 0.0f + 0.0f * I;
}

static void qr_decompose_complex64(complex64* a, complex64* q, complex64* r,
                                   int m, int n, int reduced) {
  const int k = reduced ? (m < n ? m : n) : m;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j) q[i * k + j] = (i == j) ? 1.0 : 0.0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) r[i * n + j] = a[i * n + j];
  const int lim = (m < n ? m : n);
#pragma omp parallel for if (lim > 100)
  for (int j = 0; j < lim; ++j) {
    double norm2 = 0.0;
    for (int i = j; i < m; ++i) {
      norm2 += creal(r[i * n + j]) * creal(r[i * n + j]) +
               cimag(r[i * n + j]) * cimag(r[i * n + j]);
    }
    double norm = sqrt(norm2);
    if (norm == 0.0) continue;
    double sign = (creal(r[j * n + j]) >= 0.0) ? 1.0 : -1.0;
    complex64 u1 = r[j * n + j] + sign * norm;
    complex64* v = (complex64*)calloc((size_t)m, sizeof(complex64));
    if (!v) continue;
    v[j] = 1.0;
    double tail2 = 0.0;
    for (int i = j + 1; i < m; ++i) {
      complex64 vi = r[i * n + j] / u1;
      v[i] = vi;
      tail2 += creal(vi) * creal(vi) + cimag(vi) * cimag(vi);
    }
    double beta = 2.0 / (1.0 + tail2);
    for (int col = j; col < n; ++col) {
      complex64 dot = 0.0 + 0.0 * I;
      for (int i = j; i < m; ++i) dot += conj(v[i]) * r[i * n + col];
      dot *= beta;
      for (int i = j; i < m; ++i) r[i * n + col] -= dot * v[i];
    }
    for (int col = 0; col < k; ++col) {
      complex64 dot = 0.0 + 0.0 * I;
      for (int i = j; i < m; ++i) dot += conj(v[i]) * q[i * k + col];
      dot *= beta;
      for (int i = j; i < m; ++i) q[i * k + col] -= dot * v[i];
    }
    free(v);
  }
  for (int i = 1; i < m; ++i)
    for (int j = 0; j < i && j < n; ++j) r[i * n + j] = 0.0 + 0.0 * I;
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

static int qr_decompose_complex16(caml_ba_complex16* a, caml_ba_complex16* q,
                                  caml_ba_complex16* r, int m, int n,
                                  int reduced) {
  complex32* a_complex = (complex32*)malloc(m * n * sizeof(complex32));
  int k = reduced ? (m < n ? m : n) : m;
  complex32* q_complex = (complex32*)malloc(m * k * sizeof(complex32));
  complex32* r_complex = (complex32*)malloc(m * n * sizeof(complex32));
  if (!a_complex || !q_complex || !r_complex) {
    free(a_complex);
    free(q_complex);
    free(r_complex);
    return -1;
  }
  for (int i = 0; i < m * n; i++) {
    a_complex[i] = half_to_float(a[i].re) + I * half_to_float(a[i].im);
  }
  qr_decompose_complex32(a_complex, q_complex, r_complex, m, n, reduced);
  for (int i = 0; i < m * k; i++) {
    q[i].re = float_to_half(crealf(q_complex[i]));
    q[i].im = float_to_half(cimagf(q_complex[i]));
  }
  for (int i = 0; i < m * n; i++) {
    r[i].re = float_to_half(crealf(r_complex[i]));
    r[i].im = float_to_half(cimagf(r_complex[i]));
  }
  free(a_complex);
  free(q_complex);
  free(r_complex);
  return 0;
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

// Eigenvalue decomposition implementations
static void eigh_float32(float* a, float* eigvals, float* eigvecs, int n) {
  float* diag = (float*)malloc(n * sizeof(float));
  float* offdiag = (float*)malloc((n - 1) * sizeof(float));
  float* q = eigvecs ? eigvecs : (float*)malloc(n * n * sizeof(float));
  if (!diag || !offdiag || !q) {
    free(diag);
    free(offdiag);
    if (eigvecs == NULL) free(q);
    return;
  }
  tridiagonalize_float32(a, q, diag, offdiag, n);
  qr_iteration_tridiag_float32(diag, offdiag, q, n);
  memcpy(eigvals, diag, n * sizeof(float));
  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++)
      if (eigvals[j] > eigvals[max_idx]) max_idx = j;
    if (max_idx != i) {
      float temp = eigvals[i];
      eigvals[i] = eigvals[max_idx];
      eigvals[max_idx] = temp;
      if (eigvecs) {
#pragma omp parallel for if (n > 100)
        for (int k = 0; k < n; k++) {
          temp = q[k * n + i];
          q[k * n + i] = q[k * n + max_idx];
          q[k * n + max_idx] = temp;
        }
      }
    }
  }
  free(diag);
  free(offdiag);
  if (eigvecs == NULL) free(q);
}

static void eigh_float64(double* a, double* eigvals, double* eigvecs, int n) {
  double* diag = (double*)malloc(n * sizeof(double));
  double* offdiag = (double*)malloc((n - 1) * sizeof(double));
  double* q = eigvecs ? eigvecs : (double*)malloc(n * n * sizeof(double));
  if (!diag || !offdiag || !q) {
    free(diag);
    free(offdiag);
    if (eigvecs == NULL) free(q);
    return;
  }
  tridiagonalize_float64(a, q, diag, offdiag, n);
  qr_iteration_tridiag_float64(diag, offdiag, q, n);
  memcpy(eigvals, diag, n * sizeof(double));
  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++)
      if (eigvals[j] > eigvals[max_idx]) max_idx = j;
    if (max_idx != i) {
      double temp = eigvals[i];
      eigvals[i] = eigvals[max_idx];
      eigvals[max_idx] = temp;
      if (eigvecs) {
#pragma omp parallel for if (n > 100)
        for (int k = 0; k < n; k++) {
          temp = q[k * n + i];
          q[k * n + i] = q[k * n + max_idx];
          q[k * n + max_idx] = temp;
        }
      }
    }
  }
  free(diag);
  free(offdiag);
  if (eigvecs == NULL) free(q);
}

static void eigh_complex32(complex32* a, float* eigvals, complex32* eigvecs,
                           int n) {
  float* diag = (float*)malloc(n * sizeof(float));
  float* offdiag = (float*)malloc((n - 1) * sizeof(float));
  complex32* q =
      eigvecs ? eigvecs : (complex32*)malloc(n * n * sizeof(complex32));
  if (!diag || !offdiag || !q) {
    free(diag);
    free(offdiag);
    if (eigvecs == NULL) free(q);
    return;
  }
  // Tridiagonalize Hermitian: similar to real but with conj
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      q[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
  }
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n - 2; k++) {
    float norm2 = 0.0f;
    for (int i = k + 1; i < n; i++) {
      norm2 += crealf(a[i * n + k]) * crealf(a[i * n + k]) +
               cimagf(a[i * n + k]) * cimagf(a[i * n + k]);
    }
    if (norm2 <= 0.0f) continue;
    float norm = sqrtf(norm2);
    complex32* v = (complex32*)calloc(n, sizeof(complex32));
    if (!v) continue;
    float beta = 2.0f / norm2;
    // Hermitian Householder
    for (int j = k + 1; j < n; j++) {
      complex32 gamma = 0.0f + 0.0f * I;
      for (int i = k + 1; i < n; i++)
        gamma += conjf(a[i * n + k]) * a[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) a[i * n + j] -= gamma * a[i * n + k];
      gamma = 0.0f + 0.0f * I;
      for (int i = k + 1; i < n; i++)
        gamma += conjf(a[j * n + i]) * conjf(a[k * n + i]);
      gamma *= beta;
      for (int i = k + 1; i < n; i++)
        a[j * n + i] -= conjf(gamma) * conjf(a[k * n + i]);
    }
    for (int j = 0; j < n; j++) {
      complex32 gamma = 0.0f + 0.0f * I;
      for (int i = k + 1; i < n; i++)
        gamma += conjf(a[i * n + k]) * q[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) q[i * n + j] -= gamma * a[i * n + k];
    }
    free(v);
  }
  for (int i = 0; i < n; i++) diag[i] = crealf(a[i * n + i]);
  for (int i = 0; i < n - 1; i++) offdiag[i] = crealf(a[i * n + (i + 1)]);
  qr_iteration_tridiag_float32(diag, offdiag, (float*)q, n);
  memcpy(eigvals, diag, n * sizeof(float));
  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++)
      if (eigvals[j] > eigvals[max_idx]) max_idx = j;
    if (max_idx != i) {
      float temp = eigvals[i];
      eigvals[i] = eigvals[max_idx];
      eigvals[max_idx] = temp;
      if (eigvecs) {
#pragma omp parallel for if (n > 100)
        for (int k = 0; k < n; k++) {
          complex32 ctemp = eigvecs[k * n + i];
          eigvecs[k * n + i] = eigvecs[k * n + max_idx];
          eigvecs[k * n + max_idx] = ctemp;
        }
      }
    }
  }
  free(diag);
  free(offdiag);
  if (eigvecs == NULL) free(q);
}

static void eigh_complex64(complex64* a, double* eigvals, complex64* eigvecs,
                           int n) {
  double* diag = (double*)malloc(n * sizeof(double));
  double* offdiag = (double*)malloc((n - 1) * sizeof(double));
  complex64* q =
      eigvecs ? eigvecs : (complex64*)malloc(n * n * sizeof(complex64));
  if (!diag || !offdiag || !q) {
    free(diag);
    free(offdiag);
    if (eigvecs == NULL) free(q);
    return;
  }
  // Similar to complex32 but with double
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      q[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }
#pragma omp parallel for if (n > 100)
  for (int k = 0; k < n - 2; k++) {
    double norm2 = 0.0;
    for (int i = k + 1; i < n; i++) {
      norm2 += creal(a[i * n + k]) * creal(a[i * n + k]) +
               cimag(a[i * n + k]) * cimag(a[i * n + k]);
    }
    if (norm2 <= 0.0) continue;
    double norm = sqrt(norm2);
    complex64* v = (complex64*)calloc(n, sizeof(complex64));
    if (!v) continue;
    double beta = 2.0 / norm2;
    // Hermitian Householder
    for (int j = k + 1; j < n; j++) {
      complex64 gamma = 0.0 + 0.0 * I;
      for (int i = k + 1; i < n; i++)
        gamma += conj(a[i * n + k]) * a[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) a[i * n + j] -= gamma * a[i * n + k];
      gamma = 0.0 + 0.0 * I;
      for (int i = k + 1; i < n; i++)
        gamma += conj(a[j * n + i]) * conj(a[k * n + i]);
      gamma *= beta;
      for (int i = k + 1; i < n; i++)
        a[j * n + i] -= conj(gamma) * conj(a[k * n + i]);
    }
    for (int j = 0; j < n; j++) {
      complex64 gamma = 0.0 + 0.0 * I;
      for (int i = k + 1; i < n; i++)
        gamma += conj(a[i * n + k]) * q[i * n + j];
      gamma *= beta;
      for (int i = k + 1; i < n; i++) q[i * n + j] -= gamma * a[i * n + k];
    }
    free(v);
  }
  for (int i = 0; i < n; i++) diag[i] = creal(a[i * n + i]);
  for (int i = 0; i < n - 1; i++) offdiag[i] = creal(a[i * n + (i + 1)]);
  qr_iteration_tridiag_float64(diag, offdiag, (double*)q, n);
  memcpy(eigvals, diag, n * sizeof(double));
  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++)
      if (eigvals[j] > eigvals[max_idx]) max_idx = j;
    if (max_idx != i) {
      double temp = eigvals[i];
      eigvals[i] = eigvals[max_idx];
      eigvals[max_idx] = temp;
      if (eigvecs) {
#pragma omp parallel for if (n > 100)
        for (int k = 0; k < n; k++) {
          complex64 ctemp = eigvecs[k * n + i];
          eigvecs[k * n + i] = eigvecs[k * n + max_idx];
          eigvecs[k * n + max_idx] = ctemp;
        }
      }
    }
  }
  free(diag);
  free(offdiag);
  if (eigvecs == NULL) free(q);
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

static void eigh_complex16(caml_ba_complex16* a, caml_ba_complex16* eigvals,
                           caml_ba_complex16* eigvecs, int n) {
  complex32* a_complex = (complex32*)malloc(n * n * sizeof(complex32));
  float* eigvals_float = (float*)malloc(n * sizeof(float));
  complex32* eigvecs_complex =
      eigvecs ? (complex32*)malloc(n * n * sizeof(complex32)) : NULL;
  if (!a_complex || !eigvals_float || (eigvecs && !eigvecs_complex)) {
    free(a_complex);
    free(eigvals_float);
    free(eigvecs_complex);
    return;
  }
  for (int i = 0; i < n * n; i++) {
    a_complex[i] = half_to_float(a[i].re) + I * half_to_float(a[i].im);
  }
  eigh_complex32(a_complex, eigvals_float, eigvecs_complex, n);
  for (int i = 0; i < n; i++) {
    eigvals[i].re = float_to_half(eigvals_float[i]);
    eigvals[i].im = 0.0f;
  }
  if (eigvecs) {
    for (int i = 0; i < n * n; i++) {
      eigvecs[i].re = float_to_half(crealf(eigvecs_complex[i]));
      eigvecs[i].im = float_to_half(cimagf(eigvecs_complex[i]));
    }
    free(eigvecs_complex);
  }
  free(a_complex);
  free(eigvals_float);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_op_cholesky(value v_in, value v_out, value v_upper) {
  CAMLparam3(v_in, v_out, v_upper);
  int upper = Int_val(v_upper);
  ndarray_t in = extract_ndarray(v_in);
  ndarray_t out = extract_ndarray(v_out);
  struct caml_ba_array* ba_in = Caml_ba_array_val(Field(v_in, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_out =
      Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA));
  int kind = ba_in->flags & CAML_BA_KIND_MASK;
  if (in.ndim < 2) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&out);
    caml_failwith("cholesky: input must have at least 2 dimensions");
  }
  if (in.shape[in.ndim - 1] != in.shape[in.ndim - 2]) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&out);
    caml_failwith("cholesky: input must be square matrix");
  }
  int n = in.shape[in.ndim - 1];
  int batch_size = 1;
  for (int i = 0; i < in.ndim - 2; i++) {
    batch_size *= in.shape[i];
  }
  int s_in_row = in.strides[in.ndim - 2];
  int s_in_col = in.strides[in.ndim - 1];
  int s_out_row = out.strides[out.ndim - 2];
  int s_out_col = out.strides[out.ndim - 1];
  caml_enter_blocking_section();
  for (int b = 0; b < batch_size; b++) {
    size_t off_in = in.offset;
    size_t off_out = out.offset;
    if (in.ndim > 2) {
      int remaining = b;
      for (int i = in.ndim - 3; i >= 0; i--) {
        int coord = remaining % in.shape[i];
        remaining /= in.shape[i];
        off_in += coord * in.strides[i];
        off_out += coord * out.strides[i];
      }
    }
    int status = 0;
    switch (kind) {
      case CAML_BA_FLOAT32: {
        float* base_in = (float*)ba_in->data + off_in;
        float* base_out = (float*)ba_out->data + off_out;
        float* A = (float*)malloc((size_t)n * n * sizeof(float));
        nx_pack_f32(A, base_in, n, n, s_in_row, s_in_col);
        status = cholesky_float32(A, n, upper);
        if (status == 0) {
          nx_unpack_f32(base_out, A, n, n, s_out_row, s_out_col);
        }
        free(A);
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_in = (double*)ba_in->data + off_in;
        double* base_out = (double*)ba_out->data + off_out;
        double* A = (double*)malloc((size_t)n * n * sizeof(double));
        nx_pack_f64(A, base_in, n, n, s_in_row, s_in_col);
        status = cholesky_float64(A, n, upper);
        if (status == 0) {
          nx_unpack_f64(base_out, A, n, n, s_out_row, s_out_col);
        }
        free(A);
        break;
      }
      case CAML_BA_COMPLEX32: {
        complex32* base_in = (complex32*)ba_in->data + off_in;
        complex32* base_out = (complex32*)ba_out->data + off_out;
        complex32* A = (complex32*)malloc((size_t)n * n * sizeof(complex32));
        nx_pack_c32(A, base_in, n, n, s_in_row, s_in_col);
        status = cholesky_complex32(A, n, upper);
        if (status == 0) {
          nx_unpack_c32(base_out, A, n, n, s_out_row, s_out_col);
        }
        free(A);
        break;
      }
      case CAML_BA_COMPLEX64: {
        complex64* base_in = (complex64*)ba_in->data + off_in;
        complex64* base_out = (complex64*)ba_out->data + off_out;
        complex64* A = (complex64*)malloc((size_t)n * n * sizeof(complex64));
        nx_pack_c64(A, base_in, n, n, s_in_row, s_in_col);
        status = cholesky_complex64(A, n, upper);
        if (status == 0) {
          nx_unpack_c64(base_out, A, n, n, s_out_row, s_out_col);
        }
        free(A);
        break;
      }
      case CAML_BA_FLOAT16: {
        uint16_t* base_in = (uint16_t*)ba_in->data + off_in;
        uint16_t* base_out = (uint16_t*)ba_out->data + off_out;
        uint16_t* A = (uint16_t*)malloc((size_t)n * n * sizeof(uint16_t));
        // Pack into A (copy since same type)
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = cholesky_float16(A, n, upper);
        if (status == 0) {
          // Unpack back
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_out[i * s_out_row + j * s_out_col] = A[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_BFLOAT16: {
        caml_ba_bfloat16* base_in = (caml_ba_bfloat16*)ba_in->data + off_in;
        caml_ba_bfloat16* base_out = (caml_ba_bfloat16*)ba_out->data + off_out;
        caml_ba_bfloat16* A =
            (caml_ba_bfloat16*)malloc((size_t)n * n * sizeof(caml_ba_bfloat16));
        // Pack into A
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = cholesky_bfloat16(A, n, upper);
        if (status == 0) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_out[i * s_out_row + j * s_out_col] = A[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_FP8_E4M3: {
        caml_ba_fp8_e4m3* base_in = (caml_ba_fp8_e4m3*)ba_in->data + off_in;
        caml_ba_fp8_e4m3* base_out = (caml_ba_fp8_e4m3*)ba_out->data + off_out;
        caml_ba_fp8_e4m3* A =
            (caml_ba_fp8_e4m3*)malloc((size_t)n * n * sizeof(caml_ba_fp8_e4m3));
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = cholesky_f8e4m3(A, n, upper);
        if (status == 0) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_out[i * s_out_row + j * s_out_col] = A[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_FP8_E5M2: {
        caml_ba_fp8_e5m2* base_in = (caml_ba_fp8_e5m2*)ba_in->data + off_in;
        caml_ba_fp8_e5m2* base_out = (caml_ba_fp8_e5m2*)ba_out->data + off_out;
        caml_ba_fp8_e5m2* A =
            (caml_ba_fp8_e5m2*)malloc((size_t)n * n * sizeof(caml_ba_fp8_e5m2));
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = cholesky_f8e5m2(A, n, upper);
        if (status == 0) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_out[i * s_out_row + j * s_out_col] = A[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      case NX_BA_COMPLEX16: {
        caml_ba_complex16* base_in = (caml_ba_complex16*)ba_in->data + off_in;
        caml_ba_complex16* base_out =
            (caml_ba_complex16*)ba_out->data + off_out;
        caml_ba_complex16* A = (caml_ba_complex16*)malloc(
            (size_t)n * n * sizeof(caml_ba_complex16));
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = cholesky_complex16(A, n, upper);
        if (status == 0) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              base_out[i * s_out_row + j * s_out_col] = A[i * n + j];
            }
          }
        }
        free(A);
        break;
      }
      default:
        caml_leave_blocking_section();
        cleanup_ndarray(&in);
        cleanup_ndarray(&out);
        caml_failwith("cholesky: unsupported dtype");
    }
    if (status != 0) {
      caml_leave_blocking_section();
      cleanup_ndarray(&in);
      cleanup_ndarray(&out);
      caml_failwith("cholesky: matrix is not positive definite");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&in);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_op_triangular_solve(value v_a, value v_b, value v_out,
                                           value v_upper, value v_transpose,
                                           value v_unit_diag) {
  CAMLparam5(v_a, v_b, v_out, v_upper, v_transpose);
  CAMLxparam1(v_unit_diag);
  int upper = Int_val(v_upper);
  int transpose = Int_val(v_transpose);
  int unit_diag = Int_val(v_unit_diag);
  ndarray_t a = extract_ndarray(v_a);
  ndarray_t b = extract_ndarray(v_b);
  ndarray_t out = extract_ndarray(v_out);
  struct caml_ba_array* ba_a = Caml_ba_array_val(Field(v_a, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_b = Caml_ba_array_val(Field(v_b, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_out =
      Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA));
  int kind = ba_a->flags & CAML_BA_KIND_MASK;
  if (a.ndim < 2 || b.ndim < 2) {
    cleanup_ndarray(&a);
    cleanup_ndarray(&b);
    cleanup_ndarray(&out);
    caml_failwith("triangular_solve: inputs must have at least 2 dimensions");
  }
  int m = a.shape[a.ndim - 1];
  if (a.shape[a.ndim - 2] != m) {
    cleanup_ndarray(&a);
    cleanup_ndarray(&b);
    cleanup_ndarray(&out);
    caml_failwith("triangular_solve: A must be square");
  }
  int bn = b.shape[b.ndim - 1];
  if (b.shape[b.ndim - 2] != m) {
    cleanup_ndarray(&a);
    cleanup_ndarray(&b);
    cleanup_ndarray(&out);
    caml_failwith("triangular_solve: incompatible dimensions");
  }
  int batch_size = 1;
  for (int i = 0; i < a.ndim - 2; i++) {
    batch_size *= a.shape[i];
  }
  int s_a_row = a.strides[a.ndim - 2];
  int s_a_col = a.strides[a.ndim - 1];
  int s_b_row = b.strides[b.ndim - 2];
  int s_b_col = b.strides[b.ndim - 1];
  int s_out_row = out.strides[out.ndim - 2];
  int s_out_col = out.strides[out.ndim - 1];
  caml_enter_blocking_section();
  for (int batch = 0; batch < batch_size; batch++) {
    size_t off_a = a.offset;
    size_t off_b = b.offset;
    size_t off_out = out.offset;
    if (a.ndim > 2) {
      int remaining = batch;
      for (int i = a.ndim - 3; i >= 0; i--) {
        int coord = remaining % a.shape[i];
        remaining /= a.shape[i];
        off_a += coord * a.strides[i];
        off_b += coord * b.strides[i];
        off_out += coord * out.strides[i];
      }
    }
    switch (kind) {
      case CAML_BA_FLOAT32: {
        float* base_a = (float*)ba_a->data + off_a;
        float* base_b = (float*)ba_b->data + off_b;
        float* base_out = (float*)ba_out->data + off_out;
        float* A = (float*)malloc((size_t)m * m * sizeof(float));
        float* B = (float*)malloc((size_t)m * bn * sizeof(float));
        float* X = (float*)malloc((size_t)m * bn * sizeof(float));
        nx_pack_f32(A, base_a, m, m, s_a_row, s_a_col);
        nx_pack_f32(B, base_b, m, bn, s_b_row, s_b_col);
        triangular_solve_float32(A, B, X, m, bn, upper, transpose, unit_diag);
        nx_unpack_f32(base_out, X, m, bn, s_out_row, s_out_col);
        free(A);
        free(B);
        free(X);
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_a = (double*)ba_a->data + off_a;
        double* base_b = (double*)ba_b->data + off_b;
        double* base_out = (double*)ba_out->data + off_out;
        double* A = (double*)malloc((size_t)m * m * sizeof(double));
        double* B = (double*)malloc((size_t)m * bn * sizeof(double));
        double* X = (double*)malloc((size_t)m * bn * sizeof(double));
        nx_pack_f64(A, base_a, m, m, s_a_row, s_a_col);
        nx_pack_f64(B, base_b, m, bn, s_b_row, s_b_col);
        triangular_solve_float64(A, B, X, m, bn, upper, transpose, unit_diag);
        nx_unpack_f64(base_out, X, m, bn, s_out_row, s_out_col);
        free(A);
        free(B);
        free(X);
        break;
      }
      case CAML_BA_COMPLEX32: {
        complex32* base_a = (complex32*)ba_a->data + off_a;
        complex32* base_b = (complex32*)ba_b->data + off_b;
        complex32* base_out = (complex32*)ba_out->data + off_out;
        complex32* A = (complex32*)malloc((size_t)m * m * sizeof(complex32));
        complex32* B = (complex32*)malloc((size_t)m * bn * sizeof(complex32));
        complex32* X = (complex32*)malloc((size_t)m * bn * sizeof(complex32));
        nx_pack_c32(A, base_a, m, m, s_a_row, s_a_col);
        nx_pack_c32(B, base_b, m, bn, s_b_row, s_b_col);
        triangular_solve_complex32(A, B, X, m, bn, upper, transpose, unit_diag);
        nx_unpack_c32(base_out, X, m, bn, s_out_row, s_out_col);
        free(A);
        free(B);
        free(X);
        break;
      }
      case CAML_BA_COMPLEX64: {
        complex64* base_a = (complex64*)ba_a->data + off_a;
        complex64* base_b = (complex64*)ba_b->data + off_b;
        complex64* base_out = (complex64*)ba_out->data + off_out;
        complex64* A = (complex64*)malloc((size_t)m * m * sizeof(complex64));
        complex64* B = (complex64*)malloc((size_t)m * bn * sizeof(complex64));
        complex64* X = (complex64*)malloc((size_t)m * bn * sizeof(complex64));
        nx_pack_c64(A, base_a, m, m, s_a_row, s_a_col);
        nx_pack_c64(B, base_b, m, bn, s_b_row, s_b_col);
        triangular_solve_complex64(A, B, X, m, bn, upper, transpose, unit_diag);
        nx_unpack_c64(base_out, X, m, bn, s_out_row, s_out_col);
        free(A);
        free(B);
        free(X);
        break;
      }
      case CAML_BA_FLOAT16: {
        uint16_t* base_a = (uint16_t*)ba_a->data + off_a;
        uint16_t* base_b = (uint16_t*)ba_b->data + off_b;
        uint16_t* base_out = (uint16_t*)ba_out->data + off_out;
        uint16_t* A = (uint16_t*)malloc((size_t)m * m * sizeof(uint16_t));
        uint16_t* B = (uint16_t*)malloc((size_t)m * bn * sizeof(uint16_t));
        uint16_t* X = (uint16_t*)malloc((size_t)m * bn * sizeof(uint16_t));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < m; j++) {
            A[i * m + j] = base_a[i * s_a_row + j * s_a_col];
          }
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            B[i * bn + j] = base_b[i * s_b_row + j * s_b_col];
          }
        }
        triangular_solve_float16(A, B, X, m, bn, upper, transpose, unit_diag);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            base_out[i * s_out_row + j * s_out_col] = X[i * bn + j];
          }
        }
        free(A);
        free(B);
        free(X);
        break;
      }
      case NX_BA_BFLOAT16: {
        caml_ba_bfloat16* base_a = (caml_ba_bfloat16*)ba_a->data + off_a;
        caml_ba_bfloat16* base_b = (caml_ba_bfloat16*)ba_b->data + off_b;
        caml_ba_bfloat16* base_out = (caml_ba_bfloat16*)ba_out->data + off_out;
        caml_ba_bfloat16* A =
            (caml_ba_bfloat16*)malloc((size_t)m * m * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* B = (caml_ba_bfloat16*)malloc(
            (size_t)m * bn * sizeof(caml_ba_bfloat16));
        caml_ba_bfloat16* X = (caml_ba_bfloat16*)malloc(
            (size_t)m * bn * sizeof(caml_ba_bfloat16));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < m; j++) {
            A[i * m + j] = base_a[i * s_a_row + j * s_a_col];
          }
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            B[i * bn + j] = base_b[i * s_b_row + j * s_b_col];
          }
        }
        triangular_solve_bfloat16(A, B, X, m, bn, upper, transpose, unit_diag);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            base_out[i * s_out_row + j * s_out_col] = X[i * bn + j];
          }
        }
        free(A);
        free(B);
        free(X);
        break;
      }
      case NX_BA_FP8_E4M3: {
        caml_ba_fp8_e4m3* base_a = (caml_ba_fp8_e4m3*)ba_a->data + off_a;
        caml_ba_fp8_e4m3* base_b = (caml_ba_fp8_e4m3*)ba_b->data + off_b;
        caml_ba_fp8_e4m3* base_out = (caml_ba_fp8_e4m3*)ba_out->data + off_out;
        caml_ba_fp8_e4m3* A =
            (caml_ba_fp8_e4m3*)malloc((size_t)m * m * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* B = (caml_ba_fp8_e4m3*)malloc(
            (size_t)m * bn * sizeof(caml_ba_fp8_e4m3));
        caml_ba_fp8_e4m3* X = (caml_ba_fp8_e4m3*)malloc(
            (size_t)m * bn * sizeof(caml_ba_fp8_e4m3));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < m; j++) {
            A[i * m + j] = base_a[i * s_a_row + j * s_a_col];
          }
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            B[i * bn + j] = base_b[i * s_b_row + j * s_b_col];
          }
        }
        triangular_solve_f8e4m3(A, B, X, m, bn, upper, transpose, unit_diag);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            base_out[i * s_out_row + j * s_out_col] = X[i * bn + j];
          }
        }
        free(A);
        free(B);
        free(X);
        break;
      }
      case NX_BA_FP8_E5M2: {
        caml_ba_fp8_e5m2* base_a = (caml_ba_fp8_e5m2*)ba_a->data + off_a;
        caml_ba_fp8_e5m2* base_b = (caml_ba_fp8_e5m2*)ba_b->data + off_b;
        caml_ba_fp8_e5m2* base_out = (caml_ba_fp8_e5m2*)ba_out->data + off_out;
        caml_ba_fp8_e5m2* A =
            (caml_ba_fp8_e5m2*)malloc((size_t)m * m * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* B = (caml_ba_fp8_e5m2*)malloc(
            (size_t)m * bn * sizeof(caml_ba_fp8_e5m2));
        caml_ba_fp8_e5m2* X = (caml_ba_fp8_e5m2*)malloc(
            (size_t)m * bn * sizeof(caml_ba_fp8_e5m2));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < m; j++) {
            A[i * m + j] = base_a[i * s_a_row + j * s_a_col];
          }
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            B[i * bn + j] = base_b[i * s_b_row + j * s_b_col];
          }
        }
        triangular_solve_f8e5m2(A, B, X, m, bn, upper, transpose, unit_diag);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            base_out[i * s_out_row + j * s_out_col] = X[i * bn + j];
          }
        }
        free(A);
        free(B);
        free(X);
        break;
      }
      case NX_BA_COMPLEX16: {
        caml_ba_complex16* base_a = (caml_ba_complex16*)ba_a->data + off_a;
        caml_ba_complex16* base_b = (caml_ba_complex16*)ba_b->data + off_b;
        caml_ba_complex16* base_out =
            (caml_ba_complex16*)ba_out->data + off_out;
        caml_ba_complex16* A = (caml_ba_complex16*)malloc(
            (size_t)m * m * sizeof(caml_ba_complex16));
        caml_ba_complex16* B = (caml_ba_complex16*)malloc(
            (size_t)m * bn * sizeof(caml_ba_complex16));
        caml_ba_complex16* X = (caml_ba_complex16*)malloc(
            (size_t)m * bn * sizeof(caml_ba_complex16));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < m; j++) {
            A[i * m + j] = base_a[i * s_a_row + j * s_a_col];
          }
        }
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            B[i * bn + j] = base_b[i * s_b_row + j * s_b_col];
          }
        }
        triangular_solve_complex16(A, B, X, m, bn, upper, transpose, unit_diag);
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < bn; j++) {
            base_out[i * s_out_row + j * s_out_col] = X[i * bn + j];
          }
        }
        free(A);
        free(B);
        free(X);
        break;
      }
      default:
        caml_leave_blocking_section();
        cleanup_ndarray(&a);
        cleanup_ndarray(&b);
        cleanup_ndarray(&out);
        caml_failwith("triangular_solve: unsupported dtype");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&a);
  cleanup_ndarray(&b);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);
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
  int kind = ba_in->flags & CAML_BA_KIND_MASK;
  if (in.ndim < 2) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&q_nd);
    cleanup_ndarray(&r_nd);
    caml_failwith("qr: input must have at least 2 dimensions");
  }
  int m = in.shape[in.ndim - 2];
  int n = in.shape[in.ndim - 1];
  int k = reduced ? (m < n ? m : n) : m;
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
        nx_unpack_f32(base_r, R, m, n, s_r_row, s_r_col);
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
        nx_unpack_f64(base_r, R, m, n, s_r_row, s_r_col);
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
        nx_unpack_c32(base_r, R, m, n, s_r_row, s_r_col);
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
        nx_unpack_c64(base_r, R, m, n, s_r_row, s_r_col);
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
          for (int i = 0; i < m; i++) {
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
          for (int i = 0; i < m; i++) {
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
          for (int i = 0; i < m; i++) {
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
          for (int i = 0; i < m; i++) {
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
      case NX_BA_COMPLEX16: {
        caml_ba_complex16* base_in = (caml_ba_complex16*)ba_in->data + off_in;
        caml_ba_complex16* base_q = (caml_ba_complex16*)ba_q->data + off_q;
        caml_ba_complex16* base_r = (caml_ba_complex16*)ba_r->data + off_r;
        caml_ba_complex16* A = (caml_ba_complex16*)malloc(
            (size_t)m * n * sizeof(caml_ba_complex16));
        caml_ba_complex16* Q = (caml_ba_complex16*)malloc(
            (size_t)m * k * sizeof(caml_ba_complex16));
        caml_ba_complex16* R = (caml_ba_complex16*)malloc(
            (size_t)m * n * sizeof(caml_ba_complex16));
        for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        status = qr_decompose_complex16(A, Q, R, m, n, reduced);
        if (status == 0) {
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
              base_q[i * s_q_row + j * s_q_col] = Q[i * k + j];
            }
          }
          for (int i = 0; i < m; i++) {
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
  int kind = ba_in->flags & CAML_BA_KIND_MASK;
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
  if (!symmetric) {
    cleanup_ndarray(&in);
    cleanup_ndarray(&vals);
    cleanup_ndarray(&vecs);
    caml_failwith("eig: general eigenvalue decomposition not implemented");
  }
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
        float* base_vals = (float*)ba_vals->data + off_vals;
        float* base_vecs =
            compute_vectors ? (float*)ba_vecs->data + off_vecs : NULL;
        float* A = (float*)malloc((size_t)n * n * sizeof(float));
        if (!A) continue;
        nx_pack_f32(A, base_in, n, n, s_in_row, s_in_col);
        eigh_float32(A, base_vals, base_vecs, n);
        if (compute_vectors) {
          nx_unpack_f32(base_vecs, base_vecs, n, n, s_vecs_row, s_vecs_col);
        }
        free(A);
        break;
      }
      case CAML_BA_FLOAT64: {
        double* base_in = (double*)ba_in->data + off_in;
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
      case NX_BA_COMPLEX16: {
        caml_ba_complex16* base_in = (caml_ba_complex16*)ba_in->data + off_in;
        caml_ba_complex16* base_vals =
            (caml_ba_complex16*)ba_vals->data + off_vals;
        caml_ba_complex16* base_vecs =
            compute_vectors ? (caml_ba_complex16*)ba_vecs->data + off_vecs
                            : NULL;
        caml_ba_complex16* A = (caml_ba_complex16*)malloc(
            (size_t)n * n * sizeof(caml_ba_complex16));
        if (!A) continue;
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            A[i * n + j] = base_in[i * s_in_row + j * s_in_col];
          }
        }
        eigh_complex16(A, base_vals, base_vecs, n);
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