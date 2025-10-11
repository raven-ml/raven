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

// ============================================================================
// General Linear System Solving (Ax = b)
// ============================================================================

// General solve implementations using LAPACK
static void solve_float32(float* a, float* b, int n, int nrhs) {
   // Create copies since LAPACK overwrites inputs
   float* a_copy = (float*)malloc(n * n * sizeof(float));
   float* b_copy = (float*)malloc(n * nrhs * sizeof(float));
   if (!a_copy || !b_copy) {
     free(a_copy);
     free(b_copy);
     return;
   }
   memcpy(a_copy, a, n * n * sizeof(float));
   memcpy(b_copy, b, n * nrhs * sizeof(float));

   // Pivot array for LAPACK
   int* ipiv = (int*)malloc(n * sizeof(int));
   if (!ipiv) {
     free(a_copy);
     free(b_copy);
     return;
   }

   // Solve using LAPACK
   int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, a_copy, n, ipiv, b_copy, nrhs);

   if (info == 0) {
     // Copy solution back to b
     memcpy(b, b_copy, n * nrhs * sizeof(float));
   }

   free(a_copy);
   free(b_copy);
   free(ipiv);
}

static void solve_float64(double* a, double* b, int n, int nrhs) {
   // Create copies since LAPACK overwrites inputs
   double* a_copy = (double*)malloc(n * n * sizeof(double));
   double* b_copy = (double*)malloc(n * nrhs * sizeof(double));
   if (!a_copy || !b_copy) {
     free(a_copy);
     free(b_copy);
     return;
   }
   memcpy(a_copy, a, n * n * sizeof(double));
   memcpy(b_copy, b, n * nrhs * sizeof(double));

   // Pivot array for LAPACK
   int* ipiv = (int*)malloc(n * sizeof(int));
   if (!ipiv) {
     free(a_copy);
     free(b_copy);
     return;
   }

   // Solve using LAPACK
   int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a_copy, n, ipiv, b_copy, nrhs);

   if (info == 0) {
     // Copy solution back to b
     memcpy(b, b_copy, n * nrhs * sizeof(double));
   }

   free(a_copy);
   free(b_copy);
   free(ipiv);
}

static void solve_complex32(complex32* a, complex32* b, int n, int nrhs) {
   // Create copies since LAPACK overwrites inputs
   complex32* a_copy = (complex32*)malloc(n * n * sizeof(complex32));
   complex32* b_copy = (complex32*)malloc(n * nrhs * sizeof(complex32));
   if (!a_copy || !b_copy) {
     free(a_copy);
     free(b_copy);
     return;
   }
   memcpy(a_copy, a, n * n * sizeof(complex32));
   memcpy(b_copy, b, n * nrhs * sizeof(complex32));

   // Pivot array for LAPACK
   int* ipiv = (int*)malloc(n * sizeof(int));
   if (!ipiv) {
     free(a_copy);
     free(b_copy);
     return;
   }

   // Solve using LAPACK
   int info = LAPACKE_cgesv(LAPACK_ROW_MAJOR, n, nrhs, a_copy, n, ipiv, b_copy, nrhs);

   if (info == 0) {
     // Copy solution back to b
     memcpy(b, b_copy, n * nrhs * sizeof(complex32));
   }

   free(a_copy);
   free(b_copy);
   free(ipiv);
}

static void solve_complex64(complex64* a, complex64* b, int n, int nrhs) {
   // Create copies since LAPACK overwrites inputs
   complex64* a_copy = (complex64*)malloc(n * n * sizeof(complex64));
   complex64* b_copy = (complex64*)malloc(n * nrhs * sizeof(complex64));
   if (!a_copy || !b_copy) {
     free(a_copy);
     free(b_copy);
     return;
   }
   memcpy(a_copy, a, n * n * sizeof(complex64));
   memcpy(b_copy, b, n * nrhs * sizeof(complex64));

   // Pivot array for LAPACK
   int* ipiv = (int*)malloc(n * sizeof(int));
   if (!ipiv) {
     free(a_copy);
     free(b_copy);
     return;
   }

   // Solve using LAPACK
   int info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, nrhs, a_copy, n, ipiv, b_copy, nrhs);

   if (info == 0) {
     // Copy solution back to b
     memcpy(b, b_copy, n * nrhs * sizeof(complex64));
   }

   free(a_copy);
   free(b_copy);
   free(ipiv);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================


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
  int kind = nx_ba_get_kind(ba_a);
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

// Bytecode wrapper for triangular_solve (6 arguments)
CAMLprim value caml_nx_op_triangular_solve_bc(value *argv, int argn) {
  CAMLparam0();
  (void)argn;
  value ret = caml_nx_op_triangular_solve(argv[0], argv[1], argv[2], argv[3],
                                          argv[4], argv[5]);
  CAMLreturn(ret);
}
