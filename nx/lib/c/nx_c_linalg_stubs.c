#include "nx_c_shared.h"

// Helper functions for matrix operations
static inline void* get_data_ptr(value ba, int offset, int kind) {
  char* data = (char*)Caml_ba_data_val(ba);
  int elem_size = get_element_size(kind);
  return data + offset * elem_size;
}

// Helper to copy a matrix
static void copy_matrix(void* dst, const void* src, int m, int n, int kind) {
  int elem_size = get_element_size(kind);
  memcpy(dst, src, m * n * elem_size);
}

// Helper to create identity matrix
static void eye_matrix(void* data, int n, int kind) {
  memset(data, 0, n * n * get_element_size(kind));

  switch (kind) {
    case CAML_BA_FLOAT32:
      for (int i = 0; i < n; i++) {
        ((float*)data)[i * n + i] = 1.0f;
      }
      break;
    case CAML_BA_FLOAT64:
      for (int i = 0; i < n; i++) {
        ((double*)data)[i * n + i] = 1.0;
      }
      break;
    case CAML_BA_COMPLEX32:
      for (int i = 0; i < n; i++) {
        ((c32_t*)data)[i * n + i] = 1.0f + 0.0f * I;
      }
      break;
    case CAML_BA_COMPLEX64:
      for (int i = 0; i < n; i++) {
        ((c64_t*)data)[i * n + i] = 1.0 + 0.0 * I;
      }
      break;
  }
}

// Cholesky decomposition for float32
static int cholesky_float32(float* a, int n, int upper) {
  if (upper) {
    // Upper triangular decomposition: A = U^T * U
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        sum += a[j * n + k] * a[j * n + k];
      }
      float diag = a[k * n + k] - sum;
      if (diag <= 0.0f) return -1;  // Not positive definite
      a[k * n + k] = sqrtf(diag);

      for (int i = k + 1; i < n; i++) {
        sum = 0.0f;
        for (int j = 0; j < k; j++) {
          sum += a[j * n + i] * a[j * n + k];
        }
        a[k * n + i] = (a[k * n + i] - sum) / a[k * n + k];
      }

      // Zero out lower triangle
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0f;
      }
    }
  } else {
    // Lower triangular decomposition: A = L * L^T
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        sum += a[k * n + j] * a[k * n + j];
      }
      float diag = a[k * n + k] - sum;
      if (diag <= 0.0f) return -1;  // Not positive definite
      a[k * n + k] = sqrtf(diag);

      for (int i = k + 1; i < n; i++) {
        sum = 0.0f;
        for (int j = 0; j < k; j++) {
          sum += a[i * n + j] * a[k * n + j];
        }
        a[i * n + k] = (a[i * n + k] - sum) / a[k * n + k];
      }

      // Zero out upper triangle
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0f;
      }
    }
  }
  return 0;
}

// Cholesky decomposition for float64
static int cholesky_float64(double* a, int n, int upper) {
  if (upper) {
    // Upper triangular decomposition: A = U^T * U
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        sum += a[j * n + k] * a[j * n + k];
      }
      double diag = a[k * n + k] - sum;
      if (diag <= 0.0) return -1;  // Not positive definite
      a[k * n + k] = sqrt(diag);

      for (int i = k + 1; i < n; i++) {
        sum = 0.0;
        for (int j = 0; j < k; j++) {
          sum += a[j * n + i] * a[j * n + k];
        }
        a[k * n + i] = (a[k * n + i] - sum) / a[k * n + k];
      }

      // Zero out lower triangle
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0;
      }
    }
  } else {
    // Lower triangular decomposition: A = L * L^T
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        sum += a[k * n + j] * a[k * n + j];
      }
      double diag = a[k * n + k] - sum;
      if (diag <= 0.0) return -1;  // Not positive definite
      a[k * n + k] = sqrt(diag);

      for (int i = k + 1; i < n; i++) {
        sum = 0.0;
        for (int j = 0; j < k; j++) {
          sum += a[i * n + j] * a[k * n + j];
        }
        a[i * n + k] = (a[i * n + k] - sum) / a[k * n + k];
      }

      // Zero out upper triangle
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0;
      }
    }
  }
  return 0;
}

// Cholesky decomposition for complex32
static int cholesky_complex32(c32_t* a, int n, int upper) {
  if (upper) {
    // Upper triangular decomposition: A = U^H * U
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        c32_t val = a[j * n + k];
        sum += crealf(val) * crealf(val) + cimagf(val) * cimagf(val);
      }
      float diag = crealf(a[k * n + k]) - sum;
      if (diag <= 0.0f) return -1;  // Not positive definite
      a[k * n + k] = sqrtf(diag);

      for (int i = k + 1; i < n; i++) {
        c32_t sum_c = 0.0f + 0.0f * I;
        for (int j = 0; j < k; j++) {
          sum_c += conjf(a[j * n + k]) * a[j * n + i];
        }
        a[k * n + i] = (a[k * n + i] - sum_c) / a[k * n + k];
      }

      // Zero out lower triangle
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0f + 0.0f * I;
      }
    }
  } else {
    // Lower triangular decomposition: A = L * L^H
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < k; j++) {
        c32_t val = a[k * n + j];
        sum += crealf(val) * crealf(val) + cimagf(val) * cimagf(val);
      }
      float diag = crealf(a[k * n + k]) - sum;
      if (diag <= 0.0f) return -1;  // Not positive definite
      a[k * n + k] = sqrtf(diag);

      for (int i = k + 1; i < n; i++) {
        c32_t sum_c = 0.0f + 0.0f * I;
        for (int j = 0; j < k; j++) {
          sum_c += a[i * n + j] * conjf(a[k * n + j]);
        }
        a[i * n + k] = (a[i * n + k] - sum_c) / a[k * n + k];
      }

      // Zero out upper triangle
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0f + 0.0f * I;
      }
    }
  }
  return 0;
}

// Cholesky decomposition for complex64
static int cholesky_complex64(c64_t* a, int n, int upper) {
  if (upper) {
    // Upper triangular decomposition: A = U^H * U
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        c64_t val = a[j * n + k];
        sum += creal(val) * creal(val) + cimag(val) * cimag(val);
      }
      double diag = creal(a[k * n + k]) - sum;
      if (diag <= 0.0) return -1;  // Not positive definite
      a[k * n + k] = sqrt(diag);

      for (int i = k + 1; i < n; i++) {
        c64_t sum_c = 0.0 + 0.0 * I;
        for (int j = 0; j < k; j++) {
          sum_c += conj(a[j * n + k]) * a[j * n + i];
        }
        a[k * n + i] = (a[k * n + i] - sum_c) / a[k * n + k];
      }

      // Zero out lower triangle
      for (int i = k + 1; i < n; i++) {
        a[i * n + k] = 0.0 + 0.0 * I;
      }
    }
  } else {
    // Lower triangular decomposition: A = L * L^H
    for (int k = 0; k < n; k++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        c64_t val = a[k * n + j];
        sum += creal(val) * creal(val) + cimag(val) * cimag(val);
      }
      double diag = creal(a[k * n + k]) - sum;
      if (diag <= 0.0) return -1;  // Not positive definite
      a[k * n + k] = sqrt(diag);

      for (int i = k + 1; i < n; i++) {
        c64_t sum_c = 0.0 + 0.0 * I;
        for (int j = 0; j < k; j++) {
          sum_c += a[i * n + j] * conj(a[k * n + j]);
        }
        a[i * n + k] = (a[i * n + k] - sum_c) / a[k * n + k];
      }

      // Zero out upper triangle
      for (int i = 0; i < k; i++) {
        a[i * n + k] = 0.0 + 0.0 * I;
      }
    }
  }
  return 0;
}

// OCaml interface for Cholesky decomposition
CAMLprim value caml_nx_cholesky_bc(value* argv, int argn) {
  CAMLparam0();
  CAMLlocal1(result);

  int upper = Int_val(argv[0]);

  // Input tensor
  value ba_in = argv[1];
  value v_shape = argv[2];
  value v_strides = argv[3];
  int offset_in = Int_val(argv[4]);

  // Output tensor
  value ba_out = argv[5];
  value v_out_strides = argv[6];
  int offset_out = Int_val(argv[7]);

  int kind = Caml_ba_array_val(ba_in)->flags & CAML_BA_KIND_MASK;

  // Get shape dimensions
  int ndim = Wosize_val(v_shape);
  if (ndim < 2) {
    caml_failwith("cholesky: input must have at least 2 dimensions");
  }

  // Extract dimensions
  int n = Int_val(Field(v_shape, ndim - 1));
  int m = Int_val(Field(v_shape, ndim - 2));

  if (n != m) {
    caml_failwith("cholesky: input must be square matrix");
  }

  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= Int_val(Field(v_shape, i));
  }

  // Get strides
  int stride_in = Int_val(Field(v_strides, ndim - 2));
  int stride_out = Int_val(Field(v_out_strides, ndim - 2));

  int elem_size = get_element_size(kind);

  caml_enter_blocking_section();

  // Process each matrix in the batch
  for (int b = 0; b < batch_size; b++) {
    void* data_in =
        (char*)Caml_ba_data_val(ba_in) + (offset_in + b * n * n) * elem_size;
    void* data_out =
        (char*)Caml_ba_data_val(ba_out) + (offset_out + b * n * n) * elem_size;

    // Copy input to output
    copy_matrix(data_out, data_in, n, n, kind);

    int status = 0;
    switch (kind) {
      case CAML_BA_FLOAT32:
        status = cholesky_float32((float*)data_out, n, upper);
        break;
      case CAML_BA_FLOAT64:
        status = cholesky_float64((double*)data_out, n, upper);
        break;
      case CAML_BA_COMPLEX32:
        status = cholesky_complex32((c32_t*)data_out, n, upper);
        break;
      case CAML_BA_COMPLEX64:
        status = cholesky_complex64((c64_t*)data_out, n, upper);
        break;
      default:
        caml_leave_blocking_section();
        caml_failwith("cholesky: unsupported dtype");
    }

    if (status != 0) {
      caml_leave_blocking_section();
      caml_failwith("cholesky: matrix is not positive definite");
    }
  }

  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

NATIVE_WRAPPER_8(cholesky)

// Triangular solve implementations
static void triangular_solve_float32(const float* a, const float* b, float* x,
                                     int m, int n, int upper, int transpose,
                                     int unit_diag) {
  // Copy b to x
  memcpy(x, b, m * n * sizeof(float));

  if (!transpose) {
    if (upper) {
      // Solve U * X = B
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
          for (int k = i + 1; k < m; k++) {
            x[i * n + j] -= a[i * m + k] * x[k * n + j];
          }
        }
      }
    } else {
      // Solve L * X = B
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          for (int k = 0; k < i; k++) {
            x[i * n + j] -= a[i * m + k] * x[k * n + j];
          }
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
        }
      }
    }
  } else {
    if (upper) {
      // Solve U^T * X = B
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          for (int k = 0; k < i; k++) {
            x[i * n + j] -= a[k * m + i] * x[k * n + j];
          }
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
        }
      }
    } else {
      // Solve L^T * X = B
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
          for (int k = i + 1; k < m; k++) {
            x[i * n + j] -= a[k * m + i] * x[k * n + j];
          }
        }
      }
    }
  }
}

static void triangular_solve_float64(const double* a, const double* b,
                                     double* x, int m, int n, int upper,
                                     int transpose, int unit_diag) {
  // Copy b to x
  memcpy(x, b, m * n * sizeof(double));

  if (!transpose) {
    if (upper) {
      // Solve U * X = B
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
          for (int k = i + 1; k < m; k++) {
            x[i * n + j] -= a[i * m + k] * x[k * n + j];
          }
        }
      }
    } else {
      // Solve L * X = B
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          for (int k = 0; k < i; k++) {
            x[i * n + j] -= a[i * m + k] * x[k * n + j];
          }
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
        }
      }
    }
  } else {
    if (upper) {
      // Solve U^T * X = B
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          for (int k = 0; k < i; k++) {
            x[i * n + j] -= a[k * m + i] * x[k * n + j];
          }
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
        }
      }
    } else {
      // Solve L^T * X = B
      for (int j = 0; j < n; j++) {
        for (int i = m - 1; i >= 0; i--) {
          if (!unit_diag) {
            x[i * n + j] /= a[i * m + i];
          }
          for (int k = i + 1; k < m; k++) {
            x[i * n + j] -= a[k * m + i] * x[k * n + j];
          }
        }
      }
    }
  }
}

// OCaml interface for triangular solve
CAMLprim value caml_nx_triangular_solve_bc(value* argv, int argn) {
  CAMLparam0();

  int upper = Int_val(argv[0]);
  int transpose = Int_val(argv[1]);
  int unit_diag = Int_val(argv[2]);

  // A matrix
  value ba_a = argv[3];
  value v_shape_a = argv[4];
  value v_strides_a = argv[5];
  int offset_a = Int_val(argv[6]);

  // B matrix
  value ba_b = argv[7];
  value v_shape_b = argv[8];
  value v_strides_b = argv[9];
  int offset_b = Int_val(argv[10]);

  // Output matrix
  value ba_out = argv[11];
  value v_strides_out = argv[12];
  int offset_out = Int_val(argv[13]);

  int kind = Caml_ba_array_val(ba_a)->flags & CAML_BA_KIND_MASK;

  // Get dimensions
  int ndim_a = Wosize_val(v_shape_a);
  int ndim_b = Wosize_val(v_shape_b);

  if (ndim_a < 2 || ndim_b < 2) {
    caml_failwith("triangular_solve: inputs must have at least 2 dimensions");
  }

  // Extract matrix dimensions
  int m = Int_val(Field(v_shape_a, ndim_a - 2));  // A is [..., m, m]
  int n = Int_val(Field(v_shape_b, ndim_b - 1));  // B is [..., m, n]
  int k = Int_val(Field(v_shape_b, ndim_b - 2));

  if (m != k) {
    caml_failwith("triangular_solve: incompatible dimensions");
  }

  if (m != Int_val(Field(v_shape_a, ndim_a - 1))) {
    caml_failwith("triangular_solve: A must be square");
  }

  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim_a - 2; i++) {
    batch_size *= Int_val(Field(v_shape_a, i));
  }

  int elem_size = get_element_size(kind);

  caml_enter_blocking_section();

  // Process each matrix in the batch
  for (int b = 0; b < batch_size; b++) {
    void* data_a =
        (char*)Caml_ba_data_val(ba_a) + (offset_a + b * m * m) * elem_size;
    void* data_b =
        (char*)Caml_ba_data_val(ba_b) + (offset_b + b * m * n) * elem_size;
    void* data_out =
        (char*)Caml_ba_data_val(ba_out) + (offset_out + b * m * n) * elem_size;

    switch (kind) {
      case CAML_BA_FLOAT32:
        triangular_solve_float32((float*)data_a, (float*)data_b,
                                 (float*)data_out, m, n, upper, transpose,
                                 unit_diag);
        break;
      case CAML_BA_FLOAT64:
        triangular_solve_float64((double*)data_a, (double*)data_b,
                                 (double*)data_out, m, n, upper, transpose,
                                 unit_diag);
        break;
      default:
        caml_leave_blocking_section();
        caml_failwith("triangular_solve: unsupported dtype");
    }
  }

  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

NATIVE_WRAPPER_14(triangular_solve)

// QR decomposition using Householder reflections
static void qr_decompose_float32(float* a, float* q, float* r, int m, int n,
                                 int reduced) {
  int k = reduced ? (m < n ? m : n) : m;

  // Initialize Q to identity matrix
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      q[i * k + j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  // Copy A to R
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      r[i * n + j] = a[i * n + j];
    }
  }

  // Householder QR decomposition
  for (int j = 0; j < (m < n ? m : n); j++) {
    // Compute Householder vector
    float norm = 0.0f;
    for (int i = j; i < m; i++) {
      norm += r[i * n + j] * r[i * n + j];
    }
    norm = sqrtf(norm);

    if (norm == 0.0f) continue;

    float sign = r[j * n + j] >= 0 ? 1.0f : -1.0f;
    float u1 = r[j * n + j] + sign * norm;

    // Store Householder vector
    float* v = (float*)calloc(m, sizeof(float));
    v[j] = 1.0f;
    for (int i = j + 1; i < m; i++) {
      v[i] = r[i * n + j] / u1;
    }

    // Apply Householder transformation to R
    for (int col = j; col < n; col++) {
      float dot = 0.0f;
      for (int i = j; i < m; i++) {
        dot += v[i] * r[i * n + col];
      }
      dot *= 2.0f / (1.0f + norm * norm / (u1 * u1));

      for (int i = j; i < m; i++) {
        r[i * n + col] -= dot * v[i];
      }
    }

    // Apply Householder transformation to Q
    for (int col = 0; col < k; col++) {
      float dot = 0.0f;
      for (int i = j; i < m; i++) {
        dot += v[i] * q[i * k + col];
      }
      dot *= 2.0f / (1.0f + norm * norm / (u1 * u1));

      for (int i = j; i < m; i++) {
        q[i * k + col] -= dot * v[i];
      }
    }

    free(v);
  }

  // Zero out below diagonal in R
  for (int i = 1; i < m; i++) {
    for (int j = 0; j < i && j < n; j++) {
      r[i * n + j] = 0.0f;
    }
  }
}

static void qr_decompose_float64(double* a, double* q, double* r, int m, int n,
                                 int reduced) {
  int k = reduced ? (m < n ? m : n) : m;

  // Initialize Q to identity matrix
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      q[i * k + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  // Copy A to R
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      r[i * n + j] = a[i * n + j];
    }
  }

  // Householder QR decomposition
  for (int j = 0; j < (m < n ? m : n); j++) {
    // Compute Householder vector
    double norm = 0.0;
    for (int i = j; i < m; i++) {
      norm += r[i * n + j] * r[i * n + j];
    }
    norm = sqrt(norm);

    if (norm == 0.0) continue;

    double sign = r[j * n + j] >= 0 ? 1.0 : -1.0;
    double u1 = r[j * n + j] + sign * norm;

    // Store Householder vector
    double* v = (double*)calloc(m, sizeof(double));
    v[j] = 1.0;
    for (int i = j + 1; i < m; i++) {
      v[i] = r[i * n + j] / u1;
    }

    // Apply Householder transformation to R
    for (int col = j; col < n; col++) {
      double dot = 0.0;
      for (int i = j; i < m; i++) {
        dot += v[i] * r[i * n + col];
      }
      dot *= 2.0 / (1.0 + norm * norm / (u1 * u1));

      for (int i = j; i < m; i++) {
        r[i * n + col] -= dot * v[i];
      }
    }

    // Apply Householder transformation to Q
    for (int col = 0; col < k; col++) {
      double dot = 0.0;
      for (int i = j; i < m; i++) {
        dot += v[i] * q[i * k + col];
      }
      dot *= 2.0 / (1.0 + norm * norm / (u1 * u1));

      for (int i = j; i < m; i++) {
        q[i * k + col] -= dot * v[i];
      }
    }

    free(v);
  }

  // Zero out below diagonal in R
  for (int i = 1; i < m; i++) {
    for (int j = 0; j < i && j < n; j++) {
      r[i * n + j] = 0.0;
    }
  }
}

// OCaml interface for QR decomposition
CAMLprim value caml_nx_qr_bc(value* argv, int argn) {
  CAMLparam0();

  int reduced = Int_val(argv[0]);

  // Input tensor
  value ba_in = argv[1];
  value v_shape = argv[2];
  value v_strides = argv[3];
  int offset_in = Int_val(argv[4]);

  // Q output tensor
  value ba_q = argv[5];
  value v_shape_q = argv[6];
  value v_strides_q = argv[7];
  int offset_q = Int_val(argv[8]);

  // R output tensor
  value ba_r = argv[9];
  value v_shape_r = argv[10];
  value v_strides_r = argv[11];
  int offset_r = Int_val(argv[12]);

  int kind = Caml_ba_array_val(ba_in)->flags & CAML_BA_KIND_MASK;

  // Get dimensions
  int ndim = Wosize_val(v_shape);
  if (ndim < 2) {
    caml_failwith("qr: input must have at least 2 dimensions");
  }

  int m = Int_val(Field(v_shape, ndim - 2));
  int n = Int_val(Field(v_shape, ndim - 1));

  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= Int_val(Field(v_shape, i));
  }

  int elem_size = get_element_size(kind);

  caml_enter_blocking_section();

  // Process each matrix in the batch
  for (int b = 0; b < batch_size; b++) {
    void* data_in =
        (char*)Caml_ba_data_val(ba_in) + (offset_in + b * m * n) * elem_size;
    void* data_q = (char*)Caml_ba_data_val(ba_q) +
                   (offset_q + b * Int_val(Field(v_shape_q, ndim - 2)) *
                                   Int_val(Field(v_shape_q, ndim - 1))) *
                       elem_size;
    void* data_r = (char*)Caml_ba_data_val(ba_r) +
                   (offset_r + b * Int_val(Field(v_shape_r, ndim - 2)) *
                                   Int_val(Field(v_shape_r, ndim - 1))) *
                       elem_size;

    switch (kind) {
      case CAML_BA_FLOAT32:
        qr_decompose_float32((float*)data_in, (float*)data_q, (float*)data_r, m,
                             n, reduced);
        break;
      case CAML_BA_FLOAT64:
        qr_decompose_float64((double*)data_in, (double*)data_q, (double*)data_r,
                             m, n, reduced);
        break;
      default:
        caml_leave_blocking_section();
        caml_failwith(
            "qr: unsupported dtype (only float32 and float64 supported)");
    }
  }

  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

NATIVE_WRAPPER_13(qr)

// Helper functions for SVD
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

// Givens rotation
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

// Apply Givens rotation to matrix columns
static void apply_givens_left_float32(float* a, int m, int n, int i, int j,
                                      float c, float s) {
  for (int k = 0; k < n; k++) {
    float temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -s * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

static void apply_givens_left_float64(double* a, int m, int n, int i, int j,
                                      double c, double s) {
  for (int k = 0; k < n; k++) {
    double temp = c * a[i * n + k] + s * a[j * n + k];
    a[j * n + k] = -s * a[i * n + k] + c * a[j * n + k];
    a[i * n + k] = temp;
  }
}

// Apply Givens rotation to matrix rows
static void apply_givens_right_float32(float* a, int m, int n, int i, int j,
                                       float c, float s) {
  for (int k = 0; k < m; k++) {
    float temp = c * a[k * n + i] + s * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

static void apply_givens_right_float64(double* a, int m, int n, int i, int j,
                                       double c, double s) {
  for (int k = 0; k < m; k++) {
    double temp = c * a[k * n + i] + s * a[k * n + j];
    a[k * n + j] = -s * a[k * n + i] + c * a[k * n + j];
    a[k * n + i] = temp;
  }
}

// Bidiagonalization using Householder reflections
static void bidiagonalize_float32(float* a, float* u, float* v, float* diag,
                                  float* super, int m, int n) {
  int min_mn = m < n ? m : n;

  // Initialize U and V to identity
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      u[i * m + j] = (i == j) ? 1.0f : 0.0f;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      v[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  // Bidiagonalization
  for (int k = 0; k < min_mn; k++) {
    // Left Householder transformation
    float alpha = 0.0f;
    for (int i = k; i < m; i++) {
      alpha += a[i * n + k] * a[i * n + k];
    }
    alpha = -sign_float32(a[k * n + k]) * sqrtf(alpha);

    if (alpha != 0.0f) {
      float beta = 1.0f / (alpha * (alpha - a[k * n + k]));
      a[k * n + k] -= alpha;

      // Apply to A
      for (int j = k + 1; j < n; j++) {
        float gamma = 0.0f;
        for (int i = k; i < m; i++) {
          gamma += a[i * n + k] * a[i * n + j];
        }
        gamma *= beta;
        for (int i = k; i < m; i++) {
          a[i * n + j] -= gamma * a[i * n + k];
        }
      }

      // Apply to U
      for (int j = 0; j < m; j++) {
        float gamma = 0.0f;
        for (int i = k; i < m; i++) {
          gamma += a[i * n + k] * u[i * m + j];
        }
        gamma *= beta;
        for (int i = k; i < m; i++) {
          u[i * m + j] -= gamma * a[i * n + k];
        }
      }

      diag[k] = alpha;
    } else {
      diag[k] = a[k * n + k];
    }

    if (k < min_mn - 1) {
      // Right Householder transformation
      alpha = 0.0f;
      for (int j = k + 1; j < n; j++) {
        alpha += a[k * n + j] * a[k * n + j];
      }
      alpha = -sign_float32(a[k * n + k + 1]) * sqrtf(alpha);

      if (alpha != 0.0f) {
        float beta = 1.0f / (alpha * (alpha - a[k * n + k + 1]));
        a[k * n + k + 1] -= alpha;

        // Apply to A
        for (int i = k + 1; i < m; i++) {
          float gamma = 0.0f;
          for (int j = k + 1; j < n; j++) {
            gamma += a[i * n + j] * a[k * n + j];
          }
          gamma *= beta;
          for (int j = k + 1; j < n; j++) {
            a[i * n + j] -= gamma * a[k * n + j];
          }
        }

        // Apply to V^T
        for (int i = 0; i < n; i++) {
          float gamma = 0.0f;
          for (int j = k + 1; j < n; j++) {
            gamma += v[j * n + i] * a[k * n + j];
          }
          gamma *= beta;
          for (int j = k + 1; j < n; j++) {
            v[j * n + i] -= gamma * a[k * n + j];
          }
        }

        super[k] = alpha;
      } else {
        super[k] = a[k * n + k + 1];
      }
    }
  }
}

static void bidiagonalize_float64(double* a, double* u, double* v, double* diag,
                                  double* super, int m, int n) {
  int min_mn = m < n ? m : n;

  // Initialize U and V to identity
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      u[i * m + j] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      v[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  // Bidiagonalization
  for (int k = 0; k < min_mn; k++) {
    // Left Householder transformation
    double alpha = 0.0;
    for (int i = k; i < m; i++) {
      alpha += a[i * n + k] * a[i * n + k];
    }
    alpha = -sign_float64(a[k * n + k]) * sqrt(alpha);

    if (alpha != 0.0) {
      double beta = 1.0 / (alpha * (alpha - a[k * n + k]));
      a[k * n + k] -= alpha;

      // Apply to A
      for (int j = k + 1; j < n; j++) {
        double gamma = 0.0;
        for (int i = k; i < m; i++) {
          gamma += a[i * n + k] * a[i * n + j];
        }
        gamma *= beta;
        for (int i = k; i < m; i++) {
          a[i * n + j] -= gamma * a[i * n + k];
        }
      }

      // Apply to U
      for (int j = 0; j < m; j++) {
        double gamma = 0.0;
        for (int i = k; i < m; i++) {
          gamma += a[i * n + k] * u[i * m + j];
        }
        gamma *= beta;
        for (int i = k; i < m; i++) {
          u[i * m + j] -= gamma * a[i * n + k];
        }
      }

      diag[k] = alpha;
    } else {
      diag[k] = a[k * n + k];
    }

    if (k < min_mn - 1) {
      // Right Householder transformation
      alpha = 0.0;
      for (int j = k + 1; j < n; j++) {
        alpha += a[k * n + j] * a[k * n + j];
      }
      alpha = -sign_float64(a[k * n + k + 1]) * sqrt(alpha);

      if (alpha != 0.0) {
        double beta = 1.0 / (alpha * (alpha - a[k * n + k + 1]));
        a[k * n + k + 1] -= alpha;

        // Apply to A
        for (int i = k + 1; i < m; i++) {
          double gamma = 0.0;
          for (int j = k + 1; j < n; j++) {
            gamma += a[i * n + j] * a[k * n + j];
          }
          gamma *= beta;
          for (int j = k + 1; j < n; j++) {
            a[i * n + j] -= gamma * a[k * n + j];
          }
        }

        // Apply to V^T
        for (int i = 0; i < n; i++) {
          double gamma = 0.0;
          for (int j = k + 1; j < n; j++) {
            gamma += v[j * n + i] * a[k * n + j];
          }
          gamma *= beta;
          for (int j = k + 1; j < n; j++) {
            v[j * n + i] -= gamma * a[k * n + j];
          }
        }

        super[k] = alpha;
      } else {
        super[k] = a[k * n + k + 1];
      }
    }
  }
}

// QR iteration for SVD
static void svd_qr_iteration_float32(float* diag, float* super, float* u,
                                     float* v, int m, int n, int p, int q) {
  // Wilkinson shift
  float d = (diag[q - 1] - diag[q]) / 2.0f;
  float shift =
      diag[q] - super[q - 1] * super[q - 1] /
                    (d + sign_float32(d) * hypot_float32(d, super[q - 1]));

  // Chase the bulge
  float c = 1.0f, s = 0.0f;
  float f = diag[p] - shift;
  float g = super[p];

  for (int k = p; k < q; k++) {
    givens_float32(f, g, &c, &s);
    if (k > p) super[k - 1] = hypot_float32(f, g);

    f = c * diag[k] + s * super[k];
    super[k] = c * super[k] - s * diag[k];
    g = s * diag[k + 1];
    diag[k + 1] = c * diag[k + 1];

    // Update V
    apply_givens_right_float32(v, n, n, k, k + 1, c, s);

    givens_float32(f, g, &c, &s);
    diag[k] = hypot_float32(f, g);

    f = c * super[k] + s * diag[k + 1];
    diag[k + 1] = -s * super[k] + c * diag[k + 1];
    if (k < q - 1) {
      g = s * super[k + 1];
      super[k + 1] = c * super[k + 1];
    }

    // Update U
    apply_givens_left_float32(u, m, m, k, k + 1, c, s);
  }
  super[q - 1] = f;
}

static void svd_qr_iteration_float64(double* diag, double* super, double* u,
                                     double* v, int m, int n, int p, int q) {
  // Wilkinson shift
  double d = (diag[q - 1] - diag[q]) / 2.0;
  double shift =
      diag[q] - super[q - 1] * super[q - 1] /
                    (d + sign_float64(d) * hypot_float64(d, super[q - 1]));

  // Chase the bulge
  double c = 1.0, s = 0.0;
  double f = diag[p] - shift;
  double g = super[p];

  for (int k = p; k < q; k++) {
    givens_float64(f, g, &c, &s);
    if (k > p) super[k - 1] = hypot_float64(f, g);

    f = c * diag[k] + s * super[k];
    super[k] = c * super[k] - s * diag[k];
    g = s * diag[k + 1];
    diag[k + 1] = c * diag[k + 1];

    // Update V
    apply_givens_right_float64(v, n, n, k, k + 1, c, s);

    givens_float64(f, g, &c, &s);
    diag[k] = hypot_float64(f, g);

    f = c * super[k] + s * diag[k + 1];
    diag[k + 1] = -s * super[k] + c * diag[k + 1];
    if (k < q - 1) {
      g = s * super[k + 1];
      super[k + 1] = c * super[k + 1];
    }

    // Update U
    apply_givens_left_float64(u, m, m, k, k + 1, c, s);
  }
  super[q - 1] = f;
}

// SVD using bidiagonalization and QR iteration
static void svd_float32(float* a, float* u, float* s, float* vt, int m, int n,
                        int full_matrices) {
  int min_mn = m < n ? m : n;
  int max_mn = m > n ? m : n;

  // Working arrays
  float* work_a = (float*)malloc(m * n * sizeof(float));
  float* diag = (float*)malloc(min_mn * sizeof(float));
  float* super = (float*)malloc((min_mn - 1) * sizeof(float));
  float* u_work = (float*)malloc(m * m * sizeof(float));
  float* v_work = (float*)malloc(n * n * sizeof(float));

  // Copy input matrix
  memcpy(work_a, a, m * n * sizeof(float));

  // Bidiagonalize
  bidiagonalize_float32(work_a, u_work, v_work, diag, super, m, n);

  // QR iteration
  const float eps = 1e-6f;
  const int max_iter = 100 * max_mn;

  for (int iter = 0; iter < max_iter; iter++) {
    // Find non-converged submatrix [p, q]
    int p = 0, q = min_mn - 1;

    // Find q
    for (int i = min_mn - 2; i >= 0; i--) {
      if (fabsf(super[i]) <= eps * (fabsf(diag[i]) + fabsf(diag[i + 1]))) {
        super[i] = 0.0f;
        if (i == q - 1) {
          q = i;
        }
      } else {
        break;
      }
    }

    if (q == 0) break;  // Converged

    // Find p
    for (int i = q - 1; i >= 0; i--) {
      if (i == 0 ||
          fabsf(super[i - 1]) <= eps * (fabsf(diag[i - 1]) + fabsf(diag[i]))) {
        p = i;
        break;
      }
    }

    // Apply QR iteration
    svd_qr_iteration_float32(diag, super, u_work, v_work, m, n, p, q);
  }

  // Copy singular values
  for (int i = 0; i < min_mn; i++) {
    s[i] = fabsf(diag[i]);
  }

  // Sort singular values and corresponding vectors
  for (int i = 0; i < min_mn - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < min_mn; j++) {
      if (s[j] > s[max_idx]) max_idx = j;
    }
    if (max_idx != i) {
      // Swap singular values
      float temp = s[i];
      s[i] = s[max_idx];
      s[max_idx] = temp;

      // Swap U columns
      for (int k = 0; k < m; k++) {
        temp = u_work[k * m + i];
        u_work[k * m + i] = u_work[k * m + max_idx];
        u_work[k * m + max_idx] = temp;
      }

      // Swap V columns
      for (int k = 0; k < n; k++) {
        temp = v_work[k * n + i];
        v_work[k * n + i] = v_work[k * n + max_idx];
        v_work[k * n + max_idx] = temp;
      }
    }
  }

  // Copy results
  if (full_matrices) {
    memcpy(u, u_work, m * m * sizeof(float));
    // Transpose V to get V^T
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = v_work[j * n + i];
      }
    }
  } else {
    // Copy only the first min_mn columns of U
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < min_mn; j++) {
        u[i * min_mn + j] = u_work[i * m + j];
      }
    }
    // Copy only the first min_mn rows of V^T
    for (int i = 0; i < min_mn; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = v_work[j * n + i];
      }
    }
  }

  free(work_a);
  free(diag);
  free(super);
  free(u_work);
  free(v_work);
}

static void svd_float64(double* a, double* u, double* s, double* vt, int m,
                        int n, int full_matrices) {
  int min_mn = m < n ? m : n;
  int max_mn = m > n ? m : n;

  // Working arrays
  double* work_a = (double*)malloc(m * n * sizeof(double));
  double* diag = (double*)malloc(min_mn * sizeof(double));
  double* super = (double*)malloc((min_mn - 1) * sizeof(double));
  double* u_work = (double*)malloc(m * m * sizeof(double));
  double* v_work = (double*)malloc(n * n * sizeof(double));

  // Copy input matrix
  memcpy(work_a, a, m * n * sizeof(double));

  // Bidiagonalize
  bidiagonalize_float64(work_a, u_work, v_work, diag, super, m, n);

  // QR iteration
  const double eps = 1e-14;
  const int max_iter = 100 * max_mn;

  for (int iter = 0; iter < max_iter; iter++) {
    // Find non-converged submatrix [p, q]
    int p = 0, q = min_mn - 1;

    // Find q
    for (int i = min_mn - 2; i >= 0; i--) {
      if (fabs(super[i]) <= eps * (fabs(diag[i]) + fabs(diag[i + 1]))) {
        super[i] = 0.0;
        if (i == q - 1) {
          q = i;
        }
      } else {
        break;
      }
    }

    if (q == 0) break;  // Converged

    // Find p
    for (int i = q - 1; i >= 0; i--) {
      if (i == 0 ||
          fabs(super[i - 1]) <= eps * (fabs(diag[i - 1]) + fabs(diag[i]))) {
        p = i;
        break;
      }
    }

    // Apply QR iteration
    svd_qr_iteration_float64(diag, super, u_work, v_work, m, n, p, q);
  }

  // Copy singular values
  for (int i = 0; i < min_mn; i++) {
    s[i] = fabs(diag[i]);
  }

  // Sort singular values and corresponding vectors
  for (int i = 0; i < min_mn - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < min_mn; j++) {
      if (s[j] > s[max_idx]) max_idx = j;
    }
    if (max_idx != i) {
      // Swap singular values
      double temp = s[i];
      s[i] = s[max_idx];
      s[max_idx] = temp;

      // Swap U columns
      for (int k = 0; k < m; k++) {
        temp = u_work[k * m + i];
        u_work[k * m + i] = u_work[k * m + max_idx];
        u_work[k * m + max_idx] = temp;
      }

      // Swap V columns
      for (int k = 0; k < n; k++) {
        temp = v_work[k * n + i];
        v_work[k * n + i] = v_work[k * n + max_idx];
        v_work[k * n + max_idx] = temp;
      }
    }
  }

  // Copy results
  if (full_matrices) {
    memcpy(u, u_work, m * m * sizeof(double));
    // Transpose V to get V^T
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = v_work[j * n + i];
      }
    }
  } else {
    // Copy only the first min_mn columns of U
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < min_mn; j++) {
        u[i * min_mn + j] = u_work[i * m + j];
      }
    }
    // Copy only the first min_mn rows of V^T
    for (int i = 0; i < min_mn; i++) {
      for (int j = 0; j < n; j++) {
        vt[i * n + j] = v_work[j * n + i];
      }
    }
  }

  free(work_a);
  free(diag);
  free(super);
  free(u_work);
  free(v_work);
}

// OCaml interface for SVD
CAMLprim value caml_nx_svd_bc(value* argv, int argn) {
  CAMLparam0();

  int full_matrices = Int_val(argv[0]);

  // Input tensor
  value ba_in = argv[1];
  value v_shape = argv[2];
  value v_strides = argv[3];
  int offset_in = Int_val(argv[4]);

  // U output tensor
  value ba_u = argv[5];
  value v_shape_u = argv[6];
  value v_strides_u = argv[7];
  int offset_u = Int_val(argv[8]);

  // S output tensor
  value ba_s = argv[9];
  value v_shape_s = argv[10];
  value v_strides_s = argv[11];
  int offset_s = Int_val(argv[12]);

  // VT output tensor
  value ba_vt = argv[13];
  value v_shape_vt = argv[14];
  value v_strides_vt = argv[15];
  int offset_vt = Int_val(argv[16]);

  int kind_in = Caml_ba_array_val(ba_in)->flags & CAML_BA_KIND_MASK;
  int kind_s = Caml_ba_array_val(ba_s)->flags & CAML_BA_KIND_MASK;

  // Get dimensions
  int ndim = Wosize_val(v_shape);
  if (ndim < 2) {
    caml_failwith("svd: input must have at least 2 dimensions");
  }

  int m = Int_val(Field(v_shape, ndim - 2));
  int n = Int_val(Field(v_shape, ndim - 1));

  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= Int_val(Field(v_shape, i));
  }

  int elem_size_in = get_element_size(kind_in);
  int elem_size_s = get_element_size(kind_s);

  caml_enter_blocking_section();

  // Process each matrix in the batch
  for (int b = 0; b < batch_size; b++) {
    void* data_in =
        (char*)Caml_ba_data_val(ba_in) + (offset_in + b * m * n) * elem_size_in;
    void* data_u = (char*)Caml_ba_data_val(ba_u) +
                   (offset_u + b * Int_val(Field(v_shape_u, ndim - 2)) *
                                   Int_val(Field(v_shape_u, ndim - 1))) *
                       elem_size_in;
    void* data_s =
        (char*)Caml_ba_data_val(ba_s) +
        (offset_s + b * Int_val(Field(v_shape_s, ndim - 1))) * elem_size_s;
    void* data_vt = (char*)Caml_ba_data_val(ba_vt) +
                    (offset_vt + b * Int_val(Field(v_shape_vt, ndim - 2)) *
                                     Int_val(Field(v_shape_vt, ndim - 1))) *
                        elem_size_in;

    switch (kind_in) {
      case CAML_BA_FLOAT32:
        svd_float32((float*)data_in, (float*)data_u, (float*)data_s,
                    (float*)data_vt, m, n, full_matrices);
        break;
      case CAML_BA_FLOAT64:
        svd_float64((double*)data_in, (double*)data_u, (double*)data_s,
                    (double*)data_vt, m, n, full_matrices);
        break;
      default:
        caml_leave_blocking_section();
        caml_failwith(
            "svd: unsupported dtype (only float32 and float64 supported)");
    }
  }

  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

NATIVE_WRAPPER_17(svd)

// Eigenvalue decomposition implementations

// Householder tridiagonalization for symmetric matrices
static void tridiagonalize_float32(float* a, float* q, float* diag,
                                   float* offdiag, int n) {
  // Initialize Q to identity
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      q[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  // Householder tridiagonalization (only needed for n > 2)
  if (n > 2) {
    for (int k = 0; k < n - 2; k++) {
      // Compute norm of column below diagonal
      float alpha = 0.0f;
      for (int i = k + 1; i < n; i++) {
        alpha += a[i * n + k] * a[i * n + k];
      }

      if (alpha > 0.0f) {
        alpha = sqrtf(alpha);
        if (a[(k + 1) * n + k] > 0) alpha = -alpha;

        float r = sqrtf(0.5f * (alpha * alpha - a[(k + 1) * n + k] * alpha));
        if (r == 0.0f) continue;

        // Householder vector
        float* v = (float*)calloc(n, sizeof(float));
        v[k + 1] = (a[(k + 1) * n + k] - alpha) / (2.0f * r);
        for (int i = k + 2; i < n; i++) {
          v[i] = a[i * n + k] / (2.0f * r);
        }

        // Apply transformation: A = H * A * H
        // First compute w = A * v
        float* w = (float*)calloc(n, sizeof(float));
        for (int i = 0; i < n; i++) {
          for (int j = k + 1; j < n; j++) {
            w[i] += a[i * n + j] * v[j];
          }
        }

        // Compute p = v^T * w
        float p = 0.0f;
        for (int i = k + 1; i < n; i++) {
          p += v[i] * w[i];
        }

        // Update w = w - p * v
        for (int i = 0; i < n; i++) {
          w[i] -= p * v[i];
        }

        // Update A = A - 2*v*w^T - 2*w*v^T
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            a[i * n + j] -= 2.0f * (v[i] * w[j] + w[i] * v[j]);
          }
        }

        // Update Q = Q * H
        for (int i = 0; i < n; i++) {
          float dot = 0.0f;
          for (int j = k + 1; j < n; j++) {
            dot += q[i * n + j] * v[j];
          }
          for (int j = k + 1; j < n; j++) {
            q[i * n + j] -= 2.0f * dot * v[j];
          }
        }

        free(v);
        free(w);
      }
    }
  }

  // Extract diagonal and off-diagonal
  for (int i = 0; i < n; i++) {
    diag[i] = a[i * n + i];
    if (i < n - 1) {
      // For symmetric matrices, use superdiagonal (which equals subdiagonal)
      offdiag[i] = a[i * n + (i + 1)];
    }
  }
}

// QR iteration for tridiagonal matrix
static void qr_iteration_tridiag_float32(float* diag, float* offdiag, float* q,
                                         int n) {
  const float eps = 1e-10f;
  const int max_iter = 100 * n;

  // Main iteration loop
  for (int iter = 0; iter < max_iter; iter++) {
    // Check for convergence and set small off-diagonal elements to zero
    int converged = 1;
    for (int i = 0; i < n - 1; i++) {
      if (fabsf(offdiag[i]) <= eps * (fabsf(diag[i]) + fabsf(diag[i + 1]))) {
        offdiag[i] = 0.0f;
      } else {
        converged = 0;
      }
    }
    if (converged) break;

    // Find the largest unreduced block
    int m = n - 1;
    while (m > 0 && offdiag[m - 1] == 0.0f) m--;
    if (m == 0) break;

    int l = m - 1;
    while (l > 0 && offdiag[l - 1] != 0.0f) l--;

    // Wilkinson shift
    float a = diag[m - 1];
    float b = diag[m];
    float c = m > 0 ? offdiag[m - 1] : 0.0f;
    float delta = (a - b) / 2.0f;
    float sign_delta = delta >= 0.0f ? 1.0f : -1.0f;
    float shift =
        b - c * c / (delta + sign_delta * sqrtf(delta * delta + c * c));

    // Start QR iteration
    float p = diag[l] - shift;
    float q_val = offdiag[l];

    for (int k = l; k < m; k++) {
      // Determine rotation
      float r = sqrtf(p * p + q_val * q_val);
      float c_rot = p / r;
      float s_rot = q_val / r;

      // Update off-diagonal
      if (k > l) {
        offdiag[k - 1] = r;
      }

      // Update diagonal elements
      float d1 = diag[k];
      float d2 = diag[k + 1];
      float e = offdiag[k];

      diag[k] =
          c_rot * c_rot * d1 + s_rot * s_rot * d2 - 2.0f * c_rot * s_rot * e;
      diag[k + 1] =
          s_rot * s_rot * d1 + c_rot * c_rot * d2 + 2.0f * c_rot * s_rot * e;
      offdiag[k] =
          (c_rot * c_rot - s_rot * s_rot) * e + c_rot * s_rot * (d1 - d2);

      // Update for next iteration
      if (k < m - 1) {
        p = c_rot * offdiag[k] - s_rot * offdiag[k + 1];
        q_val = s_rot * offdiag[k + 1];
        offdiag[k + 1] = c_rot * offdiag[k + 1] + s_rot * offdiag[k];
      }

      // Update eigenvectors
      for (int i = 0; i < n; i++) {
        float temp = q[i * n + k];
        q[i * n + k] = c_rot * temp - s_rot * q[i * n + k + 1];
        q[i * n + k + 1] = s_rot * temp + c_rot * q[i * n + k + 1];
      }
    }
  }

  // Final cleanup: set very small off-diagonal elements to zero
  for (int i = 0; i < n - 1; i++) {
    if (fabsf(offdiag[i]) <= eps) {
      offdiag[i] = 0.0f;
    }
  }
}

// Symmetric eigenvalue decomposition using Householder tridiagonalization + QR
static void eigh_float32(float* a, float* eigvals, float* eigvecs, int n) {
  // Working arrays
  float* work_a = (float*)malloc(n * n * sizeof(float));
  float* q = (float*)malloc(n * n * sizeof(float));
  float* diag = (float*)malloc(n * sizeof(float));
  float* offdiag = (float*)malloc((n - 1) * sizeof(float));

  // Copy input matrix for tridiagonalization
  memcpy(work_a, a, n * n * sizeof(float));

  // Tridiagonalize (modifies work_a in place and fills diag/offdiag)
  tridiagonalize_float32(work_a, q, diag, offdiag, n);

  // QR iteration on tridiagonal matrix
  qr_iteration_tridiag_float32(diag, offdiag, q, n);

  // Copy eigenvalues
  for (int i = 0; i < n; i++) {
    eigvals[i] = diag[i];
  }

  // Sort eigenvalues and eigenvectors in descending order
  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++) {
      if (eigvals[j] > eigvals[max_idx]) max_idx = j;
    }
    if (max_idx != i) {
      // Swap eigenvalues
      float temp = eigvals[i];
      eigvals[i] = eigvals[max_idx];
      eigvals[max_idx] = temp;

      // Swap eigenvector columns
      for (int k = 0; k < n; k++) {
        temp = q[k * n + i];
        q[k * n + i] = q[k * n + max_idx];
        q[k * n + max_idx] = temp;
      }
    }
  }

  // Copy eigenvectors
  if (eigvecs != NULL) {
    memcpy(eigvecs, q, n * n * sizeof(float));
  }

  free(work_a);
  free(q);
  free(diag);
  free(offdiag);
}

// Householder tridiagonalization for symmetric matrices (double precision)
static void tridiagonalize_float64(double* a, double* q, double* diag,
                                   double* offdiag, int n) {
  // Initialize Q to identity
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      q[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  // Householder tridiagonalization (only needed for n > 2)
  if (n > 2) {
    for (int k = 0; k < n - 2; k++) {
      // Compute norm of column below diagonal
      double alpha = 0.0;
      for (int i = k + 1; i < n; i++) {
        alpha += a[i * n + k] * a[i * n + k];
      }

      if (alpha > 0.0) {
        alpha = sqrt(alpha);
        if (a[(k + 1) * n + k] > 0) alpha = -alpha;

        double r = sqrt(0.5 * (alpha * alpha - a[(k + 1) * n + k] * alpha));
        if (r == 0.0) continue;

        // Householder vector
        double* v = (double*)calloc(n, sizeof(double));
        v[k + 1] = (a[(k + 1) * n + k] - alpha) / (2.0 * r);
        for (int i = k + 2; i < n; i++) {
          v[i] = a[i * n + k] / (2.0 * r);
        }

        // Apply transformation: A = H * A * H
        // First compute w = A * v
        double* w = (double*)calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) {
          for (int j = k + 1; j < n; j++) {
            w[i] += a[i * n + j] * v[j];
          }
        }

        // Compute p = v^T * w
        double p = 0.0;
        for (int i = k + 1; i < n; i++) {
          p += v[i] * w[i];
        }

        // Update w = w - p * v
        for (int i = 0; i < n; i++) {
          w[i] -= p * v[i];
        }

        // Update A = A - 2*v*w^T - 2*w*v^T
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            a[i * n + j] -= 2.0 * (v[i] * w[j] + w[i] * v[j]);
          }
        }

        // Update Q = Q * H
        for (int i = 0; i < n; i++) {
          double dot = 0.0;
          for (int j = k + 1; j < n; j++) {
            dot += q[i * n + j] * v[j];
          }
          for (int j = k + 1; j < n; j++) {
            q[i * n + j] -= 2.0 * dot * v[j];
          }
        }

        free(v);
        free(w);
      }
    }
  }

  // Extract diagonal and off-diagonal
  for (int i = 0; i < n; i++) {
    diag[i] = a[i * n + i];
    if (i < n - 1) {
      // For symmetric matrices, use superdiagonal (which equals subdiagonal)
      offdiag[i] = a[i * n + (i + 1)];
    }
  }
}

// QR iteration for tridiagonal matrix (double precision)
static void qr_iteration_tridiag_float64(double* diag, double* offdiag,
                                         double* q, int n) {
  const double eps = 1e-15;
  const int max_iter = 100 * n;

  // Main iteration loop
  for (int iter = 0; iter < max_iter; iter++) {
    // Check for convergence and set small off-diagonal elements to zero
    int converged = 1;
    for (int i = 0; i < n - 1; i++) {
      if (fabs(offdiag[i]) <= eps * (fabs(diag[i]) + fabs(diag[i + 1]))) {
        offdiag[i] = 0.0;
      } else {
        converged = 0;
      }
    }
    if (converged) break;

    // Find the largest unreduced block
    int m = n - 1;
    while (m > 0 && offdiag[m - 1] == 0.0) m--;
    if (m == 0) break;

    int l = m - 1;
    while (l > 0 && offdiag[l - 1] != 0.0) l--;

    // Wilkinson shift
    double a = diag[m - 1];
    double b = diag[m];
    double c = m > 0 ? offdiag[m - 1] : 0.0;
    double delta = (a - b) / 2.0;
    double sign_delta = delta >= 0.0 ? 1.0 : -1.0;
    double shift =
        b - c * c / (delta + sign_delta * sqrt(delta * delta + c * c));

    // Start QR iteration
    double p = diag[l] - shift;
    double q_val = offdiag[l];

    for (int k = l; k < m; k++) {
      // Determine rotation
      double r = sqrt(p * p + q_val * q_val);
      double c_rot = p / r;
      double s_rot = q_val / r;

      // Update off-diagonal
      if (k > l) {
        offdiag[k - 1] = r;
      }

      // Update diagonal elements
      double d1 = diag[k];
      double d2 = diag[k + 1];
      double e = offdiag[k];

      diag[k] =
          c_rot * c_rot * d1 + s_rot * s_rot * d2 - 2.0 * c_rot * s_rot * e;
      diag[k + 1] =
          s_rot * s_rot * d1 + c_rot * c_rot * d2 + 2.0 * c_rot * s_rot * e;
      offdiag[k] =
          (c_rot * c_rot - s_rot * s_rot) * e + c_rot * s_rot * (d1 - d2);

      // Update for next iteration
      if (k < m - 1) {
        p = c_rot * offdiag[k] - s_rot * offdiag[k + 1];
        q_val = s_rot * offdiag[k + 1];
        offdiag[k + 1] = c_rot * offdiag[k + 1] + s_rot * offdiag[k];
      }

      // Update eigenvectors
      for (int i = 0; i < n; i++) {
        double temp = q[i * n + k];
        q[i * n + k] = c_rot * temp - s_rot * q[i * n + k + 1];
        q[i * n + k + 1] = s_rot * temp + c_rot * q[i * n + k + 1];
      }
    }
  }

  // Final cleanup: set very small off-diagonal elements to zero
  for (int i = 0; i < n - 1; i++) {
    if (fabs(offdiag[i]) <= eps) {
      offdiag[i] = 0.0;
    }
  }
}

// Symmetric eigenvalue decomposition using Householder tridiagonalization + QR
static void eigh_float64(double* a, double* eigvals, double* eigvecs, int n) {
  // Working arrays
  double* work_a = (double*)malloc(n * n * sizeof(double));
  double* q = (double*)malloc(n * n * sizeof(double));
  double* diag = (double*)malloc(n * sizeof(double));
  double* offdiag = (double*)malloc((n - 1) * sizeof(double));

  // Copy input matrix
  memcpy(work_a, a, n * n * sizeof(double));

  // Tridiagonalize
  tridiagonalize_float64(work_a, q, diag, offdiag, n);

  // QR iteration on tridiagonal matrix
  qr_iteration_tridiag_float64(diag, offdiag, q, n);

  // Copy eigenvalues
  for (int i = 0; i < n; i++) {
    eigvals[i] = diag[i];
  }

  // Sort eigenvalues and eigenvectors in descending order
  for (int i = 0; i < n - 1; i++) {
    int max_idx = i;
    for (int j = i + 1; j < n; j++) {
      if (eigvals[j] > eigvals[max_idx]) max_idx = j;
    }
    if (max_idx != i) {
      // Swap eigenvalues
      double temp = eigvals[i];
      eigvals[i] = eigvals[max_idx];
      eigvals[max_idx] = temp;

      // Swap eigenvector columns
      for (int k = 0; k < n; k++) {
        temp = q[k * n + i];
        q[k * n + i] = q[k * n + max_idx];
        q[k * n + max_idx] = temp;
      }
    }
  }

  // Copy eigenvectors
  if (eigvecs != NULL) {
    memcpy(eigvecs, q, n * n * sizeof(double));
  }

  free(work_a);
  free(q);
  free(diag);
  free(offdiag);
}

// OCaml interface for eigenvalue decomposition
CAMLprim value caml_nx_eig_bc(value* argv, int argn) {
  CAMLparam0();

  int symmetric = Int_val(argv[0]);
  int compute_vectors = Int_val(argv[1]);

  // Input tensor
  value ba_in = argv[2];
  value v_shape = argv[3];
  value v_strides = argv[4];
  int offset_in = Int_val(argv[5]);

  // Eigenvalues output tensor (complex for general, real for symmetric)
  value ba_vals = argv[6];
  value v_shape_vals = argv[7];
  value v_strides_vals = argv[8];
  int offset_vals = Int_val(argv[9]);

  // Eigenvectors output tensor (optional)
  value ba_vecs = argv[10];
  value v_shape_vecs = argv[11];
  value v_strides_vecs = argv[12];
  int offset_vecs = Int_val(argv[13]);

  int kind = Caml_ba_array_val(ba_in)->flags & CAML_BA_KIND_MASK;
  int kind_vals = Caml_ba_array_val(ba_vals)->flags & CAML_BA_KIND_MASK;

  // Get dimensions
  int ndim = Wosize_val(v_shape);
  if (ndim < 2) {
    caml_failwith("eig: input must have at least 2 dimensions");
  }

  int n = Int_val(Field(v_shape, ndim - 1));
  int m = Int_val(Field(v_shape, ndim - 2));

  if (n != m) {
    caml_failwith("eig: input must be square matrix");
  }

  // Calculate batch size
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= Int_val(Field(v_shape, i));
  }

  int elem_size = get_element_size(kind);
  int elem_size_vals = get_element_size(kind_vals);

  caml_enter_blocking_section();

  // Process each matrix in the batch
  for (int b = 0; b < batch_size; b++) {
    void* data_in =
        (char*)Caml_ba_data_val(ba_in) + (offset_in + b * n * n) * elem_size;
    void* data_vecs = compute_vectors
                          ? (char*)Caml_ba_data_val(ba_vecs) +
                                (offset_vecs + b * n * n) * elem_size
                          : NULL;

    if (symmetric) {
      // Symmetric eigenvalue decomposition - eigenvalues are real
      void* data_vals = (char*)Caml_ba_data_val(ba_vals) +
                        (offset_vals + b * n) * elem_size_vals;

      switch (kind) {
        case CAML_BA_FLOAT32:
          eigh_float32((float*)data_in, (float*)data_vals, (float*)data_vecs,
                       n);
          break;
        case CAML_BA_FLOAT64:
          eigh_float64((double*)data_in, (double*)data_vals, (double*)data_vecs,
                       n);
          break;
        default:
          caml_leave_blocking_section();
          caml_failwith("eig: unsupported dtype for symmetric matrices");
      }
    } else {
      // General eigenvalue decomposition - not implemented to match native
      // backend
      caml_leave_blocking_section();
      caml_failwith("eig: general eigenvalue decomposition not implemented");
    }
  }

  caml_leave_blocking_section();

  CAMLreturn(Val_unit);
}

NATIVE_WRAPPER_14(eig)
