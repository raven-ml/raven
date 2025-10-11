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

// Machine epsilon for different precisions
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

// Helper function to check if character matches (case insensitive)
static int lsame(char ca, char cb) {
  if (ca == cb) return 1;
  int inta = (unsigned char)ca;
  int intb = (unsigned char)cb;
  return (inta >= 'A' && inta <= 'Z' ? inta + 32 : inta) ==
         (intb >= 'A' && intb <= 'Z' ? intb + 32 : intb);
}

// Cholesky decomposition implementations using LAPACK

static int cholesky_float32(float* a, int n, int upper) {
  char uplo = upper ? 'U' : 'L';
  lapack_int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, a, n);
  if (info == 0) {
    // Zero out the unused triangle
    if (upper) {
      // Zero the lower triangle
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          a[i * n + j] = 0.0f;
        }
      }
    } else {
      // Zero the upper triangle
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          a[i * n + j] = 0.0f;
        }
      }
    }
  }
  return (int)info;
}

static int cholesky_float64(double* a, int n, int upper) {
  char uplo = upper ? 'U' : 'L';
  lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, a, n);
  if (info == 0) {
    // Zero out the unused triangle
    if (upper) {
      // Zero the lower triangle
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          a[i * n + j] = 0.0;
        }
      }
    } else {
      // Zero the upper triangle
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          a[i * n + j] = 0.0;
        }
      }
    }
  }
  return (int)info;
}

static int cholesky_complex32(complex32* a, int n, int upper) {
  char uplo = upper ? 'U' : 'L';
  lapack_int info = LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, n, a, n);
  if (info == 0) {
    // Zero out the unused triangle
    if (upper) {
      // Zero the lower triangle
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          a[i * n + j] = 0.0f + 0.0f * I;
        }
      }
    } else {
      // Zero the upper triangle
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          a[i * n + j] = 0.0f + 0.0f * I;
        }
      }
    }
  }
  return (int)info;
}

static int cholesky_complex64(complex64* a, int n, int upper) {
  char uplo = upper ? 'U' : 'L';
  lapack_int info = LAPACKE_zpotrf(LAPACK_ROW_MAJOR, uplo, n, a, n);
  if (info == 0) {
    // Zero out the unused triangle
    if (upper) {
      // Zero the lower triangle
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          a[i * n + j] = 0.0 + 0.0 * I;
        }
      }
    } else {
      // Zero the upper triangle
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          a[i * n + j] = 0.0 + 0.0 * I;
        }
      }
    }
  }
  return (int)info;
}

// Lower precision implementations that upcast to float32/float64
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
    a_complex[i] = complex16_to_complex32(a[i]);
  }
  int status = cholesky_complex32(a_complex, n, upper);
  if (status == 0) {
    for (int i = 0; i < n * n; i++) {
      a[i] = complex32_to_complex16(a_complex[i]);
    }
  }
  free(a_complex);
  return status;
}

// OCaml FFI stub
CAMLprim value caml_nx_op_cholesky(value v_in, value v_out, value v_upper) {
  CAMLparam3(v_in, v_out, v_upper);
  int upper = Int_val(v_upper);
  ndarray_t in = extract_ndarray(v_in);
  ndarray_t out = extract_ndarray(v_out);
  struct caml_ba_array* ba_in = Caml_ba_array_val(Field(v_in, FFI_TENSOR_DATA));
  struct caml_ba_array* ba_out =
      Caml_ba_array_val(Field(v_out, FFI_TENSOR_DATA));
  int kind = nx_ba_get_kind(ba_in);
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
      caml_invalid_argument("cholesky: not positive-definite");
    }
  }
  caml_leave_blocking_section();
  cleanup_ndarray(&in);
  cleanup_ndarray(&out);
  CAMLreturn(Val_unit);
}
