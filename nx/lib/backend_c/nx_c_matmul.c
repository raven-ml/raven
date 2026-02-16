/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// Matrix multiplication for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <cblas.h>
#include <limits.h>

#include "nx_c_shared.h"

// Type definitions for matmul operations
typedef void (*matmul_op_t)(const ndarray_t *, const ndarray_t *, ndarray_t *);

// Dispatch table for each type
typedef struct {
  matmul_op_t i8, u8, i16, u16, i32, i64, u32, u64, inat;
  matmul_op_t f16, f32, f64;
  matmul_op_t c32, c64;
  matmul_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2;
} matmul_op_table;

// Macro to generate all standard type variants for matmul
#define GENERATE_MATMUL_OP(suffix, T, ACCUM_T, CAST) \
  MATMUL_OP_FOR_TYPE(suffix, T, ACCUM_T, CAST)

// Helper to iterate over batch dimensions with a kernel function for matmul
typedef void (*matmul_kernel_t)(void *, long, long, long, void *, long, long,
                                long, void *, long, long, long, long, long,
                                long);

static inline void iterate_batch(
    const long *batch_shape, int batch_nd, const long *batch_strides_a,
    const long *batch_strides_b, const long *batch_strides_c, void *a_data,
    void *b_data, void *c_data, long a_off, long b_off, long c_off, long a_rs,
    long a_cs, long b_rs, long b_cs, long c_rs, long c_cs, long m, long k,
    long n, matmul_kernel_t kernel) {
  if (batch_nd <= 0) {
    kernel(a_data, a_off, a_rs, a_cs, b_data, b_off, b_rs, b_cs, c_data, c_off,
           c_rs, c_cs, m, k, n);
    return;
  }

  int coords_buf[MAX_NDIM];
  int *coords = coords_buf; // batch_nd <= MAX_NDIM by construction
  for (int i = 0; i < batch_nd; ++i) coords[i] = 0;

  bool done = false;
  while (!done) {
    long a_batch_off = a_off;
    long b_batch_off = b_off;
    long c_batch_off = c_off;

    for (int i = 0; i < batch_nd; i++) {
      a_batch_off += coords[i] * batch_strides_a[i];
      b_batch_off += coords[i] * batch_strides_b[i];
      c_batch_off += coords[i] * batch_strides_c[i];
    }

    kernel(a_data, a_batch_off, a_rs, a_cs, b_data, b_batch_off, b_rs, b_cs,
           c_data, c_batch_off, c_rs, c_cs, m, k, n);

    // Advance to next position
    done = true;
    for (int i = batch_nd - 1; i >= 0; i--) {
      coords[i]++;
      if (coords[i] < batch_shape[i]) {
        done = false;
        break;
      }
      coords[i] = 0;
    }
  }
}

// Generic matmul kernel
#define MATMUL_OP_KERNEL(suffix, T, ACCUM_T, CAST)                                  \
  static void nx_c_matmul_##suffix##_kernel(                                        \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,                 \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,        \
      long c_cs, long m, long k, long n) {                                          \
    T *restrict a = (T *)a_data;                                                    \
    T *restrict b = (T *)b_data;                                                    \
    T *restrict c = (T *)c_data;                                                    \
    /* Generic kernel (naive triple loop). Specific types may override               \
       with specialized kernels below. */                                            \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)")                        \
    for (long i = 0; i < m; i++) {                                                  \
      for (long j = 0; j < n; j++) {                                                \
        ACCUM_T sum = 0;                                                            \
        for (long p = 0; p < k; p++) {                                              \
          sum += (ACCUM_T)a[a_off + i * a_rs + p * a_cs] *                          \
                 (ACCUM_T)b[b_off + p * b_rs + j * b_cs];                           \
        }                                                                           \
        c[c_off + i * c_rs + j * c_cs] = CAST(sum);                                 \
      }                                                                             \
    }                                                                               \
  }

// Generic matmul implementation
#define MATMUL_OP_IMPL(suffix, ELEM_SIZE)                                      \
  static void nx_c_matmul_##suffix(const ndarray_t *a, const ndarray_t *b,     \
                                   ndarray_t *c) {                             \
    if (!a || !b || !c) {                                                      \
      caml_failwith("nx_c_matmul_" #suffix ": null pointer");                  \
    }                                                                          \
    int nd = a->ndim > b->ndim ? a->ndim : b->ndim;                            \
    if (c->ndim != nd) {                                                       \
      caml_failwith("nx_c_matmul_" #suffix ": output ndim mismatch");          \
    }                                                                          \
    if (a->ndim < 2 || b->ndim < 2) {                                          \
      caml_failwith("nx_c_matmul_" #suffix ": input ndim < 2");                \
    }                                                                          \
    long m = a->shape[a->ndim - 2];                                            \
    long k = a->shape[a->ndim - 1];                                            \
    long kk = b->shape[b->ndim - 2];                                           \
    long n = b->shape[b->ndim - 1];                                            \
    if (k != kk) {                                                             \
      /* Build shape strings for error message */                              \
      char shape_a_str[256] = "[";                                             \
      char shape_b_str[256] = "[";                                             \
      for (int i = 0; i < a->ndim; i++) {                                      \
        char buf[32];                                                          \
        snprintf(buf, sizeof(buf), "%s%d", i > 0 ? "," : "", a->shape[i]);   \
        strcat(shape_a_str, buf);                                              \
      }                                                                        \
      strcat(shape_a_str, "]");                                                \
      for (int i = 0; i < b->ndim; i++) {                                      \
        char buf[32];                                                          \
        snprintf(buf, sizeof(buf), "%s%d", i > 0 ? "," : "", b->shape[i]);   \
        strcat(shape_b_str, buf);                                              \
      }                                                                        \
      strcat(shape_b_str, "]");                                                \
      char msg[512];                                                           \
      snprintf(msg, sizeof(msg),                                               \
               "dot: cannot contract %s (last axis: %ld) to %s (axis %d: %ld) "\
               "(size %ld≠%ld)",                                               \
               shape_a_str, k, shape_b_str, b->ndim - 2, kk, k, kk);           \
      caml_invalid_argument(msg);                                              \
    }                                                                          \
    if (c->shape[c->ndim - 2] != m || c->shape[c->ndim - 1] != n) {            \
      caml_failwith("nx_c_matmul_" #suffix ": output shape mismatch");         \
    }                                                                          \
    int batch_nd = nd - 2;                                                     \
    long batch_shape_buf[MAX_NDIM];                                            \
    long batch_strides_a_buf[MAX_NDIM];                                        \
    long batch_strides_b_buf[MAX_NDIM];                                        \
    long batch_strides_c_buf[MAX_NDIM];                                        \
    long *batch_shape = batch_shape_buf;                                       \
    long *batch_strides_a = batch_strides_a_buf;                                \
    long *batch_strides_b = batch_strides_b_buf;                                \
    long *batch_strides_c = batch_strides_c_buf;                                \
    int a_batch_offset = nd - a->ndim;                                         \
    int b_batch_offset = nd - b->ndim;                                         \
    for (int i = 0; i < batch_nd; i++) {                                       \
      long sa = 1, sb = 1;                                                     \
      long stra = 0, strb = 0;                                                 \
      if (i >= a_batch_offset) {                                               \
        int a_i = i - a_batch_offset;                                          \
        sa = a->shape[a_i];                                                    \
        stra = a->strides[a_i];                                                \
      }                                                                        \
      if (i >= b_batch_offset) {                                               \
        int b_i = i - b_batch_offset;                                          \
        sb = b->shape[b_i];                                                    \
        strb = b->strides[b_i];                                                \
      }                                                                        \
      if (sa != sb && sa != 1 && sb != 1) {                                    \
        caml_failwith("nx_c_matmul_" #suffix ": batch shape mismatch");        \
      }                                                                        \
      long s = sa > sb ? sa : sb;                                              \
      batch_shape[i] = s;                                                      \
      batch_strides_a[i] = (sa == 1) ? 0 : stra;                               \
      batch_strides_b[i] = (sb == 1) ? 0 : strb;                               \
      batch_strides_c[i] = c->strides[i];                                      \
      if (c->shape[i] != s) {                                                  \
        caml_failwith("nx_c_matmul_" #suffix ": output batch shape mismatch"); \
      }                                                                        \
    }                                                                          \
    long a_rs = a->strides[a->ndim - 2];                                       \
    long a_cs = a->strides[a->ndim - 1];                                       \
    long b_rs = b->strides[b->ndim - 2];                                       \
    long b_cs = b->strides[b->ndim - 1];                                       \
    long c_rs = c->strides[c->ndim - 2];                                       \
    long c_cs = c->strides[c->ndim - 1];                                       \
    void *a_data = (char *)a->data + (ELEM_SIZE ? a->offset * ELEM_SIZE : a->offset / 2);  \
    void *b_data = (char *)b->data + (ELEM_SIZE ? b->offset * ELEM_SIZE : b->offset / 2);  \
    void *c_data = (char *)c->data + (ELEM_SIZE ? c->offset * ELEM_SIZE : c->offset / 2);  \
    caml_enter_blocking_section();                                             \
    iterate_batch(batch_shape, batch_nd, batch_strides_a, batch_strides_b,     \
                  batch_strides_c, a_data, b_data, c_data, 0, 0, 0, a_rs,      \
                  a_cs, b_rs, b_cs, c_rs, c_cs, m, k, n,                       \
                  nx_c_matmul_##suffix##_kernel);                              \
    caml_leave_blocking_section();                                             \
  }

// Macro to generate both kernel and implementation for matmul
#define MATMUL_OP_FOR_TYPE(suffix, T, ACCUM_T, CAST) \
  MATMUL_OP_KERNEL(suffix, T, ACCUM_T, CAST)         \
  MATMUL_OP_IMPL(suffix, sizeof(T))

// Low-precision float kernel (convert to float for mul/acc)
#define LOW_PREC_MATMUL_KERNEL(suffix, T, TO_FLOAT, FROM_FLOAT)               \
  static void nx_c_matmul_##suffix##_kernel(                                  \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,           \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,  \
      long c_cs, long m, long k, long n) {                                    \
    T *a = (T *)a_data;                                                       \
    T *b = (T *)b_data;                                                       \
    T *c = (T *)c_data;                                                       \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)") for (long i = 0; \
                                                                  i < m;      \
                                                                  i++) {      \
      for (long j = 0; j < n; j++) {                                          \
        float sum = 0.0f;                                                     \
        for (long p = 0; p < k; p++) {                                        \
          float aa = TO_FLOAT(a[a_off + i * a_rs + p * a_cs]);                \
          float bb = TO_FLOAT(b[b_off + p * b_rs + j * b_cs]);                \
          sum += aa * bb;                                                     \
        }                                                                     \
        c[c_off + i * c_rs + j * c_cs] = FROM_FLOAT(sum);                     \
      }                                                                       \
    }                                                                         \
  }

// For low-precision, use the impl with the special kernel
#define LOW_PREC_MATMUL_IMPL(suffix, T) MATMUL_OP_IMPL(suffix, sizeof(T))

// Special implementation for int4 (packed, unpack/mul/acc/pack with saturation)
#define INT4_MATMUL_IMPL(signedness, suffix)                                   \
  static void nx_c_matmul_##suffix##_kernel(                                   \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,            \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,   \
      long c_cs, long m, long k, long n) {                                     \
    uint8_t *a = (uint8_t *)a_data;                                            \
    uint8_t *b = (uint8_t *)b_data;                                            \
    uint8_t *c = (uint8_t *)c_data;                                            \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)") for (long i = 0;  \
                                                                  i < m;       \
                                                                  i++) {       \
      for (long j = 0; j < n; j++) {                                           \
        int32_t sum = 0;                                                       \
        for (long p = 0; p < k; p++) {                                         \
          long a_idx = a_off + i * a_rs + p * a_cs;                            \
          long a_byte_off = a_idx / 2;                                         \
          int a_nib_off = a_idx % 2;                                           \
          int aa =                                                             \
              a_nib_off                                                        \
                  ? (signedness ? (int8_t)(a[a_byte_off] >> 4)                 \
                                : ((a[a_byte_off] >> 4) & 0x0F))               \
                  : (signedness ? (int8_t)(((a[a_byte_off] & 0x0F) << 4) >> 4) \
                                : (a[a_byte_off] & 0x0F));                     \
          long b_idx = b_off + p * b_rs + j * b_cs;                            \
          long b_byte_off = b_idx / 2;                                         \
          int b_nib_off = b_idx % 2;                                           \
          int bb =                                                             \
              b_nib_off                                                        \
                  ? (signedness ? (int8_t)(b[b_byte_off] >> 4)                 \
                                : ((b[b_byte_off] >> 4) & 0x0F))               \
                  : (signedness ? (int8_t)(((b[b_byte_off] & 0x0F) << 4) >> 4) \
                                : (b[b_byte_off] & 0x0F));                     \
          sum += aa * bb;                                                      \
        }                                                                      \
        int res = signedness ? CLAMP_I4(sum) : CLAMP_U4(sum);                  \
        uint8_t nib = (uint8_t)res & 0x0F;                                     \
        long c_idx = c_off + i * c_rs + j * c_cs;                              \
        long c_byte_off = c_idx / 2;                                           \
        int c_nib_off = c_idx % 2;                                             \
        if (c_nib_off) {                                                       \
          c[c_byte_off] = (c[c_byte_off] & 0x0F) | (nib << 4);                 \
        } else {                                                               \
          c[c_byte_off] = (c[c_byte_off] & 0xF0) | nib;                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  MATMUL_OP_IMPL(suffix, 0)  /* int4 offset is in nibbles, handled in kernel */

// Generate for integer types with wider accumulation
GENERATE_MATMUL_OP(i8, int8_t, int64_t, (int8_t))
GENERATE_MATMUL_OP(u8, uint8_t, uint64_t, (uint8_t))
GENERATE_MATMUL_OP(i16, int16_t, int64_t, (int16_t))
GENERATE_MATMUL_OP(u16, uint16_t, uint64_t, (uint16_t))
GENERATE_MATMUL_OP(i32, int32_t, int64_t, (int32_t))
GENERATE_MATMUL_OP(i64, int64_t, int64_t, (int64_t))
GENERATE_MATMUL_OP(u32, uint32_t, uint64_t, (uint32_t))
GENERATE_MATMUL_OP(u64, uint64_t, uint64_t, (uint64_t))
GENERATE_MATMUL_OP(inat, intnat, int64_t, (intnat))
GENERATE_MATMUL_OP(bool_, caml_ba_bool, uint64_t, (caml_ba_bool))

// Float types with same-type accumulation
/* BLAS-based GEMM kernels for float32/float64 using CBLAS.
   These use optimized BLAS routines when possible, falling back to
   packing for non-contiguous strides. */

static inline int setup_blas_row_major_params(long rows, long cols, long row_stride,
                                              long col_stride, CBLAS_TRANSPOSE *trans,
                                              int *ld) {
  if (row_stride <= 0 || col_stride <= 0) return 0;
  if (col_stride == 1) {
    if (row_stride < cols || row_stride > INT_MAX) return 0;
    *trans = CblasNoTrans;
    *ld = (int)row_stride;
    return 1;
  }
  if (row_stride == 1) {
    if (col_stride < rows || col_stride > INT_MAX) return 0;
    *trans = CblasTrans;
    *ld = (int)col_stride;
    return 1;
  }
  return 0;
}

static inline int setup_blas_row_major_output(long rows, long cols, long row_stride,
                                              long col_stride, int *ld) {
  if (col_stride != 1) return 0;
  if (row_stride < cols || row_stride > INT_MAX) return 0;
  *ld = (int)row_stride;
  return 1;
}

static void nx_c_matmul_f32_kernel(void *a_data, long a_off, long a_rs,
                                   long a_cs, void *b_data, long b_off,
                                   long b_rs, long b_cs, void *c_data,
                                   long c_off, long c_rs, long c_cs, long m,
                                   long k, long n) {
  float *restrict a = (float *)a_data;
  float *restrict b = (float *)b_data;
  float *restrict c = (float *)c_data;

  int use_blas_direct = 0;
  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasNoTrans;
  int lda = 0, ldb = 0, ldc = 0;

  if (setup_blas_row_major_params(m, k, a_rs, a_cs, &trans_a, &lda) &&
      setup_blas_row_major_params(k, n, b_rs, b_cs, &trans_b, &ldb) &&
      setup_blas_row_major_output(m, n, c_rs, c_cs, &ldc)) {
    use_blas_direct = 1;
  }

  if (use_blas_direct) {
    cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, 1.0f,
                a + a_off, lda, b + b_off, ldb, 0.0f, c + c_off, ldc);
  } else {
    /* Non-contiguous layout: pack matrices first */
    float *a_packed = (float *)malloc(m * k * sizeof(float));
    float *b_packed = (float *)malloc(k * n * sizeof(float));
    float *c_packed = (float *)malloc(m * n * sizeof(float));
    if (!a_packed || !b_packed || !c_packed) {
      free(a_packed);
      free(b_packed);
      free(c_packed);
      return;
    }

    /* Pack A and B */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < k; j++) {
        a_packed[i * k + j] = a[a_off + i * a_rs + j * a_cs];
      }
    }
    for (long i = 0; i < k; i++) {
      for (long j = 0; j < n; j++) {
        b_packed[i * n + j] = b[b_off + i * b_rs + j * b_cs];
      }
    }

    /* Compute using BLAS */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                a_packed, k, b_packed, n, 0.0f, c_packed, n);

    /* Unpack C */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < n; j++) {
        c[c_off + i * c_rs + j * c_cs] = c_packed[i * n + j];
      }
    }

    free(a_packed);
    free(b_packed);
    free(c_packed);
  }
}

static void nx_c_matmul_f64_kernel(void *a_data, long a_off, long a_rs,
                                   long a_cs, void *b_data, long b_off,
                                   long b_rs, long b_cs, void *c_data,
                                   long c_off, long c_rs, long c_cs, long m,
                                   long k, long n) {
  double *restrict a = (double *)a_data;
  double *restrict b = (double *)b_data;
  double *restrict c = (double *)c_data;

  int use_blas_direct = 0;
  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasNoTrans;
  int lda = 0, ldb = 0, ldc = 0;

  if (setup_blas_row_major_params(m, k, a_rs, a_cs, &trans_a, &lda) &&
      setup_blas_row_major_params(k, n, b_rs, b_cs, &trans_b, &ldb) &&
      setup_blas_row_major_output(m, n, c_rs, c_cs, &ldc)) {
    use_blas_direct = 1;
  }

  if (use_blas_direct) {
    cblas_dgemm(CblasRowMajor, trans_a, trans_b, m, n, k, 1.0,
                a + a_off, lda, b + b_off, ldb, 0.0, c + c_off, ldc);
  } else {
    /* Non-contiguous layout: pack matrices first */
    double *a_packed = (double *)malloc(m * k * sizeof(double));
    double *b_packed = (double *)malloc(k * n * sizeof(double));
    double *c_packed = (double *)malloc(m * n * sizeof(double));
    if (!a_packed || !b_packed || !c_packed) {
      free(a_packed);
      free(b_packed);
      free(c_packed);
      return;
    }

    /* Pack A and B */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < k; j++) {
        a_packed[i * k + j] = a[a_off + i * a_rs + j * a_cs];
      }
    }
    for (long i = 0; i < k; i++) {
      for (long j = 0; j < n; j++) {
        b_packed[i * n + j] = b[b_off + i * b_rs + j * b_cs];
      }
    }

    /* Compute using BLAS */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                a_packed, k, b_packed, n, 0.0, c_packed, n);

    /* Unpack C */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < n; j++) {
        c[c_off + i * c_rs + j * c_cs] = c_packed[i * n + j];
      }
    }

    free(a_packed);
    free(b_packed);
    free(c_packed);
  }
}

/* Use the optimized kernels for f32/f64 and the generic implementation glue */
MATMUL_OP_IMPL(f32, sizeof(float))
MATMUL_OP_IMPL(f64, sizeof(double))

// Complex types with BLAS GEMM
static void nx_c_matmul_c32_kernel(void *a_data, long a_off, long a_rs,
                                   long a_cs, void *b_data, long b_off,
                                   long b_rs, long b_cs, void *c_data,
                                   long c_off, long c_rs, long c_cs, long m,
                                   long k, long n) {
  complex32 *restrict a = (complex32 *)a_data;
  complex32 *restrict b = (complex32 *)b_data;
  complex32 *restrict c = (complex32 *)c_data;

  complex32 alpha = 1.0f + 0.0f * I;
  complex32 beta = 0.0f + 0.0f * I;

  int use_blas_direct = 0;
  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasNoTrans;
  int lda = 0, ldb = 0, ldc = 0;

  if (setup_blas_row_major_params(m, k, a_rs, a_cs, &trans_a, &lda) &&
      setup_blas_row_major_params(k, n, b_rs, b_cs, &trans_b, &ldb) &&
      setup_blas_row_major_output(m, n, c_rs, c_cs, &ldc)) {
    use_blas_direct = 1;
  }

  if (use_blas_direct) {
    cblas_cgemm(CblasRowMajor, trans_a, trans_b, m, n, k, &alpha,
                a + a_off, lda, b + b_off, ldb, &beta, c + c_off, ldc);
  } else {
    /* Non-contiguous layout: pack matrices first */
    complex32 *a_packed = (complex32 *)malloc(m * k * sizeof(complex32));
    complex32 *b_packed = (complex32 *)malloc(k * n * sizeof(complex32));
    complex32 *c_packed = (complex32 *)malloc(m * n * sizeof(complex32));
    if (!a_packed || !b_packed || !c_packed) {
      free(a_packed);
      free(b_packed);
      free(c_packed);
      return;
    }

    /* Pack A and B */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < k; j++) {
        a_packed[i * k + j] = a[a_off + i * a_rs + j * a_cs];
      }
    }
    for (long i = 0; i < k; i++) {
      for (long j = 0; j < n; j++) {
        b_packed[i * n + j] = b[b_off + i * b_rs + j * b_cs];
      }
    }

    /* Compute using BLAS */
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                a_packed, k, b_packed, n, &beta, c_packed, n);

    /* Unpack C */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < n; j++) {
        c[c_off + i * c_rs + j * c_cs] = c_packed[i * n + j];
      }
    }

    free(a_packed);
    free(b_packed);
    free(c_packed);
  }
}

static void nx_c_matmul_c64_kernel(void *a_data, long a_off, long a_rs,
                                   long a_cs, void *b_data, long b_off,
                                   long b_rs, long b_cs, void *c_data,
                                   long c_off, long c_rs, long c_cs, long m,
                                   long k, long n) {
  complex64 *restrict a = (complex64 *)a_data;
  complex64 *restrict b = (complex64 *)b_data;
  complex64 *restrict c = (complex64 *)c_data;

  complex64 alpha = 1.0 + 0.0 * I;
  complex64 beta = 0.0 + 0.0 * I;

  int use_blas_direct = 0;
  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasNoTrans;
  int lda = 0, ldb = 0, ldc = 0;

  if (setup_blas_row_major_params(m, k, a_rs, a_cs, &trans_a, &lda) &&
      setup_blas_row_major_params(k, n, b_rs, b_cs, &trans_b, &ldb) &&
      setup_blas_row_major_output(m, n, c_rs, c_cs, &ldc)) {
    use_blas_direct = 1;
  }

  if (use_blas_direct) {
    cblas_zgemm(CblasRowMajor, trans_a, trans_b, m, n, k, &alpha,
                a + a_off, lda, b + b_off, ldb, &beta, c + c_off, ldc);
  } else {
    /* Non-contiguous layout: pack matrices first */
    complex64 *a_packed = (complex64 *)malloc(m * k * sizeof(complex64));
    complex64 *b_packed = (complex64 *)malloc(k * n * sizeof(complex64));
    complex64 *c_packed = (complex64 *)malloc(m * n * sizeof(complex64));
    if (!a_packed || !b_packed || !c_packed) {
      free(a_packed);
      free(b_packed);
      free(c_packed);
      return;
    }

    /* Pack A and B */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < k; j++) {
        a_packed[i * k + j] = a[a_off + i * a_rs + j * a_cs];
      }
    }
    for (long i = 0; i < k; i++) {
      for (long j = 0; j < n; j++) {
        b_packed[i * n + j] = b[b_off + i * b_rs + j * b_cs];
      }
    }

    /* Compute using BLAS */
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                a_packed, k, b_packed, n, &beta, c_packed, n);

    /* Unpack C */
    for (long i = 0; i < m; i++) {
      for (long j = 0; j < n; j++) {
        c[c_off + i * c_rs + j * c_cs] = c_packed[i * n + j];
      }
    }

    free(a_packed);
    free(b_packed);
    free(c_packed);
  }
}

MATMUL_OP_IMPL(c32, sizeof(complex32))
MATMUL_OP_IMPL(c64, sizeof(complex64))

// Low-precision floats
LOW_PREC_MATMUL_KERNEL(f16, uint16_t, half_to_float, float_to_half)
LOW_PREC_MATMUL_IMPL(f16, uint16_t)
LOW_PREC_MATMUL_KERNEL(bf16, caml_ba_bfloat16, bfloat16_to_float,
                       float_to_bfloat16)
LOW_PREC_MATMUL_IMPL(bf16, caml_ba_bfloat16)
LOW_PREC_MATMUL_KERNEL(f8e4m3, caml_ba_fp8_e4m3, fp8_e4m3_to_float,
                       float_to_fp8_e4m3)
LOW_PREC_MATMUL_IMPL(f8e4m3, caml_ba_fp8_e4m3)
LOW_PREC_MATMUL_KERNEL(f8e5m2, caml_ba_fp8_e5m2, fp8_e5m2_to_float,
                       float_to_fp8_e5m2)
LOW_PREC_MATMUL_IMPL(f8e5m2, caml_ba_fp8_e5m2)

// Int4/Uint4
INT4_MATMUL_IMPL(1, i4)
INT4_MATMUL_IMPL(0, u4)

// Build dispatch table
#define BUILD_DISPATCH_TABLE(name)                                             \
  static const matmul_op_table name##_table = {.i8 = nx_c_##name##_i8,         \
                                               .u8 = nx_c_##name##_u8,         \
                                               .i16 = nx_c_##name##_i16,       \
                                               .u16 = nx_c_##name##_u16,       \
                                               .i32 = nx_c_##name##_i32,       \
                                               .i64 = nx_c_##name##_i64,       \
                                               .u32 = nx_c_##name##_u32,       \
                                               .u64 = nx_c_##name##_u64,       \
                                               .inat = nx_c_##name##_inat,     \
                                               .f16 = nx_c_##name##_f16,       \
                                               .f32 = nx_c_##name##_f32,       \
                                               .f64 = nx_c_##name##_f64,       \
                                               .c32 = nx_c_##name##_c32,       \
                                               .c64 = nx_c_##name##_c64,       \
                                               .bf16 = nx_c_##name##_bf16,     \
                                               .bool_ = nx_c_##name##_bool_,   \
                                               .i4 = nx_c_##name##_i4,         \
                                               .u4 = nx_c_##name##_u4,         \
                                               .f8e4m3 = nx_c_##name##_f8e4m3, \
                                               .f8e5m2 = nx_c_##name##_f8e5m2}

BUILD_DISPATCH_TABLE(matmul);

// Generic dispatch function for matmul operations
static void dispatch_matmul_op(value v_a, value v_b, value v_c,
                               const matmul_op_table *table,
                               const char *op_name) {
  // Extract ndarrays using stack-allocated buffers (no malloc)
  int sa[MAX_NDIM], stra[MAX_NDIM];
  int sb[MAX_NDIM], strb[MAX_NDIM];
  int sc[MAX_NDIM], strc[MAX_NDIM];
  ndarray_t A = extract_ndarray_stack(v_a, sa, stra);
  ndarray_t B = extract_ndarray_stack(v_b, sb, strb);
  ndarray_t C = extract_ndarray_stack(v_c, sc, strc);

  // Get bigarray kind from the data field
  struct caml_ba_array *ba = Caml_ba_array_val(Field(v_a, FFI_TENSOR_DATA));
  int kind = nx_ba_get_kind(ba);

  // Check kinds match for b and c
  int kind_b = nx_ba_get_kind(Caml_ba_array_val(Field(v_b, FFI_TENSOR_DATA)));
  int kind_c = nx_ba_get_kind(Caml_ba_array_val(Field(v_c, FFI_TENSOR_DATA)));
  if (kind != kind_b || kind != kind_c) {
    caml_failwith("dtype mismatch");
  }

  // Select operation based on dtype
  matmul_op_t op = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      op = table->i8;
      break;
    case CAML_BA_UINT8:
      op = table->u8;
      break;
    case CAML_BA_SINT16:
      op = table->i16;
      break;
    case CAML_BA_UINT16:
      op = table->u16;
      break;
    case CAML_BA_INT32:
      op = table->i32;
      break;
    case CAML_BA_INT64:
      op = table->i64;
      break;
    case NX_BA_UINT32:
      op = table->u32;
      break;
    case NX_BA_UINT64:
      op = table->u64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      op = table->inat;
      break;
    case CAML_BA_FLOAT16:
      op = table->f16;
      break;
    case CAML_BA_FLOAT32:
      op = table->f32;
      break;
    case CAML_BA_FLOAT64:
      op = table->f64;
      break;
    case CAML_BA_COMPLEX32:
      op = table->c32;
      break;
    case CAML_BA_COMPLEX64:
      op = table->c64;
      break;
    case NX_BA_BFLOAT16:
      op = table->bf16;
      break;
    case NX_BA_BOOL:
      op = table->bool_;
      break;
    case NX_BA_INT4:
      op = table->i4;
      break;
    case NX_BA_UINT4:
      op = table->u4;
      break;
    case NX_BA_FP8_E4M3:
      op = table->f8e4m3;
      break;
    case NX_BA_FP8_E5M2:
      op = table->f8e5m2;
      break;
    default:
      caml_failwith("dispatch_matmul_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    caml_failwith(msg);
  }

  // Perform the operation (no cleanup needed — stack-allocated)
  op(&A, &B, &C);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_matmul(value v_a, value v_b, value v_c) {
  CAMLparam3(v_a, v_b, v_c);
  dispatch_matmul_op(v_a, v_b, v_c, &matmul_table, "matmul");
  CAMLreturn(Val_unit);
}

