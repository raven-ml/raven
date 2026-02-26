/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// Cast operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Type definitions for cast operations
typedef void (*cast_op_t)(const ndarray_t *, ndarray_t *);

// Enum for dtypes to index the table
typedef enum {
  NX_DTYPE_I8 = 0,
  NX_DTYPE_U8,
  NX_DTYPE_I16,
  NX_DTYPE_U16,
  NX_DTYPE_I32,
  NX_DTYPE_I64,
  NX_DTYPE_U32,
  NX_DTYPE_U64,
  NX_DTYPE_INAT,
  NX_DTYPE_F16,
  NX_DTYPE_F32,
  NX_DTYPE_F64,
  NX_DTYPE_C32,
  NX_DTYPE_C64,
  NX_DTYPE_BF16,
  NX_DTYPE_BOOL,
  NX_DTYPE_I4,
  NX_DTYPE_U4,
  NX_DTYPE_F8E4M3,
  NX_DTYPE_F8E5M2,
  NX_NUM_DTYPES
} nx_dtype;

// Map caml_ba_kind to nx_dtype
static nx_dtype kind_to_dtype(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
      return NX_DTYPE_I8;
    case CAML_BA_UINT8:
      return NX_DTYPE_U8;
    case CAML_BA_SINT16:
      return NX_DTYPE_I16;
    case CAML_BA_UINT16:
      return NX_DTYPE_U16;
    case CAML_BA_INT32:
      return NX_DTYPE_I32;
    case CAML_BA_INT64:
      return NX_DTYPE_I64;
    case NX_BA_UINT32:
      return NX_DTYPE_U32;
    case NX_BA_UINT64:
      return NX_DTYPE_U64;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      return NX_DTYPE_INAT;
    case CAML_BA_FLOAT16:
      return NX_DTYPE_F16;
    case CAML_BA_FLOAT32:
      return NX_DTYPE_F32;
    case CAML_BA_FLOAT64:
      return NX_DTYPE_F64;
    case CAML_BA_COMPLEX32:
      return NX_DTYPE_C32;
    case CAML_BA_COMPLEX64:
      return NX_DTYPE_C64;
    case NX_BA_BFLOAT16:
      return NX_DTYPE_BF16;
    case NX_BA_BOOL:
      return NX_DTYPE_BOOL;
    case NX_BA_INT4:
      return NX_DTYPE_I4;
    case NX_BA_UINT4:
      return NX_DTYPE_U4;
    case NX_BA_FP8_E4M3:
      return NX_DTYPE_F8E4M3;
    case NX_BA_FP8_E5M2:
      return NX_DTYPE_F8E5M2;
    default:
      return NX_NUM_DTYPES;
  }
}

// Helper to iterate over inner dimensions for unary (cast) operations
typedef void (*kernel_fn)(void *, void *, long, long);

static inline void iterate_inner_dims2(const ndarray_t *x, const ndarray_t *z,
                                       long outer_idx, kernel_fn kernel,
                                       void *x_data, void *z_data) {
  if (x->ndim <= 1) {
    kernel(x_data, z_data, outer_idx * x->strides[0],
           outer_idx * z->strides[0]);
    return;
  }

  long x_base = outer_idx * x->strides[0];
  long z_base = outer_idx * z->strides[0];

  int inner_ndim = x->ndim - 1;
  int *coords = (int *)calloc(inner_ndim, sizeof(int));
  if (!coords) {
    caml_failwith("iterate_inner_dims2: allocation failed");
  }

  bool done = false;
  while (!done) {
    long x_off = x_base;
    long z_off = z_base;

    for (int i = 0; i < inner_ndim; i++) {
      x_off += coords[i] * x->strides[i + 1];
      z_off += coords[i] * z->strides[i + 1];
    }

    kernel(x_data, z_data, x_off, z_off);

    done = true;
    for (int i = inner_ndim - 1; i >= 0; i--) {
      coords[i]++;
      if (coords[i] < x->shape[i + 1]) {
        done = false;
        break;
      }
      coords[i] = 0;
    }
  }

  free(coords);
}

// Generic cast implementation macro
#define CAST_IMPL(src_suffix, dst_suffix)                                      \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix(const ndarray_t *src,   \
                                                       ndarray_t *dst) {       \
    if (!src || !dst) {                                                        \
      caml_failwith("nx_c_cast_" #src_suffix "_to_" #dst_suffix                \
                    ": null pointer");                                         \
    }                                                                          \
    long total = total_elements_safe(src);                                     \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_contiguous(src) && is_contiguous(dst)) {                            \
      _Pragma("omp parallel for simd if(total > 1000)") for (long i = 0;       \
                                                             i < total; i++) { \
        nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(                     \
            src->data, dst->data, src->offset + i, dst->offset + i);           \
      }                                                                        \
    } else if (src->shape[0] > 1 && total / src->shape[0] > 50) {              \
      _Pragma("omp parallel for if(src->shape[0] > 4)") for (long i = 0;       \
                                                             i <               \
                                                             src->shape[0];    \
                                                             i++) {            \
        iterate_inner_dims2(src, dst, i,                                       \
                            nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel, \
                            src->data, dst->data);                             \
      }                                                                        \
    } else {                                                                   \
      nd_copy_iterator_t it;                                                   \
      nd_copy_iterator_init(&it, src, dst);                                    \
      do {                                                                     \
        long src_off, dst_off;                                                 \
        nd_copy_iterator_get_offsets(&it, &src_off, &dst_off);                 \
        nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(                     \
            src->data, dst->data, src->offset + src_off,                       \
            dst->offset + dst_off);                                            \
      } while (nd_copy_iterator_next(&it));                                    \
      nd_copy_iterator_destroy(&it);                                           \
    }                                                                          \
  }

// Note: Assumes is_fully_contiguous accepts 3 arguments, with NULL for y.
// Assume shared.h has overload or ignore NULL.

// Standard real types (integer and float, excluding bool, low prec, complex,
// packed)
#define STANDARD_REAL_TYPES \
  SR_TYPE(i8, int8_t)       \
  SR_TYPE(u8, uint8_t)      \
  SR_TYPE(i16, int16_t)     \
  SR_TYPE(u16, uint16_t)    \
  SR_TYPE(i32, int32_t)     \
  SR_TYPE(i64, int64_t)     \
  SR_TYPE(u32, uint32_t)    \
  SR_TYPE(u64, uint64_t)    \
  SR_TYPE(inat, intnat)     \
  SR_TYPE(f32, float)       \
  SR_TYPE(f64, double)

// Low precision float types
#define LOW_PREC_TYPES                                                    \
  LP_TYPE(f16, uint16_t, half_to_float, float_to_half)                    \
  LP_TYPE(bf16, caml_ba_bfloat16, bfloat16_to_float, float_to_bfloat16)   \
  LP_TYPE(f8e4m3, caml_ba_fp8_e4m3, fp8_e4m3_to_float, float_to_fp8_e4m3) \
  LP_TYPE(f8e5m2, caml_ba_fp8_e5m2, fp8_e5m2_to_float, float_to_fp8_e5m2)

// Complex types
#define COMPLEX_TYPES                                                        \
  CP_TYPE(c32, complex32, crealf(src[src_off]), cimagf(src[src_off]), float) \
  CP_TYPE(c64, complex64, creal(src[src_off]), cimag(src[src_off]), double)

// Generate cast for standard real to standard real

#define GEN_CAST_STANDARD_TO_STANDARD(src_suffix, dst_suffix, src_t, dst_t) \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(            \
      void *src_data, void *dst_data, long src_off, long dst_off) {         \
    src_t *src = (src_t *)src_data;                                         \
    dst_t *dst = (dst_t *)dst_data;                                         \
    dst[dst_off] = (dst_t)src[src_off];                                     \
  }                                                                         \
  CAST_IMPL(src_suffix, dst_suffix)

// Generate all standard-to-standard cast combinations
#define GEN_ALL_STANDARD_CASTS(src_suffix, src_t)                 \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, i8, src_t, int8_t)    \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, u8, src_t, uint8_t)   \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, i16, src_t, int16_t)  \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, u16, src_t, uint16_t) \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, i32, src_t, int32_t)  \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, i64, src_t, int64_t)  \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, u32, src_t, uint32_t) \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, u64, src_t, uint64_t) \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, inat, src_t, intnat)  \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, f32, src_t, float)    \
  GEN_CAST_STANDARD_TO_STANDARD(src_suffix, f64, src_t, double)

// Generate casts for each standard type
GEN_ALL_STANDARD_CASTS(i8, int8_t)
GEN_ALL_STANDARD_CASTS(u8, uint8_t)
GEN_ALL_STANDARD_CASTS(i16, int16_t)
GEN_ALL_STANDARD_CASTS(u16, uint16_t)
GEN_ALL_STANDARD_CASTS(i32, int32_t)
GEN_ALL_STANDARD_CASTS(i64, int64_t)
GEN_ALL_STANDARD_CASTS(u32, uint32_t)
GEN_ALL_STANDARD_CASTS(u64, uint64_t)
GEN_ALL_STANDARD_CASTS(inat, intnat)
GEN_ALL_STANDARD_CASTS(f32, float)
GEN_ALL_STANDARD_CASTS(f64, double)

// Generate cast for bool to standard real

#define GEN_CAST_BOOL_TO_STANDARD(dst_suffix, dst_t)                \
  static void nx_c_cast_bool_to_##dst_suffix##_kernel(              \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    uint8_t *src = (uint8_t *)src_data;                             \
    dst_t *dst = (dst_t *)dst_data;                                 \
    dst[dst_off] = (dst_t)src[src_off];                             \
  }                                                                 \
  CAST_IMPL(bool, dst_suffix)

#define SR_TYPE(dst_suffix, dst_t) GEN_CAST_BOOL_TO_STANDARD(dst_suffix, dst_t)
STANDARD_REAL_TYPES
#undef SR_TYPE

// Generate cast for standard real to bool

#define GEN_CAST_STANDARD_TO_BOOL(src_suffix, src_t)                \
  static void nx_c_cast_##src_suffix##_to_bool_kernel(              \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    src_t *src = (src_t *)src_data;                                 \
    uint8_t *dst = (uint8_t *)dst_data;                             \
    dst[dst_off] = (src[src_off] != 0) ? 1 : 0;                     \
  }                                                                 \
  CAST_IMPL(src_suffix, bool)

#define SR_TYPE(src_suffix, src_t) GEN_CAST_STANDARD_TO_BOOL(src_suffix, src_t)
STANDARD_REAL_TYPES
#undef SR_TYPE

// Bool to bool

static void nx_c_cast_bool_to_bool_kernel(void *src_data, void *dst_data,
                                          long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  dst[dst_off] = (src[src_off] != 0) ? 1 : 0;
}
CAST_IMPL(bool, bool)

// Generate cast for low prec to standard real

#define GEN_CAST_LP_TO_STANDARD(src_suffix, dst_suffix, src_t, dst_t, \
                                TO_FLOAT)                             \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(      \
      void *src_data, void *dst_data, long src_off, long dst_off) {   \
    src_t *src = (src_t *)src_data;                                   \
    dst_t *dst = (dst_t *)dst_data;                                   \
    float temp = TO_FLOAT(src[src_off]);                              \
    dst[dst_off] = (dst_t)temp;                                       \
  }                                                                   \
  CAST_IMPL(src_suffix, dst_suffix)

// Generate all low-precision to standard casts
#define GEN_ALL_LP_TO_STANDARD_CASTS(src_suffix, src_t, TO_FLOAT)     \
  GEN_CAST_LP_TO_STANDARD(src_suffix, i8, src_t, int8_t, TO_FLOAT)    \
  GEN_CAST_LP_TO_STANDARD(src_suffix, u8, src_t, uint8_t, TO_FLOAT)   \
  GEN_CAST_LP_TO_STANDARD(src_suffix, i16, src_t, int16_t, TO_FLOAT)  \
  GEN_CAST_LP_TO_STANDARD(src_suffix, u16, src_t, uint16_t, TO_FLOAT) \
  GEN_CAST_LP_TO_STANDARD(src_suffix, i32, src_t, int32_t, TO_FLOAT)  \
  GEN_CAST_LP_TO_STANDARD(src_suffix, i64, src_t, int64_t, TO_FLOAT)  \
  GEN_CAST_LP_TO_STANDARD(src_suffix, u32, src_t, uint32_t, TO_FLOAT) \
  GEN_CAST_LP_TO_STANDARD(src_suffix, u64, src_t, uint64_t, TO_FLOAT) \
  GEN_CAST_LP_TO_STANDARD(src_suffix, inat, src_t, intnat, TO_FLOAT)  \
  GEN_CAST_LP_TO_STANDARD(src_suffix, f32, src_t, float, TO_FLOAT)    \
  GEN_CAST_LP_TO_STANDARD(src_suffix, f64, src_t, double, TO_FLOAT)

// Generate casts for each low-precision type
GEN_ALL_LP_TO_STANDARD_CASTS(f16, uint16_t, half_to_float)
GEN_ALL_LP_TO_STANDARD_CASTS(bf16, caml_ba_bfloat16, bfloat16_to_float)
GEN_ALL_LP_TO_STANDARD_CASTS(f8e4m3, caml_ba_fp8_e4m3, fp8_e4m3_to_float)
GEN_ALL_LP_TO_STANDARD_CASTS(f8e5m2, caml_ba_fp8_e5m2, fp8_e5m2_to_float)

// Generate cast for standard real to low prec

#define GEN_CAST_STANDARD_TO_LP(src_suffix, dst_suffix, src_t, dst_t, \
                                FROM_FLOAT)                           \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(      \
      void *src_data, void *dst_data, long src_off, long dst_off) {   \
    src_t *src = (src_t *)src_data;                                   \
    dst_t *dst = (dst_t *)dst_data;                                   \
    float temp = (float)src[src_off];                                 \
    dst[dst_off] = FROM_FLOAT(temp);                                  \
  }                                                                   \
  CAST_IMPL(src_suffix, dst_suffix)

// Generate all standard to low-precision casts
#define GEN_ALL_STANDARD_TO_LP_CASTS(dst_suffix, dst_t, FROM_FLOAT)     \
  GEN_CAST_STANDARD_TO_LP(i8, dst_suffix, int8_t, dst_t, FROM_FLOAT)    \
  GEN_CAST_STANDARD_TO_LP(u8, dst_suffix, uint8_t, dst_t, FROM_FLOAT)   \
  GEN_CAST_STANDARD_TO_LP(i16, dst_suffix, int16_t, dst_t, FROM_FLOAT)  \
  GEN_CAST_STANDARD_TO_LP(u16, dst_suffix, uint16_t, dst_t, FROM_FLOAT) \
  GEN_CAST_STANDARD_TO_LP(i32, dst_suffix, int32_t, dst_t, FROM_FLOAT)  \
  GEN_CAST_STANDARD_TO_LP(i64, dst_suffix, int64_t, dst_t, FROM_FLOAT)  \
  GEN_CAST_STANDARD_TO_LP(u32, dst_suffix, uint32_t, dst_t, FROM_FLOAT) \
  GEN_CAST_STANDARD_TO_LP(u64, dst_suffix, uint64_t, dst_t, FROM_FLOAT) \
  GEN_CAST_STANDARD_TO_LP(inat, dst_suffix, intnat, dst_t, FROM_FLOAT)  \
  GEN_CAST_STANDARD_TO_LP(f32, dst_suffix, float, dst_t, FROM_FLOAT)    \
  GEN_CAST_STANDARD_TO_LP(f64, dst_suffix, double, dst_t, FROM_FLOAT)

// Generate casts for each low-precision type
GEN_ALL_STANDARD_TO_LP_CASTS(f16, uint16_t, float_to_half)
GEN_ALL_STANDARD_TO_LP_CASTS(bf16, caml_ba_bfloat16, float_to_bfloat16)
GEN_ALL_STANDARD_TO_LP_CASTS(f8e4m3, caml_ba_fp8_e4m3, float_to_fp8_e4m3)
GEN_ALL_STANDARD_TO_LP_CASTS(f8e5m2, caml_ba_fp8_e5m2, float_to_fp8_e5m2)

// Bool to low prec
#define GEN_CAST_BOOL_TO_LP(dst_suffix, dst_t, FROM_FLOAT)            \
  static void nx_c_cast_bool_to_##dst_suffix##_kernel(                \
      void *src_data, void *dst_data, long src_off, long dst_off) {   \
    uint8_t *src = (uint8_t *)src_data;                               \
    dst_t *dst = (dst_t *)dst_data;                                   \
    float temp = (src[src_off] != 0) ? 1.0f : 0.0f;                   \
    dst[dst_off] = FROM_FLOAT(temp);                                  \
  }                                                                   \
  CAST_IMPL(bool, dst_suffix)

GEN_CAST_BOOL_TO_LP(f16, uint16_t, float_to_half)
GEN_CAST_BOOL_TO_LP(bf16, caml_ba_bfloat16, float_to_bfloat16)
GEN_CAST_BOOL_TO_LP(f8e4m3, caml_ba_fp8_e4m3, float_to_fp8_e4m3)
GEN_CAST_BOOL_TO_LP(f8e5m2, caml_ba_fp8_e5m2, float_to_fp8_e5m2)

// Generate cast for low prec to bool

#define GEN_CAST_LP_TO_BOOL(src_suffix, src_t, TO_FLOAT)            \
  static void nx_c_cast_##src_suffix##_to_bool_kernel(              \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    src_t *src = (src_t *)src_data;                                 \
    uint8_t *dst = (uint8_t *)dst_data;                             \
    float temp = TO_FLOAT(src[src_off]);                            \
    dst[dst_off] = (temp != 0.0f) ? 1 : 0;                          \
  }                                                                 \
  CAST_IMPL(src_suffix, bool)

#define LP_TYPE(suffix, t, to_f, from_f) GEN_CAST_LP_TO_BOOL(suffix, t, to_f)
LOW_PREC_TYPES
#undef LP_TYPE

// Generate cast for low prec to low prec

#define GEN_CAST_LP_TO_LP(src_suffix, dst_suffix, src_t, dst_t, TO_FLOAT, \
                          FROM_FLOAT)                                     \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(          \
      void *src_data, void *dst_data, long src_off, long dst_off) {       \
    src_t *src = (src_t *)src_data;                                       \
    dst_t *dst = (dst_t *)dst_data;                                       \
    float temp = TO_FLOAT(src[src_off]);                                  \
    dst[dst_off] = FROM_FLOAT(temp);                                      \
  }                                                                       \
  CAST_IMPL(src_suffix, dst_suffix)

// Generate all low-precision to low-precision casts
// Identity casts for low-precision types
static void nx_c_cast_f16_to_f16_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  uint16_t *src = (uint16_t *)src_data;
  uint16_t *dst = (uint16_t *)dst_data;
  dst[dst_off] = src[src_off];
}
CAST_IMPL(f16, f16)

GEN_CAST_LP_TO_LP(f16, bf16, uint16_t, caml_ba_bfloat16, half_to_float,
                  float_to_bfloat16)
GEN_CAST_LP_TO_LP(f16, f8e4m3, uint16_t, caml_ba_fp8_e4m3, half_to_float,
                  float_to_fp8_e4m3)
GEN_CAST_LP_TO_LP(f16, f8e5m2, uint16_t, caml_ba_fp8_e5m2, half_to_float,
                  float_to_fp8_e5m2)
GEN_CAST_LP_TO_LP(bf16, f16, caml_ba_bfloat16, uint16_t, bfloat16_to_float,
                  float_to_half)
GEN_CAST_LP_TO_LP(bf16, f8e4m3, caml_ba_bfloat16, caml_ba_fp8_e4m3,
                  bfloat16_to_float, float_to_fp8_e4m3)
GEN_CAST_LP_TO_LP(bf16, f8e5m2, caml_ba_bfloat16, caml_ba_fp8_e5m2,
                  bfloat16_to_float, float_to_fp8_e5m2)
GEN_CAST_LP_TO_LP(f8e4m3, f16, caml_ba_fp8_e4m3, uint16_t, fp8_e4m3_to_float,
                  float_to_half)
GEN_CAST_LP_TO_LP(f8e4m3, bf16, caml_ba_fp8_e4m3, caml_ba_bfloat16,
                  fp8_e4m3_to_float, float_to_bfloat16)
GEN_CAST_LP_TO_LP(f8e4m3, f8e5m2, caml_ba_fp8_e4m3, caml_ba_fp8_e5m2,
                  fp8_e4m3_to_float, float_to_fp8_e5m2)
GEN_CAST_LP_TO_LP(f8e5m2, f16, caml_ba_fp8_e5m2, uint16_t, fp8_e5m2_to_float,
                  float_to_half)
GEN_CAST_LP_TO_LP(f8e5m2, bf16, caml_ba_fp8_e5m2, caml_ba_bfloat16,
                  fp8_e5m2_to_float, float_to_bfloat16)
GEN_CAST_LP_TO_LP(f8e5m2, f8e4m3, caml_ba_fp8_e5m2, caml_ba_fp8_e4m3,
                  fp8_e5m2_to_float, float_to_fp8_e4m3)

// Generate cast for complex to standard real

#define GEN_CAST_CP_TO_STANDARD(src_suffix, dst_suffix, src_t, dst_t, RE_FN) \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(             \
      void *src_data, void *dst_data, long src_off, long dst_off) {          \
    src_t *src = (src_t *)src_data;                                          \
    dst_t *dst = (dst_t *)dst_data;                                          \
    double temp = RE_FN;                                                     \
    dst[dst_off] = (dst_t)temp;                                              \
  }                                                                          \
  CAST_IMPL(src_suffix, dst_suffix)

// Generate all complex to standard casts
#define GEN_ALL_CP_TO_STANDARD_CASTS(src_suffix, src_t, RE_FN)     \
  GEN_CAST_CP_TO_STANDARD(src_suffix, i8, src_t, int8_t, RE_FN)    \
  GEN_CAST_CP_TO_STANDARD(src_suffix, u8, src_t, uint8_t, RE_FN)   \
  GEN_CAST_CP_TO_STANDARD(src_suffix, i16, src_t, int16_t, RE_FN)  \
  GEN_CAST_CP_TO_STANDARD(src_suffix, u16, src_t, uint16_t, RE_FN) \
  GEN_CAST_CP_TO_STANDARD(src_suffix, i32, src_t, int32_t, RE_FN)  \
  GEN_CAST_CP_TO_STANDARD(src_suffix, i64, src_t, int64_t, RE_FN)  \
  GEN_CAST_CP_TO_STANDARD(src_suffix, u32, src_t, uint32_t, RE_FN) \
  GEN_CAST_CP_TO_STANDARD(src_suffix, u64, src_t, uint64_t, RE_FN) \
  GEN_CAST_CP_TO_STANDARD(src_suffix, inat, src_t, intnat, RE_FN)  \
  GEN_CAST_CP_TO_STANDARD(src_suffix, f32, src_t, float, RE_FN)    \
  GEN_CAST_CP_TO_STANDARD(src_suffix, f64, src_t, double, RE_FN)

GEN_ALL_CP_TO_STANDARD_CASTS(c32, complex32, crealf(src[src_off]))
GEN_ALL_CP_TO_STANDARD_CASTS(c64, complex64, creal(src[src_off]))

// Generate cast for complex to bool

#define GEN_CAST_CP_TO_BOOL(src_suffix, src_t, RE_FN, IM_FN)        \
  static void nx_c_cast_##src_suffix##_to_bool_kernel(              \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    src_t *src = (src_t *)src_data;                                 \
    uint8_t *dst = (uint8_t *)dst_data;                             \
    double re = RE_FN;                                              \
    double im = IM_FN;                                              \
    dst[dst_off] = (re != 0 || im != 0) ? 1 : 0;                    \
  }                                                                 \
  CAST_IMPL(src_suffix, bool)

#define CP_TYPE(suffix, t, re_fn, im_fn, base_t) \
  GEN_CAST_CP_TO_BOOL(suffix, t, re_fn, im_fn)
COMPLEX_TYPES
#undef CP_TYPE

// Generate cast for complex to low prec float

#define GEN_CAST_CP_TO_LP(src_suffix, dst_suffix, src_t, dst_t, RE_FN, \
                          FROM_FLOAT)                                  \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(       \
      void *src_data, void *dst_data, long src_off, long dst_off) {    \
    src_t *src = (src_t *)src_data;                                    \
    dst_t *dst = (dst_t *)dst_data;                                    \
    float temp = RE_FN;                                                \
    dst[dst_off] = FROM_FLOAT(temp);                                   \
  }                                                                    \
  CAST_IMPL(src_suffix, dst_suffix)

// Generate all complex to low-precision casts
#define GEN_ALL_CP_TO_LP_CASTS(src_suffix, src_t, RE_FN)                    \
  GEN_CAST_CP_TO_LP(src_suffix, f16, src_t, uint16_t, RE_FN, float_to_half) \
  GEN_CAST_CP_TO_LP(src_suffix, bf16, src_t, caml_ba_bfloat16, RE_FN,       \
                    float_to_bfloat16)                                      \
  GEN_CAST_CP_TO_LP(src_suffix, f8e4m3, src_t, caml_ba_fp8_e4m3, RE_FN,     \
                    float_to_fp8_e4m3)                                      \
  GEN_CAST_CP_TO_LP(src_suffix, f8e5m2, src_t, caml_ba_fp8_e5m2, RE_FN,     \
                    float_to_fp8_e5m2)

GEN_ALL_CP_TO_LP_CASTS(c32, complex32, crealf(src[src_off]))
GEN_ALL_CP_TO_LP_CASTS(c64, complex64, creal(src[src_off]))

// Generate cast for standard real to c32/c64

#define GEN_CAST_STANDARD_TO_C32_C64(src_suffix, dst_suffix, src_t, dst_t, \
                                     BASE_T)                               \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(           \
      void *src_data, void *dst_data, long src_off, long dst_off) {        \
    src_t *src = (src_t *)src_data;                                        \
    dst_t *dst = (dst_t *)dst_data;                                        \
    BASE_T temp = (BASE_T)src[src_off];                                    \
    dst[dst_off] = temp + 0.0 * I;                                         \
  }                                                                        \
  CAST_IMPL(src_suffix, dst_suffix)


// Generate all standard to complex casts for c32/c64
#define GEN_ALL_STANDARD_TO_C32_C64_CASTS(dst_suffix, dst_t, BASE_T)     \
  GEN_CAST_STANDARD_TO_C32_C64(i8, dst_suffix, int8_t, dst_t, BASE_T)    \
  GEN_CAST_STANDARD_TO_C32_C64(u8, dst_suffix, uint8_t, dst_t, BASE_T)   \
  GEN_CAST_STANDARD_TO_C32_C64(i16, dst_suffix, int16_t, dst_t, BASE_T)  \
  GEN_CAST_STANDARD_TO_C32_C64(u16, dst_suffix, uint16_t, dst_t, BASE_T) \
  GEN_CAST_STANDARD_TO_C32_C64(i32, dst_suffix, int32_t, dst_t, BASE_T)  \
  GEN_CAST_STANDARD_TO_C32_C64(i64, dst_suffix, int64_t, dst_t, BASE_T)  \
  GEN_CAST_STANDARD_TO_C32_C64(u32, dst_suffix, uint32_t, dst_t, BASE_T) \
  GEN_CAST_STANDARD_TO_C32_C64(u64, dst_suffix, uint64_t, dst_t, BASE_T) \
  GEN_CAST_STANDARD_TO_C32_C64(inat, dst_suffix, intnat, dst_t, BASE_T)  \
  GEN_CAST_STANDARD_TO_C32_C64(f32, dst_suffix, float, dst_t, BASE_T)    \
  GEN_CAST_STANDARD_TO_C32_C64(f64, dst_suffix, double, dst_t, BASE_T)

GEN_ALL_STANDARD_TO_C32_C64_CASTS(c32, complex32, float)
GEN_ALL_STANDARD_TO_C32_C64_CASTS(c64, complex64, double)


// Removed - Already generated above

// Bool to c32/c64

#define GEN_CAST_BOOL_TO_CP(dst_suffix, dst_t, BASE_T)              \
  static void nx_c_cast_bool_to_##dst_suffix##_kernel(              \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    uint8_t *src = (uint8_t *)src_data;                             \
    dst_t *dst = (dst_t *)dst_data;                                 \
    BASE_T temp = (BASE_T)src[src_off];                             \
    dst[dst_off] = temp + 0.0 * I;                                  \
  }                                                                 \
  CAST_IMPL(bool, dst_suffix)

GEN_CAST_BOOL_TO_CP(c32, complex32, float)
GEN_CAST_BOOL_TO_CP(c64, complex64, double)

// Low prec to c32/c64

#define GEN_CAST_LP_TO_CP(src_suffix, src_t, dst_suffix, dst_t, BASE_T, \
                          TO_FLOAT)                                     \
  static void nx_c_cast_##src_suffix##_to_##dst_suffix##_kernel(        \
      void *src_data, void *dst_data, long src_off, long dst_off) {     \
    src_t *src = (src_t *)src_data;                                     \
    dst_t *dst = (dst_t *)dst_data;                                     \
    BASE_T temp = (BASE_T)TO_FLOAT(src[src_off]);                       \
    dst[dst_off] = temp + 0.0 * I;                                      \
  }                                                                     \
  CAST_IMPL(src_suffix, dst_suffix)

#define DEFINE_CASTS_LP_TO_CP(src_suffix, src_t, TO_FLOAT)              \
  GEN_CAST_LP_TO_CP(src_suffix, src_t, c32, complex32, float, TO_FLOAT) \
  GEN_CAST_LP_TO_CP(src_suffix, src_t, c64, complex64, double, TO_FLOAT)

#define LP_TYPE(suffix, t, to_f, from_f) DEFINE_CASTS_LP_TO_CP(suffix, t, to_f)
LOW_PREC_TYPES
#undef LP_TYPE

// Complex to complex pairs (individual)

static void nx_c_cast_c32_to_c32_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  complex32 *src = (complex32 *)src_data;
  complex32 *dst = (complex32 *)dst_data;
  dst[dst_off] = src[src_off];
}
CAST_IMPL(c32, c32)

static void nx_c_cast_c32_to_c64_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  complex32 *src = (complex32 *)src_data;
  complex64 *dst = (complex64 *)dst_data;
  dst[dst_off] = (complex64)src[src_off];
}
CAST_IMPL(c32, c64)


static void nx_c_cast_c64_to_c32_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  complex64 *src = (complex64 *)src_data;
  complex32 *dst = (complex32 *)dst_data;
  dst[dst_off] = (complex32)src[src_off];
}
CAST_IMPL(c64, c32)

static void nx_c_cast_c64_to_c64_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  complex64 *src = (complex64 *)src_data;
  complex64 *dst = (complex64 *)dst_data;
  dst[dst_off] = src[src_off];
}
CAST_IMPL(c64, c64)


// Generate cast for i4 to standard real

#define GEN_CAST_I4_TO_STANDARD(dst_suffix, dst_t)                  \
  static void nx_c_cast_i4_to_##dst_suffix##_kernel(                \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    uint8_t *src = (uint8_t *)src_data;                             \
    dst_t *dst = (dst_t *)dst_data;                                 \
    long byte_off = src_off / 2;                                    \
    int nib_off = src_off % 2;                                      \
    int a = nib_off ? ((int8_t)src[byte_off] >> 4)                  \
                    : (int8_t)((src[byte_off] & 0x0F) << 4) >> 4;   \
    dst[dst_off] = (dst_t)a;                                        \
  }                                                                 \
  CAST_IMPL(i4, dst_suffix)

#define SR_TYPE(dst_suffix, dst_t) GEN_CAST_I4_TO_STANDARD(dst_suffix, dst_t)
STANDARD_REAL_TYPES
#undef SR_TYPE

// Generate cast for u4 to standard real

#define GEN_CAST_U4_TO_STANDARD(dst_suffix, dst_t)                        \
  static void nx_c_cast_u4_to_##dst_suffix##_kernel(                      \
      void *src_data, void *dst_data, long src_off, long dst_off) {       \
    uint8_t *src = (uint8_t *)src_data;                                   \
    dst_t *dst = (dst_t *)dst_data;                                       \
    long byte_off = src_off / 2;                                          \
    int nib_off = src_off % 2;                                            \
    int a = nib_off ? (src[byte_off] >> 4) & 0x0F : src[byte_off] & 0x0F; \
    dst[dst_off] = (dst_t)a;                                              \
  }                                                                       \
  CAST_IMPL(u4, dst_suffix)

#define SR_TYPE(dst_suffix, dst_t) GEN_CAST_U4_TO_STANDARD(dst_suffix, dst_t)
STANDARD_REAL_TYPES
#undef SR_TYPE

// i4 to bool

static void nx_c_cast_i4_to_bool_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  long byte_off = src_off / 2;
  int nib_off = src_off % 2;
  int a = nib_off ? ((int8_t)src[byte_off] >> 4)
                  : (int8_t)((src[byte_off] & 0x0F) << 4) >> 4;
  dst[dst_off] = (a != 0) ? 1 : 0;
}
CAST_IMPL(i4, bool)

// u4 to bool

static void nx_c_cast_u4_to_bool_kernel(void *src_data, void *dst_data,
                                        long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  long byte_off = src_off / 2;
  int nib_off = src_off % 2;
  int a = nib_off ? (src[byte_off] >> 4) & 0x0F : src[byte_off] & 0x0F;
  dst[dst_off] = (a != 0) ? 1 : 0;
}
CAST_IMPL(u4, bool)

// Generate cast for i4 to low prec

#define GEN_CAST_I4_TO_LP(dst_suffix, dst_t, FROM_FLOAT)            \
  static void nx_c_cast_i4_to_##dst_suffix##_kernel(                \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    uint8_t *src = (uint8_t *)src_data;                             \
    dst_t *dst = (dst_t *)dst_data;                                 \
    long byte_off = src_off / 2;                                    \
    int nib_off = src_off % 2;                                      \
    int a = nib_off ? ((int8_t)src[byte_off] >> 4)                  \
                    : (int8_t)((src[byte_off] & 0x0F) << 4) >> 4;   \
    float temp = (float)a;                                          \
    dst[dst_off] = FROM_FLOAT(temp);                                \
  }                                                                 \
  CAST_IMPL(i4, dst_suffix)

#define LP_TYPE(suffix, t, to_f, from_f) GEN_CAST_I4_TO_LP(suffix, t, from_f)
LOW_PREC_TYPES
#undef LP_TYPE

// Generate cast for u4 to low prec

#define GEN_CAST_U4_TO_LP(dst_suffix, dst_t, FROM_FLOAT)                  \
  static void nx_c_cast_u4_to_##dst_suffix##_kernel(                      \
      void *src_data, void *dst_data, long src_off, long dst_off) {       \
    uint8_t *src = (uint8_t *)src_data;                                   \
    dst_t *dst = (dst_t *)dst_data;                                       \
    long byte_off = src_off / 2;                                          \
    int nib_off = src_off % 2;                                            \
    int a = nib_off ? (src[byte_off] >> 4) & 0x0F : src[byte_off] & 0x0F; \
    float temp = (float)a;                                                \
    dst[dst_off] = FROM_FLOAT(temp);                                      \
  }                                                                       \
  CAST_IMPL(u4, dst_suffix)

#define LP_TYPE(suffix, t, to_f, from_f) GEN_CAST_U4_TO_LP(suffix, t, from_f)
LOW_PREC_TYPES
#undef LP_TYPE

// Generate cast for i4 to c32/c64

#define GEN_CAST_I4_TO_CP(dst_suffix, dst_t, BASE_T)                \
  static void nx_c_cast_i4_to_##dst_suffix##_kernel(                \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    uint8_t *src = (uint8_t *)src_data;                             \
    dst_t *dst = (dst_t *)dst_data;                                 \
    long byte_off = src_off / 2;                                    \
    int nib_off = src_off % 2;                                      \
    int a = nib_off ? ((int8_t)src[byte_off] >> 4)                  \
                    : (int8_t)((src[byte_off] & 0x0F) << 4) >> 4;   \
    BASE_T temp = (BASE_T)a;                                        \
    dst[dst_off] = temp + 0.0 * I;                                  \
  }                                                                 \
  CAST_IMPL(i4, dst_suffix)

GEN_CAST_I4_TO_CP(c32, complex32, float)
GEN_CAST_I4_TO_CP(c64, complex64, double)

// Similar for u4 to c32/c64

#define GEN_CAST_U4_TO_CP(dst_suffix, dst_t, BASE_T)                      \
  static void nx_c_cast_u4_to_##dst_suffix##_kernel(                      \
      void *src_data, void *dst_data, long src_off, long dst_off) {       \
    uint8_t *src = (uint8_t *)src_data;                                   \
    dst_t *dst = (dst_t *)dst_data;                                       \
    long byte_off = src_off / 2;                                          \
    int nib_off = src_off % 2;                                            \
    int a = nib_off ? (src[byte_off] >> 4) & 0x0F : src[byte_off] & 0x0F; \
    BASE_T temp = (BASE_T)a;                                              \
    dst[dst_off] = temp + 0.0 * I;                                        \
  }                                                                       \
  CAST_IMPL(u4, dst_suffix)

GEN_CAST_U4_TO_CP(c32, complex32, float)
GEN_CAST_U4_TO_CP(c64, complex64, double)

// Generate cast for standard real to i4/u4

#define GEN_CAST_STANDARD_TO_PACKED(src_suffix, src_t, packed_suffix) \
  static void nx_c_cast_##src_suffix##_to_##packed_suffix##_kernel(   \
      void *src_data, void *dst_data, long src_off, long dst_off) {   \
    src_t *src = (src_t *)src_data;                                   \
    uint8_t *dst = (uint8_t *)dst_data;                               \
    long byte_off = dst_off / 2;                                      \
    int nib_off = dst_off % 2;                                        \
    int res = (int)src[src_off];                                      \
    uint8_t nib = (uint8_t)res & 0x0F;                                \
    if (nib_off) {                                                    \
      dst[byte_off] = (dst[byte_off] & 0x0F) | (nib << 4);            \
    } else {                                                          \
      dst[byte_off] = (dst[byte_off] & 0xF0) | nib;                   \
    }                                                                 \
  }                                                                   \
  CAST_IMPL(src_suffix, packed_suffix)

// Generate standard to packed casts
GEN_CAST_STANDARD_TO_PACKED(i8, int8_t, i4)
GEN_CAST_STANDARD_TO_PACKED(u8, uint8_t, i4)
GEN_CAST_STANDARD_TO_PACKED(i16, int16_t, i4)
GEN_CAST_STANDARD_TO_PACKED(u16, uint16_t, i4)
GEN_CAST_STANDARD_TO_PACKED(i32, int32_t, i4)
GEN_CAST_STANDARD_TO_PACKED(i64, int64_t, i4)
GEN_CAST_STANDARD_TO_PACKED(u32, uint32_t, i4)
GEN_CAST_STANDARD_TO_PACKED(u64, uint64_t, i4)
GEN_CAST_STANDARD_TO_PACKED(inat, intnat, i4)
GEN_CAST_STANDARD_TO_PACKED(f32, float, i4)
GEN_CAST_STANDARD_TO_PACKED(f64, double, i4)

GEN_CAST_STANDARD_TO_PACKED(i8, int8_t, u4)
GEN_CAST_STANDARD_TO_PACKED(u8, uint8_t, u4)
GEN_CAST_STANDARD_TO_PACKED(i16, int16_t, u4)
GEN_CAST_STANDARD_TO_PACKED(u16, uint16_t, u4)
GEN_CAST_STANDARD_TO_PACKED(i32, int32_t, u4)
GEN_CAST_STANDARD_TO_PACKED(i64, int64_t, u4)
GEN_CAST_STANDARD_TO_PACKED(u32, uint32_t, u4)
GEN_CAST_STANDARD_TO_PACKED(u64, uint64_t, u4)
GEN_CAST_STANDARD_TO_PACKED(inat, intnat, u4)
GEN_CAST_STANDARD_TO_PACKED(f32, float, u4)
GEN_CAST_STANDARD_TO_PACKED(f64, double, u4)

// Bool to i4/u4

#define GEN_CAST_BOOL_TO_PACKED(packed_suffix)                      \
  static void nx_c_cast_bool_to_##packed_suffix##_kernel(           \
      void *src_data, void *dst_data, long src_off, long dst_off) { \
    uint8_t *src = (uint8_t *)src_data;                             \
    uint8_t *dst = (uint8_t *)dst_data;                             \
    long byte_off = dst_off / 2;                                    \
    int nib_off = dst_off % 2;                                      \
    int res = (int)src[src_off];                                    \
    uint8_t nib = (uint8_t)res & 0x0F;                              \
    if (nib_off) {                                                  \
      dst[byte_off] = (dst[byte_off] & 0x0F) | (nib << 4);          \
    } else {                                                        \
      dst[byte_off] = (dst[byte_off] & 0xF0) | nib;                 \
    }                                                               \
  }                                                                 \
  CAST_IMPL(bool, packed_suffix)

GEN_CAST_BOOL_TO_PACKED(i4)
GEN_CAST_BOOL_TO_PACKED(u4)

// Low prec to i4/u4

#define GEN_CAST_LP_TO_PACKED(src_suffix, src_t, TO_FLOAT, packed_suffix) \
  static void nx_c_cast_##src_suffix##_to_##packed_suffix##_kernel(       \
      void *src_data, void *dst_data, long src_off, long dst_off) {       \
    src_t *src = (src_t *)src_data;                                       \
    uint8_t *dst = (uint8_t *)dst_data;                                   \
    long byte_off = dst_off / 2;                                          \
    int nib_off = dst_off % 2;                                            \
    float temp = TO_FLOAT(src[src_off]);                                  \
    int res = (int)temp;                                                  \
    uint8_t nib = (uint8_t)res & 0x0F;                                    \
    if (nib_off) {                                                        \
      dst[byte_off] = (dst[byte_off] & 0x0F) | (nib << 4);                \
    } else {                                                              \
      dst[byte_off] = (dst[byte_off] & 0xF0) | nib;                       \
    }                                                                     \
  }                                                                       \
  CAST_IMPL(src_suffix, packed_suffix)

// Generate low-precision to packed casts
GEN_CAST_LP_TO_PACKED(f16, uint16_t, half_to_float, i4)
GEN_CAST_LP_TO_PACKED(bf16, caml_ba_bfloat16, bfloat16_to_float, i4)
GEN_CAST_LP_TO_PACKED(f8e4m3, caml_ba_fp8_e4m3, fp8_e4m3_to_float, i4)
GEN_CAST_LP_TO_PACKED(f8e5m2, caml_ba_fp8_e5m2, fp8_e5m2_to_float, i4)

GEN_CAST_LP_TO_PACKED(f16, uint16_t, half_to_float, u4)
GEN_CAST_LP_TO_PACKED(bf16, caml_ba_bfloat16, bfloat16_to_float, u4)
GEN_CAST_LP_TO_PACKED(f8e4m3, caml_ba_fp8_e4m3, fp8_e4m3_to_float, u4)
GEN_CAST_LP_TO_PACKED(f8e5m2, caml_ba_fp8_e5m2, fp8_e5m2_to_float, u4)

// Complex to i4/u4

#define GEN_CAST_CP_TO_PACKED(src_suffix, src_t, RE_FN, packed_suffix) \
  static void nx_c_cast_##src_suffix##_to_##packed_suffix##_kernel(    \
      void *src_data, void *dst_data, long src_off, long dst_off) {    \
    src_t *src = (src_t *)src_data;                                    \
    uint8_t *dst = (uint8_t *)dst_data;                                \
    long byte_off = dst_off / 2;                                       \
    int nib_off = dst_off % 2;                                         \
    float temp = RE_FN;                                                \
    int res = (int)temp;                                               \
    uint8_t nib = (uint8_t)res & 0x0F;                                 \
    if (nib_off) {                                                     \
      dst[byte_off] = (dst[byte_off] & 0x0F) | (nib << 4);             \
    } else {                                                           \
      dst[byte_off] = (dst[byte_off] & 0xF0) | nib;                    \
    }                                                                  \
  }                                                                    \
  CAST_IMPL(src_suffix, packed_suffix)

// Generate complex to packed casts
GEN_CAST_CP_TO_PACKED(c32, complex32, crealf(src[src_off]), i4)
GEN_CAST_CP_TO_PACKED(c64, complex64, creal(src[src_off]), i4)

GEN_CAST_CP_TO_PACKED(c32, complex32, crealf(src[src_off]), u4)
GEN_CAST_CP_TO_PACKED(c64, complex64, creal(src[src_off]), u4)

// Identity casts for other low-precision and special types
static void nx_c_cast_bf16_to_bf16_kernel(void *src_data, void *dst_data,
                                          long src_off, long dst_off) {
  caml_ba_bfloat16 *src = (caml_ba_bfloat16 *)src_data;
  caml_ba_bfloat16 *dst = (caml_ba_bfloat16 *)dst_data;
  dst[dst_off] = src[src_off];
}
CAST_IMPL(bf16, bf16)

static void nx_c_cast_f8e4m3_to_f8e4m3_kernel(void *src_data, void *dst_data,
                                              long src_off, long dst_off) {
  caml_ba_fp8_e4m3 *src = (caml_ba_fp8_e4m3 *)src_data;
  caml_ba_fp8_e4m3 *dst = (caml_ba_fp8_e4m3 *)dst_data;
  dst[dst_off] = src[src_off];
}
CAST_IMPL(f8e4m3, f8e4m3)

static void nx_c_cast_f8e5m2_to_f8e5m2_kernel(void *src_data, void *dst_data,
                                              long src_off, long dst_off) {
  caml_ba_fp8_e5m2 *src = (caml_ba_fp8_e5m2 *)src_data;
  caml_ba_fp8_e5m2 *dst = (caml_ba_fp8_e5m2 *)dst_data;
  dst[dst_off] = src[src_off];
}
CAST_IMPL(f8e5m2, f8e5m2)

// Packed to packed

static void nx_c_cast_i4_to_i4_kernel(void *src_data, void *dst_data,
                                      long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  long byte_off = src_off / 2;
  int nib_off = src_off % 2;
  int a = nib_off ? ((int8_t)src[byte_off] >> 4)
                  : (int8_t)((src[byte_off] & 0x0F) << 4) >> 4;
  uint8_t nib = (uint8_t)a & 0x0F;
  long d_byte_off = dst_off / 2;
  int d_nib_off = dst_off % 2;
  if (d_nib_off) {
    dst[d_byte_off] = (dst[d_byte_off] & 0x0F) | (nib << 4);
  } else {
    dst[d_byte_off] = (dst[d_byte_off] & 0xF0) | nib;
  }
}
CAST_IMPL(i4, i4)

static void nx_c_cast_i4_to_u4_kernel(void *src_data, void *dst_data,
                                      long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  long byte_off = src_off / 2;
  int nib_off = src_off % 2;
  int a = nib_off ? ((int8_t)src[byte_off] >> 4)
                  : (int8_t)((src[byte_off] & 0x0F) << 4) >> 4;
  uint8_t nib = (uint8_t)a & 0x0F;
  long d_byte_off = dst_off / 2;
  int d_nib_off = dst_off % 2;
  if (d_nib_off) {
    dst[d_byte_off] = (dst[d_byte_off] & 0x0F) | (nib << 4);
  } else {
    dst[d_byte_off] = (dst[d_byte_off] & 0xF0) | nib;
  }
}
CAST_IMPL(i4, u4)

static void nx_c_cast_u4_to_i4_kernel(void *src_data, void *dst_data,
                                      long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  long byte_off = src_off / 2;
  int nib_off = src_off % 2;
  int a = nib_off ? (src[byte_off] >> 4) & 0x0F : src[byte_off] & 0x0F;
  uint8_t nib = (uint8_t)a & 0x0F;
  long d_byte_off = dst_off / 2;
  int d_nib_off = dst_off % 2;
  if (d_nib_off) {
    dst[d_byte_off] = (dst[d_byte_off] & 0x0F) | (nib << 4);
  } else {
    dst[d_byte_off] = (dst[d_byte_off] & 0xF0) | nib;
  }
}
CAST_IMPL(u4, i4)

static void nx_c_cast_u4_to_u4_kernel(void *src_data, void *dst_data,
                                      long src_off, long dst_off) {
  uint8_t *src = (uint8_t *)src_data;
  uint8_t *dst = (uint8_t *)dst_data;
  long byte_off = src_off / 2;
  int nib_off = src_off % 2;
  int a = nib_off ? (src[byte_off] >> 4) & 0x0F : src[byte_off] & 0x0F;
  uint8_t nib = (uint8_t)a & 0x0F;
  long d_byte_off = dst_off / 2;
  int d_nib_off = dst_off % 2;
  if (d_nib_off) {
    dst[d_byte_off] = (dst[d_byte_off] & 0x0F) | (nib << 4);
  } else {
    dst[d_byte_off] = (dst[d_byte_off] & 0xF0) | nib;
  }
}
CAST_IMPL(u4, u4)

// Dispatch table

static const cast_op_t cast_table[NX_NUM_DTYPES][NX_NUM_DTYPES] =
    {
        [NX_DTYPE_I8] =
            {
                [NX_DTYPE_I8] = nx_c_cast_i8_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_i8_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_i8_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_i8_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_i8_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_i8_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_i8_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_i8_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_i8_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_i8_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_i8_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_i8_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_i8_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_i8_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_i8_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_i8_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_i8_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_i8_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_i8_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_i8_to_f8e5m2,
            },
        [NX_DTYPE_U8] =
            {
                [NX_DTYPE_I8] = nx_c_cast_u8_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_u8_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_u8_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_u8_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_u8_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_u8_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_u8_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_u8_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_u8_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_u8_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_u8_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_u8_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_u8_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_u8_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_u8_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_u8_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_u8_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_u8_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_u8_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_u8_to_f8e5m2,
            },
        [NX_DTYPE_I16] =
            {
                [NX_DTYPE_I8] = nx_c_cast_i16_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_i16_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_i16_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_i16_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_i16_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_i16_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_i16_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_i16_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_i16_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_i16_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_i16_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_i16_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_i16_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_i16_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_i16_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_i16_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_i16_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_i16_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_i16_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_i16_to_f8e5m2,
            },
        [NX_DTYPE_U16] =
            {
                [NX_DTYPE_I8] = nx_c_cast_u16_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_u16_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_u16_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_u16_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_u16_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_u16_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_u16_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_u16_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_u16_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_u16_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_u16_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_u16_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_u16_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_u16_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_u16_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_u16_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_u16_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_u16_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_u16_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_u16_to_f8e5m2,
            },
        [NX_DTYPE_I32] =
            {
                [NX_DTYPE_I8] = nx_c_cast_i32_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_i32_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_i32_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_i32_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_i32_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_i32_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_i32_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_i32_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_i32_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_i32_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_i32_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_i32_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_i32_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_i32_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_i32_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_i32_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_i32_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_i32_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_i32_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_i32_to_f8e5m2,
            },
        [NX_DTYPE_I64] =
            {
                [NX_DTYPE_I8] = nx_c_cast_i64_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_i64_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_i64_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_i64_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_i64_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_i64_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_i64_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_i64_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_i64_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_i64_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_i64_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_i64_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_i64_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_i64_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_i64_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_i64_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_i64_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_i64_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_i64_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_i64_to_f8e5m2,
            },
        [NX_DTYPE_U32] =
            {
                [NX_DTYPE_I8] = nx_c_cast_u32_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_u32_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_u32_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_u32_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_u32_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_u32_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_u32_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_u32_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_u32_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_u32_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_u32_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_u32_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_u32_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_u32_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_u32_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_u32_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_u32_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_u32_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_u32_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_u32_to_f8e5m2,
            },
        [NX_DTYPE_U64] =
            {
                [NX_DTYPE_I8] = nx_c_cast_u64_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_u64_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_u64_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_u64_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_u64_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_u64_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_u64_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_u64_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_u64_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_u64_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_u64_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_u64_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_u64_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_u64_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_u64_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_u64_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_u64_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_u64_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_u64_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_u64_to_f8e5m2,
            },
        [NX_DTYPE_INAT] =
            {
                [NX_DTYPE_I8] = nx_c_cast_inat_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_inat_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_inat_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_inat_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_inat_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_inat_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_inat_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_inat_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_inat_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_inat_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_inat_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_inat_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_inat_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_inat_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_inat_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_inat_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_inat_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_inat_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_inat_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_inat_to_f8e5m2,
            },
        [NX_DTYPE_F16] =
            {
                [NX_DTYPE_I8] = nx_c_cast_f16_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_f16_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_f16_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_f16_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_f16_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_f16_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_f16_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_f16_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_f16_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_f16_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_f16_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_f16_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_f16_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_f16_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_f16_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_f16_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_f16_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_f16_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_f16_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_f16_to_f8e5m2,
            },
        [NX_DTYPE_F32] =
            {
                [NX_DTYPE_I8] = nx_c_cast_f32_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_f32_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_f32_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_f32_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_f32_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_f32_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_f32_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_f32_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_f32_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_f32_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_f32_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_f32_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_f32_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_f32_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_f32_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_f32_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_f32_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_f32_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_f32_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_f32_to_f8e5m2,
            },
        [NX_DTYPE_F64] =
            {
                [NX_DTYPE_I8] = nx_c_cast_f64_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_f64_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_f64_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_f64_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_f64_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_f64_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_f64_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_f64_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_f64_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_f64_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_f64_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_f64_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_f64_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_f64_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_f64_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_f64_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_f64_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_f64_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_f64_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_f64_to_f8e5m2,
            },
        [NX_DTYPE_C32] =
            {
                [NX_DTYPE_I8] = nx_c_cast_c32_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_c32_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_c32_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_c32_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_c32_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_c32_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_c32_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_c32_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_c32_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_c32_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_c32_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_c32_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_c32_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_c32_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_c32_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_c32_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_c32_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_c32_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_c32_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_c32_to_f8e5m2,
            },
        [NX_DTYPE_C64] =
            {
                [NX_DTYPE_I8] = nx_c_cast_c64_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_c64_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_c64_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_c64_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_c64_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_c64_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_c64_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_c64_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_c64_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_c64_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_c64_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_c64_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_c64_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_c64_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_c64_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_c64_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_c64_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_c64_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_c64_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_c64_to_f8e5m2,
            },
        [NX_DTYPE_BF16] =
            {
                [NX_DTYPE_I8] = nx_c_cast_bf16_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_bf16_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_bf16_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_bf16_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_bf16_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_bf16_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_bf16_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_bf16_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_bf16_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_bf16_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_bf16_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_bf16_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_bf16_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_bf16_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_bf16_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_bf16_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_bf16_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_bf16_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_bf16_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_bf16_to_f8e5m2,
            },
        [NX_DTYPE_BOOL] =
            {
                [NX_DTYPE_I8] = nx_c_cast_bool_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_bool_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_bool_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_bool_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_bool_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_bool_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_bool_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_bool_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_bool_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_bool_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_bool_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_bool_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_bool_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_bool_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_bool_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_bool_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_bool_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_bool_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_bool_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_bool_to_f8e5m2,
            },
        [NX_DTYPE_I4] =
            {
                [NX_DTYPE_I8] = nx_c_cast_i4_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_i4_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_i4_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_i4_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_i4_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_i4_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_i4_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_i4_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_i4_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_i4_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_i4_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_i4_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_i4_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_i4_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_i4_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_i4_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_i4_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_i4_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_i4_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_i4_to_f8e5m2,
            },
        [NX_DTYPE_U4] =
            {
                [NX_DTYPE_I8] = nx_c_cast_u4_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_u4_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_u4_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_u4_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_u4_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_u4_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_u4_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_u4_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_u4_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_u4_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_u4_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_u4_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_u4_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_u4_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_u4_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_u4_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_u4_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_u4_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_u4_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_u4_to_f8e5m2,
            },
        [NX_DTYPE_F8E4M3] =
            {
                [NX_DTYPE_I8] = nx_c_cast_f8e4m3_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_f8e4m3_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_f8e4m3_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_f8e4m3_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_f8e4m3_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_f8e4m3_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_f8e4m3_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_f8e4m3_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_f8e4m3_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_f8e4m3_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_f8e4m3_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_f8e4m3_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_f8e4m3_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_f8e4m3_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_f8e4m3_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_f8e4m3_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_f8e4m3_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_f8e4m3_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_f8e4m3_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_f8e4m3_to_f8e5m2,
            },
        [NX_DTYPE_F8E5M2] =
            {
                [NX_DTYPE_I8] = nx_c_cast_f8e5m2_to_i8,
                [NX_DTYPE_U8] = nx_c_cast_f8e5m2_to_u8,
                [NX_DTYPE_I16] = nx_c_cast_f8e5m2_to_i16,
                [NX_DTYPE_U16] = nx_c_cast_f8e5m2_to_u16,
                [NX_DTYPE_I32] = nx_c_cast_f8e5m2_to_i32,
                [NX_DTYPE_I64] = nx_c_cast_f8e5m2_to_i64,
                [NX_DTYPE_U32] = nx_c_cast_f8e5m2_to_u32,
                [NX_DTYPE_U64] = nx_c_cast_f8e5m2_to_u64,
                [NX_DTYPE_INAT] = nx_c_cast_f8e5m2_to_inat,
                [NX_DTYPE_F16] = nx_c_cast_f8e5m2_to_f16,
                [NX_DTYPE_F32] = nx_c_cast_f8e5m2_to_f32,
                [NX_DTYPE_F64] = nx_c_cast_f8e5m2_to_f64,
                [NX_DTYPE_C32] = nx_c_cast_f8e5m2_to_c32,
                [NX_DTYPE_C64] = nx_c_cast_f8e5m2_to_c64,
                [NX_DTYPE_BF16] = nx_c_cast_f8e5m2_to_bf16,
                [NX_DTYPE_BOOL] = nx_c_cast_f8e5m2_to_bool,
                [NX_DTYPE_I4] = nx_c_cast_f8e5m2_to_i4,
                [NX_DTYPE_U4] = nx_c_cast_f8e5m2_to_u4,
                [NX_DTYPE_F8E4M3] = nx_c_cast_f8e5m2_to_f8e4m3,
                [NX_DTYPE_F8E5M2] = nx_c_cast_f8e5m2_to_f8e5m2,
            }
};

// Dispatch function for cast operations
static void dispatch_cast(value v_src, value v_dst) {
  ndarray_t src = extract_ndarray(v_src);
  ndarray_t dst = extract_ndarray(v_dst);

  if (src.ndim != dst.ndim) {
    cleanup_ndarray(&src);
    cleanup_ndarray(&dst);
    caml_failwith("shape mismatch");
  }
  for (int i = 0; i < src.ndim; i++) {
    if (src.shape[i] != dst.shape[i]) {
      cleanup_ndarray(&src);
      cleanup_ndarray(&dst);
      caml_failwith("shape mismatch");
    }
  }

  value v_src_data = Field(v_src, FFI_TENSOR_DATA);
  value v_dst_data = Field(v_dst, FFI_TENSOR_DATA);

  int src_kind = nx_buffer_get_kind(Caml_ba_array_val(v_src_data));
  int dst_kind = nx_buffer_get_kind(Caml_ba_array_val(v_dst_data));

  nx_dtype src_dtype = kind_to_dtype(src_kind);
  nx_dtype dst_dtype = kind_to_dtype(dst_kind);

  if (src_dtype == NX_NUM_DTYPES || dst_dtype == NX_NUM_DTYPES) {
    cleanup_ndarray(&src);
    cleanup_ndarray(&dst);
    caml_failwith("unsupported dtype");
  }

  cast_op_t op = cast_table[src_dtype][dst_dtype];

  if (!op) {
    cleanup_ndarray(&src);
    cleanup_ndarray(&dst);
    caml_failwith("cast not supported for this dtype combination");
  }

  caml_enter_blocking_section();
  op(&src, &dst);
  caml_leave_blocking_section();

  cleanup_ndarray(&src);
  cleanup_ndarray(&dst);
}

// OCaml FFI Stub
CAMLprim value caml_nx_cast(value v_src, value v_dst) {
  CAMLparam2(v_src, v_dst);
  dispatch_cast(v_src, v_dst);
  CAMLreturn(Val_unit);
}
