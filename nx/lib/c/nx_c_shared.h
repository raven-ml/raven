// nx_c_shared.h

#ifndef NX_C_SHARED_H
#define NX_C_SHARED_H

#include <bigarray_ext_stubs.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Common complex types
typedef float _Complex c32_t;
typedef double _Complex c64_t;

// Common struct for passing array info
typedef struct {
  void *data;
  int ndim;
  const int *shape;
  const int *strides;
  int offset;
} ndarray_t;

// Define INTNAT_MIN based on word size
#if SIZEOF_PTR == 8
#define INTNAT_MIN INT64_MIN
#else
#define INTNAT_MIN INT32_MIN
#endif

// Math constant fallback
#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

// Native wrapper macros for bytecode/native code compatibility
#define NATIVE_WRAPPER_6(name)                                          \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4, \
                                value v5, value v6) {                   \
    value argv[6] = {v1, v2, v3, v4, v5, v6};                           \
    return caml_nx_##name##_bc(argv, 6);                                \
  }

#define NATIVE_WRAPPER_8(name)                                            \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4,   \
                                value v5, value v6, value v7, value v8) { \
    value argv[8] = {v1, v2, v3, v4, v5, v6, v7, v8};                     \
    return caml_nx_##name##_bc(argv, 8);                                  \
  }

#define NATIVE_WRAPPER_9(name)                                          \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4, \
                                value v5, value v6, value v7, value v8, \
                                value v9) {                             \
    value argv[9] = {v1, v2, v3, v4, v5, v6, v7, v8, v9};               \
    return caml_nx_##name##_bc(argv, 9);                                \
  }

#define NATIVE_WRAPPER_10(name)                                         \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4, \
                                value v5, value v6, value v7, value v8, \
                                value v9, value v10) {                  \
    value argv[10] = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10};         \
    return caml_nx_##name##_bc(argv, 10);                               \
  }

#define REDUCE_NATIVE_WRAPPER_10(name)                                         \
  CAMLprim value caml_nx_reduce_##name(value v1, value v2, value v3, value v4, \
                                       value v5, value v6, value v7, value v8, \
                                       value v9, value v10) {                  \
    value argv[10] = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10};                \
    return caml_nx_reduce_##name##_bc(argv, 10);                               \
  }

#define NATIVE_WRAPPER_11(name)                                         \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4, \
                                value v5, value v6, value v7, value v8, \
                                value v9, value v10, value v11) {       \
    value argv[11] = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11};    \
    return caml_nx_##name##_bc(argv, 11);                               \
  }

#define NATIVE_WRAPPER_12(name)                                              \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4,      \
                                value v5, value v6, value v7, value v8,      \
                                value v9, value v10, value v11, value v12) { \
    value argv[12] = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12};    \
    return caml_nx_##name##_bc(argv, 12);                                    \
  }

#define NATIVE_WRAPPER_13(name)                                               \
  CAMLprim value caml_nx_##name(                                              \
      value arg1, value arg2, value arg3, value arg4, value arg5, value arg6, \
      value arg7, value arg8, value arg9, value arg10, value arg11,           \
      value arg12, value arg13) {                                             \
    value argv[13] = {arg1, arg2, arg3,  arg4,  arg5,  arg6, arg7,            \
                      arg8, arg9, arg10, arg11, arg12, arg13};                \
    return caml_nx_##name##_bc(argv, 13);                                     \
  }

#define NATIVE_WRAPPER_14(name)                                            \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4,    \
                                value v5, value v6, value v7, value v8,    \
                                value v9, value v10, value v11, value v12, \
                                value v13, value v14) {                    \
    value argv[14] = {v1, v2, v3,  v4,  v5,  v6,  v7,                      \
                      v8, v9, v10, v11, v12, v13, v14};                    \
    return caml_nx_##name##_bc(argv, 14);                                  \
  }

#define NATIVE_WRAPPER_15(name)                                            \
  CAMLprim value caml_nx_##name(value v1, value v2, value v3, value v4,    \
                                value v5, value v6, value v7, value v8,    \
                                value v9, value v10, value v11, value v12, \
                                value v13, value v14, value v15) {         \
    value argv[15] = {v1, v2,  v3,  v4,  v5,  v6,  v7, v8,                 \
                      v9, v10, v11, v12, v13, v14, v15};                   \
    return caml_nx_##name##_bc(argv, 15);                                  \
  }

#define NATIVE_WRAPPER_17(name)                                             \
  CAMLprim value caml_nx_##name(                                            \
      value v1, value v2, value v3, value v4, value v5, value v6, value v7, \
      value v8, value v9, value v10, value v11, value v12, value v13,       \
      value v14, value v15, value v16, value v17) {                         \
    value argv[17] = {v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8, v9,            \
                      v10, v11, v12, v13, v14, v15, v16, v17};              \
    return caml_nx_##name##_bc(argv, 17);                                   \
  }

// N-dimensional iterator for efficient array traversal
typedef struct {
  int ndim;
  const int *shape;
  const int *x_strides;
  const int *y_strides;  // May be NULL for unary ops
  const int *z_strides;
  long x_offset;
  long y_offset;  // May be unused for unary ops
  long z_offset;
  long indices[16];    // Stack allocation for common cases (ndim <= 16)
  long *heap_indices;  // Heap allocation for larger ndim
} nd_iterator_t;

// Inline helper functions
static inline size_t get_element_size(int kind) {
  switch (kind) {
    case CAML_BA_FLOAT32:
      return sizeof(float);
    case CAML_BA_FLOAT64:
      return sizeof(double);
    case CAML_BA_SINT8:
      return sizeof(int8_t);
    case CAML_BA_UINT8:
      return sizeof(uint8_t);
    case CAML_BA_SINT16:
      return sizeof(int16_t);
    case CAML_BA_UINT16:
      return sizeof(uint16_t);
    case CAML_BA_INT32:
      return sizeof(int32_t);
    case CAML_BA_INT64:
      return sizeof(int64_t);
    case CAML_BA_CAML_INT:
      return sizeof(intnat);
    case CAML_BA_NATIVE_INT:
      return sizeof(intnat);
    case CAML_BA_COMPLEX32:
      return sizeof(c32_t);
    case CAML_BA_COMPLEX64:
      return sizeof(c64_t);
    case CAML_BA_FLOAT16:
      return sizeof(uint16_t);  // Float16
    case NX_BA_BFLOAT16:
      return sizeof(uint16_t);  // BFloat16
    case NX_BA_BOOL:
      return sizeof(uint8_t);  // Bool
    case NX_BA_INT4:
      return 1;  // Int4 (2 values packed per byte)
    case NX_BA_UINT4:
      return 1;  // UInt4 (2 values packed per byte)
    case NX_BA_FP8_E4M3:
      return sizeof(uint8_t);  // Float8_e4m3
    case NX_BA_FP8_E5M2:
      return sizeof(uint8_t);  // Float8_e5m2
    case NX_BA_COMPLEX16:
      return 4;  // Complex16 (2 x bfloat16)
    case NX_BA_QINT8:
      return sizeof(int8_t);  // QInt8
    case NX_BA_QUINT8:
      return sizeof(uint8_t);  // QUInt8
    default:
      return 0;  // Should not happen with supported types
  }
}

static inline long total_elements(const ndarray_t *arr) {
  if (arr->ndim == 0) return 1;
  long total = 1;
  for (int i = 0; i < arr->ndim; i++) total *= arr->shape[i];
  return total;
}

static inline int is_c_contiguous(const ndarray_t *arr) {
  long expected_stride = 1;
  for (int i = arr->ndim - 1; i >= 0; i--) {
    if (arr->shape[i] > 1 && arr->strides[i] != expected_stride) return 0;
    expected_stride *= arr->shape[i];
  }
  return 1;
}

// N-dimensional iterator initialization and management
static inline void nd_iterator_init(nd_iterator_t *it, const ndarray_t *x,
                                    const ndarray_t *y, const ndarray_t *z) {
  it->ndim = x->ndim;
  it->shape = x->shape;
  it->x_strides = x->strides;
  it->y_strides = y ? y->strides : NULL;
  it->z_strides = z->strides;
  it->x_offset = x->offset;
  it->y_offset = y ? y->offset : 0;
  it->z_offset = z->offset;

  // Initialize indices to 0
  if (it->ndim <= 16) {
    it->heap_indices = NULL;
    for (int i = 0; i < it->ndim; i++) it->indices[i] = 0;
  } else {
    it->heap_indices = calloc(it->ndim, sizeof(long));
    if (it->heap_indices == NULL) {
      caml_failwith("nd_iterator_init: failed to allocate memory for indices");
    }
  }
}

static inline void nd_iterator_destroy(nd_iterator_t *it) {
  if (it->heap_indices) {
    free(it->heap_indices);
    it->heap_indices = NULL;
  }
}

static inline int nd_iterator_next(nd_iterator_t *it) {
  long *indices = it->heap_indices ? it->heap_indices : it->indices;

  // Increment indices from innermost dimension
  for (int d = it->ndim - 1; d >= 0; d--) {
    indices[d]++;
    if (indices[d] < it->shape[d]) {
      return 1;  // More elements to process
    }
    // Reset this dimension and continue to next
    indices[d] = 0;
  }

  // All indices have wrapped around - we're done
  return 0;
}

static inline void nd_iterator_get_offsets(const nd_iterator_t *it, long *x_off,
                                           long *y_off, long *z_off) {
  const long *indices = it->heap_indices ? it->heap_indices : it->indices;

  *x_off = it->x_offset;
  if (y_off) *y_off = it->y_offset;
  *z_off = it->z_offset;

  for (int d = 0; d < it->ndim; d++) {
    *x_off += indices[d] * it->x_strides[d];
    if (y_off && it->y_strides) *y_off += indices[d] * it->y_strides[d];
    *z_off += indices[d] * it->z_strides[d];
  }
}

#endif  // NX_C_SHARED_H
