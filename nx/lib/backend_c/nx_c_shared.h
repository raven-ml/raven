#ifndef NX_C_SHARED_H
#define NX_C_SHARED_H

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bigarray_ext_stubs.h"  // For extended kinds, caml_ba_* typedefs, and conversions

#ifdef _OPENMP
#include <omp.h>
#endif

// Maximum number of dimensions supported
#define MAX_NDIM 32

// FFI tensor field indices (matches OCaml record)
// type ffi_tensor = {
//   data : buffer;       (* Field 0 *)
//   shape : int array;   (* Field 1 *)
//   strides : int array; (* Field 2 *)
//   offset : int;        (* Field 3 *)
// }
#define FFI_TENSOR_DATA 0
#define FFI_TENSOR_SHAPE 1
#define FFI_TENSOR_STRIDES 2
#define FFI_TENSOR_OFFSET 3

typedef float _Complex complex32;
typedef double _Complex complex64;

// Int4/uint4 clamping macros for saturation
#define CLAMP_I4(x) ((x) < -8 ? -8 : ((x) > 7 ? 7 : (x)))
#define CLAMP_U4(x) ((x) < 0 ? 0 : ((x) > 15 ? 15 : (x)))

static inline int int4_get(const uint8_t *data, long offset, bool is_signed) {
  long byte_off = offset / 2;
  int nibble_off = offset % 2;
  uint8_t byte = data[byte_off];
  if (is_signed) {
    if (nibble_off) {
      return (int8_t)(byte & 0xF0) >> 4;
    } else {
      return (int8_t)((byte & 0x0F) << 4) >> 4;
    }
  } else {
    if (nibble_off) {
      return (byte >> 4) & 0x0F;
    } else {
      return byte & 0x0F;
    }
  }
}

static inline void int4_set(uint8_t *data, long offset, int value,
                            bool is_signed) {
  int clamped = is_signed ? CLAMP_I4(value) : CLAMP_U4(value);
  uint8_t nibble = (uint8_t)(clamped & 0x0F);
  long byte_off = offset / 2;
  int nibble_off = offset % 2;
  if (nibble_off) {
    data[byte_off] = (data[byte_off] & 0x0F) | (nibble << 4);
  } else {
    data[byte_off] = (data[byte_off] & 0xF0) | nibble;
  }
}

// Complex arithmetic operations
#define COMPLEX_ADD(a, b) ((a) + (b))
#define COMPLEX_MUL(a, b) ((a) * (b))

// Complex comparison operations (lexicographic order)
static inline complex32 complex_max(complex32 a, complex32 b) {
  float a_real = crealf(a), a_imag = cimagf(a);
  float b_real = crealf(b), b_imag = cimagf(b);
  if (a_real > b_real) return a;
  if (a_real < b_real) return b;
  return (a_imag >= b_imag) ? a : b;
}

static inline complex64 complex64_max(complex64 a, complex64 b) {
  double a_real = creal(a), a_imag = cimag(a);
  double b_real = creal(b), b_imag = cimag(b);
  if (a_real > b_real) return a;
  if (a_real < b_real) return b;
  return (a_imag >= b_imag) ? a : b;
}

static inline complex32 complex_min(complex32 a, complex32 b) {
  float a_real = crealf(a), a_imag = cimagf(a);
  float b_real = crealf(b), b_imag = cimagf(b);
  if (a_real < b_real) return a;
  if (a_real > b_real) return b;
  return (a_imag < b_imag) ? a : b;
}

static inline complex64 complex64_min(complex64 a, complex64 b) {
  double a_real = creal(a), a_imag = cimag(a);
  double b_real = creal(b), b_imag = cimag(b);
  if (a_real < b_real) return a;
  if (a_real > b_real) return b;
  return (a_imag < b_imag) ? a : b;
}

// Core ndarray structure for strided array operations
typedef struct {
  void *data;
  int ndim;
  int *shape;
  int *strides;
  int offset;
} ndarray_t;

// Iterator for n-dimensional arrays (binary operations)
typedef struct {
  int ndim;
  int *shape;
  int *coords;
  int *x_strides;
  int *y_strides;
  int *z_strides;
} nd_iterator_t;

// Iterator for copying between two arrays
typedef struct {
  int ndim;
  int *shape;
  int *coords;
  int *src_strides;
  int *dst_strides;
} nd_copy_iterator_t;

// Single array iterator for unary operations
typedef struct {
  int ndim;
  int *shape;
  int *coords;
  int *strides;
} nd_single_iterator_t;

// Macro to iterate over all types (extended to include bigarray_ext types)
// Note: int4/uint4 need special handling (2 values per byte)
// Note: float16 uses caml_ba_uint16 like the standard library
#define FOR_EACH_TYPE(MACRO)                      \
  MACRO(int8_t, i8, CAML_BA_SINT8)                \
  MACRO(uint8_t, u8, CAML_BA_UINT8)               \
  MACRO(int16_t, i16, CAML_BA_SINT16)             \
  MACRO(uint16_t, u16, CAML_BA_UINT16)            \
  MACRO(int32_t, i32, CAML_BA_INT32)              \
  MACRO(int64_t, i64, CAML_BA_INT64)              \
  MACRO(caml_ba_uint32, u32, NX_BA_UINT32)        \
  MACRO(caml_ba_uint64, u64, NX_BA_UINT64)        \
  MACRO(intnat, inat, CAML_BA_NATIVE_INT)         \
  MACRO(uint16_t, f16, CAML_BA_FLOAT16)           \
  MACRO(float, f32, CAML_BA_FLOAT32)              \
  MACRO(double, f64, CAML_BA_FLOAT64)             \
  MACRO(complex32, c32, CAML_BA_COMPLEX32)        \
  MACRO(complex64, c64, CAML_BA_COMPLEX64)        \
  MACRO(caml_ba_bfloat16, bf16, NX_BA_BFLOAT16)   \
  MACRO(caml_ba_bool, bool_, NX_BA_BOOL)          \
  MACRO(uint8_t, i4, NX_BA_INT4)                  \
  MACRO(uint8_t, u4, NX_BA_UINT4)                 \
  MACRO(caml_ba_fp8_e4m3, f8e4m3, NX_BA_FP8_E4M3) \
  MACRO(caml_ba_fp8_e5m2, f8e5m2, NX_BA_FP8_E5M2)

// Helper functions for safe operations
static inline long total_elements_safe(const ndarray_t *arr) {
  if (!arr || arr->ndim == 0) return 1;
  long total = 1;
  for (int i = 0; i < arr->ndim; i++) {
    long dim = arr->shape[i];
    if (dim <= 0) return 0;
    if (total > LONG_MAX / dim) {
      caml_failwith("total_elements_safe: integer overflow");
    }
    total *= dim;
  }
  return total;
}

static inline bool is_fully_contiguous(const ndarray_t *x, const ndarray_t *y,
                                       const ndarray_t *z) {
  if (!x || !y || !z || x->ndim != y->ndim || x->ndim != z->ndim) return false;
  if (x->ndim == 0) return true;

  // Check C-contiguous layout
  int expected_stride = 1;
  for (int i = x->ndim - 1; i >= 0; i--) {
    if (x->strides[i] != expected_stride || y->strides[i] != expected_stride ||
        z->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= x->shape[i];
  }
  return true;
}

static inline bool is_contiguous(const ndarray_t *x) {
  if (!x || x->ndim == 0) return true;
  
  // Check C-contiguous layout
  int expected_stride = 1;
  for (int i = x->ndim - 1; i >= 0; i--) {
    if (x->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= x->shape[i];
  }
  return true;
}

static inline void nd_iterator_init_safe(nd_iterator_t *it, const ndarray_t *x,
                                         const ndarray_t *y,
                                         const ndarray_t *z) {
  if (!it || !x || !y || !z) {
    caml_failwith("nd_iterator_init_safe: null pointer");
  }
  if (x->ndim != y->ndim || x->ndim != z->ndim) {
    caml_failwith("nd_iterator_init_safe: dimension mismatch");
  }
  it->ndim = x->ndim;
  it->shape = x->shape;
  it->coords = (int *)calloc(x->ndim, sizeof(int));
  it->x_strides = x->strides;
  it->y_strides = y->strides;
  it->z_strides = z->strides;
  if (!it->coords) {
    caml_failwith("nd_iterator_init_safe: allocation failed");
  }
}

static inline void nd_iterator_get_offsets(const nd_iterator_t *it, long *x_off,
                                           long *y_off, long *z_off) {
  *x_off = 0;
  *y_off = 0;
  *z_off = 0;
  for (int i = 0; i < it->ndim; i++) {
    *x_off += it->coords[i] * it->x_strides[i];
    *y_off += it->coords[i] * it->y_strides[i];
    *z_off += it->coords[i] * it->z_strides[i];
  }
}

static inline bool nd_iterator_next(nd_iterator_t *it) {
  for (int i = it->ndim - 1; i >= 0; i--) {
    it->coords[i]++;
    if (it->coords[i] < it->shape[i]) {
      return true;
    }
    it->coords[i] = 0;
  }
  return false;
}

static inline void nd_iterator_destroy(nd_iterator_t *it) {
  if (it && it->coords) {
    free(it->coords);
    it->coords = NULL;
  }
}

// Single array iterator functions
static inline void nd_iterator_init(nd_single_iterator_t *it, const ndarray_t *arr) {
  if (!it || !arr) {
    caml_failwith("nd_iterator_init: null pointer");
  }
  it->ndim = arr->ndim;
  it->shape = arr->shape;
  it->coords = (int *)calloc(arr->ndim, sizeof(int));
  it->strides = arr->strides;
  if (!it->coords) {
    caml_failwith("nd_iterator_init: allocation failed");
  }
}

static inline void nd_iterator_get_offset(const nd_single_iterator_t *it, long *offset) {
  *offset = 0;
  for (int i = 0; i < it->ndim; i++) {
    *offset += it->coords[i] * it->strides[i];
  }
}

static inline bool nd_single_iterator_next(nd_single_iterator_t *it) {
  for (int i = it->ndim - 1; i >= 0; i--) {
    it->coords[i]++;
    if (it->coords[i] < it->shape[i]) {
      return true;
    }
    it->coords[i] = 0;
  }
  return false;
}

static inline void nd_single_iterator_destroy(nd_single_iterator_t *it) {
  if (it && it->coords) {
    free(it->coords);
    it->coords = NULL;
  }
}

// Copy iterator functions
static inline void nd_copy_iterator_init(nd_copy_iterator_t *it, const ndarray_t *src, const ndarray_t *dst) {
  if (!it || !src || !dst) {
    caml_failwith("nd_copy_iterator_init: null pointer");
  }
  if (src->ndim != dst->ndim) {
    caml_failwith("nd_copy_iterator_init: dimension mismatch");
  }
  it->ndim = src->ndim;
  it->shape = src->shape;
  it->coords = (int *)calloc(src->ndim, sizeof(int));
  it->src_strides = src->strides;
  it->dst_strides = dst->strides;
  if (!it->coords) {
    caml_failwith("nd_copy_iterator_init: allocation failed");
  }
}

static inline void nd_copy_iterator_get_offsets(const nd_copy_iterator_t *it, long *src_off, long *dst_off) {
  *src_off = 0;
  *dst_off = 0;
  for (int i = 0; i < it->ndim; i++) {
    *src_off += it->coords[i] * it->src_strides[i];
    *dst_off += it->coords[i] * it->dst_strides[i];
  }
}

static inline bool nd_copy_iterator_next(nd_copy_iterator_t *it) {
  for (int i = it->ndim - 1; i >= 0; i--) {
    it->coords[i]++;
    if (it->coords[i] < it->shape[i]) {
      return true;
    }
    it->coords[i] = 0;
  }
  return false;
}

static inline void nd_copy_iterator_destroy(nd_copy_iterator_t *it) {
  if (it && it->coords) {
    free(it->coords);
    it->coords = NULL;
  }
}

// Helper to extract ndarray from FFI tensor
static inline ndarray_t extract_ndarray(value v_ffi_tensor) {
  value v_data = Field(v_ffi_tensor, FFI_TENSOR_DATA);
  value v_shape = Field(v_ffi_tensor, FFI_TENSOR_SHAPE);
  value v_strides = Field(v_ffi_tensor, FFI_TENSOR_STRIDES);
  int offset = Int_val(Field(v_ffi_tensor, FFI_TENSOR_OFFSET));

  struct caml_ba_array *ba = Caml_ba_array_val(v_data);
  void *data = ba->data;

  int ndim = Wosize_val(v_shape);

  // Always allocate on heap to avoid stack corruption
  int *shape = (int *)malloc(ndim * sizeof(int));
  int *strides = (int *)malloc(ndim * sizeof(int));

  if (!shape || !strides) {
    if (shape) free(shape);
    if (strides) free(strides);
    caml_failwith("extract_ndarray: allocation failed");
  }

  // Extract shape and strides
  for (int i = 0; i < ndim; i++) {
    shape[i] = Int_val(Field(v_shape, i));
    strides[i] = Int_val(Field(v_strides, i));
  }

  ndarray_t arr = {data, ndim, shape, strides, offset};
  return arr;
}

// Clean up heap-allocated arrays if needed
static inline void cleanup_ndarray(ndarray_t *arr) {
  // Always free since we always allocate on heap now
  if (arr->shape) free(arr->shape);
  if (arr->strides) free(arr->strides);
}

#endif  // NX_C_SHARED_H
