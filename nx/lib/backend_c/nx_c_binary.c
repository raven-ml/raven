// Binary operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <string.h>

#include "nx_c_shared.h"

#if defined(_OPENMP)
#define NX_PARALLEL_THRESHOLD 32768
#define NX_FOR_EACH_ELEM(total, BODY)                                           \
  do {                                                                          \
    if ((total) >= NX_PARALLEL_THRESHOLD) {                                     \
      _Pragma("omp parallel for simd schedule(static)")                         \
      for (long i = 0; i < (total); ++i) {                                      \
        BODY;                                                                   \
      }                                                                         \
    } else {                                                                    \
      _Pragma("omp simd")                                                       \
      for (long i = 0; i < (total); ++i) {                                      \
        BODY;                                                                   \
      }                                                                         \
    }                                                                           \
  } while (0)
#else
#define NX_FOR_EACH_ELEM(total, BODY)                                           \
  do {                                                                          \
    for (long i = 0; i < (total); ++i) {                                        \
      BODY;                                                                     \
    }                                                                           \
  } while (0)
#endif

// Type definitions for binary operations
typedef void (*binary_op_t)(const ndarray_t *, const ndarray_t *, ndarray_t *);

// Dispatch table for each type
typedef struct {
  binary_op_t i8, u8, i16, u16, i32, i64, u32, u64, inat;
  binary_op_t f16, f32, f64;
  binary_op_t c32, c64;
  binary_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2;
} binary_op_table;

// Macro to generate all standard type variants for an operation
// Note: float16, bfloat16, fp8 types need special handling with conversion
#define GENERATE_BINARY_OP(name, OP_EXPR)               \
  BINARY_OP_FOR_TYPE(name, int8_t, i8, OP_EXPR)         \
  BINARY_OP_FOR_TYPE(name, uint8_t, u8, OP_EXPR)        \
  BINARY_OP_FOR_TYPE(name, int16_t, i16, OP_EXPR)       \
  BINARY_OP_FOR_TYPE(name, uint16_t, u16, OP_EXPR)      \
  BINARY_OP_FOR_TYPE(name, int32_t, i32, OP_EXPR)       \
  BINARY_OP_FOR_TYPE(name, int64_t, i64, OP_EXPR)       \
  BINARY_OP_FOR_TYPE(name, uint32_t, u32, OP_EXPR)      \
  BINARY_OP_FOR_TYPE(name, uint64_t, u64, OP_EXPR)      \
  BINARY_OP_FOR_TYPE(name, intnat, inat, OP_EXPR)       \
  BINARY_OP_FOR_TYPE(name, float, f32, OP_EXPR)         \
  BINARY_OP_FOR_TYPE(name, double, f64, OP_EXPR)

// Macro to build dispatch table
#define BUILD_DISPATCH_TABLE(name)                                             \
  static const binary_op_table name##_table = {.i8 = nx_c_##name##_i8,         \
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

// Helper to iterate over inner dimensions with a kernel function for binary
// operations
typedef void (*kernel_fn)(void *, void *, void *, long, long, long);

static inline void iterate_inner_dims(const ndarray_t *x, const ndarray_t *y,
                                      const ndarray_t *z, long outer_idx,
                                      kernel_fn kernel, void *x_data,
                                      void *y_data, void *z_data) {
  if (x->ndim <= 1) {
    kernel(x_data, y_data, z_data, outer_idx * x->strides[0],
           outer_idx * y->strides[0], outer_idx * z->strides[0]);
    return;
  }

  long x_base = outer_idx * x->strides[0];
  long y_base = outer_idx * y->strides[0];
  long z_base = outer_idx * z->strides[0];

  // Create temporary iterator for inner dimensions
  int inner_ndim = x->ndim - 1;
  int coords_stack[MAX_NDIM];
  int *coords = coords_stack;
  bool heap_alloc = false;
  if (inner_ndim > MAX_NDIM) {
    coords = (int *)calloc(inner_ndim, sizeof(int));
    if (!coords) {
      caml_failwith("iterate_inner_dims: allocation failed");
    }
    heap_alloc = true;
  } else {
    memset(coords_stack, 0, inner_ndim * sizeof(int));
  }

  // Iterate over inner dimensions
  bool done = false;
  while (!done) {
    long x_off = x_base;
    long y_off = y_base;
    long z_off = z_base;

    for (int i = 0; i < inner_ndim; i++) {
      x_off += coords[i] * x->strides[i + 1];
      y_off += coords[i] * y->strides[i + 1];
      z_off += coords[i] * z->strides[i + 1];
    }

    kernel(x_data, y_data, z_data, x_off, y_off, z_off);

    // Advance to next position
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

  if (heap_alloc) free(coords);
}

// Generic binary operation kernel
#define BINARY_OP_KERNEL(name, T, suffix, OP)                             \
  static inline void nx_c_##name##_##suffix##_kernel(void *x_data, void *y_data, \
                                              void *z_data, long x_off,   \
                                              long y_off, long z_off) {   \
    T *x = (T *)x_data;                                                   \
    T *y = (T *)y_data;                                                   \
    T *z = (T *)z_data;                                                   \
    z[z_off] = OP(x[x_off], y[y_off]);                                    \
  }

// Generic binary operation implementation
#define BINARY_OP_IMPL(name, T, suffix)                                        \
  static void nx_c_##name##_##suffix(const ndarray_t *x, const ndarray_t *y,   \
                                     ndarray_t *z) {                           \
    if (!x || !y || !z) {                                                      \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    long total = total_elements_safe(x);                                       \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_fully_contiguous(x, y, z)) {                                        \
      T *restrict xs = (T *)x->data + x->offset;                               \
      T *restrict ys = (T *)y->data + y->offset;                               \
      T *restrict zs = (T *)z->data + z->offset;                               \
      NX_FOR_EACH_ELEM(total, nx_c_##name##_##suffix##_kernel(xs, ys, zs, i, i, i)); \
    } else if (x->shape[0] > 1 && total / x->shape[0] > 50) {                  \
      _Pragma("omp parallel for if(x->shape[0] > 4)") for (long i = 0;         \
                                                           i < x->shape[0];    \
                                                           i++) {              \
        iterate_inner_dims(x, y, z, i, nx_c_##name##_##suffix##_kernel,        \
                           x->data, y->data, z->data);                         \
      }                                                                        \
    } else {                                                                   \
      nd_iterator_t it;                                                        \
      nd_iterator_init_safe(&it, x, y, z);                                     \
      do {                                                                     \
        long x_off, y_off, z_off;                                              \
        nd_iterator_get_offsets(&it, &x_off, &y_off, &z_off);                  \
        nx_c_##name##_##suffix##_kernel(x->data, y->data, z->data,             \
                                        x->offset + x_off, y->offset + y_off,  \
                                        z->offset + z_off);                    \
      } while (nd_iterator_next(&it));                                         \
      nd_iterator_destroy(&it);                                                \
    }                                                                          \
  }

// Macro to generate both kernel and implementation for an operation
#define BINARY_OP_FOR_TYPE(name, T, suffix, OP) \
  BINARY_OP_KERNEL(name, T, suffix, OP)         \
  BINARY_OP_IMPL(name, T, suffix)

// Low-precision float kernel (convert to float for op)
#define LOW_PREC_OP_KERNEL(name, T, suffix, OP, TO_FLOAT, FROM_FLOAT)     \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *y_data, \
                                              void *z_data, long x_off,   \
                                              long y_off, long z_off) {   \
    T *x = (T *)x_data;                                                   \
    T *y = (T *)y_data;                                                   \
    T *z = (T *)z_data;                                                   \
    float a = TO_FLOAT(x[x_off]);                                         \
    float b = TO_FLOAT(y[y_off]);                                         \
    z[z_off] = FROM_FLOAT(OP(a, b));                                      \
  }

// For floating-point division, no zero check - let IEEE 754 semantics apply
// (produces inf/-inf/NaN as appropriate)

// For low-precision, use the impl with the special kernel
#define LOW_PREC_OP_IMPL(name, T, suffix) BINARY_OP_IMPL(name, T, suffix)

// Complex OP for arithmetic - reuse from nx_c_shared.h where possible
// COMPLEX_ADD and COMPLEX_MUL are defined in nx_c_shared.h
#define COMPLEX_SUB(x, y) ((x) - (y))
#define COMPLEX_DIV(x, y) ((x) / (y))

// Helper macros for int4 saturation
#define CLAMP_I4(x) ((x) < -8 ? -8 : ((x) > 7 ? 7 : (x)))
#define CLAMP_U4(x) ((x) < 0 ? 0 : ((x) > 15 ? 15 : (x)))

// Special implementation for int4 (packed, unpack/op/pack with saturation)
#define INT4_OP_IMPL(name, signedness, suffix, OP)                            \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *y_data,     \
                                              void *z_data, long x_off,       \
                                              long y_off, long z_off) {       \
    uint8_t *x = (uint8_t *)x_data;                                           \
    uint8_t *y = (uint8_t *)y_data;                                           \
    uint8_t *z = (uint8_t *)z_data;                                           \
    long byte_off = x_off / 2;                                                \
    int nib_off = x_off % 2;                                                  \
    int a = nib_off ? (signedness ? (int8_t)(x[byte_off] >> 4)                \
                                  : (x[byte_off] >> 4) & 0x0F)                \
                    : (signedness ? (int8_t)((x[byte_off] & 0x0F) << 4) >> 4  \
                                  : x[byte_off] & 0x0F);                      \
    int b = nib_off ? (signedness ? (int8_t)(y[byte_off] >> 4)                \
                                  : (y[byte_off] >> 4) & 0x0F)                \
                    : (signedness ? (int8_t)((y[byte_off] & 0x0F) << 4) >> 4  \
                                  : y[byte_off] & 0x0F);                      \
    int res = OP(a, b);                                                       \
    /* Saturate to 4-bit range */                                             \
    res = signedness ? CLAMP_I4(res) : CLAMP_U4(res);                         \
    uint8_t nib = (uint8_t)res & 0x0F;                                        \
    if (nib_off) {                                                            \
      z[byte_off] = (z[byte_off] & 0x0F) | (nib << 4);                        \
    } else {                                                                  \
      z[byte_off] = (z[byte_off] & 0xF0) | nib;                               \
    }                                                                         \
  }                                                                           \
  static void nx_c_##name##_##suffix(const ndarray_t *x, const ndarray_t *y,  \
                                     ndarray_t *z) {                          \
    if (is_fully_contiguous(x, y, z)) {                                       \
      long total = total_elements_safe(x);                                    \
      void *x_data = x->data + x->offset;                                     \
      void *y_data = y->data + y->offset;                                     \
      void *z_data = z->data + z->offset;                                     \
      _Pragma("omp parallel for if(total > 10000)") for (long i = 0;          \
                                                         i < total; i++) {    \
        nx_c_##name##_##suffix##_kernel(x_data, y_data, z_data, i, i, i);     \
      }                                                                       \
    } else {                                                                  \
      nd_iterator_t it;                                                       \
      nd_iterator_init_safe(&it, x, y, z);                                    \
      void *x_data = x->data;                                                 \
      void *y_data = y->data;                                                 \
      void *z_data = z->data;                                                 \
      do {                                                                    \
        long x_off, y_off, z_off;                                             \
        nd_iterator_get_offsets(&it, &x_off, &y_off, &z_off);                 \
        nx_c_##name##_##suffix##_kernel(x_data, y_data, z_data,               \
                                        x_off + x->offset, y_off + y->offset, \
                                        z_off + z->offset);                   \
      } while (nd_iterator_next(&it));                                        \
      nd_iterator_destroy(&it);                                               \
    }                                                                         \
  }

// For bool, treat as uint8_t with standard arithmetic
// Note: Results may exceed 0/1 range (e.g., 1+1=2), stored in uint8_t

// Generate for all ops
// Addition
#define ADD_OP(x, y) ((x) + (y))
GENERATE_BINARY_OP(add, ADD_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(add, uint16_t, f16, ADD_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(add, uint16_t, f16)
LOW_PREC_OP_KERNEL(add, caml_ba_bfloat16, bf16, ADD_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(add, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(add, caml_ba_fp8_e4m3, f8e4m3, ADD_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(add, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(add, caml_ba_fp8_e5m2, f8e5m2, ADD_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(add, caml_ba_fp8_e5m2, f8e5m2)

BINARY_OP_FOR_TYPE(add, complex32, c32, COMPLEX_ADD)
BINARY_OP_FOR_TYPE(add, complex64, c64, COMPLEX_ADD)

INT4_OP_IMPL(add, 1, i4, ADD_OP)
INT4_OP_IMPL(add, 0, u4, ADD_OP)
BINARY_OP_FOR_TYPE(add, caml_ba_bool, bool_, ADD_OP)  // Standard arithmetic
BUILD_DISPATCH_TABLE(add);

// Subtraction
#define SUB_OP(x, y) ((x) - (y))
GENERATE_BINARY_OP(sub, SUB_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(sub, uint16_t, f16, SUB_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(sub, uint16_t, f16)
LOW_PREC_OP_KERNEL(sub, caml_ba_bfloat16, bf16, SUB_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(sub, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(sub, caml_ba_fp8_e4m3, f8e4m3, SUB_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(sub, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(sub, caml_ba_fp8_e5m2, f8e5m2, SUB_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(sub, caml_ba_fp8_e5m2, f8e5m2)

BINARY_OP_FOR_TYPE(sub, complex32, c32, COMPLEX_SUB)
BINARY_OP_FOR_TYPE(sub, complex64, c64, COMPLEX_SUB)

INT4_OP_IMPL(sub, 1, i4, SUB_OP)
INT4_OP_IMPL(sub, 0, u4, SUB_OP)
BINARY_OP_FOR_TYPE(sub, caml_ba_bool, bool_,
                   SUB_OP)  // Standard arithmetic (may wrap)
BUILD_DISPATCH_TABLE(sub);

// Multiplication
#define MUL_OP(x, y) ((x) * (y))
GENERATE_BINARY_OP(mul, MUL_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(mul, uint16_t, f16, MUL_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(mul, uint16_t, f16)
LOW_PREC_OP_KERNEL(mul, caml_ba_bfloat16, bf16, MUL_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(mul, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(mul, caml_ba_fp8_e4m3, f8e4m3, MUL_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(mul, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(mul, caml_ba_fp8_e5m2, f8e5m2, MUL_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(mul, caml_ba_fp8_e5m2, f8e5m2)

BINARY_OP_FOR_TYPE(mul, complex32, c32, COMPLEX_MUL)
BINARY_OP_FOR_TYPE(mul, complex64, c64, COMPLEX_MUL)

INT4_OP_IMPL(mul, 1, i4, MUL_OP)
INT4_OP_IMPL(mul, 0, u4, MUL_OP)
BINARY_OP_FOR_TYPE(mul, caml_ba_bool, bool_, MUL_OP)  // Standard arithmetic
BUILD_DISPATCH_TABLE(mul);

// Integer division - truncates and checks for zero
#define INT_DIV_OP(x, y) \
  ((y) == 0 ? (caml_failwith("division by zero"), (x)) : ((x) / (y)))

// Integer types
BINARY_OP_FOR_TYPE(idiv, int8_t, i8, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, uint8_t, u8, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, int16_t, i16, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, uint16_t, u16, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, int32_t, i32, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, int64_t, i64, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, uint32_t, u32, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, uint64_t, u64, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, intnat, inat, INT_DIV_OP)

// For float types, idiv truncates the result
#define FLOAT_IDIV_OP(x, y) (trunc((x) / (y)))
BINARY_OP_FOR_TYPE(idiv, float, f32, FLOAT_IDIV_OP)
BINARY_OP_FOR_TYPE(idiv, double, f64, FLOAT_IDIV_OP)

LOW_PREC_OP_KERNEL(idiv, uint16_t, f16, FLOAT_IDIV_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(idiv, uint16_t, f16)
LOW_PREC_OP_KERNEL(idiv, caml_ba_bfloat16, bf16, FLOAT_IDIV_OP,
                   bfloat16_to_float, float_to_bfloat16)
LOW_PREC_OP_IMPL(idiv, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(idiv, caml_ba_fp8_e4m3, f8e4m3, FLOAT_IDIV_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(idiv, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(idiv, caml_ba_fp8_e5m2, f8e5m2, FLOAT_IDIV_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(idiv, caml_ba_fp8_e5m2, f8e5m2)

// Complex idiv also truncates both real and imaginary parts
static void nx_c_idiv_c32_kernel(void *x_data, void *y_data, void *z_data,
                                 long x_off, long y_off, long z_off) {
  complex32 *x = (complex32 *)x_data;
  complex32 *y = (complex32 *)y_data;
  complex32 *z = (complex32 *)z_data;
  complex32 res = x[x_off] / y[y_off];
  z[z_off] = truncf(crealf(res)) + I * truncf(cimagf(res));
}

static void nx_c_idiv_c64_kernel(void *x_data, void *y_data, void *z_data,
                                 long x_off, long y_off, long z_off) {
  complex64 *x = (complex64 *)x_data;
  complex64 *y = (complex64 *)y_data;
  complex64 *z = (complex64 *)z_data;
  complex64 res = x[x_off] / y[y_off];
  z[z_off] = trunc(creal(res)) + I * trunc(cimag(res));
}

BINARY_OP_IMPL(idiv, complex32, c32)
BINARY_OP_IMPL(idiv, complex64, c64)


INT4_OP_IMPL(idiv, 1, i4, INT_DIV_OP)
INT4_OP_IMPL(idiv, 0, u4, INT_DIV_OP)
BINARY_OP_FOR_TYPE(idiv, caml_ba_bool, bool_, INT_DIV_OP)
BUILD_DISPATCH_TABLE(idiv);

// Floating-point division - follows IEEE 754 (inf/NaN for division by zero)
#define FLOAT_DIV_OP(x, y) ((x) / (y))

// Integer types converted to float for fdiv
BINARY_OP_FOR_TYPE(fdiv, int8_t, i8, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, uint8_t, u8, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, int16_t, i16, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, uint16_t, u16, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, int32_t, i32, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, int64_t, i64, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, uint32_t, u32, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, uint64_t, u64, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, intnat, inat, FLOAT_DIV_OP)

// Floating-point types use IEEE 754 semantics
BINARY_OP_FOR_TYPE(fdiv, float, f32, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, double, f64, FLOAT_DIV_OP)

LOW_PREC_OP_KERNEL(fdiv, uint16_t, f16, FLOAT_DIV_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(fdiv, uint16_t, f16)
LOW_PREC_OP_KERNEL(fdiv, caml_ba_bfloat16, bf16, FLOAT_DIV_OP,
                   bfloat16_to_float, float_to_bfloat16)
LOW_PREC_OP_IMPL(fdiv, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(fdiv, caml_ba_fp8_e4m3, f8e4m3, FLOAT_DIV_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(fdiv, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(fdiv, caml_ba_fp8_e5m2, f8e5m2, FLOAT_DIV_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(fdiv, caml_ba_fp8_e5m2, f8e5m2)

BINARY_OP_FOR_TYPE(fdiv, complex32, c32, COMPLEX_DIV)
BINARY_OP_FOR_TYPE(fdiv, complex64, c64, COMPLEX_DIV)

INT4_OP_IMPL(fdiv, 1, i4, FLOAT_DIV_OP)
INT4_OP_IMPL(fdiv, 0, u4, FLOAT_DIV_OP)
BINARY_OP_FOR_TYPE(fdiv, caml_ba_bool, bool_, FLOAT_DIV_OP)
BUILD_DISPATCH_TABLE(fdiv);

// Max/Min with special complex
#define MAX_OP(x, y) ((x) > (y) ? (x) : (y))
GENERATE_BINARY_OP(max, MAX_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(max, uint16_t, f16, MAX_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(max, uint16_t, f16)
LOW_PREC_OP_KERNEL(max, caml_ba_bfloat16, bf16, MAX_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(max, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(max, caml_ba_fp8_e4m3, f8e4m3, MAX_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(max, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(max, caml_ba_fp8_e5m2, f8e5m2, MAX_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(max, caml_ba_fp8_e5m2, f8e5m2)

BINARY_OP_FOR_TYPE(max, complex32, c32, complex_max)
BINARY_OP_FOR_TYPE(max, complex64, c64, complex64_max)
INT4_OP_IMPL(max, 1, i4, MAX_OP)
INT4_OP_IMPL(max, 0, u4, MAX_OP)
BINARY_OP_FOR_TYPE(max, caml_ba_bool, bool_, MAX_OP)  // Standard comparison
BUILD_DISPATCH_TABLE(max);

#define MIN_OP(x, y) ((x) < (y) ? (x) : (y))
GENERATE_BINARY_OP(min, MIN_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(min, uint16_t, f16, MIN_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(min, uint16_t, f16)
LOW_PREC_OP_KERNEL(min, caml_ba_bfloat16, bf16, MIN_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(min, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(min, caml_ba_fp8_e4m3, f8e4m3, MIN_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(min, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(min, caml_ba_fp8_e5m2, f8e5m2, MIN_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(min, caml_ba_fp8_e5m2, f8e5m2)

BINARY_OP_FOR_TYPE(min, complex32, c32, complex_min)
BINARY_OP_FOR_TYPE(min, complex64, c64, complex64_min)
INT4_OP_IMPL(min, 1, i4, MIN_OP)
INT4_OP_IMPL(min, 0, u4, MIN_OP)
BINARY_OP_FOR_TYPE(min, caml_ba_bool, bool_, MIN_OP)  // Standard comparison
BUILD_DISPATCH_TABLE(min);

// =========== MODULO ===========
#define MOD_OP(x, y) \
  ((y) == 0 ? (caml_failwith("modulo by zero"), 0) : ((x) % (y)))
#define FMOD_OP(x, y) (fmod((x), (y)))

// Integer modulo
BINARY_OP_FOR_TYPE(mod, int8_t, i8, MOD_OP)
BINARY_OP_FOR_TYPE(mod, uint8_t, u8, MOD_OP)
BINARY_OP_FOR_TYPE(mod, int16_t, i16, MOD_OP)
BINARY_OP_FOR_TYPE(mod, uint16_t, u16, MOD_OP)
BINARY_OP_FOR_TYPE(mod, int32_t, i32, MOD_OP)
BINARY_OP_FOR_TYPE(mod, int64_t, i64, MOD_OP)
BINARY_OP_FOR_TYPE(mod, uint32_t, u32, MOD_OP)
BINARY_OP_FOR_TYPE(mod, uint64_t, u64, MOD_OP)
BINARY_OP_FOR_TYPE(mod, intnat, inat, MOD_OP)

// Float modulo uses fmod
BINARY_OP_FOR_TYPE(mod, float, f32, FMOD_OP)
BINARY_OP_FOR_TYPE(mod, double, f64, FMOD_OP)
LOW_PREC_OP_KERNEL(mod, uint16_t, f16, FMOD_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(mod, uint16_t, f16)
LOW_PREC_OP_KERNEL(mod, caml_ba_bfloat16, bf16, FMOD_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(mod, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(mod, caml_ba_fp8_e4m3, f8e4m3, FMOD_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(mod, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(mod, caml_ba_fp8_e5m2, f8e5m2, FMOD_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(mod, caml_ba_fp8_e5m2, f8e5m2)

// Complex modulo not well-defined

INT4_OP_IMPL(mod, 1, i4, MOD_OP)
INT4_OP_IMPL(mod, 0, u4, MOD_OP)
BINARY_OP_FOR_TYPE(mod, caml_ba_bool, bool_, MOD_OP)

// Build dispatch table with NULL for unsupported complex types
static const binary_op_table mod_table = {.i8 = nx_c_mod_i8,
                                          .u8 = nx_c_mod_u8,
                                          .i16 = nx_c_mod_i16,
                                          .u16 = nx_c_mod_u16,
                                          .i32 = nx_c_mod_i32,
                                          .i64 = nx_c_mod_i64,
                                          .u32 = nx_c_mod_u32,
                                          .u64 = nx_c_mod_u64,
                                          .inat = nx_c_mod_inat,
                                          .f16 = nx_c_mod_f16,
                                          .f32 = nx_c_mod_f32,
                                          .f64 = nx_c_mod_f64,
                                          .c32 = NULL,
                                          .c64 = NULL,
                                          .bf16 = nx_c_mod_bf16,
                                          .bool_ = nx_c_mod_bool_,
                                          .i4 = nx_c_mod_i4,
                                          .u4 = nx_c_mod_u4,
                                          .f8e4m3 = nx_c_mod_f8e4m3,
                                          .f8e5m2 = nx_c_mod_f8e5m2};

// =========== POWER ===========
#define POW_OP(x, y) (pow((double)(x), (double)(y)))
#define FPOW_OP(x, y) (powf((x), (y)))
#define DPOW_OP(x, y) (pow((x), (y)))

// All types use pow, converting to appropriate precision
BINARY_OP_FOR_TYPE(pow, int8_t, i8, POW_OP)
BINARY_OP_FOR_TYPE(pow, uint8_t, u8, POW_OP)
BINARY_OP_FOR_TYPE(pow, int16_t, i16, POW_OP)
BINARY_OP_FOR_TYPE(pow, uint16_t, u16, POW_OP)
BINARY_OP_FOR_TYPE(pow, int32_t, i32, POW_OP)
BINARY_OP_FOR_TYPE(pow, int64_t, i64, POW_OP)
BINARY_OP_FOR_TYPE(pow, uint32_t, u32, POW_OP)
BINARY_OP_FOR_TYPE(pow, uint64_t, u64, POW_OP)
BINARY_OP_FOR_TYPE(pow, intnat, inat, POW_OP)

BINARY_OP_FOR_TYPE(pow, float, f32, FPOW_OP)
BINARY_OP_FOR_TYPE(pow, double, f64, DPOW_OP)
LOW_PREC_OP_KERNEL(pow, uint16_t, f16, FPOW_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(pow, uint16_t, f16)
LOW_PREC_OP_KERNEL(pow, caml_ba_bfloat16, bf16, FPOW_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(pow, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(pow, caml_ba_fp8_e4m3, f8e4m3, FPOW_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(pow, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(pow, caml_ba_fp8_e5m2, f8e5m2, FPOW_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(pow, caml_ba_fp8_e5m2, f8e5m2)

// Complex power using cpow
#define CPOW32_OP(x, y) (cpowf((x), (y)))
#define CPOW64_OP(x, y) (cpow((x), (y)))
BINARY_OP_FOR_TYPE(pow, complex32, c32, CPOW32_OP)
BINARY_OP_FOR_TYPE(pow, complex64, c64, CPOW64_OP)


INT4_OP_IMPL(pow, 1, i4, POW_OP)
INT4_OP_IMPL(pow, 0, u4, POW_OP)
BINARY_OP_FOR_TYPE(pow, caml_ba_bool, bool_, POW_OP)
BUILD_DISPATCH_TABLE(pow);

// =========== COMPARISON - LESS THAN ===========
#define CMPLT_OP(x, y) ((x) < (y) ? 1 : 0)

// Comparison operations that output uint8
#define COMPARISON_OP_FOR_TYPE(name, T, suffix, OP)                            \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *y_data,      \
                                              void *z_data, long x_off,        \
                                              long y_off, long z_off) {        \
    T *x = (T *)x_data;                                                        \
    T *y = (T *)y_data;                                                        \
    uint8_t *z = (uint8_t *)z_data;                                            \
    z[z_off] = OP(x[x_off], y[y_off]);                                         \
  }                                                                            \
  static void nx_c_##name##_##suffix(const ndarray_t *x, const ndarray_t *y,   \
                                     ndarray_t *z) {                           \
    if (!x || !y || !z) {                                                      \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    long total = total_elements_safe(x);                                       \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_fully_contiguous(x, y, z)) {                                        \
      _Pragma("omp parallel for simd if(total > 1000)") for (long i = 0;       \
                                                             i < total; i++) { \
        nx_c_##name##_##suffix##_kernel(x->data, y->data, z->data,             \
                                        x->offset + i, y->offset + i,          \
                                        z->offset + i);                        \
      }                                                                        \
    } else if (x->shape[0] > 1 && total / x->shape[0] > 50) {                  \
      _Pragma("omp parallel for if(x->shape[0] > 4)") for (long i = 0;         \
                                                           i < x->shape[0];    \
                                                           i++) {              \
        iterate_inner_dims(x, y, z, i, nx_c_##name##_##suffix##_kernel,        \
                           x->data, y->data, z->data);                         \
      }                                                                        \
    } else {                                                                   \
      nd_iterator_t it;                                                        \
      nd_iterator_init_safe(&it, x, y, z);                                     \
      do {                                                                     \
        long x_off, y_off, z_off;                                              \
        nd_iterator_get_offsets(&it, &x_off, &y_off, &z_off);                  \
        nx_c_##name##_##suffix##_kernel(x->data, y->data, z->data,             \
                                        x->offset + x_off, y->offset + y_off,  \
                                        z->offset + z_off);                    \
      } while (nd_iterator_next(&it));                                         \
      nd_iterator_destroy(&it);                                                \
    }                                                                          \
  }

COMPARISON_OP_FOR_TYPE(cmplt, int8_t, i8, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, uint8_t, u8, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, int16_t, i16, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, uint16_t, u16, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, int32_t, i32, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, int64_t, i64, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, uint32_t, u32, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, uint64_t, u64, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, intnat, inat, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, float, f32, CMPLT_OP)
COMPARISON_OP_FOR_TYPE(cmplt, double, f64, CMPLT_OP)

// Low precision comparisons
static void nx_c_cmplt_f16_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  uint16_t *x = (uint16_t *)x_data;
  uint16_t *y = (uint16_t *)y_data;
  bool *z = (bool *)z_data;
  float a = half_to_float(x[x_off]);
  float b = half_to_float(y[y_off]);
  z[z_off] = a < b ? 1 : 0;
}
BINARY_OP_IMPL(cmplt, uint16_t, f16)

// Similar for other low-precision types
#define LOW_PREC_CMP_KERNEL(name, T, suffix, OP, TO_FLOAT)                \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *y_data, \
                                              void *z_data, long x_off,   \
                                              long y_off, long z_off) {   \
    T *x = (T *)x_data;                                                   \
    T *y = (T *)y_data;                                                   \
    bool *z = (bool *)z_data;                                       \
    float a = TO_FLOAT(x[x_off]);                                         \
    float b = TO_FLOAT(y[y_off]);                                         \
    z[z_off] = OP(a, b);                                                  \
  }                                                                       \
  BINARY_OP_IMPL(name, T, suffix)

LOW_PREC_CMP_KERNEL(cmplt, caml_ba_bfloat16, bf16, CMPLT_OP, bfloat16_to_float)
LOW_PREC_CMP_KERNEL(cmplt, caml_ba_fp8_e4m3, f8e4m3, CMPLT_OP,
                    fp8_e4m3_to_float)
LOW_PREC_CMP_KERNEL(cmplt, caml_ba_fp8_e5m2, f8e5m2, CMPLT_OP,
                    fp8_e5m2_to_float)

// Complex comparison not well-defined

// Int4 comparison implementation - unpacks 4-bit values and outputs uint8
#define INT4_COMPARISON_OP_IMPL(name, signedness, suffix, OP)                  \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *y_data,      \
                                              void *z_data, long x_off,        \
                                              long y_off, long z_off) {        \
    uint8_t *x = (uint8_t *)x_data;                                            \
    uint8_t *y = (uint8_t *)y_data;                                            \
    bool *z = (bool *)z_data;                                            \
    /* Unpack x value */                                                       \
    long x_byte_off = x_off / 2;                                               \
    int x_nib_off = x_off % 2;                                                 \
    int a = x_nib_off ? (signedness ? (int8_t)(x[x_byte_off] >> 4)             \
                                    : (x[x_byte_off] >> 4) & 0x0F)             \
                      : (signedness ? (int8_t)((x[x_byte_off] & 0x0F) << 4) >> 4 \
                                    : x[x_byte_off] & 0x0F);                    \
    /* Unpack y value */                                                       \
    long y_byte_off = y_off / 2;                                               \
    int y_nib_off = y_off % 2;                                                 \
    int b = y_nib_off ? (signedness ? (int8_t)(y[y_byte_off] >> 4)             \
                                    : (y[y_byte_off] >> 4) & 0x0F)             \
                      : (signedness ? (int8_t)((y[y_byte_off] & 0x0F) << 4) >> 4 \
                                    : y[y_byte_off] & 0x0F);                    \
    /* Store comparison result as uint8 (0 or 1) */                            \
    z[z_off] = OP(a, b);                                                       \
  }                                                                            \
  static void nx_c_##name##_##suffix(const ndarray_t *x, const ndarray_t *y,   \
                                     ndarray_t *z) {                           \
    if (is_fully_contiguous(x, y, z)) {                                        \
      long total = total_elements_safe(x);                                     \
      void *x_data = x->data + x->offset;                                      \
      void *y_data = y->data + y->offset;                                      \
      void *z_data = z->data + z->offset;                                      \
      _Pragma("omp parallel for if(total > 10000)") for (long i = 0;           \
                                                         i < total; i++) {     \
        nx_c_##name##_##suffix##_kernel(x_data, y_data, z_data, i, i, i);      \
      }                                                                        \
    } else {                                                                   \
      nd_iterator_t it;                                                        \
      nd_iterator_init_safe(&it, x, y, z);                                     \
      void *x_data = x->data;                                                  \
      void *y_data = y->data;                                                  \
      void *z_data = z->data;                                                  \
      do {                                                                     \
        long x_off, y_off, z_off;                                              \
        nd_iterator_get_offsets(&it, &x_off, &y_off, &z_off);                  \
        nx_c_##name##_##suffix##_kernel(x_data, y_data, z_data,                \
                                        x_off + x->offset, y_off + y->offset,  \
                                        z_off + z->offset);                    \
      } while (nd_iterator_next(&it));                                         \
      nd_iterator_destroy(&it);                                                \
    }                                                                          \
  }

// Define comparison operators
#define CMPGT_OP(x, y) ((x) > (y) ? true : false)
#define CMPLE_OP(x, y) ((x) <= (y) ? true : false)
#define CMPGE_OP(x, y) ((x) >= (y) ? true : false)
#define CMPEQ_OP(x, y) ((x) == (y) ? true : false)
#define CMPNE_OP(x, y) ((x) != (y) ? true : false)

// Generate int4/uint4 comparison operations
INT4_COMPARISON_OP_IMPL(cmplt, 1, i4, CMPLT_OP)
INT4_COMPARISON_OP_IMPL(cmplt, 0, u4, CMPLT_OP)
INT4_COMPARISON_OP_IMPL(cmpgt, 1, i4, CMPGT_OP)
INT4_COMPARISON_OP_IMPL(cmpgt, 0, u4, CMPGT_OP)
INT4_COMPARISON_OP_IMPL(cmple, 1, i4, CMPLE_OP)
INT4_COMPARISON_OP_IMPL(cmple, 0, u4, CMPLE_OP)
INT4_COMPARISON_OP_IMPL(cmpge, 1, i4, CMPGE_OP)
INT4_COMPARISON_OP_IMPL(cmpge, 0, u4, CMPGE_OP)
INT4_COMPARISON_OP_IMPL(cmpeq, 1, i4, CMPEQ_OP)
INT4_COMPARISON_OP_IMPL(cmpeq, 0, u4, CMPEQ_OP)
INT4_COMPARISON_OP_IMPL(cmpne, 1, i4, CMPNE_OP)
INT4_COMPARISON_OP_IMPL(cmpne, 0, u4, CMPNE_OP)

COMPARISON_OP_FOR_TYPE(cmplt, caml_ba_bool, bool_, CMPLT_OP)

// Build dispatch table with NULL for unsupported complex and int4/uint4 types
static const binary_op_table cmplt_table = {.i8 = nx_c_cmplt_i8,
                                            .u8 = nx_c_cmplt_u8,
                                            .i16 = nx_c_cmplt_i16,
                                            .u16 = nx_c_cmplt_u16,
                                            .i32 = nx_c_cmplt_i32,
                                            .i64 = nx_c_cmplt_i64,
                                            .u32 = nx_c_cmplt_u32,
                                            .u64 = nx_c_cmplt_u64,
                                            .inat = nx_c_cmplt_inat,
                                            .f16 = nx_c_cmplt_f16,
                                            .f32 = nx_c_cmplt_f32,
                                            .f64 = nx_c_cmplt_f64,
                                            .c32 = NULL,
                                            .c64 = NULL,
                                            .bf16 = nx_c_cmplt_bf16,
                                            .bool_ = nx_c_cmplt_bool_,
                                            .i4 = nx_c_cmplt_i4,
                                            .u4 = nx_c_cmplt_u4,
                                            .f8e4m3 = nx_c_cmplt_f8e4m3,
                                            .f8e5m2 = nx_c_cmplt_f8e5m2};

// =========== COMPARISON - NOT EQUAL ===========
COMPARISON_OP_FOR_TYPE(cmpne, int8_t, i8, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, uint8_t, u8, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, int16_t, i16, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, uint16_t, u16, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, int32_t, i32, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, int64_t, i64, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, uint32_t, u32, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, uint64_t, u64, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, intnat, inat, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, float, f32, CMPNE_OP)
COMPARISON_OP_FOR_TYPE(cmpne, double, f64, CMPNE_OP)

// Low precision
static void nx_c_cmpne_f16_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  uint16_t *x = (uint16_t *)x_data;
  uint16_t *y = (uint16_t *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  float a = half_to_float(x[x_off]);
  float b = half_to_float(y[y_off]);
  z[z_off] = a != b ? 1 : 0;
}
BINARY_OP_IMPL(cmpne, uint16_t, f16)

LOW_PREC_CMP_KERNEL(cmpne, caml_ba_bfloat16, bf16, CMPNE_OP, bfloat16_to_float)
LOW_PREC_CMP_KERNEL(cmpne, caml_ba_fp8_e4m3, f8e4m3, CMPNE_OP,
                    fp8_e4m3_to_float)
LOW_PREC_CMP_KERNEL(cmpne, caml_ba_fp8_e5m2, f8e5m2, CMPNE_OP,
                    fp8_e5m2_to_float)

// Complex comparison for equality
static void nx_c_cmpne_c32_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  complex32 *x = (complex32 *)x_data;
  complex32 *y = (complex32 *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  z[z_off] = (x[x_off] != y[y_off]) ? 1 : 0;
}
BINARY_OP_IMPL(cmpne, complex32, c32)

static void nx_c_cmpne_c64_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  complex64 *x = (complex64 *)x_data;
  complex64 *y = (complex64 *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  z[z_off] = (x[x_off] != y[y_off]) ? 1 : 0;
}
BINARY_OP_IMPL(cmpne, complex64, c64)

// Int4 comparison not yet implemented

COMPARISON_OP_FOR_TYPE(cmpne, caml_ba_bool, bool_, CMPNE_OP)

// Build dispatch table with NULL for unsupported int4/uint4 types
static const binary_op_table cmpne_table = {.i8 = nx_c_cmpne_i8,
                                            .u8 = nx_c_cmpne_u8,
                                            .i16 = nx_c_cmpne_i16,
                                            .u16 = nx_c_cmpne_u16,
                                            .i32 = nx_c_cmpne_i32,
                                            .i64 = nx_c_cmpne_i64,
                                            .u32 = nx_c_cmpne_u32,
                                            .u64 = nx_c_cmpne_u64,
                                            .inat = nx_c_cmpne_inat,
                                            .f16 = nx_c_cmpne_f16,
                                            .f32 = nx_c_cmpne_f32,
                                            .f64 = nx_c_cmpne_f64,
                                            .c32 = nx_c_cmpne_c32,
                                            .c64 = nx_c_cmpne_c64,
                                            .bf16 = nx_c_cmpne_bf16,
                                            .bool_ = nx_c_cmpne_bool_,
                                            .i4 = nx_c_cmpne_i4,
                                            .u4 = nx_c_cmpne_u4,
                                            .f8e4m3 = nx_c_cmpne_f8e4m3,
                                            .f8e5m2 = nx_c_cmpne_f8e5m2};

// =========== COMPARISON - EQUAL ===========
COMPARISON_OP_FOR_TYPE(cmpeq, int8_t, i8, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, uint8_t, u8, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, int16_t, i16, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, uint16_t, u16, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, int32_t, i32, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, int64_t, i64, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, uint32_t, u32, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, uint64_t, u64, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, intnat, inat, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, float, f32, CMPEQ_OP)
COMPARISON_OP_FOR_TYPE(cmpeq, double, f64, CMPEQ_OP)

// Low precision
static void nx_c_cmpeq_f16_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  uint16_t *x = (uint16_t *)x_data;
  uint16_t *y = (uint16_t *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  float a = half_to_float(x[x_off]);
  float b = half_to_float(y[y_off]);
  z[z_off] = a == b ? 1 : 0;
}
BINARY_OP_IMPL(cmpeq, uint16_t, f16)

LOW_PREC_CMP_KERNEL(cmpeq, caml_ba_bfloat16, bf16, CMPEQ_OP, bfloat16_to_float)
LOW_PREC_CMP_KERNEL(cmpeq, caml_ba_fp8_e4m3, f8e4m3, CMPEQ_OP,
                    fp8_e4m3_to_float)
LOW_PREC_CMP_KERNEL(cmpeq, caml_ba_fp8_e5m2, f8e5m2, CMPEQ_OP,
                    fp8_e5m2_to_float)

// Complex comparison for equality
static void nx_c_cmpeq_c32_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  complex32 *x = (complex32 *)x_data;
  complex32 *y = (complex32 *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  z[z_off] = (x[x_off] == y[y_off]) ? 1 : 0;
}
BINARY_OP_IMPL(cmpeq, complex32, c32)

static void nx_c_cmpeq_c64_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  complex64 *x = (complex64 *)x_data;
  complex64 *y = (complex64 *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  z[z_off] = (x[x_off] == y[y_off]) ? 1 : 0;
}
BINARY_OP_IMPL(cmpeq, complex64, c64)

COMPARISON_OP_FOR_TYPE(cmpeq, caml_ba_bool, bool_, CMPEQ_OP)

static const binary_op_table cmpeq_table = {.i8 = nx_c_cmpeq_i8,
                                            .u8 = nx_c_cmpeq_u8,
                                            .i16 = nx_c_cmpeq_i16,
                                            .u16 = nx_c_cmpeq_u16,
                                            .i32 = nx_c_cmpeq_i32,
                                            .i64 = nx_c_cmpeq_i64,
                                            .u32 = nx_c_cmpeq_u32,
                                            .u64 = nx_c_cmpeq_u64,
                                            .inat = nx_c_cmpeq_inat,
                                            .f16 = nx_c_cmpeq_f16,
                                            .f32 = nx_c_cmpeq_f32,
                                            .f64 = nx_c_cmpeq_f64,
                                            .c32 = nx_c_cmpeq_c32,
                                            .c64 = nx_c_cmpeq_c64,
                                            .bf16 = nx_c_cmpeq_bf16,
                                            .bool_ = nx_c_cmpeq_bool_,
                                            .i4 = nx_c_cmpeq_i4,
                                            .u4 = nx_c_cmpeq_u4,
                                            .f8e4m3 = nx_c_cmpeq_f8e4m3,
                                            .f8e5m2 = nx_c_cmpeq_f8e5m2};

// =========== COMPARISON - LESS THAN OR EQUAL ===========
COMPARISON_OP_FOR_TYPE(cmple, int8_t, i8, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, uint8_t, u8, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, int16_t, i16, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, uint16_t, u16, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, int32_t, i32, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, int64_t, i64, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, uint32_t, u32, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, uint64_t, u64, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, intnat, inat, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, float, f32, CMPLE_OP)
COMPARISON_OP_FOR_TYPE(cmple, double, f64, CMPLE_OP)

// Low precision comparisons
static void nx_c_cmple_f16_kernel(void *x_data, void *y_data, void *z_data,
                                  long x_off, long y_off, long z_off) {
  uint16_t *x = (uint16_t *)x_data;
  uint16_t *y = (uint16_t *)y_data;
  uint8_t *z = (uint8_t *)z_data;
  float a = half_to_float(x[x_off]);
  float b = half_to_float(y[y_off]);
  z[z_off] = a <= b ? 1 : 0;
}
BINARY_OP_IMPL(cmple, uint16_t, f16)

LOW_PREC_CMP_KERNEL(cmple, caml_ba_bfloat16, bf16, CMPLE_OP, bfloat16_to_float)
LOW_PREC_CMP_KERNEL(cmple, caml_ba_fp8_e4m3, f8e4m3, CMPLE_OP,
                    fp8_e4m3_to_float)
LOW_PREC_CMP_KERNEL(cmple, caml_ba_fp8_e5m2, f8e5m2, CMPLE_OP,
                    fp8_e5m2_to_float)

COMPARISON_OP_FOR_TYPE(cmple, caml_ba_bool, bool_, CMPLE_OP)

// Build dispatch table with NULL for unsupported complex types
static const binary_op_table cmple_table = {.i8 = nx_c_cmple_i8,
                                            .u8 = nx_c_cmple_u8,
                                            .i16 = nx_c_cmple_i16,
                                            .u16 = nx_c_cmple_u16,
                                            .i32 = nx_c_cmple_i32,
                                            .i64 = nx_c_cmple_i64,
                                            .u32 = nx_c_cmple_u32,
                                            .u64 = nx_c_cmple_u64,
                                            .inat = nx_c_cmple_inat,
                                            .f16 = nx_c_cmple_f16,
                                            .f32 = nx_c_cmple_f32,
                                            .f64 = nx_c_cmple_f64,
                                            .c32 = NULL,
                                            .c64 = NULL,
                                            .bf16 = nx_c_cmple_bf16,
                                            .bool_ = nx_c_cmple_bool_,
                                            .i4 = nx_c_cmple_i4,
                                            .u4 = nx_c_cmple_u4,
                                            .f8e4m3 = nx_c_cmple_f8e4m3,
                                            .f8e5m2 = nx_c_cmple_f8e5m2};

// =========== BITWISE XOR ===========
#define XOR_OP(x, y) ((x) ^ (y))

// Bitwise operations only for integer types
BINARY_OP_FOR_TYPE(xor, int8_t, i8, XOR_OP)
BINARY_OP_FOR_TYPE(xor, uint8_t, u8, XOR_OP)
BINARY_OP_FOR_TYPE(xor, int16_t, i16, XOR_OP)
BINARY_OP_FOR_TYPE(xor, uint16_t, u16, XOR_OP)
BINARY_OP_FOR_TYPE(xor, int32_t, i32, XOR_OP)
BINARY_OP_FOR_TYPE(xor, int64_t, i64, XOR_OP)
BINARY_OP_FOR_TYPE(xor, uint32_t, u32, XOR_OP)
BINARY_OP_FOR_TYPE(xor, uint64_t, u64, XOR_OP)
BINARY_OP_FOR_TYPE(xor, intnat, inat, XOR_OP)

// Float bitwise operations not well-defined

INT4_OP_IMPL(xor, 1, i4, XOR_OP)
INT4_OP_IMPL(xor, 0, u4, XOR_OP)
BINARY_OP_FOR_TYPE(xor, caml_ba_bool, bool_, XOR_OP)

// Build dispatch table with NULL for unsupported float/complex types
static const binary_op_table xor_table = {.i8 = nx_c_xor_i8,
                                          .u8 = nx_c_xor_u8,
                                          .i16 = nx_c_xor_i16,
                                          .u16 = nx_c_xor_u16,
                                          .i32 = nx_c_xor_i32,
                                          .i64 = nx_c_xor_i64,
                                          .u32 = nx_c_xor_u32,
                                          .u64 = nx_c_xor_u64,
                                          .inat = nx_c_xor_inat,
                                          .f16 = NULL,
                                          .f32 = NULL,
                                          .f64 = NULL,
                                          .c32 = NULL,
                                          .c64 = NULL,
                                          .bf16 = NULL,
                                          .bool_ = nx_c_xor_bool_,
                                          .i4 = nx_c_xor_i4,
                                          .u4 = nx_c_xor_u4,
                                          .f8e4m3 = NULL,
                                          .f8e5m2 = NULL};

// =========== BITWISE OR ===========
#define OR_OP(x, y) ((x) | (y))

BINARY_OP_FOR_TYPE(or, int8_t, i8, OR_OP)
BINARY_OP_FOR_TYPE(or, uint8_t, u8, OR_OP)
BINARY_OP_FOR_TYPE(or, int16_t, i16, OR_OP)
BINARY_OP_FOR_TYPE(or, uint16_t, u16, OR_OP)
BINARY_OP_FOR_TYPE(or, int32_t, i32, OR_OP)
BINARY_OP_FOR_TYPE(or, int64_t, i64, OR_OP)
BINARY_OP_FOR_TYPE(or, uint32_t, u32, OR_OP)
BINARY_OP_FOR_TYPE(or, uint64_t, u64, OR_OP)
BINARY_OP_FOR_TYPE(or, intnat, inat, OR_OP)

// Float bitwise operations not well-defined

INT4_OP_IMPL(or, 1, i4, OR_OP)
INT4_OP_IMPL(or, 0, u4, OR_OP)
BINARY_OP_FOR_TYPE(or, caml_ba_bool, bool_, OR_OP)

// Build dispatch table with NULL for unsupported float/complex types
static const binary_op_table or_table = {.i8 = nx_c_or_i8,
                                         .u8 = nx_c_or_u8,
                                         .i16 = nx_c_or_i16,
                                         .u16 = nx_c_or_u16,
                                         .i32 = nx_c_or_i32,
                                         .i64 = nx_c_or_i64,
                                         .u32 = nx_c_or_u32,
                                         .u64 = nx_c_or_u64,
                                         .inat = nx_c_or_inat,
                                         .f16 = NULL,
                                         .f32 = NULL,
                                         .f64 = NULL,
                                         .c32 = NULL,
                                         .c64 = NULL,
                                         .bf16 = NULL,
                                         .bool_ = nx_c_or_bool_,
                                         .i4 = nx_c_or_i4,
                                         .u4 = nx_c_or_u4,
                                         .f8e4m3 = NULL,
                                         .f8e5m2 = NULL};

// =========== BITWISE AND ===========
#define AND_OP(x, y) ((x) & (y))

BINARY_OP_FOR_TYPE(and, int8_t, i8, AND_OP)
BINARY_OP_FOR_TYPE(and, uint8_t, u8, AND_OP)
BINARY_OP_FOR_TYPE(and, int16_t, i16, AND_OP)
BINARY_OP_FOR_TYPE(and, uint16_t, u16, AND_OP)
BINARY_OP_FOR_TYPE(and, int32_t, i32, AND_OP)
BINARY_OP_FOR_TYPE(and, int64_t, i64, AND_OP)
BINARY_OP_FOR_TYPE(and, uint32_t, u32, AND_OP)
BINARY_OP_FOR_TYPE(and, uint64_t, u64, AND_OP)
BINARY_OP_FOR_TYPE(and, intnat, inat, AND_OP)

// Float bitwise operations not well-defined

INT4_OP_IMPL(and, 1, i4, AND_OP)
INT4_OP_IMPL(and, 0, u4, AND_OP)
BINARY_OP_FOR_TYPE(and, caml_ba_bool, bool_, AND_OP)

// Build dispatch table with NULL for unsupported float/complex types
static const binary_op_table and_table = {.i8 = nx_c_and_i8,
                                          .u8 = nx_c_and_u8,
                                          .i16 = nx_c_and_i16,
                                          .u16 = nx_c_and_u16,
                                          .i32 = nx_c_and_i32,
                                          .i64 = nx_c_and_i64,
                                          .u32 = nx_c_and_u32,
                                          .u64 = nx_c_and_u64,
                                          .inat = nx_c_and_inat,
                                          .f16 = NULL,
                                          .f32 = NULL,
                                          .f64 = NULL,
                                          .c32 = NULL,
                                          .c64 = NULL,
                                          .bf16 = NULL,
                                          .bool_ = nx_c_and_bool_,
                                          .i4 = nx_c_and_i4,
                                          .u4 = nx_c_and_u4,
                                          .f8e4m3 = NULL,
                                          .f8e5m2 = NULL};

// Shared dispatch infrastructure

// Generic dispatch function for binary operations
static void dispatch_binary_op(value v_x, value v_y, value v_z,
                               const binary_op_table *table,
                               const char *op_name) {
  // Extract ndarrays from FFI tensors
  ndarray_t x = extract_ndarray(v_x);
  ndarray_t y = extract_ndarray(v_y);
  ndarray_t z = extract_ndarray(v_z);

  // Check shapes match
  if (x.ndim != y.ndim || x.ndim != z.ndim) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("shape mismatch");
  }
  for (int i = 0; i < x.ndim; i++) {
    if (x.shape[i] != y.shape[i] || x.shape[i] != z.shape[i]) {
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("shape mismatch");
    }
  }

  // Get bigarray kind from the data field
  value v_x_data = Field(v_x, FFI_TENSOR_DATA);
  value v_y_data = Field(v_y, FFI_TENSOR_DATA);
  value v_z_data = Field(v_z, FFI_TENSOR_DATA);

  struct caml_ba_array *ba = Caml_ba_array_val(v_x_data);
  int kind = nx_ba_get_kind(ba);

  // Check kinds match for y and z
  int kind_y = nx_ba_get_kind(Caml_ba_array_val(v_y_data));
  int kind_z = nx_ba_get_kind(Caml_ba_array_val(v_z_data));
  if (kind != kind_y || kind != kind_z) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("dtype mismatch");
  }

  // Select operation based on dtype
  binary_op_t op = NULL;
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
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("dispatch_binary_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith(msg);
  }

  // Enter blocking section for potentially long computation
  caml_enter_blocking_section();
  op(&x, &y, &z);
  caml_leave_blocking_section();

  // Clean up if heap allocated
  cleanup_ndarray(&x);
  cleanup_ndarray(&y);
  cleanup_ndarray(&z);
}

// Generic dispatch function for comparison operations (output is always bool)
static void dispatch_comparison_op(value v_x, value v_y, value v_z,
                                   const binary_op_table *table,
                                   const char *op_name) {
  // Extract ndarrays from FFI tensors
  ndarray_t x = extract_ndarray(v_x);
  ndarray_t y = extract_ndarray(v_y);
  ndarray_t z = extract_ndarray(v_z);

  // Check shapes match
  if (x.ndim != y.ndim || x.ndim != z.ndim) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("shape mismatch");
  }
  for (int i = 0; i < x.ndim; i++) {
    if (x.shape[i] != y.shape[i] || x.shape[i] != z.shape[i]) {
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("shape mismatch");
    }
  }

  // Get bigarray kind from the data field
  value v_x_data = Field(v_x, FFI_TENSOR_DATA);
  value v_y_data = Field(v_y, FFI_TENSOR_DATA);
  value v_z_data = Field(v_z, FFI_TENSOR_DATA);

  struct caml_ba_array *ba = Caml_ba_array_val(v_x_data);
  int kind = nx_ba_get_kind(ba);

  // Check input kinds match
  int kind_y = nx_ba_get_kind(Caml_ba_array_val(v_y_data));
  if (kind != kind_y) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("dtype mismatch: comparison inputs must have same dtype");
  }
  
  // Check output is uint8
  int kind_z = nx_ba_get_kind(Caml_ba_array_val(v_z_data));
  if (kind_z != NX_BA_BOOL) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("dtype mismatch: comparison output must be bool");
  }

  // Select operation based on input dtype
  binary_op_t op = NULL;
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
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("dispatch_comparison_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith(msg);
  }

  // Enter blocking section for potentially long computation
  caml_enter_blocking_section();
  op(&x, &y, &z);
  caml_leave_blocking_section();

  // Clean up if heap allocated
  cleanup_ndarray(&x);
  cleanup_ndarray(&y);
  cleanup_ndarray(&z);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

// Macro to define FFI stub for each operation
#define DEFINE_FFI_STUB(name)                                      \
  CAMLprim value caml_nx_##name(value v_x, value v_y, value v_z) { \
    CAMLparam3(v_x, v_y, v_z);                                     \
    dispatch_binary_op(v_x, v_y, v_z, &name##_table, #name);       \
    CAMLreturn(Val_unit);                                          \
  }

// Macro to define FFI stub for comparison operations
#define DEFINE_CMP_FFI_STUB(name)                                  \
  CAMLprim value caml_nx_##name(value v_x, value v_y, value v_z) { \
    CAMLparam3(v_x, v_y, v_z);                                     \
    dispatch_comparison_op(v_x, v_y, v_z, &name##_table, #name);   \
    CAMLreturn(Val_unit);                                          \
  }

DEFINE_FFI_STUB(add)
DEFINE_FFI_STUB(sub)
DEFINE_FFI_STUB(mul)
DEFINE_FFI_STUB(idiv)
DEFINE_FFI_STUB(fdiv)
DEFINE_FFI_STUB(max)
DEFINE_FFI_STUB(min)
DEFINE_FFI_STUB(mod)
DEFINE_FFI_STUB(pow)
DEFINE_CMP_FFI_STUB(cmpeq)
DEFINE_CMP_FFI_STUB(cmpne)
DEFINE_CMP_FFI_STUB(cmplt)
DEFINE_CMP_FFI_STUB(cmple)
DEFINE_FFI_STUB(xor)
DEFINE_FFI_STUB(or)
DEFINE_FFI_STUB(and)
