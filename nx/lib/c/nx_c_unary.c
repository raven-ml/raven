// Unary operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <complex.h>
#include <math.h>

#include "nx_c_shared.h"

// Type definitions for unary operations
typedef void (*unary_op_t)(const ndarray_t *, ndarray_t *);

// Dispatch table for each type
typedef struct {
  unary_op_t i8, u8, i16, u16, i32, i64, inat;
  unary_op_t f16, f32, f64;
  unary_op_t c32, c64;
  unary_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2, c16, qi8, qu8;
} unary_op_table;

// Macro to generate all standard type variants for an operation (ints and
// floats) Note: float16, bfloat16, fp8 types need special handling with
// conversion
#define GENERATE_UNARY_OP(name, OP_EXPR)               \
  UNARY_OP_FOR_TYPE(name, int8_t, i8, OP_EXPR)         \
  UNARY_OP_FOR_TYPE(name, uint8_t, u8, OP_EXPR)        \
  UNARY_OP_FOR_TYPE(name, int16_t, i16, OP_EXPR)       \
  UNARY_OP_FOR_TYPE(name, uint16_t, u16, OP_EXPR)      \
  UNARY_OP_FOR_TYPE(name, int32_t, i32, OP_EXPR)       \
  UNARY_OP_FOR_TYPE(name, int64_t, i64, OP_EXPR)       \
  UNARY_OP_FOR_TYPE(name, intnat, inat, OP_EXPR)       \
  UNARY_OP_FOR_TYPE(name, float, f32, OP_EXPR)         \
  UNARY_OP_FOR_TYPE(name, double, f64, OP_EXPR)        \
  UNARY_OP_FOR_TYPE(name, caml_ba_qint8, qi8, OP_EXPR) \
  UNARY_OP_FOR_TYPE(name, caml_ba_quint8, qu8, OP_EXPR)

// Macro to generate floating-point only variants
#define GENERATE_UNARY_FLOAT_OP(name, OP_FLOAT, OP_DOUBLE) \
  UNARY_OP_FOR_TYPE(name, float, f32, OP_FLOAT)            \
  UNARY_OP_FOR_TYPE(name, double, f64, OP_DOUBLE)

// Macro to build dispatch table
#define BUILD_DISPATCH_TABLE(name)                                            \
  static const unary_op_table name##_table = {.i8 = nx_c_##name##_i8,         \
                                              .u8 = nx_c_##name##_u8,         \
                                              .i16 = nx_c_##name##_i16,       \
                                              .u16 = nx_c_##name##_u16,       \
                                              .i32 = nx_c_##name##_i32,       \
                                              .i64 = nx_c_##name##_i64,       \
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
                                              .f8e5m2 = nx_c_##name##_f8e5m2, \
                                              .c16 = nx_c_##name##_c16,       \
                                              .qi8 = nx_c_##name##_qi8,       \
                                              .qu8 = nx_c_##name##_qu8}

// Helper function to iterate over inner dimensions for unary operations
typedef void (*unary_kernel_fn)(void *, void *, long, long);

static inline void iterate_inner_dims_unary(const ndarray_t *x,
                                            const ndarray_t *z, long outer_idx,
                                            unary_kernel_fn kernel,
                                            void *x_data, void *z_data) {
  if (x->ndim <= 1) {
    kernel(x_data, z_data, outer_idx * x->strides[0],
           outer_idx * z->strides[0]);
    return;
  }

  long x_base = outer_idx * x->strides[0];
  long z_base = outer_idx * z->strides[0];

  // Create temporary iterator for inner dimensions
  int inner_ndim = x->ndim - 1;
  int *coords = (int *)calloc(inner_ndim, sizeof(int));
  if (!coords) {
    caml_failwith("iterate_inner_dims_unary: allocation failed");
  }

  // Iterate over inner dimensions
  bool done = false;
  while (!done) {
    long x_off = x_base;
    long z_off = z_base;

    for (int i = 0; i < inner_ndim; i++) {
      x_off += coords[i] * x->strides[i + 1];
      z_off += coords[i] * z->strides[i + 1];
    }

    kernel(x_data, z_data, x_off, z_off);

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

  free(coords);
}

// Generic unary operation kernel
#define UNARY_OP_KERNEL(name, T, suffix, OP)                              \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *z_data, \
                                              long x_off, long z_off) {   \
    T *x = (T *)x_data;                                                   \
    T *z = (T *)z_data;                                                   \
    z[z_off] = OP(x[x_off]);                                              \
  }

// Generic unary operation implementation
#define UNARY_OP_IMPL(name, T, suffix)                                         \
  static void nx_c_##name##_##suffix(const ndarray_t *x, ndarray_t *z) {       \
    if (!x || !z) {                                                            \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    long total = total_elements_safe(x);                                       \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_contiguous(x) && is_contiguous(z)) {                                \
      _Pragma("omp parallel for simd if(total > 1000)") for (long i = 0;       \
                                                             i < total; i++) { \
        nx_c_##name##_##suffix##_kernel(x->data, z->data, x->offset + i,       \
                                        z->offset + i);                        \
      }                                                                        \
    } else if (x->shape[0] > 1 && total / x->shape[0] > 50) {                  \
      _Pragma("omp parallel for if(x->shape[0] > 4)") for (long i = 0;         \
                                                           i < x->shape[0];    \
                                                           i++) {              \
        iterate_inner_dims_unary(x, z, i, nx_c_##name##_##suffix##_kernel,     \
                                 x->data, z->data);                            \
      }                                                                        \
    } else {                                                                   \
      nd_copy_iterator_t it;                                                   \
      nd_copy_iterator_init(&it, x, z);                                        \
      do {                                                                     \
        long x_off, z_off;                                                     \
        nd_copy_iterator_get_offsets(&it, &x_off, &z_off);                     \
        nx_c_##name##_##suffix##_kernel(x->data, z->data, x->offset + x_off,   \
                                        z->offset + z_off);                    \
      } while (nd_copy_iterator_next(&it));                                    \
      nd_copy_iterator_destroy(&it);                                           \
    }                                                                          \
  }

// Macro to generate both kernel and implementation for an operation
#define UNARY_OP_FOR_TYPE(name, T, suffix, OP) \
  UNARY_OP_KERNEL(name, T, suffix, OP)         \
  UNARY_OP_IMPL(name, T, suffix)

// Low-precision float kernel (convert to float for op)
#define LOW_PREC_OP_KERNEL(name, T, suffix, OP, TO_FLOAT, FROM_FLOAT)     \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *z_data, \
                                              long x_off, long z_off) {   \
    T *x = (T *)x_data;                                                   \
    T *z = (T *)z_data;                                                   \
    float a = TO_FLOAT(x[x_off]);                                         \
    z[z_off] = FROM_FLOAT(OP(a));                                         \
  }

// For low-precision, use the impl with the special kernel
#define LOW_PREC_OP_IMPL(name, T, suffix) UNARY_OP_IMPL(name, T, suffix)

// Complex16 operations using conversion approach
#define COMPLEX16_OP_KERNEL(name, OP)                                          \
  static void nx_c_##name##_c16_kernel(void *x_data, void *z_data, long x_off, \
                                       long z_off) {                           \
    caml_ba_complex16 *x = (caml_ba_complex16 *)x_data;                        \
    caml_ba_complex16 *z = (caml_ba_complex16 *)z_data;                        \
    complex32 a = half_to_float(x[x_off].re) + I * half_to_float(x[x_off].im); \
    complex32 res = OP(a);                                                     \
    z[z_off].re = float_to_half(crealf(res));                                  \
    z[z_off].im = float_to_half(cimagf(res));                                  \
  }

// Helper macros for int4 saturation
#define CLAMP_I4(x) ((x) < -8 ? -8 : ((x) > 7 ? 7 : (x)))
#define CLAMP_U4(x) ((x) < 0 ? 0 : ((x) > 15 ? 15 : (x)))

// Special implementation for int4 (packed, unpack/op/pack with saturation)
#define INT4_UNARY_IMPL(name, signedness, suffix, OP)                        \
  static void nx_c_##name##_##suffix##_kernel(void *x_data, void *z_data,    \
                                              long x_off, long z_off) {      \
    uint8_t *x = (uint8_t *)x_data;                                          \
    uint8_t *z = (uint8_t *)z_data;                                          \
    long byte_off = x_off / 2;                                               \
    int nib_off = x_off % 2;                                                 \
    int a = nib_off ? (signedness ? (int8_t)(x[byte_off] >> 4)               \
                                  : (x[byte_off] >> 4) & 0x0F)               \
                    : (signedness ? (int8_t)((x[byte_off] & 0x0F) << 4) >> 4 \
                                  : x[byte_off] & 0x0F);                     \
    int res = OP(a);                                                         \
    /* Saturate to 4-bit range */                                            \
    res = signedness ? CLAMP_I4(res) : CLAMP_U4(res);                        \
    uint8_t nib = (uint8_t)res & 0x0F;                                       \
    if (nib_off) {                                                           \
      z[byte_off] = (z[byte_off] & 0x0F) | (nib << 4);                       \
    } else {                                                                 \
      z[byte_off] = (z[byte_off] & 0xF0) | nib;                              \
    }                                                                        \
  }                                                                          \
  static void nx_c_##name##_##suffix(const ndarray_t *x, ndarray_t *z) {     \
    if (is_contiguous(x) && is_contiguous(z)) {                              \
      long total = total_elements_safe(x);                                   \
      void *x_data = x->data + x->offset;                                    \
      void *z_data = z->data + z->offset;                                    \
      _Pragma("omp parallel for if(total > 10000)") for (long i = 0;         \
                                                         i < total; i++) {   \
        nx_c_##name##_##suffix##_kernel(x_data, z_data, i, i);               \
      }                                                                      \
    } else {                                                                 \
      nd_copy_iterator_t it;                                                 \
      nd_copy_iterator_init(&it, x, z);                                      \
      void *x_data = x->data;                                                \
      void *z_data = z->data;                                                \
      do {                                                                   \
        long x_off, z_off;                                                   \
        nd_copy_iterator_get_offsets(&it, &x_off, &z_off);                   \
        nx_c_##name##_##suffix##_kernel(x_data, z_data, x_off + x->offset,   \
                                        z_off + z->offset);                  \
      } while (nd_copy_iterator_next(&it));                                  \
      nd_copy_iterator_destroy(&it);                                         \
    }                                                                        \
  }

// Generate for all ops
// Negation
#define NEG_OP(x) (-(x))
#define NEG_BOOL_OP(x) (!(x))
GENERATE_UNARY_OP(neg, NEG_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(neg, uint16_t, f16, NEG_OP, half_to_float, float_to_half)
LOW_PREC_OP_IMPL(neg, uint16_t, f16)
LOW_PREC_OP_KERNEL(neg, caml_ba_bfloat16, bf16, NEG_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(neg, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(neg, caml_ba_fp8_e4m3, f8e4m3, NEG_OP, fp8_e4m3_to_float,
                   float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(neg, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(neg, caml_ba_fp8_e5m2, f8e5m2, NEG_OP, fp8_e5m2_to_float,
                   float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(neg, caml_ba_fp8_e5m2, f8e5m2)

UNARY_OP_FOR_TYPE(neg, complex32, c32, NEG_OP)
UNARY_OP_FOR_TYPE(neg, complex64, c64, NEG_OP)

COMPLEX16_OP_KERNEL(neg, NEG_OP)
UNARY_OP_IMPL(neg, caml_ba_complex16, c16)
INT4_UNARY_IMPL(neg, 1, i4, NEG_OP)
INT4_UNARY_IMPL(neg, 0, u4, NEG_OP)
UNARY_OP_FOR_TYPE(neg, caml_ba_bool, bool_, NEG_BOOL_OP)
BUILD_DISPATCH_TABLE(neg);

// Log2 (floating-point and complex only)
#define LOG2_FLOAT_OP(x) (log2f(x))
#define LOG2_DOUBLE_OP(x) (log2(x))
#define COMPLEX32_LOG2_OP(x) (clogf(x) / logf(2.0f))
#define COMPLEX64_LOG2_OP(x) (clog(x) / log(2.0))
GENERATE_UNARY_FLOAT_OP(log2, LOG2_FLOAT_OP, LOG2_DOUBLE_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(log2, uint16_t, f16, LOG2_FLOAT_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(log2, uint16_t, f16)
LOW_PREC_OP_KERNEL(log2, caml_ba_bfloat16, bf16, LOG2_FLOAT_OP,
                   bfloat16_to_float, float_to_bfloat16)
LOW_PREC_OP_IMPL(log2, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(log2, caml_ba_fp8_e4m3, f8e4m3, LOG2_FLOAT_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(log2, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(log2, caml_ba_fp8_e5m2, f8e5m2, LOG2_FLOAT_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(log2, caml_ba_fp8_e5m2, f8e5m2)

UNARY_OP_FOR_TYPE(log2, complex32, c32, COMPLEX32_LOG2_OP)
UNARY_OP_FOR_TYPE(log2, complex64, c64, COMPLEX64_LOG2_OP)

COMPLEX16_OP_KERNEL(log2, COMPLEX32_LOG2_OP)
UNARY_OP_IMPL(log2, caml_ba_complex16, c16)

// Build dispatch table with only float types (integers not supported)
static const unary_op_table log2_table = {.i8 = NULL,
                                          .u8 = NULL,
                                          .i16 = NULL,
                                          .u16 = NULL,
                                          .i32 = NULL,
                                          .i64 = NULL,
                                          .inat = NULL,
                                          .f16 = nx_c_log2_f16,
                                          .f32 = nx_c_log2_f32,
                                          .f64 = nx_c_log2_f64,
                                          .c32 = nx_c_log2_c32,
                                          .c64 = nx_c_log2_c64,
                                          .bf16 = nx_c_log2_bf16,
                                          .bool_ = NULL,
                                          .i4 = NULL,
                                          .u4 = NULL,
                                          .f8e4m3 = nx_c_log2_f8e4m3,
                                          .f8e5m2 = nx_c_log2_f8e5m2,
                                          .c16 = nx_c_log2_c16,
                                          .qi8 = NULL,
                                          .qu8 = NULL};

// Exp2 (floating-point and complex only)
#define EXP2_FLOAT_OP(x) (exp2f(x))
#define EXP2_DOUBLE_OP(x) (exp2(x))
#define COMPLEX32_EXP2_OP(x) (cpowf(2.0f, x))
#define COMPLEX64_EXP2_OP(x) (cpow(2.0, x))
GENERATE_UNARY_FLOAT_OP(exp2, EXP2_FLOAT_OP, EXP2_DOUBLE_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(exp2, uint16_t, f16, EXP2_FLOAT_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(exp2, uint16_t, f16)
LOW_PREC_OP_KERNEL(exp2, caml_ba_bfloat16, bf16, EXP2_FLOAT_OP,
                   bfloat16_to_float, float_to_bfloat16)
LOW_PREC_OP_IMPL(exp2, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(exp2, caml_ba_fp8_e4m3, f8e4m3, EXP2_FLOAT_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(exp2, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(exp2, caml_ba_fp8_e5m2, f8e5m2, EXP2_FLOAT_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(exp2, caml_ba_fp8_e5m2, f8e5m2)

UNARY_OP_FOR_TYPE(exp2, complex32, c32, COMPLEX32_EXP2_OP)
UNARY_OP_FOR_TYPE(exp2, complex64, c64, COMPLEX64_EXP2_OP)

COMPLEX16_OP_KERNEL(exp2, COMPLEX32_EXP2_OP)
UNARY_OP_IMPL(exp2, caml_ba_complex16, c16)

// Build dispatch table with only float types (integers not supported)
static const unary_op_table exp2_table = {.i8 = NULL,
                                          .u8 = NULL,
                                          .i16 = NULL,
                                          .u16 = NULL,
                                          .i32 = NULL,
                                          .i64 = NULL,
                                          .inat = NULL,
                                          .f16 = nx_c_exp2_f16,
                                          .f32 = nx_c_exp2_f32,
                                          .f64 = nx_c_exp2_f64,
                                          .c32 = nx_c_exp2_c32,
                                          .c64 = nx_c_exp2_c64,
                                          .bf16 = nx_c_exp2_bf16,
                                          .bool_ = NULL,
                                          .i4 = NULL,
                                          .u4 = NULL,
                                          .f8e4m3 = nx_c_exp2_f8e4m3,
                                          .f8e5m2 = nx_c_exp2_f8e5m2,
                                          .c16 = nx_c_exp2_c16,
                                          .qi8 = NULL,
                                          .qu8 = NULL};

// Sin (floating-point and complex only)
#define SIN_FLOAT_OP(x) (sinf(x))
#define SIN_DOUBLE_OP(x) (sin(x))
#define COMPLEX32_SIN_OP(x) (csinf(x))
#define COMPLEX64_SIN_OP(x) (csin(x))
GENERATE_UNARY_FLOAT_OP(sin, SIN_FLOAT_OP, SIN_DOUBLE_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(sin, uint16_t, f16, SIN_FLOAT_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(sin, uint16_t, f16)
LOW_PREC_OP_KERNEL(sin, caml_ba_bfloat16, bf16, SIN_FLOAT_OP, bfloat16_to_float,
                   float_to_bfloat16)
LOW_PREC_OP_IMPL(sin, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(sin, caml_ba_fp8_e4m3, f8e4m3, SIN_FLOAT_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(sin, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(sin, caml_ba_fp8_e5m2, f8e5m2, SIN_FLOAT_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(sin, caml_ba_fp8_e5m2, f8e5m2)

UNARY_OP_FOR_TYPE(sin, complex32, c32, COMPLEX32_SIN_OP)
UNARY_OP_FOR_TYPE(sin, complex64, c64, COMPLEX64_SIN_OP)

COMPLEX16_OP_KERNEL(sin, COMPLEX32_SIN_OP)
UNARY_OP_IMPL(sin, caml_ba_complex16, c16)

// Build dispatch table with only float types (integers not supported)
static const unary_op_table sin_table = {.i8 = NULL,
                                         .u8 = NULL,
                                         .i16 = NULL,
                                         .u16 = NULL,
                                         .i32 = NULL,
                                         .i64 = NULL,
                                         .inat = NULL,
                                         .f16 = nx_c_sin_f16,
                                         .f32 = nx_c_sin_f32,
                                         .f64 = nx_c_sin_f64,
                                         .c32 = nx_c_sin_c32,
                                         .c64 = nx_c_sin_c64,
                                         .bf16 = nx_c_sin_bf16,
                                         .bool_ = NULL,
                                         .i4 = NULL,
                                         .u4 = NULL,
                                         .f8e4m3 = nx_c_sin_f8e4m3,
                                         .f8e5m2 = nx_c_sin_f8e5m2,
                                         .c16 = nx_c_sin_c16,
                                         .qi8 = NULL,
                                         .qu8 = NULL};

// Sqrt (floating-point and complex only)
#define SQRT_FLOAT_OP(x) (sqrtf(x))
#define SQRT_DOUBLE_OP(x) (sqrt(x))
#define COMPLEX32_SQRT_OP(x) (csqrtf(x))
#define COMPLEX64_SQRT_OP(x) (csqrt(x))
GENERATE_UNARY_FLOAT_OP(sqrt, SQRT_FLOAT_OP, SQRT_DOUBLE_OP)

// Float16, BFloat16, FP8 variants need conversion
LOW_PREC_OP_KERNEL(sqrt, uint16_t, f16, SQRT_FLOAT_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(sqrt, uint16_t, f16)
LOW_PREC_OP_KERNEL(sqrt, caml_ba_bfloat16, bf16, SQRT_FLOAT_OP,
                   bfloat16_to_float, float_to_bfloat16)
LOW_PREC_OP_IMPL(sqrt, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(sqrt, caml_ba_fp8_e4m3, f8e4m3, SQRT_FLOAT_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(sqrt, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(sqrt, caml_ba_fp8_e5m2, f8e5m2, SQRT_FLOAT_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(sqrt, caml_ba_fp8_e5m2, f8e5m2)

UNARY_OP_FOR_TYPE(sqrt, complex32, c32, COMPLEX32_SQRT_OP)
UNARY_OP_FOR_TYPE(sqrt, complex64, c64, COMPLEX64_SQRT_OP)

COMPLEX16_OP_KERNEL(sqrt, COMPLEX32_SQRT_OP)
UNARY_OP_IMPL(sqrt, caml_ba_complex16, c16)

// Build dispatch table with only float types (integers not supported)
static const unary_op_table sqrt_table = {.i8 = NULL,
                                          .u8 = NULL,
                                          .i16 = NULL,
                                          .u16 = NULL,
                                          .i32 = NULL,
                                          .i64 = NULL,
                                          .inat = NULL,
                                          .f16 = nx_c_sqrt_f16,
                                          .f32 = nx_c_sqrt_f32,
                                          .f64 = nx_c_sqrt_f64,
                                          .c32 = nx_c_sqrt_c32,
                                          .c64 = nx_c_sqrt_c64,
                                          .bf16 = nx_c_sqrt_bf16,
                                          .bool_ = NULL,
                                          .i4 = NULL,
                                          .u4 = NULL,
                                          .f8e4m3 = nx_c_sqrt_f8e4m3,
                                          .f8e5m2 = nx_c_sqrt_f8e5m2,
                                          .c16 = nx_c_sqrt_c16,
                                          .qi8 = NULL,
                                          .qu8 = NULL};

// Reciprocal - separate handling for integers (check zero) vs floats (IEEE 754)
#define INT_RECIP_OP(x) \
  ((x) == 0 ? (caml_failwith("division by zero"), (x)) : (1 / (x)))
#define FLOAT_RECIP_OP(x) (1 / (x))
#define COMPLEX32_RECIP_OP(x) (1.0f / (x))
#define COMPLEX64_RECIP_OP(x) (1.0 / (x))

// Integer types need zero check
UNARY_OP_FOR_TYPE(recip, int8_t, i8, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, uint8_t, u8, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, int16_t, i16, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, uint16_t, u16, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, int32_t, i32, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, int64_t, i64, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, intnat, inat, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, caml_ba_qint8, qi8, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, caml_ba_quint8, qu8, INT_RECIP_OP)

// Floating-point types use IEEE 754 semantics (no zero check)
UNARY_OP_FOR_TYPE(recip, float, f32, FLOAT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, double, f64, FLOAT_RECIP_OP)

// Float16, BFloat16, FP8 variants - no zero check, let IEEE semantics apply
LOW_PREC_OP_KERNEL(recip, uint16_t, f16, FLOAT_RECIP_OP, half_to_float,
                   float_to_half)
LOW_PREC_OP_IMPL(recip, uint16_t, f16)
LOW_PREC_OP_KERNEL(recip, caml_ba_bfloat16, bf16, FLOAT_RECIP_OP,
                   bfloat16_to_float, float_to_bfloat16)
LOW_PREC_OP_IMPL(recip, caml_ba_bfloat16, bf16)
LOW_PREC_OP_KERNEL(recip, caml_ba_fp8_e4m3, f8e4m3, FLOAT_RECIP_OP,
                   fp8_e4m3_to_float, float_to_fp8_e4m3)
LOW_PREC_OP_IMPL(recip, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_OP_KERNEL(recip, caml_ba_fp8_e5m2, f8e5m2, FLOAT_RECIP_OP,
                   fp8_e5m2_to_float, float_to_fp8_e5m2)
LOW_PREC_OP_IMPL(recip, caml_ba_fp8_e5m2, f8e5m2)

UNARY_OP_FOR_TYPE(recip, complex32, c32, COMPLEX32_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, complex64, c64, COMPLEX64_RECIP_OP)

COMPLEX16_OP_KERNEL(recip, COMPLEX32_RECIP_OP)
UNARY_OP_IMPL(recip, caml_ba_complex16, c16)
INT4_UNARY_IMPL(recip, 1, i4, INT_RECIP_OP)
INT4_UNARY_IMPL(recip, 0, u4, INT_RECIP_OP)
UNARY_OP_FOR_TYPE(recip, caml_ba_bool, bool_, INT_RECIP_OP)
BUILD_DISPATCH_TABLE(recip);

// Shared dispatch infrastructure

// Generic dispatch function for unary operations
static void dispatch_unary_op(value v_x, value v_z, const unary_op_table *table,
                              const char *op_name) {
  // Extract ndarrays from FFI tensors
  ndarray_t x = extract_ndarray(v_x);
  ndarray_t z = extract_ndarray(v_z);

  // Check shapes match
  if (x.ndim != z.ndim) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&z);
    caml_failwith("shape mismatch");
  }
  for (int i = 0; i < x.ndim; i++) {
    if (x.shape[i] != z.shape[i]) {
      cleanup_ndarray(&x);
      cleanup_ndarray(&z);
      caml_failwith("shape mismatch");
    }
  }

  // Get bigarray kind from the data field
  value v_x_data = Field(v_x, FFI_TENSOR_DATA);
  value v_z_data = Field(v_z, FFI_TENSOR_DATA);

  struct caml_ba_array *ba = Caml_ba_array_val(v_x_data);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  // Check kinds match for z
  int kind_z = Caml_ba_array_val(v_z_data)->flags & CAML_BA_KIND_MASK;
  if (kind != kind_z) {
    cleanup_ndarray(&x);
    cleanup_ndarray(&z);
    caml_failwith("dtype mismatch");
  }

  // Select operation based on dtype
  unary_op_t op = NULL;
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
    case NX_BA_COMPLEX16:
      op = table->c16;
      break;
    case NX_BA_QINT8:
      op = table->qi8;
      break;
    case NX_BA_QUINT8:
      op = table->qu8;
      break;
    default:
      cleanup_ndarray(&x);
      cleanup_ndarray(&z);
      caml_failwith("dispatch_unary_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    cleanup_ndarray(&x);
    cleanup_ndarray(&z);
    caml_failwith(msg);
  }

  // Enter blocking section for potentially long computation
  caml_enter_blocking_section();
  op(&x, &z);
  caml_leave_blocking_section();

  // Clean up if heap allocated
  cleanup_ndarray(&x);
  cleanup_ndarray(&z);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

// Macro to define FFI stub for each operation
#define DEFINE_FFI_STUB(name)                           \
  CAMLprim value caml_nx_##name(value v_x, value v_z) { \
    CAMLparam2(v_x, v_z);                               \
    dispatch_unary_op(v_x, v_z, &name##_table, #name);  \
    CAMLreturn(Val_unit);                               \
  }

DEFINE_FFI_STUB(neg)
DEFINE_FFI_STUB(log2)
DEFINE_FFI_STUB(exp2)
DEFINE_FFI_STUB(sin)
DEFINE_FFI_STUB(sqrt)
DEFINE_FFI_STUB(recip)
