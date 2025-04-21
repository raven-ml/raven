#include <metal_stdlib>
using namespace metal;

// Maximum number of dimensions supported (consistent with binary)
#define MAX_DIMS 8

// Macro for contiguous unary operations
#define UNARY_OP_KERNEL_CONTIGUOUS(op_name, op_expr, dtype, dtype_name) \
  kernel void tensor_##op_name##_##dtype_name##_contiguous(             \
      device const dtype* a [[buffer(0)]],                              \
      constant uint& total_elements [[buffer(1)]],                      \
      device dtype* out [[buffer(2)]],                                  \
      uint thread_position_in_grid [[thread_position_in_grid]]) {       \
    uint linear_index = thread_position_in_grid;                        \
    if (linear_index < total_elements) {                                \
      out[linear_index] = op_expr;                                      \
    }                                                                   \
  }

// Macro for general unary operations (handles strides)
#define UNARY_OP_KERNEL_GENERAL(op_name, op_expr, dtype, dtype_name)           \
  kernel void tensor_##op_name##_##dtype_name##_general(                       \
      device const dtype* a [[buffer(0)]],                                     \
      constant uint& total_elements [[buffer(1)]],                             \
      device dtype* out [[buffer(2)]], constant uint& ndim [[buffer(3)]],      \
      constant uint* shape [[buffer(4)]], /* Use shape for indexing */         \
      constant long* a_strides [[buffer(5)]],                                  \
      uint thread_position_in_grid [[thread_position_in_grid]]) {              \
    uint linear_index = thread_position_in_grid;                               \
    if (linear_index < total_elements && ndim <= MAX_DIMS) {                   \
      long element_index_a = 0;                                                \
      uint temp = linear_index;                                                \
      /* Calculate input index based on strides */                             \
      /* Note: out is indexed linearly, a is indexed via strides */            \
      for (uint dim_idx = 0; dim_idx < ndim; ++dim_idx) {                      \
        /* Calculate multidimensional index from linear index */               \
        uint current_dim_size = shape[ndim - 1 - dim_idx];                     \
        /* Handle dim size 0 or 1 for safety, though total_elements check      \
         * helps                                                               \
         */                                                                    \
        if (current_dim_size == 0) break; /* Should not happen if total > 0 */ \
        uint index_in_dim =                                                    \
            (current_dim_size == 1) ? 0 : (temp % current_dim_size);           \
        temp /= (current_dim_size == 1) ? 1 : current_dim_size;                \
        /* Add stride contribution for this dimension */                       \
        element_index_a += (long)index_in_dim * a_strides[ndim - 1 - dim_idx]; \
      }                                                                        \
      /* op_expr now needs careful definition to work for both macros */       \
      /* We'll define VAL slightly differently */                              \
      out[linear_index] = op_expr;                                             \
    }                                                                          \
  }

// Operation expressions
// For CONTIGUOUS: VAL accesses a[linear_index]
// For GENERAL: VAL accesses a[element_index_a]
// We achieve this by defining VAL differently within the scope of each macro
// usage
#define NEG_EXPR(VAL) (-VAL)
#define ABS_EXPR(VAL) abs(VAL)
#define SIGN_EXPR(VAL) \
  (VAL > 0 ? 1 : (VAL < 0 ? -1 : 0))  // Standard sign definition
#define SQRT_EXPR(VAL) sqrt(VAL)
#define SQRT_EXPR_INT(VAL) ((int)sqrt((float)VAL))  // Cast for int sqrt
#define SQUARE_EXPR(VAL) (VAL * VAL)  // Added for Var/Std calculation
#define EXP_EXPR(VAL) exp(VAL)
#define LOG_EXPR(VAL) log(VAL)                    // Metal log is natural log
#define LOG_EXPR_INT(VAL) ((int)log((float)VAL))  // Cast for int log
#define SIN_EXPR(VAL) sin(VAL)
#define COS_EXPR(VAL) cos(VAL)
#define TAN_EXPR(VAL) tan(VAL)
#define ASIN_EXPR(VAL) asin(VAL)
#define ACOS_EXPR(VAL) acos(VAL)
#define ATAN_EXPR(VAL) atan(VAL)
#define SINH_EXPR(VAL) sinh(VAL)
#define COSH_EXPR(VAL) cosh(VAL)
#define TANH_EXPR(VAL) tanh(VAL)
#define ASINH_EXPR(VAL) asinh(VAL)
#define ACOSH_EXPR(VAL) acosh(VAL)
#define ATANH_EXPR(VAL) atanh(VAL)

// --- Generate Float Kernels ---
#define VAL_CONTIGUOUS a[linear_index]
#define VAL_GENERAL a[element_index_a]

UNARY_OP_KERNEL_CONTIGUOUS(neg, NEG_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(neg, NEG_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(abs, ABS_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(abs, ABS_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(sign, SIGN_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(sign, SIGN_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(sqrt, SQRT_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(sqrt, SQRT_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(square, SQUARE_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(square, SQUARE_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(exp, EXP_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(exp, EXP_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(log, LOG_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(log, LOG_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(sin, SIN_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(sin, SIN_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(cos, COS_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(cos, COS_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(tan, TAN_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(tan, TAN_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(asin, ASIN_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(asin, ASIN_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(acos, ACOS_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(acos, ACOS_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(atan, ATAN_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(atan, ATAN_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(sinh, SINH_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(sinh, SINH_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(cosh, COSH_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(cosh, COSH_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(tanh, TANH_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(tanh, TANH_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(asinh, ASINH_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(asinh, ASINH_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(acosh, ACOSH_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(acosh, ACOSH_EXPR(VAL_GENERAL), float, float32)
UNARY_OP_KERNEL_CONTIGUOUS(atanh, ATANH_EXPR(VAL_CONTIGUOUS), float, float32)
UNARY_OP_KERNEL_GENERAL(atanh, ATANH_EXPR(VAL_GENERAL), float, float32)

// --- Generate Int Kernels ---
UNARY_OP_KERNEL_CONTIGUOUS(neg, NEG_EXPR(VAL_CONTIGUOUS), int, int32)
UNARY_OP_KERNEL_GENERAL(neg, NEG_EXPR(VAL_GENERAL), int, int32)
UNARY_OP_KERNEL_CONTIGUOUS(abs, ABS_EXPR(VAL_CONTIGUOUS), int, int32)
UNARY_OP_KERNEL_GENERAL(abs, ABS_EXPR(VAL_GENERAL), int, int32)
UNARY_OP_KERNEL_CONTIGUOUS(sign, SIGN_EXPR(VAL_CONTIGUOUS), int, int32)
UNARY_OP_KERNEL_GENERAL(sign, SIGN_EXPR(VAL_GENERAL), int, int32)
UNARY_OP_KERNEL_CONTIGUOUS(square, SQUARE_EXPR(VAL_CONTIGUOUS), int, int32)
UNARY_OP_KERNEL_GENERAL(square, SQUARE_EXPR(VAL_GENERAL), int, int32)
// Add other int unary ops if needed (e.g., bitwise not)

#undef VAL_CONTIGUOUS
#undef VAL_GENERAL
