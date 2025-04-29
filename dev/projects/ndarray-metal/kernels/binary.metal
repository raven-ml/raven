#include <metal_stdlib>
using namespace metal;

// Maximum number of dimensions supported
#define MAX_DIMS 8

// Macro for general broadcasting binary operations (output dtype is same as
// input)
#define BINARY_OP_KERNEL_GENERAL(op_name, op_expr, dtype, dtype_name)          \
  kernel void tensor_##op_name##_##dtype_name##_general(                       \
      device const dtype* a [[buffer(0)]],                                     \
      device const dtype* b [[buffer(1)]],                                     \
      constant uint& total_elements [[buffer(2)]],                             \
      device dtype* out [[buffer(3)]], constant uint& ndim [[buffer(4)]],      \
      constant uint* out_shape [[buffer(5)]],                                  \
      constant long* a_strides [[buffer(6)]],                                  \
      constant long* b_strides [[buffer(7)]],                                  \
      uint thread_position_in_grid [[thread_position_in_grid]]) {              \
    uint linear_index = thread_position_in_grid;                               \
    if (linear_index < total_elements && ndim <= MAX_DIMS) {                   \
      long element_index_a = 0;                                                \
      long element_index_b = 0;                                                \
      uint temp = linear_index;                                                \
      /* Calculate indices using strides (your existing logic) */              \
      for (uint dim_idx = 0; dim_idx < ndim; ++dim_idx) {                      \
        uint current_dim_size = out_shape[ndim - 1 - dim_idx];                 \
        uint index_in_dim = temp % current_dim_size;                           \
        temp /= current_dim_size;                                              \
        element_index_a += (long)index_in_dim * a_strides[ndim - 1 - dim_idx]; \
        element_index_b += (long)index_in_dim * b_strides[ndim - 1 - dim_idx]; \
      }                                                                        \
      /* Fetch operands using calculated indices */                            \
      dtype operand_a = a[element_index_a];                                    \
      dtype operand_b = b[element_index_b];                                    \
      /* Perform the operation using op_expr which uses operand_a/b */         \
      out[linear_index] = op_expr;                                             \
    }                                                                          \
  }

// Macro for contiguous binary operations (output dtype is same as input)
#define BINARY_OP_KERNEL_CONTIGUOUS(op_name, op_expr, dtype, dtype_name) \
  kernel void tensor_##op_name##_##dtype_name##_contiguous(              \
      device const dtype* a [[buffer(0)]],                               \
      device const dtype* b [[buffer(1)]],                               \
      constant uint& total_elements [[buffer(2)]],                       \
      device dtype* out [[buffer(3)]],                                   \
      uint thread_position_in_grid [[thread_position_in_grid]]) {        \
    uint linear_index = thread_position_in_grid;                         \
    if (linear_index < total_elements) {                                 \
      /* Fetch operands using linear index */                            \
      dtype operand_a = a[linear_index];                                 \
      dtype operand_b = b[linear_index];                                 \
      /* Perform the operation using op_expr which uses operand_a/b */   \
      out[linear_index] = op_expr;                                       \
    }                                                                    \
  }

// Operation expressions (defined in terms of operand_a and operand_b)
#define ADD_EXPR (operand_a + operand_b)
#define SUB_EXPR (operand_a - operand_b)
#define MUL_EXPR (operand_a * operand_b)
#define DIV_EXPR (operand_a / operand_b)
#define POW_EXPR pow(operand_a, operand_b)
#define POW_EXPR_INT \
  ((int)pow((float)operand_a, (float)operand_b))  // Cast to float for pow
#define MAX_EXPR max(operand_a, operand_b)
#define MIN_EXPR min(operand_a, operand_b)
// Rely on implicit bool -> number conversion (true->1/1.0, false->0/0.0)
#define EQUAL_EXPR (operand_a == operand_b)
#define GREATER_EXPR (operand_a > operand_b)
#define GREATER_EQUAL_EXPR (operand_a >= operand_b)
#define LESS_EXPR (operand_a < operand_b)
#define LESS_EQUAL_EXPR (operand_a <= operand_b)
#define REMAINDER_EXPR (operand_a % operand_b)
#define REMAINDER_EXPR_FLOAT fmod(operand_a, operand_b)

// Generate kernels for float (output type is float)
BINARY_OP_KERNEL_GENERAL(add, ADD_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(add, ADD_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(sub, SUB_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(sub, SUB_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(mul, MUL_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(mul, MUL_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(div, DIV_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(div, DIV_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(pow, POW_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(pow, POW_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(remainder, REMAINDER_EXPR_FLOAT, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(remainder, REMAINDER_EXPR_FLOAT, float, float32)
BINARY_OP_KERNEL_GENERAL(max, MAX_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(max, MAX_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(min, MIN_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(min, MIN_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(equal, EQUAL_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(equal, EQUAL_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(greater, GREATER_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(greater, GREATER_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(greater_equal, GREATER_EQUAL_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(greater_equal, GREATER_EQUAL_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(less, LESS_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(less, LESS_EXPR, float, float32)
BINARY_OP_KERNEL_GENERAL(less_equal, LESS_EQUAL_EXPR, float, float32)
BINARY_OP_KERNEL_CONTIGUOUS(less_equal, LESS_EQUAL_EXPR, float, float32)

// Generate kernels for int (output type is int)
BINARY_OP_KERNEL_GENERAL(add, ADD_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(add, ADD_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(sub, SUB_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(sub, SUB_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(mul, MUL_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(mul, MUL_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(div, DIV_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(div, DIV_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(pow, POW_EXPR_INT, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(pow, POW_EXPR_INT, int, int32)
BINARY_OP_KERNEL_GENERAL(remainder, REMAINDER_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(remainder, REMAINDER_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(max, MAX_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(max, MAX_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(min, MIN_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(min, MIN_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(equal, EQUAL_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(equal, EQUAL_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(greater, GREATER_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(greater, GREATER_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(greater_equal, GREATER_EQUAL_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(greater_equal, GREATER_EQUAL_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(less, LESS_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(less, LESS_EXPR, int, int32)
BINARY_OP_KERNEL_GENERAL(less_equal, LESS_EQUAL_EXPR, int, int32)
BINARY_OP_KERNEL_CONTIGUOUS(less_equal, LESS_EQUAL_EXPR, int, int32)
