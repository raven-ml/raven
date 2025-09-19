#include "nx_c_shared.h"

typedef void (*scan_fn_t)(const ndarray_t *, ndarray_t *, int axis);

typedef struct {
  int axis_len;
  long axis_stride_in;
  long axis_stride_out;
  int outer_dims;
  int *outer_axes;
  int *outer_coord;
  const char *op_name;
} scan_plan_t;

static scan_plan_t scan_prepare(const ndarray_t *input, const ndarray_t *output,
                                int axis, const char *op_name) {
  scan_plan_t plan;
  plan.axis_len = 0;
  plan.axis_stride_in = 0;
  plan.axis_stride_out = 0;
  plan.outer_dims = 0;
  plan.outer_axes = NULL;
  plan.outer_coord = NULL;
  plan.op_name = op_name;

  if (!input || !output) {
    caml_failwith("associative_scan: null tensor");
  }

  if (input->ndim != output->ndim) {
    char msg[128];
    snprintf(msg, sizeof(msg), "%s: rank mismatch", op_name);
    caml_failwith(msg);
  }

  if (input->ndim <= 0) {
    char msg[128];
    snprintf(msg, sizeof(msg), "%s: tensor rank must be >= 1", op_name);
    caml_failwith(msg);
  }

  if (axis < 0 || axis >= input->ndim) {
    char msg[128];
    snprintf(msg, sizeof(msg), "%s: axis %d out of bounds for rank %d",
             op_name, axis, input->ndim);
    caml_failwith(msg);
  }

  for (int i = 0; i < input->ndim; ++i) {
    if (input->shape[i] != output->shape[i]) {
      char msg[128];
      snprintf(msg, sizeof(msg), "%s: shape mismatch on dim %d", op_name, i);
      caml_failwith(msg);
    }
  }

  plan.axis_len = input->shape[axis];
  plan.axis_stride_in = input->strides[axis];
  plan.axis_stride_out = output->strides[axis];
  plan.outer_dims = input->ndim - 1;

  if (plan.outer_dims > 0) {
    plan.outer_axes = (int *)malloc(plan.outer_dims * sizeof(int));
    plan.outer_coord = (int *)calloc(plan.outer_dims, sizeof(int));
    if (!plan.outer_axes || !plan.outer_coord) {
      if (plan.outer_axes) free(plan.outer_axes);
      if (plan.outer_coord) free(plan.outer_coord);
      caml_failwith("associative_scan: allocation failed");
    }
    int idx = 0;
    for (int i = 0; i < input->ndim; ++i) {
      if (i != axis) {
        plan.outer_axes[idx++] = i;
      }
    }
  }

  return plan;
}

static void scan_plan_destroy(scan_plan_t *plan) {
  if (plan->outer_axes) free(plan->outer_axes);
  if (plan->outer_coord) free(plan->outer_coord);
}

static bool advance_outer_coords(const ndarray_t *input, const int *outer_axes,
                                 int *outer_coord, int outer_dims) {
  if (outer_dims == 0) return false;
  for (int idx = outer_dims - 1; idx >= 0; --idx) {
    int axis = outer_axes[idx];
    outer_coord[idx]++;
    if (outer_coord[idx] < input->shape[axis]) {
      return true;
    }
    outer_coord[idx] = 0;
  }
  return false;
}

#define SUM_EXPR(acc, val) ((acc) + (val))
#define PROD_EXPR(acc, val) ((acc) * (val))
#define MAX_EXPR(acc, val) ((acc) > (val) ? (acc) : (val))
#define MIN_EXPR(acc, val) ((acc) < (val) ? (acc) : (val))
#define MAX_FLOAT_EXPR(acc, val)                                                \
  (isnan((double)(acc)) || isnan((double)(val))                                  \
       ? NAN                                                                     \
       : ((acc) > (val) ? (acc) : (val)))
#define MIN_FLOAT_EXPR(acc, val)                                                \
  (isnan((double)(acc)) || isnan((double)(val))                                  \
       ? NAN                                                                     \
       : ((acc) < (val) ? (acc) : (val)))
#define MAX_COMPLEX32_EXPR(acc, val) complex_max(acc, val)
#define MIN_COMPLEX32_EXPR(acc, val) complex_min(acc, val)
#define MAX_COMPLEX64_EXPR(acc, val) complex64_max(acc, val)
#define MIN_COMPLEX64_EXPR(acc, val) complex64_min(acc, val)

#define DEFINE_SCAN_DIRECT(OPNAME, TYPE, SUFFIX, ACC_EXPR)                      \
  static void nx_c_scan_##OPNAME##_##SUFFIX(const ndarray_t *input,            \
                                            ndarray_t *output, int axis) {     \
    scan_plan_t plan = scan_prepare(input, output, axis, "scan_" #OPNAME);     \
    if (plan.axis_len <= 0) {                                                  \
      scan_plan_destroy(&plan);                                                \
      return;                                                                  \
    }                                                                          \
    TYPE *in_data = (TYPE *)input->data;                                       \
    TYPE *out_data = (TYPE *)output->data;                                     \
    const int outer_dims = plan.outer_dims;                                    \
    const int *outer_axes = plan.outer_axes;                                   \
    int *outer_coord = plan.outer_coord;                                       \
    const long axis_stride_in = plan.axis_stride_in;                           \
    const long axis_stride_out = plan.axis_stride_out;                         \
    while (true) {                                                             \
      long in_base = input->offset;                                            \
      long out_base = output->offset;                                          \
      for (int i = 0; i < outer_dims; ++i) {                                  \
        int ax = outer_axes[i];                                                \
        long coord = outer_coord[i];                                           \
        in_base += coord * input->strides[ax];                                 \
        out_base += coord * output->strides[ax];                               \
      }                                                                        \
      long in_off = in_base;                                                   \
      long out_off = out_base;                                                 \
      TYPE acc = in_data[in_off];                                              \
      out_data[out_off] = acc;                                                 \
      for (int k = 1; k < plan.axis_len; ++k) {                                \
        in_off += axis_stride_in;                                              \
        out_off += axis_stride_out;                                            \
        TYPE val = in_data[in_off];                                            \
        acc = ACC_EXPR;                                                        \
        out_data[out_off] = acc;                                               \
      }                                                                        \
      if (outer_dims == 0) break;                                              \
      if (!advance_outer_coords(input, outer_axes, outer_coord, outer_dims))   \
        break;                                                                 \
    }                                                                          \
    scan_plan_destroy(&plan);                                                  \
  }

#define DEFINE_SCAN_LOW_PREC(OPNAME, STORAGE_TYPE, SUFFIX, ACC_EXPR, TO_FLOAT,  \
                             FROM_FLOAT)                                       \
  static void nx_c_scan_##OPNAME##_##SUFFIX(const ndarray_t *input,            \
                                            ndarray_t *output, int axis) {     \
    scan_plan_t plan = scan_prepare(input, output, axis, "scan_" #OPNAME);     \
    if (plan.axis_len <= 0) {                                                  \
      scan_plan_destroy(&plan);                                                \
      return;                                                                  \
    }                                                                          \
    STORAGE_TYPE *in_data = (STORAGE_TYPE *)input->data;                       \
    STORAGE_TYPE *out_data = (STORAGE_TYPE *)output->data;                     \
    const int outer_dims = plan.outer_dims;                                    \
    const int *outer_axes = plan.outer_axes;                                   \
    int *outer_coord = plan.outer_coord;                                       \
    const long axis_stride_in = plan.axis_stride_in;                           \
    const long axis_stride_out = plan.axis_stride_out;                         \
    while (true) {                                                             \
      long in_base = input->offset;                                            \
      long out_base = output->offset;                                          \
      for (int i = 0; i < outer_dims; ++i) {                                  \
        int ax = outer_axes[i];                                                \
        long coord = outer_coord[i];                                           \
        in_base += coord * input->strides[ax];                                 \
        out_base += coord * output->strides[ax];                               \
      }                                                                        \
      long in_off = in_base;                                                   \
      long out_off = out_base;                                                 \
      float acc = TO_FLOAT(in_data[in_off]);                                   \
      out_data[out_off] = FROM_FLOAT(acc);                                     \
      for (int k = 1; k < plan.axis_len; ++k) {                                \
        in_off += axis_stride_in;                                              \
        out_off += axis_stride_out;                                            \
        float val = TO_FLOAT(in_data[in_off]);                                  \
        acc = ACC_EXPR;                                                        \
        out_data[out_off] = FROM_FLOAT(acc);                                   \
      }                                                                        \
      if (outer_dims == 0) break;                                              \
      if (!advance_outer_coords(input, outer_axes, outer_coord, outer_dims))   \
        break;                                                                 \
    }                                                                          \
    scan_plan_destroy(&plan);                                                  \
  }

#define DEFINE_SCAN_COMPLEX16(OPNAME, ACC_EXPR)                                \
  static void nx_c_scan_##OPNAME##_c16(const ndarray_t *input,                \
                                       ndarray_t *output, int axis) {         \
    scan_plan_t plan = scan_prepare(input, output, axis, "scan_" #OPNAME);     \
    if (plan.axis_len <= 0) {                                                  \
      scan_plan_destroy(&plan);                                                \
      return;                                                                  \
    }                                                                          \
    caml_ba_complex16 *in_data = (caml_ba_complex16 *)input->data;            \
    caml_ba_complex16 *out_data = (caml_ba_complex16 *)output->data;          \
    const int outer_dims = plan.outer_dims;                                    \
    const int *outer_axes = plan.outer_axes;                                   \
    int *outer_coord = plan.outer_coord;                                       \
    const long axis_stride_in = plan.axis_stride_in;                           \
    const long axis_stride_out = plan.axis_stride_out;                         \
    while (true) {                                                             \
      long in_base = input->offset;                                            \
      long out_base = output->offset;                                          \
      for (int i = 0; i < outer_dims; ++i) {                                  \
        int ax = outer_axes[i];                                                \
        long coord = outer_coord[i];                                           \
        in_base += coord * input->strides[ax];                                 \
        out_base += coord * output->strides[ax];                               \
      }                                                                        \
      long in_off = in_base;                                                   \
      long out_off = out_base;                                                 \
      complex32 acc = complex16_to_complex32(in_data[in_off]);                 \
      out_data[out_off] = complex32_to_complex16(acc);                         \
      for (int k = 1; k < plan.axis_len; ++k) {                                \
        in_off += axis_stride_in;                                              \
        out_off += axis_stride_out;                                            \
        complex32 val = complex16_to_complex32(in_data[in_off]);               \
        acc = ACC_EXPR;                                                        \
        out_data[out_off] = complex32_to_complex16(acc);                       \
      }                                                                        \
      if (outer_dims == 0) break;                                              \
      if (!advance_outer_coords(input, outer_axes, outer_coord, outer_dims))   \
        break;                                                                 \
    }                                                                          \
    scan_plan_destroy(&plan);                                                  \
  }

#define DEFINE_SCAN_INT4(OPNAME, SUFFIX, IS_SIGNED, ACC_EXPR)                  \
  static void nx_c_scan_##OPNAME##_##SUFFIX(const ndarray_t *input,            \
                                            ndarray_t *output, int axis) {     \
    scan_plan_t plan = scan_prepare(input, output, axis, "scan_" #OPNAME);     \
    if (plan.axis_len <= 0) {                                                  \
      scan_plan_destroy(&plan);                                                \
      return;                                                                  \
    }                                                                          \
    uint8_t *in_data = (uint8_t *)input->data;                                 \
    uint8_t *out_data = (uint8_t *)output->data;                               \
    const bool is_signed = (IS_SIGNED);                                        \
    const int outer_dims = plan.outer_dims;                                    \
    const int *outer_axes = plan.outer_axes;                                   \
    int *outer_coord = plan.outer_coord;                                       \
    const long axis_stride_in = plan.axis_stride_in;                           \
    const long axis_stride_out = plan.axis_stride_out;                         \
    while (true) {                                                             \
      long in_base = input->offset;                                            \
      long out_base = output->offset;                                          \
      for (int i = 0; i < outer_dims; ++i) {                                  \
        int ax = outer_axes[i];                                                \
        long coord = outer_coord[i];                                           \
        in_base += coord * input->strides[ax];                                 \
        out_base += coord * output->strides[ax];                               \
      }                                                                        \
      long in_off = in_base;                                                   \
      long out_off = out_base;                                                 \
      int acc = int4_get(in_data, in_off, is_signed);                          \
      int4_set(out_data, out_off, acc, is_signed);                             \
      for (int k = 1; k < plan.axis_len; ++k) {                                \
        in_off += axis_stride_in;                                              \
        out_off += axis_stride_out;                                            \
        int val = int4_get(in_data, in_off, is_signed);                        \
        acc = ACC_EXPR;                                                        \
        int4_set(out_data, out_off, acc, is_signed);                           \
      }                                                                        \
      if (outer_dims == 0) break;                                              \
      if (!advance_outer_coords(input, outer_axes, outer_coord, outer_dims))   \
        break;                                                                 \
    }                                                                          \
    scan_plan_destroy(&plan);                                                  \
  }

typedef struct {
  scan_fn_t i8;
  scan_fn_t u8;
  scan_fn_t i16;
  scan_fn_t u16;
  scan_fn_t i32;
  scan_fn_t i64;
  scan_fn_t inat;
  scan_fn_t f16;
  scan_fn_t f32;
  scan_fn_t f64;
  scan_fn_t c32;
  scan_fn_t c64;
  scan_fn_t bf16;
  scan_fn_t bool_;
  scan_fn_t i4;
  scan_fn_t u4;
  scan_fn_t f8e4m3;
  scan_fn_t f8e5m2;
  scan_fn_t c16;
  scan_fn_t qi8;
  scan_fn_t qu8;
} scan_dispatch_table;

// Sum implementations
DEFINE_SCAN_DIRECT(sum, int8_t, i8, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, uint8_t, u8, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, int16_t, i16, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, uint16_t, u16, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, int32_t, i32, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, int64_t, i64, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, intnat, inat, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, float, f32, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, double, f64, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, complex32, c32, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, complex64, c64, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, caml_ba_bool, bool_, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, caml_ba_qint8, qi8, SUM_EXPR(acc, val))
DEFINE_SCAN_DIRECT(sum, caml_ba_quint8, qu8, SUM_EXPR(acc, val))
DEFINE_SCAN_LOW_PREC(sum, uint16_t, f16, SUM_EXPR(acc, val), half_to_float,
                     float_to_half)
DEFINE_SCAN_LOW_PREC(sum, caml_ba_bfloat16, bf16, SUM_EXPR(acc, val),
                     bfloat16_to_float, float_to_bfloat16)
DEFINE_SCAN_LOW_PREC(sum, caml_ba_fp8_e4m3, f8e4m3, SUM_EXPR(acc, val),
                     fp8_e4m3_to_float, float_to_fp8_e4m3)
DEFINE_SCAN_LOW_PREC(sum, caml_ba_fp8_e5m2, f8e5m2, SUM_EXPR(acc, val),
                     fp8_e5m2_to_float, float_to_fp8_e5m2)
DEFINE_SCAN_COMPLEX16(sum, SUM_EXPR(acc, val))
DEFINE_SCAN_INT4(sum, i4, true, SUM_EXPR(acc, val))
DEFINE_SCAN_INT4(sum, u4, false, SUM_EXPR(acc, val))

static const scan_dispatch_table scan_sum_table = {
    .i8 = nx_c_scan_sum_i8,
    .u8 = nx_c_scan_sum_u8,
    .i16 = nx_c_scan_sum_i16,
    .u16 = nx_c_scan_sum_u16,
    .i32 = nx_c_scan_sum_i32,
    .i64 = nx_c_scan_sum_i64,
    .inat = nx_c_scan_sum_inat,
    .f16 = nx_c_scan_sum_f16,
    .f32 = nx_c_scan_sum_f32,
    .f64 = nx_c_scan_sum_f64,
    .c32 = nx_c_scan_sum_c32,
    .c64 = nx_c_scan_sum_c64,
    .bf16 = nx_c_scan_sum_bf16,
    .bool_ = nx_c_scan_sum_bool_,
    .i4 = nx_c_scan_sum_i4,
    .u4 = nx_c_scan_sum_u4,
    .f8e4m3 = nx_c_scan_sum_f8e4m3,
    .f8e5m2 = nx_c_scan_sum_f8e5m2,
    .c16 = nx_c_scan_sum_c16,
    .qi8 = nx_c_scan_sum_qi8,
    .qu8 = nx_c_scan_sum_qu8};

// Prod implementations
DEFINE_SCAN_DIRECT(prod, int8_t, i8, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, uint8_t, u8, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, int16_t, i16, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, uint16_t, u16, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, int32_t, i32, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, int64_t, i64, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, intnat, inat, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, float, f32, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, double, f64, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, complex32, c32, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, complex64, c64, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, caml_ba_bool, bool_, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, caml_ba_qint8, qi8, PROD_EXPR(acc, val))
DEFINE_SCAN_DIRECT(prod, caml_ba_quint8, qu8, PROD_EXPR(acc, val))
DEFINE_SCAN_LOW_PREC(prod, uint16_t, f16, PROD_EXPR(acc, val), half_to_float,
                     float_to_half)
DEFINE_SCAN_LOW_PREC(prod, caml_ba_bfloat16, bf16, PROD_EXPR(acc, val),
                     bfloat16_to_float, float_to_bfloat16)
DEFINE_SCAN_LOW_PREC(prod, caml_ba_fp8_e4m3, f8e4m3, PROD_EXPR(acc, val),
                     fp8_e4m3_to_float, float_to_fp8_e4m3)
DEFINE_SCAN_LOW_PREC(prod, caml_ba_fp8_e5m2, f8e5m2, PROD_EXPR(acc, val),
                     fp8_e5m2_to_float, float_to_fp8_e5m2)
DEFINE_SCAN_COMPLEX16(prod, PROD_EXPR(acc, val))
DEFINE_SCAN_INT4(prod, i4, true, PROD_EXPR(acc, val))
DEFINE_SCAN_INT4(prod, u4, false, PROD_EXPR(acc, val))

static const scan_dispatch_table scan_prod_table = {
    .i8 = nx_c_scan_prod_i8,
    .u8 = nx_c_scan_prod_u8,
    .i16 = nx_c_scan_prod_i16,
    .u16 = nx_c_scan_prod_u16,
    .i32 = nx_c_scan_prod_i32,
    .i64 = nx_c_scan_prod_i64,
    .inat = nx_c_scan_prod_inat,
    .f16 = nx_c_scan_prod_f16,
    .f32 = nx_c_scan_prod_f32,
    .f64 = nx_c_scan_prod_f64,
    .c32 = nx_c_scan_prod_c32,
    .c64 = nx_c_scan_prod_c64,
    .bf16 = nx_c_scan_prod_bf16,
    .bool_ = nx_c_scan_prod_bool_,
    .i4 = nx_c_scan_prod_i4,
    .u4 = nx_c_scan_prod_u4,
    .f8e4m3 = nx_c_scan_prod_f8e4m3,
    .f8e5m2 = nx_c_scan_prod_f8e5m2,
    .c16 = nx_c_scan_prod_c16,
    .qi8 = nx_c_scan_prod_qi8,
    .qu8 = nx_c_scan_prod_qu8};

// Max implementations
DEFINE_SCAN_DIRECT(max, int8_t, i8, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, uint8_t, u8, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, int16_t, i16, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, uint16_t, u16, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, int32_t, i32, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, int64_t, i64, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, intnat, inat, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, float, f32, MAX_FLOAT_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, double, f64, MAX_FLOAT_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, complex32, c32, MAX_COMPLEX32_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, complex64, c64, MAX_COMPLEX64_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, caml_ba_bool, bool_, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, caml_ba_qint8, qi8, MAX_EXPR(acc, val))
DEFINE_SCAN_DIRECT(max, caml_ba_quint8, qu8, MAX_EXPR(acc, val))
DEFINE_SCAN_LOW_PREC(max, uint16_t, f16, MAX_FLOAT_EXPR(acc, val), half_to_float,
                     float_to_half)
DEFINE_SCAN_LOW_PREC(max, caml_ba_bfloat16, bf16, MAX_FLOAT_EXPR(acc, val),
                     bfloat16_to_float, float_to_bfloat16)
DEFINE_SCAN_LOW_PREC(max, caml_ba_fp8_e4m3, f8e4m3, MAX_FLOAT_EXPR(acc, val),
                     fp8_e4m3_to_float, float_to_fp8_e4m3)
DEFINE_SCAN_LOW_PREC(max, caml_ba_fp8_e5m2, f8e5m2, MAX_FLOAT_EXPR(acc, val),
                     fp8_e5m2_to_float, float_to_fp8_e5m2)
DEFINE_SCAN_COMPLEX16(max, MAX_COMPLEX32_EXPR(acc, val))
DEFINE_SCAN_INT4(max, i4, true, MAX_EXPR(acc, val))
DEFINE_SCAN_INT4(max, u4, false, MAX_EXPR(acc, val))

static const scan_dispatch_table scan_max_table = {
    .i8 = nx_c_scan_max_i8,
    .u8 = nx_c_scan_max_u8,
    .i16 = nx_c_scan_max_i16,
    .u16 = nx_c_scan_max_u16,
    .i32 = nx_c_scan_max_i32,
    .i64 = nx_c_scan_max_i64,
    .inat = nx_c_scan_max_inat,
    .f16 = nx_c_scan_max_f16,
    .f32 = nx_c_scan_max_f32,
    .f64 = nx_c_scan_max_f64,
    .c32 = nx_c_scan_max_c32,
    .c64 = nx_c_scan_max_c64,
    .bf16 = nx_c_scan_max_bf16,
    .bool_ = nx_c_scan_max_bool_,
    .i4 = nx_c_scan_max_i4,
    .u4 = nx_c_scan_max_u4,
    .f8e4m3 = nx_c_scan_max_f8e4m3,
    .f8e5m2 = nx_c_scan_max_f8e5m2,
    .c16 = nx_c_scan_max_c16,
    .qi8 = nx_c_scan_max_qi8,
    .qu8 = nx_c_scan_max_qu8};

// Min implementations
DEFINE_SCAN_DIRECT(min, int8_t, i8, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, uint8_t, u8, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, int16_t, i16, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, uint16_t, u16, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, int32_t, i32, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, int64_t, i64, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, intnat, inat, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, float, f32, MIN_FLOAT_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, double, f64, MIN_FLOAT_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, complex32, c32, MIN_COMPLEX32_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, complex64, c64, MIN_COMPLEX64_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, caml_ba_bool, bool_, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, caml_ba_qint8, qi8, MIN_EXPR(acc, val))
DEFINE_SCAN_DIRECT(min, caml_ba_quint8, qu8, MIN_EXPR(acc, val))
DEFINE_SCAN_LOW_PREC(min, uint16_t, f16, MIN_FLOAT_EXPR(acc, val), half_to_float,
                     float_to_half)
DEFINE_SCAN_LOW_PREC(min, caml_ba_bfloat16, bf16, MIN_FLOAT_EXPR(acc, val),
                     bfloat16_to_float, float_to_bfloat16)
DEFINE_SCAN_LOW_PREC(min, caml_ba_fp8_e4m3, f8e4m3, MIN_FLOAT_EXPR(acc, val),
                     fp8_e4m3_to_float, float_to_fp8_e4m3)
DEFINE_SCAN_LOW_PREC(min, caml_ba_fp8_e5m2, f8e5m2, MIN_FLOAT_EXPR(acc, val),
                     fp8_e5m2_to_float, float_to_fp8_e5m2)
DEFINE_SCAN_COMPLEX16(min, MIN_COMPLEX32_EXPR(acc, val))
DEFINE_SCAN_INT4(min, i4, true, MIN_EXPR(acc, val))
DEFINE_SCAN_INT4(min, u4, false, MIN_EXPR(acc, val))

static const scan_dispatch_table scan_min_table = {
    .i8 = nx_c_scan_min_i8,
    .u8 = nx_c_scan_min_u8,
    .i16 = nx_c_scan_min_i16,
    .u16 = nx_c_scan_min_u16,
    .i32 = nx_c_scan_min_i32,
    .i64 = nx_c_scan_min_i64,
    .inat = nx_c_scan_min_inat,
    .f16 = nx_c_scan_min_f16,
    .f32 = nx_c_scan_min_f32,
    .f64 = nx_c_scan_min_f64,
    .c32 = nx_c_scan_min_c32,
    .c64 = nx_c_scan_min_c64,
    .bf16 = nx_c_scan_min_bf16,
    .bool_ = nx_c_scan_min_bool_,
    .i4 = nx_c_scan_min_i4,
    .u4 = nx_c_scan_min_u4,
    .f8e4m3 = nx_c_scan_min_f8e4m3,
    .f8e5m2 = nx_c_scan_min_f8e5m2,
    .c16 = nx_c_scan_min_c16,
    .qi8 = nx_c_scan_min_qi8,
    .qu8 = nx_c_scan_min_qu8};

static void dispatch_scan_op(value v_input, value v_output, int axis,
                             const scan_dispatch_table *table,
                             const char *op_name) {
  ndarray_t input = extract_ndarray(v_input);
  ndarray_t output = extract_ndarray(v_output);

  value v_input_data = Field(v_input, FFI_TENSOR_DATA);
  value v_output_data = Field(v_output, FFI_TENSOR_DATA);
  struct caml_ba_array *ba_input = Caml_ba_array_val(v_input_data);
  struct caml_ba_array *ba_output = Caml_ba_array_val(v_output_data);
  int kind_input = ba_input->flags & CAML_BA_KIND_MASK;
  int kind_output = ba_output->flags & CAML_BA_KIND_MASK;

  if (kind_input != kind_output) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("associative_scan: dtype mismatch");
  }

  scan_fn_t fn = NULL;
  switch (kind_input) {
    case CAML_BA_SINT8:
      fn = table->i8;
      break;
    case CAML_BA_UINT8:
      fn = table->u8;
      break;
    case CAML_BA_SINT16:
      fn = table->i16;
      break;
    case CAML_BA_UINT16:
      fn = table->u16;
      break;
    case CAML_BA_INT32:
      fn = table->i32;
      break;
    case CAML_BA_INT64:
      fn = table->i64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      fn = table->inat;
      break;
    case CAML_BA_FLOAT16:
      fn = table->f16;
      break;
    case CAML_BA_FLOAT32:
      fn = table->f32;
      break;
    case CAML_BA_FLOAT64:
      fn = table->f64;
      break;
    case CAML_BA_COMPLEX32:
      fn = table->c32;
      break;
    case CAML_BA_COMPLEX64:
      fn = table->c64;
      break;
    case NX_BA_BFLOAT16:
      fn = table->bf16;
      break;
    case NX_BA_BOOL:
      fn = table->bool_;
      break;
    case NX_BA_INT4:
      fn = table->i4;
      break;
    case NX_BA_UINT4:
      fn = table->u4;
      break;
    case NX_BA_FP8_E4M3:
      fn = table->f8e4m3;
      break;
    case NX_BA_FP8_E5M2:
      fn = table->f8e5m2;
      break;
    case NX_BA_COMPLEX16:
      fn = table->c16;
      break;
    case NX_BA_QINT8:
      fn = table->qi8;
      break;
    case NX_BA_QUINT8:
      fn = table->qu8;
      break;
    default:
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("associative_scan: unsupported dtype");
  }

  if (!fn) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("associative_scan: operation not supported for dtype");
  }

  caml_enter_blocking_section();
  fn(&input, &output, axis);
  caml_leave_blocking_section();

  cleanup_ndarray(&input);
  cleanup_ndarray(&output);
}

CAMLprim value caml_nx_associative_scan(value v_input, value v_output,
                                        value v_axis, value v_op_tag) {
  CAMLparam4(v_input, v_output, v_axis, v_op_tag);
  int axis = Int_val(v_axis);
  int op_tag = Int_val(v_op_tag);
  const scan_dispatch_table *table = NULL;
  const char *op_name = NULL;
  switch (op_tag) {
    case 0:
      table = &scan_sum_table;
      op_name = "scan_sum";
      break;
    case 1:
      table = &scan_prod_table;
      op_name = "scan_prod";
      break;
    case 2:
      table = &scan_max_table;
      op_name = "scan_max";
      break;
    case 3:
      table = &scan_min_table;
      op_name = "scan_min";
      break;
    default:
      caml_failwith("associative_scan: invalid operation tag");
  }

  dispatch_scan_op(v_input, v_output, axis, table, op_name);

  CAMLreturn(Val_unit);
}
