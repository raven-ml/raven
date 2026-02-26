/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// Gather and scatter operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Helper to check if shapes are equal
static bool shape_equal(const int *shape1, const int *shape2, int ndim) {
  for (int i = 0; i < ndim; i++) {
    if (shape1[i] != shape2[i]) return false;
  }
  return true;
}

// Helper to check if two ndarrays have the same shape
static bool same_shape(const ndarray_t *a, const ndarray_t *b) {
  if (a->ndim != b->ndim) return false;
  return shape_equal(a->shape, b->shape, a->ndim);
}

// Forward declaration - implementation after multi_iterator definition
static void copy_ndarray(const ndarray_t *src, ndarray_t *dst, int kind);

// Type definitions for element-wise in-place operations (for scatter modes)
typedef void (*elem_op_fn)(void *, long, void *, long);

// Dispatch table for each type
typedef struct {
  elem_op_fn i8, u8, i16, u16, i32, i64, u32, u64, inat;
  elem_op_fn f16, f32, f64;
  elem_op_fn c32, c64;
  elem_op_fn bf16, bool_, i4, u4, f8e4m3, f8e5m2;
} elem_op_table;

// Macro to generate all standard type variants for an element operation
#define GENERATE_ELEM_OP(name, OP_EXPR)               \
  ELEM_OP_FOR_TYPE(name, int8_t, i8, OP_EXPR)         \
  ELEM_OP_FOR_TYPE(name, uint8_t, u8, OP_EXPR)        \
  ELEM_OP_FOR_TYPE(name, int16_t, i16, OP_EXPR)       \
  ELEM_OP_FOR_TYPE(name, uint16_t, u16, OP_EXPR)      \
  ELEM_OP_FOR_TYPE(name, int32_t, i32, OP_EXPR)       \
  ELEM_OP_FOR_TYPE(name, int64_t, i64, OP_EXPR)       \
  ELEM_OP_FOR_TYPE(name, uint32_t, u32, OP_EXPR)      \
  ELEM_OP_FOR_TYPE(name, uint64_t, u64, OP_EXPR)      \
  ELEM_OP_FOR_TYPE(name, intnat, inat, OP_EXPR)       \
  ELEM_OP_FOR_TYPE(name, float, f32, OP_EXPR)         \
  ELEM_OP_FOR_TYPE(name, double, f64, OP_EXPR)

// Macro to build dispatch table
#define BUILD_ELEM_OP_TABLE(name)             \
  static const elem_op_table name##_table = { \
      .i8 = nx_c_elem_##name##_i8,            \
      .u8 = nx_c_elem_##name##_u8,            \
      .i16 = nx_c_elem_##name##_i16,          \
      .u16 = nx_c_elem_##name##_u16,          \
      .i32 = nx_c_elem_##name##_i32,          \
      .i64 = nx_c_elem_##name##_i64,          \
      .u32 = nx_c_elem_##name##_u32,          \
      .u64 = nx_c_elem_##name##_u64,          \
      .inat = nx_c_elem_##name##_inat,        \
      .f16 = nx_c_elem_##name##_f16,          \
      .f32 = nx_c_elem_##name##_f32,          \
      .f64 = nx_c_elem_##name##_f64,          \
      .c32 = nx_c_elem_##name##_c32,          \
      .c64 = nx_c_elem_##name##_c64,          \
      .bf16 = nx_c_elem_##name##_bf16,        \
      .bool_ = nx_c_elem_##name##_bool_,      \
      .i4 = nx_c_elem_##name##_i4,            \
      .u4 = nx_c_elem_##name##_u4,            \
      .f8e4m3 = nx_c_elem_##name##_f8e4m3,    \
      .f8e5m2 = nx_c_elem_##name##_f8e5m2}

// Generic element operation
#define ELEM_OP_FOR_TYPE(name, T, suffix, OP_EXPR)                            \
  static void nx_c_elem_##name##_##suffix(void *src, long src_off, void *dst, \
                                          long dst_off) {                     \
    T *s = (T *)src;                                                          \
    T *d = (T *)dst;                                                          \
    T a = s[src_off];                                                         \
    T b = d[dst_off];                                                         \
    d[dst_off] = OP_EXPR(a, b);                                               \
  }

// Low-precision float elem op (convert to float)
#define LOW_PREC_ELEM_OP(name, T, suffix, OP_EXPR, TO_FLOAT, FROM_FLOAT)      \
  static void nx_c_elem_##name##_##suffix(void *src, long src_off, void *dst, \
                                          long dst_off) {                     \
    T *s = (T *)src;                                                          \
    T *d = (T *)dst;                                                          \
    float a = TO_FLOAT(s[src_off]);                                           \
    float b = TO_FLOAT(d[dst_off]);                                           \
    d[dst_off] = FROM_FLOAT(OP_EXPR(a, b));                                   \
  }

// Complex elem op
#define COMPLEX_ELEM_OP_FOR_TYPE(name, T, suffix, OP_EXPR)                    \
  static void nx_c_elem_##name##_##suffix(void *src, long src_off, void *dst, \
                                          long dst_off) {                     \
    T *s = (T *)src;                                                          \
    T *d = (T *)dst;                                                          \
    T a = s[src_off];                                                         \
    T b = d[dst_off];                                                         \
    d[dst_off] = OP_EXPR(a, b);                                               \
  }

// Int4 elem op (packed)
#define INT4_ELEM_OP(name, signedness, suffix, OP_EXPR)                       \
  static void nx_c_elem_##name##_##suffix(void *src, long src_off, void *dst, \
                                          long dst_off) {                     \
    uint8_t *s = (uint8_t *)src;                                              \
    uint8_t *d = (uint8_t *)dst;                                              \
    long s_byte = src_off / 2;                                                \
    int s_nib = src_off % 2;                                                  \
    int a = s_nib ? (signedness ? (int8_t)(s[s_byte] >> 4)                    \
                                : (s[s_byte] >> 4) & 0x0F)                    \
                  : (signedness ? (int8_t)((s[s_byte] & 0x0F) << 4) >> 4      \
                                : s[s_byte] & 0x0F);                          \
    long d_byte = dst_off / 2;                                                \
    int d_nib = dst_off % 2;                                                  \
    int b = d_nib ? (signedness ? (int8_t)(d[d_byte] >> 4)                    \
                                : (d[d_byte] >> 4) & 0x0F)                    \
                  : (signedness ? (int8_t)((d[d_byte] & 0x0F) << 4) >> 4      \
                                : d[d_byte] & 0x0F);                          \
    int res = OP_EXPR(a, b);                                                  \
    res = signedness ? CLAMP_I4(res) : CLAMP_U4(res);                         \
    uint8_t nib = (uint8_t)res & 0x0F;                                        \
    if (d_nib) {                                                              \
      d[d_byte] = (d[d_byte] & 0x0F) | (nib << 4);                            \
    } else {                                                                  \
      d[d_byte] = (d[d_byte] & 0xF0) | nib;                                   \
    }                                                                         \
  }

// Generate for set (assign)
#define SET_EXPR(a, b) (a)
GENERATE_ELEM_OP(set, SET_EXPR)

LOW_PREC_ELEM_OP(set, uint16_t, f16, SET_EXPR, half_to_float, float_to_half)
LOW_PREC_ELEM_OP(set, caml_ba_bfloat16, bf16, SET_EXPR, bfloat16_to_float,
                 float_to_bfloat16)
LOW_PREC_ELEM_OP(set, caml_ba_fp8_e4m3, f8e4m3, SET_EXPR, fp8_e4m3_to_float,
                 float_to_fp8_e4m3)
LOW_PREC_ELEM_OP(set, caml_ba_fp8_e5m2, f8e5m2, SET_EXPR, fp8_e5m2_to_float,
                 float_to_fp8_e5m2)

COMPLEX_ELEM_OP_FOR_TYPE(set, complex32, c32, SET_EXPR)
COMPLEX_ELEM_OP_FOR_TYPE(set, complex64, c64, SET_EXPR)
INT4_ELEM_OP(set, 1, i4, SET_EXPR)
INT4_ELEM_OP(set, 0, u4, SET_EXPR)
ELEM_OP_FOR_TYPE(set, caml_ba_bool, bool_, SET_EXPR)
BUILD_ELEM_OP_TABLE(set);

// Generate for add (accumulate)
#define ADD_EXPR(a, b) ((a) + (b))
GENERATE_ELEM_OP(add, ADD_EXPR)

LOW_PREC_ELEM_OP(add, uint16_t, f16, ADD_EXPR, half_to_float, float_to_half)
LOW_PREC_ELEM_OP(add, caml_ba_bfloat16, bf16, ADD_EXPR, bfloat16_to_float,
                 float_to_bfloat16)
LOW_PREC_ELEM_OP(add, caml_ba_fp8_e4m3, f8e4m3, ADD_EXPR, fp8_e4m3_to_float,
                 float_to_fp8_e4m3)
LOW_PREC_ELEM_OP(add, caml_ba_fp8_e5m2, f8e5m2, ADD_EXPR, fp8_e5m2_to_float,
                 float_to_fp8_e5m2)

COMPLEX_ELEM_OP_FOR_TYPE(add, complex32, c32, ADD_EXPR)
COMPLEX_ELEM_OP_FOR_TYPE(add, complex64, c64, ADD_EXPR)
INT4_ELEM_OP(add, 1, i4, ADD_EXPR)
INT4_ELEM_OP(add, 0, u4, ADD_EXPR)
ELEM_OP_FOR_TYPE(add, caml_ba_bool, bool_, ADD_EXPR)
BUILD_ELEM_OP_TABLE(add);

// Multi-dimensional iterator for shapes
typedef struct {
  int ndim;
  long *shape;
  long *coords;  // Changed to long to handle large dimensions
  int has_elements;
} multi_iterator_t;

static void multi_iterator_init(multi_iterator_t *it, const ndarray_t *nd) {
  it->ndim = nd->ndim;
  it->shape = (long *)caml_stat_alloc(it->ndim * sizeof(long));
  it->coords = (long *)caml_stat_alloc(it->ndim * sizeof(long));
  it->has_elements = 1;
  for (int i = 0; i < it->ndim; i++) {
    long dim = nd->shape[i];
    it->shape[i] = dim;
    it->coords[i] = 0;
    if (dim == 0) it->has_elements = 0;
  }
  if (it->ndim == 0) it->has_elements = 1;
}

static int multi_iterator_next(multi_iterator_t *it) {
  for (int i = it->ndim - 1; i >= 0; i--) {
    it->coords[i]++;
    if (it->coords[i] < it->shape[i]) return 1;
    it->coords[i] = 0;
  }
  return 0;
}

static void multi_iterator_destroy(multi_iterator_t *it) {
  caml_stat_free(it->shape);
  caml_stat_free(it->coords);
}

static long compute_offset(const ndarray_t *nd, const long *coords) {
  long off = 0;
  for (int i = 0; i < nd->ndim; i++) {
    off += coords[i] * nd->strides[i];
  }
  return off;
}

// Helper to get element byte size for memset (returns bytes per element, 0.5
// approximated as special case)
static double get_elem_byte_size(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
    case CAML_BA_UINT8:
    case NX_BA_BOOL:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
      return 1.0;
    case CAML_BA_SINT16:
    case CAML_BA_UINT16:
    case CAML_BA_FLOAT16:
    case NX_BA_BFLOAT16:
      return 2.0;
    case CAML_BA_INT32:
    case CAML_BA_FLOAT32:
    case NX_BA_UINT32:
      return 4.0;
    case CAML_BA_INT64:
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
    case CAML_BA_FLOAT64:
    case NX_BA_UINT64:
      return 8.0;
    case CAML_BA_COMPLEX32:
      return 8.0;
    case CAML_BA_COMPLEX64:
      return 16.0;
    case NX_BA_INT4:
    case NX_BA_UINT4:
      return 0.5;
    default:
      caml_failwith("unsupported kind");
      return 0;
  }
}

// Integer element size in bytes for common kinds. Returns 0 if unsupported
static inline size_t elem_size_from_kind(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
    case CAML_BA_UINT8:
    case NX_BA_BOOL:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
      return 1;
    case CAML_BA_SINT16:
    case CAML_BA_UINT16:
    case CAML_BA_FLOAT16:
    case NX_BA_BFLOAT16:
      return 2;
    case CAML_BA_INT32:
    case CAML_BA_FLOAT32:
    case NX_BA_UINT32:
      return 4;
    case CAML_BA_INT64:
    case CAML_BA_FLOAT64:
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
    case NX_BA_UINT64:
      return 8;
    case CAML_BA_COMPLEX32:
      return 8;  // 2 * float32
    case CAML_BA_COMPLEX64:
      return 16; // 2 * float64
    default:
      return 0;
  }
}

// Zero the output array - requires passing the value to access bigarray
static void zero_ndarray(ndarray_t *nd, value v_tensor, int kind) {
  value v_data = Field(v_tensor, FFI_TENSOR_DATA);
  struct caml_ba_array *ba = Caml_ba_array_val(v_data);
  long total_elems = total_elements_safe(nd);
  double bytes_per_elem = get_elem_byte_size(kind);
  long total_bytes;
  if (kind == NX_BA_INT4 || kind == NX_BA_UINT4) {
    total_bytes = total_elems / 2;
  } else {
    total_bytes = (long)(total_elems * bytes_per_elem);
  }
  memset(ba->data, 0, total_bytes);
}

// Helper to copy data from one ndarray to another (assuming same shape)
static void copy_ndarray(const ndarray_t *src, ndarray_t *dst, int kind) {
  if (!src || !dst) return;
  if (!same_shape(src, dst)) return;
  
  // Get element size based on kind
  size_t elem_size = 1;
  switch (kind) {
    case CAML_BA_SINT8:
    case CAML_BA_UINT8:
    case NX_BA_BOOL:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
      elem_size = 1;
      break;
    case CAML_BA_SINT16:
    case CAML_BA_UINT16:
    case CAML_BA_FLOAT16:
    case NX_BA_BFLOAT16:
      elem_size = 2;
      break;
    case CAML_BA_INT32:
    case CAML_BA_FLOAT32:
    case NX_BA_UINT32:
      elem_size = 4;
      break;
    case CAML_BA_INT64:
    case CAML_BA_FLOAT64:
    case NX_BA_UINT64:
      elem_size = 8;
      break;
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
      elem_size = sizeof(intnat);
      break;
    case CAML_BA_COMPLEX32:
      elem_size = 8;  // 2 * float32
      break;
    case CAML_BA_COMPLEX64:
      elem_size = 16;  // 2 * float64
      break;
    default:
      return;  // Unsupported type
  }
  
  // Use multi-iterator to copy elements
  multi_iterator_t it;
  multi_iterator_init(&it, src);

  if (it.has_elements) {
    do {
      long src_off = compute_offset(src, it.coords);
      long dst_off = compute_offset(dst, it.coords);

      memcpy(dst->data + (dst->offset + dst_off) * elem_size,
             src->data + (src->offset + src_off) * elem_size,
             elem_size);
    } while (multi_iterator_next(&it));
  }

  multi_iterator_destroy(&it);
}

// Generic gather implementation
static const char *generic_gather(const ndarray_t *data,
                                  const ndarray_t *indices, ndarray_t *out,
                                  int axis, elem_op_fn op) {
  const char *error_msg = NULL;

  if (data->ndim != indices->ndim || data->ndim != out->ndim) {
    error_msg = "ndim mismatch";
    return error_msg;
  }
  for (int i = 0; i < data->ndim; i++) {
    if (i != axis) {
      if (indices->shape[i] != data->shape[i]) {
        error_msg = "shape mismatch on non-axis dims";
        return error_msg;
      }
    }
  }
  if (!shape_equal(indices->shape, out->shape, data->ndim)) {
    error_msg = "output shape must match indices";
    return error_msg;
  }

  if (total_elements_safe(indices) == 0) {
    return NULL;
  }

  multi_iterator_t it;
  multi_iterator_init(&it, indices);

  if (it.has_elements) {
    do {
      long indices_off = compute_offset(indices, it.coords);
      int32_t index = *
          ((int32_t *)(indices->data
                        + (indices->offset + indices_off) * sizeof(int32_t)));
      // Handle negative indices (Python-style)
      if (index < 0) {
        index += data->shape[axis];
      }
      if (index < 0 || index >= data->shape[axis]) {
        error_msg = "index out of bounds";
        break;
      }

      long data_coords[32];  // Stack buffer for coordinates
      for (int i = 0; i < it.ndim; i++) {
        data_coords[i] = (i == axis) ? index : it.coords[i];
      }
      long data_off = compute_offset(data, data_coords);
      long out_off = compute_offset(out, it.coords);

      // Apply set op (copy)
      op(data->data, data->offset + data_off, out->data,
         out->offset + out_off);
    } while (multi_iterator_next(&it));
  }
  multi_iterator_destroy(&it);

  return error_msg;
}

// Generic scatter implementation
static const char *generic_scatter(const ndarray_t *template,
                                   const ndarray_t *indices,
                                   const ndarray_t *updates, ndarray_t *out,
                                   value v_out, int axis, elem_op_fn op,
                                   int unique, int kind, int mode) {
  const char *error_msg = NULL;

  // NULL checks
  if (!template || !indices || !updates || !out) {
    error_msg = "generic_scatter: NULL pointer";
    return error_msg;
  }
  if (!template->shape || !indices->shape || !updates->shape || !out->shape) {
    error_msg = "generic_scatter: NULL shape array";
    return error_msg;
  }

  if (template->ndim != indices->ndim || template->ndim != updates->ndim ||
      template->ndim != out->ndim) {
    error_msg = "ndim mismatch";
    return error_msg;
  }
  for (int i = 0; i < template->ndim; i++) {
    if (i != axis) {
      if (indices->shape[i] != template->shape[i] ||
          updates->shape[i] != indices->shape[i]) {
        error_msg = "shape mismatch on non-axis dims";
        return error_msg;
      }
    } else {
      if (indices->shape[i] != updates->shape[i]) {
        error_msg = "indices and updates mismatch on axis";
        return error_msg;
      }
    }
  }
  if (!shape_equal(template->shape, out->shape, template->ndim)) {
    error_msg = "output shape must match template";
    return error_msg;
  }

  // For Set mode (0), copy template to output first
  // For Add mode (1), zero the output
  if (mode == 0) {
    // Set mode - copy template data to output to preserve existing values
    copy_ndarray(template, out, kind);
  } else {
    // Add mode - zero the output
    zero_ndarray(out, v_out, kind);
  }

  multi_iterator_t it;
  multi_iterator_init(&it, indices);

  if (it.has_elements) {
    do {
      long indices_off = compute_offset(indices, it.coords);
      int32_t index = *
          ((int32_t *)(indices->data
                        + (indices->offset + indices_off) * sizeof(int32_t)));
      // Handle negative indices (Python-style)
      if (index < 0) {
        index += template->shape[axis];
      }
      if (index < 0 || index >= template->shape[axis]) {
        error_msg = "index out of bounds";
        break;
      }

      long out_coords[32];  // Stack buffer for coordinates
      for (int i = 0; i < it.ndim; i++) {
        out_coords[i] = (i == axis) ? index : it.coords[i];
      }
      long out_off = compute_offset(out, out_coords);
      long updates_off = compute_offset(updates, it.coords);

      // Apply op
      op(updates->data, updates->offset + updates_off, out->data,
         out->offset + out_off);
    } while (multi_iterator_next(&it));
  }
  multi_iterator_destroy(&it);

  return error_msg;
}

// Dispatch for gather
static void dispatch_gather(value v_data, value v_indices, value v_out,
                            int axis) {
  ndarray_t data = extract_ndarray(v_data);
  ndarray_t indices = extract_ndarray(v_indices);
  ndarray_t out = extract_ndarray(v_out);

  value v_data_data = Field(v_data, FFI_TENSOR_DATA);
  value v_out_data = Field(v_out, FFI_TENSOR_DATA);
  value v_indices_data = Field(v_indices, FFI_TENSOR_DATA);

  struct caml_ba_array *ba_data = Caml_ba_array_val(v_data_data);
  struct caml_ba_array *ba_indices = Caml_ba_array_val(v_indices_data);
  int kind = nx_buffer_get_kind(ba_data);
  if (kind != nx_buffer_get_kind(Caml_ba_array_val(v_out_data)))
    caml_failwith("dtype mismatch");
  if (nx_buffer_get_kind(ba_indices) != CAML_BA_INT32)
    caml_failwith("indices must be int32");

  elem_op_fn op = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      op = set_table.i8;
      break;
    case CAML_BA_UINT8:
      op = set_table.u8;
      break;
    case CAML_BA_SINT16:
      op = set_table.i16;
      break;
    case CAML_BA_UINT16:
      op = set_table.u16;
      break;
    case CAML_BA_INT32:
      op = set_table.i32;
      break;
    case CAML_BA_INT64:
      op = set_table.i64;
      break;
    case NX_BA_UINT32:
      op = set_table.u32;
      break;
    case NX_BA_UINT64:
      op = set_table.u64;
      break;
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
      op = set_table.inat;
      break;
    case CAML_BA_FLOAT16:
      op = set_table.f16;
      break;
    case CAML_BA_FLOAT32:
      op = set_table.f32;
      break;
    case CAML_BA_FLOAT64:
      op = set_table.f64;
      break;
    case CAML_BA_COMPLEX32:
      op = set_table.c32;
      break;
    case CAML_BA_COMPLEX64:
      op = set_table.c64;
      break;
    case NX_BA_BFLOAT16:
      op = set_table.bf16;
      break;
    case NX_BA_BOOL:
      op = set_table.bool_;
      break;
    case NX_BA_INT4:
      op = set_table.i4;
      break;
    case NX_BA_UINT4:
      op = set_table.u4;
      break;
    case NX_BA_FP8_E4M3:
      op = set_table.f8e4m3;
      break;
    case NX_BA_FP8_E5M2:
      op = set_table.f8e5m2;
      break;
    default:
      caml_failwith("unsupported dtype for gather");
  }

  if (!op) caml_failwith("gather not supported for dtype");

  // Fast path: 2D gather along axis 0 with broadcasted indices on dim 1,
  // contiguous data/out -> memcpy whole rows
  const char *error = NULL;
  size_t elem_size = elem_size_from_kind(kind);
  if (axis == 0 && data.ndim == 2 && indices.ndim == 2 && out.ndim == 2 &&
      elem_size > 0 && is_contiguous(&data) && is_contiguous(&out)) {
    // Broadcasting along dim 1 is represented by stride==0 on indices dim 1
    if (indices.strides[1] == 0 &&
        data.shape[1] == out.shape[1] && indices.shape[0] == out.shape[0]) {
      long n = out.shape[0];
      long d = out.shape[1];
      long data_row_stride = data.strides[0]; // in elements
      long out_row_stride = out.strides[0];   // in elements
      char *restrict data_ptr = (char *)data.data;
      char *restrict out_ptr = (char *)out.data;
      int32_t *restrict idx_ptr = (int32_t *)indices.data;
      long idx_off0 = indices.offset; // element offset
      long idx_row_stride = indices.strides[0]; // in elements
      long data_base = data.offset; // element offset
      long out_base = out.offset;   // element offset
      size_t row_bytes = (size_t)d * elem_size;

      caml_enter_blocking_section();
      for (long i = 0; i < n; i++) {
        long idx_eoff = idx_off0 + i * idx_row_stride; // indices strides[1]==0
        int32_t index = idx_ptr[idx_eoff];
        if (index < 0) index += data.shape[0];
        if (index < 0 || index >= data.shape[0]) {
          error = "index out of bounds";
          break;
        }
        long src_eoff = data_base + index * data_row_stride;
        long dst_eoff = out_base + i * out_row_stride;
        memcpy(out_ptr + (size_t)dst_eoff * elem_size,
               data_ptr + (size_t)src_eoff * elem_size, row_bytes);
      }
      caml_leave_blocking_section();
      cleanup_ndarray(&data);
      cleanup_ndarray(&indices);
      cleanup_ndarray(&out);
      if (error) caml_failwith(error);
      return;
    }
  }

  caml_enter_blocking_section();
  error = generic_gather(&data, &indices, &out, axis, op);
  caml_leave_blocking_section();

  cleanup_ndarray(&data);
  cleanup_ndarray(&indices);
  cleanup_ndarray(&out);

  if (error) caml_failwith(error);
}

// Dispatch for scatter
static void dispatch_scatter(value v_template, value v_indices, value v_updates,
                             value v_out, int axis, value v_mode,
                             value v_unique) {
  ndarray_t templ = extract_ndarray(v_template);
  ndarray_t indices = extract_ndarray(v_indices);
  ndarray_t updates = extract_ndarray(v_updates);

  // Check if v_out is None (represented as 0 in OCaml)
  ndarray_t out;
  if (v_out == Val_int(0)) {
    // If v_out is None, use template as output
    out = templ;
  } else {
    out = extract_ndarray(v_out);
  }

  value v_template_data = Field(v_template, FFI_TENSOR_DATA);
  value v_updates_data = Field(v_updates, FFI_TENSOR_DATA);
  value v_out_data =
      (v_out == Val_int(0)) ? v_template_data : Field(v_out, FFI_TENSOR_DATA);
  value v_indices_data = Field(v_indices, FFI_TENSOR_DATA);

  struct caml_ba_array *ba_templ = Caml_ba_array_val(v_template_data);
  int kind = nx_buffer_get_kind(ba_templ);
  if (kind != nx_buffer_get_kind(Caml_ba_array_val(v_updates_data)) ||
      kind != nx_buffer_get_kind(Caml_ba_array_val(v_out_data)))
    caml_failwith("dtype mismatch");
  if (nx_buffer_get_kind(Caml_ba_array_val(v_indices_data)) != CAML_BA_INT32)
    caml_failwith("indices must be int32");

  const elem_op_table *table = Int_val(v_mode) == 0 ? &set_table : &add_table;

  elem_op_fn op = NULL;
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
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
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
      caml_failwith("unsupported dtype for scatter");
  }

  if (!op) caml_failwith("scatter not supported for dtype");

  int unique = Bool_val(
      v_unique);  // Currently ignored, but can be used for future optimizations

  caml_enter_blocking_section();
  value actual_v_out = (v_out == Val_int(0)) ? v_template : v_out;
  const char *error = generic_scatter(&templ, &indices, &updates, &out,
                                      actual_v_out, axis, op, unique, kind, Int_val(v_mode));
  caml_leave_blocking_section();

  cleanup_ndarray(&indices);
  cleanup_ndarray(&updates);
  // Only cleanup templ and out if they're different
  if (v_out != Val_int(0)) {
    cleanup_ndarray(&templ);
    cleanup_ndarray(&out);
  } else {
    // When v_out is None, out == templ, so only cleanup once
    cleanup_ndarray(&templ);
  }

  if (error) caml_failwith(error);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_op_gather(value v_data, value v_indices, value v_out,
                                 value v_axis) {
  CAMLparam4(v_data, v_indices, v_out, v_axis);
  dispatch_gather(v_data, v_indices, v_out, Int_val(v_axis));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_op_scatter(value v_template, value v_indices,
                                  value v_updates, value v_axis, value v_out,
                                  value v_mode, value v_unique) {
  CAMLparam0();  // More params, but camlparam max 5, use 0 for simplicity
  dispatch_scatter(v_template, v_indices, v_updates, v_out, Int_val(v_axis),
                   v_mode, v_unique);
  CAMLreturn(Val_unit);
}

// Bytecode wrapper for scatter (7 arguments)
CAMLprim value caml_nx_op_scatter_bc(value *argv, int argn) {
  CAMLparam0();
  (void)argn;
  value ret = caml_nx_op_scatter(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5], argv[6]);
  CAMLreturn(ret);
}
