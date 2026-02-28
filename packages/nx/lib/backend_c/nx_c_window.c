/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

// Window operations for nx C backend (unfold/fold)
//
// Generalized sliding-window extraction and its inverse.
// The last K dimensions of the input are treated as spatial; all preceding
// dimensions ("leading") are preserved as-is.
//
// unfold: (*leading, *spatial_K) -> (*leading, prod(kernel_size), L)
// fold:   (*leading, prod(kernel_size), L) -> (*leading, *output_size)

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Max supported spatial dimensions (for stack arrays)
#define MAX_SPATIAL_DIMS 32

// Type operations for element-wise copy, zero, add
typedef void (*elem_add_t)(void*, long, void*, long);
typedef void (*elem_copy_t)(void*, long, void*, long);
typedef void (*elem_zero_t)(void*, long);

// Table for type-specific operations
typedef struct {
  elem_add_t add;
  elem_copy_t copy;
  elem_zero_t zero;
} type_ops_t;

// Dispatch table for each type
typedef struct {
  type_ops_t i8, u8, i16, u16, i32, i64, u32, u64, inat;
  type_ops_t f16, f32, f64;
  type_ops_t c32, c64;
  type_ops_t bf16, bool_, i4, u4, f8e4m3, f8e5m2;
} type_ops_table_t;

// Macros for standard types
#define STANDARD_ADD(T, suffix)                                                \
  static void add_elem_##suffix(void* out, long o_off, void* in, long i_off) { \
    T* ot = (T*)out;                                                           \
    T* it = (T*)in;                                                            \
    ot[o_off] += it[i_off];                                                    \
  }

#define STANDARD_COPY(T, suffix)                                  \
  static void copy_elem_##suffix(void* out, long o_off, void* in, \
                                 long i_off) {                    \
    T* ot = (T*)out;                                              \
    T* it = (T*)in;                                               \
    ot[o_off] = it[i_off];                                        \
  }

#define STANDARD_ZERO(T, suffix)                          \
  static void zero_elem_##suffix(void* out, long o_off) { \
    T* ot = (T*)out;                                      \
    ot[o_off] = (T)0;                                     \
  }

// Generate for standard types
#define GENERATE_STANDARD_OPS(T, suffix) \
  STANDARD_ADD(T, suffix)                \
  STANDARD_COPY(T, suffix)               \
  STANDARD_ZERO(T, suffix)

GENERATE_STANDARD_OPS(int8_t, i8)
GENERATE_STANDARD_OPS(uint8_t, u8)
GENERATE_STANDARD_OPS(int16_t, i16)
GENERATE_STANDARD_OPS(uint16_t, u16)
GENERATE_STANDARD_OPS(int32_t, i32)
GENERATE_STANDARD_OPS(int64_t, i64)
GENERATE_STANDARD_OPS(uint32_t, u32)
GENERATE_STANDARD_OPS(uint64_t, u64)
GENERATE_STANDARD_OPS(intnat, inat)
GENERATE_STANDARD_OPS(float, f32)
GENERATE_STANDARD_OPS(double, f64)
GENERATE_STANDARD_OPS(complex32, c32)
GENERATE_STANDARD_OPS(complex64, c64)
GENERATE_STANDARD_OPS(caml_ba_bool, bool_)

// For low-precision floats: add requires conversion, copy/zero are direct
STANDARD_COPY(uint16_t, f16)
STANDARD_ZERO(uint16_t, f16)
static void add_elem_f16(void* out, long o_off, void* in, long i_off) {
  uint16_t* ot = (uint16_t*)out;
  uint16_t* it = (uint16_t*)in;
  float a = half_to_float(ot[o_off]);
  float b = half_to_float(it[i_off]);
  ot[o_off] = float_to_half(a + b);
}

STANDARD_COPY(caml_ba_bfloat16, bf16)
STANDARD_ZERO(caml_ba_bfloat16, bf16)
static void add_elem_bf16(void* out, long o_off, void* in, long i_off) {
  caml_ba_bfloat16* ot = (caml_ba_bfloat16*)out;
  caml_ba_bfloat16* it = (caml_ba_bfloat16*)in;
  float a = bfloat16_to_float(ot[o_off]);
  float b = bfloat16_to_float(it[i_off]);
  ot[o_off] = float_to_bfloat16(a + b);
}

STANDARD_COPY(caml_ba_fp8_e4m3, f8e4m3)
STANDARD_ZERO(caml_ba_fp8_e4m3, f8e4m3)
static void add_elem_f8e4m3(void* out, long o_off, void* in, long i_off) {
  caml_ba_fp8_e4m3* ot = (caml_ba_fp8_e4m3*)out;
  caml_ba_fp8_e4m3* it = (caml_ba_fp8_e4m3*)in;
  float a = fp8_e4m3_to_float(ot[o_off]);
  float b = fp8_e4m3_to_float(it[i_off]);
  ot[o_off] = float_to_fp8_e4m3(a + b);
}

STANDARD_COPY(caml_ba_fp8_e5m2, f8e5m2)
STANDARD_ZERO(caml_ba_fp8_e5m2, f8e5m2)
static void add_elem_f8e5m2(void* out, long o_off, void* in, long i_off) {
  caml_ba_fp8_e5m2* ot = (caml_ba_fp8_e5m2*)out;
  caml_ba_fp8_e5m2* it = (caml_ba_fp8_e5m2*)in;
  float a = fp8_e5m2_to_float(ot[o_off]);
  float b = fp8_e5m2_to_float(it[i_off]);
  ot[o_off] = float_to_fp8_e5m2(a + b);
}

// For int4/uint4 (packed nibbles)
static void zero_elem_i4(void* out, long o_off) {
  uint8_t* ot = (uint8_t*)out;
  long byte_off = o_off / 2;
  int nib_off = o_off % 2;
  if (nib_off) {
    ot[byte_off] &= 0x0F;
  } else {
    ot[byte_off] &= 0xF0;
  }
}

static void copy_elem_i4(void* out, long o_off, void* in, long i_off) {
  uint8_t* oi = (uint8_t*)in;
  uint8_t* oo = (uint8_t*)out;
  long byte_i = i_off / 2;
  int nib_i = i_off % 2;
  long byte_o = o_off / 2;
  int nib_o = o_off % 2;
  int8_t val = nib_i ? (oi[byte_i] >> 4) : ((oi[byte_i] & 0x0F) << 4) >> 4;
  uint8_t nib = (uint8_t)val & 0x0F;
  if (nib_o) {
    oo[byte_o] = (oo[byte_o] & 0x0F) | (nib << 4);
  } else {
    oo[byte_o] = (oo[byte_o] & 0xF0) | nib;
  }
}

static void add_elem_i4(void* out, long o_off, void* in, long i_off) {
  uint8_t* od = (uint8_t*)out;
  uint8_t* id = (uint8_t*)in;
  long byte_o = o_off / 2;
  int nib_o = o_off % 2;
  long byte_i = i_off / 2;
  int nib_i = i_off % 2;
  int8_t a = nib_o ? (od[byte_o] >> 4) : ((od[byte_o] & 0x0F) << 4) >> 4;
  int8_t b = nib_i ? (id[byte_i] >> 4) : ((id[byte_i] & 0x0F) << 4) >> 4;
  int res = (int)a + (int)b;
  res = CLAMP_I4(res);
  uint8_t nib = (uint8_t)res & 0x0F;
  if (nib_o) {
    od[byte_o] = (od[byte_o] & 0x0F) | (nib << 4);
  } else {
    od[byte_o] = (od[byte_o] & 0xF0) | nib;
  }
}

static void zero_elem_u4(void* out, long o_off) {
  uint8_t* ot = (uint8_t*)out;
  long byte_off = o_off / 2;
  int nib_off = o_off % 2;
  if (nib_off) {
    ot[byte_off] &= 0x0F;
  } else {
    ot[byte_off] &= 0xF0;
  }
}

static void copy_elem_u4(void* out, long o_off, void* in, long i_off) {
  uint8_t* oi = (uint8_t*)in;
  uint8_t* oo = (uint8_t*)out;
  long byte_i = i_off / 2;
  int nib_i = i_off % 2;
  long byte_o = o_off / 2;
  int nib_o = o_off % 2;
  uint8_t val = nib_i ? (oi[byte_i] >> 4) & 0x0F : oi[byte_i] & 0x0F;
  uint8_t nib = val & 0x0F;
  if (nib_o) {
    oo[byte_o] = (oo[byte_o] & 0x0F) | (nib << 4);
  } else {
    oo[byte_o] = (oo[byte_o] & 0xF0) | nib;
  }
}

static void add_elem_u4(void* out, long o_off, void* in, long i_off) {
  uint8_t* od = (uint8_t*)out;
  uint8_t* id = (uint8_t*)in;
  long byte_o = o_off / 2;
  int nib_o = o_off % 2;
  long byte_i = i_off / 2;
  int nib_i = i_off % 2;
  uint8_t a = nib_o ? (od[byte_o] >> 4) & 0x0F : od[byte_o] & 0x0F;
  uint8_t b = nib_i ? (id[byte_i] >> 4) & 0x0F : id[byte_i] & 0x0F;
  int res = (int)a + (int)b;
  res = CLAMP_U4(res);
  uint8_t nib = (uint8_t)res & 0x0F;
  if (nib_o) {
    od[byte_o] = (od[byte_o] & 0x0F) | (nib << 4);
  } else {
    od[byte_o] = (od[byte_o] & 0xF0) | nib;
  }
}

// Build dispatch table
static const type_ops_table_t type_ops_table = {
    .i8 = {add_elem_i8, copy_elem_i8, zero_elem_i8},
    .u8 = {add_elem_u8, copy_elem_u8, zero_elem_u8},
    .i16 = {add_elem_i16, copy_elem_i16, zero_elem_i16},
    .u16 = {add_elem_u16, copy_elem_u16, zero_elem_u16},
    .i32 = {add_elem_i32, copy_elem_i32, zero_elem_i32},
    .i64 = {add_elem_i64, copy_elem_i64, zero_elem_i64},
    .u32 = {add_elem_u32, copy_elem_u32, zero_elem_u32},
    .u64 = {add_elem_u64, copy_elem_u64, zero_elem_u64},
    .inat = {add_elem_inat, copy_elem_inat, zero_elem_inat},
    .f16 = {add_elem_f16, copy_elem_f16, zero_elem_f16},
    .f32 = {add_elem_f32, copy_elem_f32, zero_elem_f32},
    .f64 = {add_elem_f64, copy_elem_f64, zero_elem_f64},
    .c32 = {add_elem_c32, copy_elem_c32, zero_elem_c32},
    .c64 = {add_elem_c64, copy_elem_c64, zero_elem_c64},
    .bf16 = {add_elem_bf16, copy_elem_bf16, zero_elem_bf16},
    .bool_ = {add_elem_bool_, copy_elem_bool_, zero_elem_bool_},
    .i4 = {add_elem_i4, copy_elem_i4, zero_elem_i4},
    .u4 = {add_elem_u4, copy_elem_u4, zero_elem_u4},
    .f8e4m3 = {add_elem_f8e4m3, copy_elem_f8e4m3, zero_elem_f8e4m3},
    .f8e5m2 = {add_elem_f8e5m2, copy_elem_f8e5m2, zero_elem_f8e5m2}};

// Helper to get elem_size (for memset, etc.)
static size_t get_elem_size(int kind) {
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
      return 4;
    case CAML_BA_INT64:
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
    case CAML_BA_FLOAT64:
    case CAML_BA_COMPLEX32:
      return 8;
    case CAML_BA_COMPLEX64:
      return 16;
    case NX_BA_INT4:
    case NX_BA_UINT4:
      return 0;  // Special handling
    default:
      return 0;
  }
}

// Compute the flat offset into a tensor for a given leading index.
// leading_idx is a flat index into the collapsed leading dims.
// Returns the strided offset for that leading position.
static long leading_offset(const ndarray_t* t, int leading_ndim,
                           long leading_idx) {
  long off = 0;
  long rem = leading_idx;
  for (int d = leading_ndim - 1; d >= 0; d--) {
    long coord = rem % t->shape[d];
    rem /= t->shape[d];
    off += coord * t->strides[d];
  }
  return off;
}

// Implementation for unfold
// input:  (*leading, *spatial_K)
// output: (*leading, kernel_prod, L)
static void nx_c_unfold_impl(const ndarray_t* in, ndarray_t* out,
                             int K, const long* kernel_size,
                             const long* stride_arr, const long* dilation_arr,
                             const long* pad_before, const long* pad_after,
                             int leading_ndim, const type_ops_t* ops,
                             size_t elem_size) {
  long* out_spatial = (long*)calloc(K, sizeof(long));

  // Compute leading_size = product of all leading dimensions
  long leading_size = 1;
  for (int d = 0; d < leading_ndim; d++) leading_size *= in->shape[d];

  long kernel_prod = 1;
  bool no_padding = true;
  for (int d = 0; d < K; d++) {
    long effective_ker = dilation_arr[d] * (kernel_size[d] - 1) + 1;
    long padded = in->shape[leading_ndim + d] + pad_before[d] + pad_after[d];
    long diff = padded - effective_ker;
    out_spatial[d] = (diff / stride_arr[d]) + 1;
    kernel_prod *= kernel_size[d];
    if (pad_before[d] != 0 || pad_after[d] != 0) no_padding = false;
  }
  if (kernel_prod == 0) no_padding = false;
  long L = 1;
  for (int d = 0; d < K; d++) L *= out_spatial[d];

  // Output shape: (*leading, kernel_prod, L)
  // out dims: leading_ndim + 2
  // out->strides[leading_ndim] is stride for kernel_prod axis
  // out->strides[leading_ndim + 1] is stride for L axis

  long* out_cumprod = (long*)calloc(K, sizeof(long));
  if (K > 0) {
    out_cumprod[K - 1] = 1;
    for (int i = K - 2; i >= 0; i--)
      out_cumprod[i] = out_cumprod[i + 1] * out_spatial[i + 1];
  }

  // Pre-compute kernel offsets for the no-padding fast path
  long* kernel_offsets = NULL;
  if (no_padding && kernel_prod > 0) {
    kernel_offsets = (long*)malloc(kernel_prod * sizeof(long));
    if (kernel_offsets) {
      long coords[MAX_SPATIAL_DIMS] = {0};
      for (long kf = 0; kf < kernel_prod; ++kf) {
        long offset = 0;
        for (int d = 0; d < K; ++d)
          offset += coords[d] * dilation_arr[d] * in->strides[leading_ndim + d];
        kernel_offsets[kf] = offset;
        for (int d = K - 1; d >= 0; --d) {
          coords[d]++;
          if (coords[d] < kernel_size[d]) break;
          coords[d] = 0;
        }
      }
    } else {
      no_padding = false;
    }
  }

  if (no_padding) {
    long stride_steps[MAX_SPATIAL_DIMS];
    for (int d = 0; d < K; ++d)
      stride_steps[d] = stride_arr[d] * in->strides[leading_ndim + d];

    for (long lead = 0; lead < leading_size; ++lead) {
      long base_in = leading_offset(in, leading_ndim, lead);
      long base_out = leading_offset(out, leading_ndim, lead);
      long block_coords[MAX_SPATIAL_DIMS] = {0};
      long block_offset = 0;
      for (long l = 0; l < L; ++l) {
        long out_l_base = base_out + l * out->strides[leading_ndim + 1];
        long in_block_base = base_in + block_offset;
        for (long kf = 0; kf < kernel_prod; ++kf) {
          long out_off = out_l_base + kf * out->strides[leading_ndim];
          long in_off = in_block_base + kernel_offsets[kf];
          ops->copy(out->data, out->offset + out_off, in->data,
                    in->offset + in_off);
        }
        for (int d = K - 1; d >= 0; --d) {
          block_coords[d]++;
          block_offset += stride_steps[d];
          if (block_coords[d] < out_spatial[d]) break;
          block_offset -= out_spatial[d] * stride_steps[d];
          block_coords[d] = 0;
        }
      }
    }
    goto cleanup;
  }

  // General path with padding
#pragma omp parallel for collapse(2) if (leading_size * L > 1000)
  for (long lead = 0; lead < leading_size; lead++) {
    for (long l = 0; l < L; l++) {
      long base_in = leading_offset(in, leading_ndim, lead);
      long base_out = leading_offset(out, leading_ndim, lead);
      long temp = l;
      long block_pos[MAX_SPATIAL_DIMS];
      for (int d = 0; d < K; d++) {
        block_pos[d] = temp / out_cumprod[d];
        temp %= out_cumprod[d];
      }
      for (long kf = 0; kf < kernel_prod; kf++) {
        long k_temp = kf;
        long k_pos[MAX_SPATIAL_DIMS];
        for (int d = K - 1; d >= 0; d--) {
          k_pos[d] = k_temp % kernel_size[d];
          k_temp /= kernel_size[d];
        }
        long in_off = base_in;
        bool valid = true;
        for (int d = 0; d < K; d++) {
          long sp =
              block_pos[d] * stride_arr[d] + k_pos[d] * dilation_arr[d] - pad_before[d];
          if (sp < 0 || sp >= in->shape[leading_ndim + d]) {
            valid = false;
            break;
          }
          in_off += sp * in->strides[leading_ndim + d];
        }
        long out_off = base_out + kf * out->strides[leading_ndim] +
                       l * out->strides[leading_ndim + 1];
        if (valid) {
          ops->copy(out->data, out->offset + out_off, in->data,
                    in->offset + in_off);
        } else {
          ops->zero(out->data, out->offset + out_off);
        }
      }
    }
  }

cleanup:
  free(out_spatial);
  free(out_cumprod);
  if (kernel_offsets) free(kernel_offsets);
}

// Implementation for fold
// input:  (*leading, kernel_prod, L)
// output: (*leading, *output_size)
static void nx_c_fold_impl(const ndarray_t* in, ndarray_t* out,
                           int K, const long* output_size,
                           const long* kernel_size, const long* stride_arr,
                           const long* dilation_arr, const long* pad_before,
                           const long* pad_after, int leading_ndim,
                           const type_ops_t* ops, size_t elem_size) {

  // Compute leading_size = product of all leading dimensions
  long leading_size = 1;
  for (int d = 0; d < leading_ndim; d++) leading_size *= in->shape[d];

  long kernel_prod = in->shape[leading_ndim];
  long L = in->shape[leading_ndim + 1];

  long expected_block[MAX_SPATIAL_DIMS];
  long expected_L = 1;
  for (int d = 0; d < K; d++) {
    long effective_ker = dilation_arr[d] * (kernel_size[d] - 1) + 1;
    long padded = output_size[d] + pad_before[d] + pad_after[d];
    long diff = padded - effective_ker;
    expected_block[d] = (diff / stride_arr[d]) + 1;
    expected_L *= expected_block[d];
  }

  long* out_cumprod = (long*)calloc(K, sizeof(long));
  if (K > 0) {
    out_cumprod[K - 1] = 1;
    for (int i = K - 2; i >= 0; i--)
      out_cumprod[i] = out_cumprod[i + 1] * expected_block[i + 1];
  }

  // Zero the output (bytes zero works for all types)
  long total_out = total_elements_safe(out);
  memset((char*)out->data + out->offset * elem_size, 0, total_out * elem_size);

#pragma omp parallel for collapse(2) if (leading_size * L > 1000)
  for (long lead = 0; lead < leading_size; lead++) {
    for (long l = 0; l < L; l++) {
      long base_in = leading_offset(in, leading_ndim, lead);
      long base_out = leading_offset(out, leading_ndim, lead);
      long temp = l;
      long block_pos[MAX_SPATIAL_DIMS];
      for (int d = 0; d < K; d++) {
        block_pos[d] = temp / out_cumprod[d];
        temp %= out_cumprod[d];
      }
      for (long kf = 0; kf < kernel_prod; kf++) {
        long k_temp = kf;
        long k_pos[MAX_SPATIAL_DIMS];
        for (int d = K - 1; d >= 0; d--) {
          k_pos[d] = k_temp % kernel_size[d];
          k_temp /= kernel_size[d];
        }
        long out_off = base_out;
        bool valid = true;
        for (int d = 0; d < K; d++) {
          long sp =
              block_pos[d] * stride_arr[d] + k_pos[d] * dilation_arr[d] - pad_before[d];
          if (sp < 0 || sp >= out->shape[leading_ndim + d]) {
            valid = false;
            break;
          }
          out_off += sp * out->strides[leading_ndim + d];
        }
        long in_off = base_in + kf * in->strides[leading_ndim] +
                      l * in->strides[leading_ndim + 1];
        if (valid) {
          ops->add(out->data, out->offset + out_off, in->data,
                   in->offset + in_off);
        }
      }
    }
  }

  free(out_cumprod);
}

// Dispatch helper
static const type_ops_t* get_type_ops(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
      return &type_ops_table.i8;
    case CAML_BA_UINT8:
      return &type_ops_table.u8;
    case CAML_BA_SINT16:
      return &type_ops_table.i16;
    case CAML_BA_UINT16:
      return &type_ops_table.u16;
    case CAML_BA_INT32:
      return &type_ops_table.i32;
    case CAML_BA_INT64:
      return &type_ops_table.i64;
    case NX_BA_UINT32:
      return &type_ops_table.u32;
    case NX_BA_UINT64:
      return &type_ops_table.u64;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      return &type_ops_table.inat;
    case CAML_BA_FLOAT16:
      return &type_ops_table.f16;
    case CAML_BA_FLOAT32:
      return &type_ops_table.f32;
    case CAML_BA_FLOAT64:
      return &type_ops_table.f64;
    case CAML_BA_COMPLEX32:
      return &type_ops_table.c32;
    case CAML_BA_COMPLEX64:
      return &type_ops_table.c64;
    case NX_BA_BFLOAT16:
      return &type_ops_table.bf16;
    case NX_BA_BOOL:
      return &type_ops_table.bool_;
    case NX_BA_INT4:
      return &type_ops_table.i4;
    case NX_BA_UINT4:
      return &type_ops_table.u4;
    case NX_BA_FP8_E4M3:
      return &type_ops_table.f8e4m3;
    case NX_BA_FP8_E5M2:
      return &type_ops_table.f8e5m2;
    default:
      return NULL;
  }
}

// OCaml FFI Stubs

CAMLprim value caml_nx_op_unfold(value v_in, value v_kernel_size,
                                 value v_stride, value v_dilation,
                                 value v_padding, value v_out) {
  CAMLparam5(v_in, v_kernel_size, v_stride, v_dilation, v_padding);
  CAMLxparam1(v_out);

  ndarray_t input = extract_ndarray(v_in);
  ndarray_t output = extract_ndarray(v_out);

  value v_in_data = Field(v_in, FFI_TENSOR_DATA);
  struct caml_ba_array* ba_in = Caml_ba_array_val(v_in_data);
  int kind = nx_buffer_get_kind(ba_in);

  value v_out_data = Field(v_out, FFI_TENSOR_DATA);
  int kind_out = nx_buffer_get_kind(Caml_ba_array_val(v_out_data));
  if (kind != kind_out) caml_failwith("dtype mismatch");

  const type_ops_t* ops = get_type_ops(kind);
  if (!ops) caml_failwith("unsupported dtype");

  size_t elem_size = get_elem_size(kind);

  // Validate parameters before entering blocking section
  int K = Wosize_val(v_kernel_size);
  if (K > MAX_SPATIAL_DIMS) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("too many spatial dimensions");
  }
  if (Wosize_val(v_stride) != K || Wosize_val(v_dilation) != K ||
      Wosize_val(v_padding) != 2 * K) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("parameter length mismatch");
  }
  if (input.ndim < K) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("unfold: input must have at least K dimensions");
  }

  int leading_ndim = input.ndim - K;

  // Validate output ndim: should be leading_ndim + 2 (kernel_prod, L)
  if (output.ndim != leading_ndim + 2) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("unfold: output ndim mismatch");
  }

  // Validate leading dims match
  for (int d = 0; d < leading_ndim; d++) {
    if (output.shape[d] != input.shape[d]) {
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("unfold: leading dimension mismatch");
    }
  }

  // Extract OCaml arrays into C arrays BEFORE releasing the runtime lock.
  long* c_kernel_size = (long*)calloc(K, sizeof(long));
  long* c_stride = (long*)calloc(K, sizeof(long));
  long* c_dilation = (long*)calloc(K, sizeof(long));
  long* c_pad_before = (long*)calloc(K, sizeof(long));
  long* c_pad_after = (long*)calloc(K, sizeof(long));
  for (int d = 0; d < K; d++) {
    c_kernel_size[d] = Long_val(Field(v_kernel_size, d));
    c_stride[d] = Long_val(Field(v_stride, d));
    c_dilation[d] = Long_val(Field(v_dilation, d));
    c_pad_before[d] = Long_val(Field(v_padding, 2 * d));
    c_pad_after[d] = Long_val(Field(v_padding, 2 * d + 1));
  }

  caml_enter_blocking_section();
  nx_c_unfold_impl(&input, &output, K, c_kernel_size, c_stride, c_dilation,
                   c_pad_before, c_pad_after, leading_ndim, ops, elem_size);
  caml_leave_blocking_section();

  free(c_kernel_size);
  free(c_stride);
  free(c_dilation);
  free(c_pad_before);
  free(c_pad_after);

  cleanup_ndarray(&input);
  cleanup_ndarray(&output);

  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_op_fold(value v_in, value v_output_size,
                               value v_kernel_size, value v_stride,
                               value v_dilation, value v_padding, value v_out) {
  CAMLparam5(v_in, v_output_size, v_kernel_size, v_stride, v_dilation);
  CAMLxparam2(v_padding, v_out);

  ndarray_t input = extract_ndarray(v_in);
  ndarray_t output = extract_ndarray(v_out);

  value v_in_data = Field(v_in, FFI_TENSOR_DATA);
  struct caml_ba_array* ba_in = Caml_ba_array_val(v_in_data);
  int kind = nx_buffer_get_kind(ba_in);

  value v_out_data = Field(v_out, FFI_TENSOR_DATA);
  int kind_out = nx_buffer_get_kind(Caml_ba_array_val(v_out_data));
  if (kind != kind_out) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("dtype mismatch");
  }

  const type_ops_t* ops = get_type_ops(kind);
  if (!ops) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("unsupported dtype");
  }

  size_t elem_size = get_elem_size(kind);

  // Validate parameters before entering blocking section
  int K = Wosize_val(v_kernel_size);
  if (K > MAX_SPATIAL_DIMS) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("too many spatial dimensions");
  }
  if (Wosize_val(v_output_size) != K || Wosize_val(v_stride) != K ||
      Wosize_val(v_dilation) != K || Wosize_val(v_padding) != 2 * K) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("parameter length mismatch");
  }

  // Input must have at least 2 dims (kernel_prod, L) plus optional leading
  if (input.ndim < 2) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("fold: input must have at least 2 dimensions");
  }

  int leading_ndim = input.ndim - 2;

  // Output must have leading_ndim + K dims
  if (output.ndim != leading_ndim + K) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("fold: output ndim mismatch");
  }

  // Validate leading dims match
  for (int d = 0; d < leading_ndim; d++) {
    if (output.shape[d] != input.shape[d]) {
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("fold: leading dimension mismatch");
    }
  }

  // Validate spatial dimensions match output_size
  for (int d = 0; d < K; d++) {
    if (output.shape[leading_ndim + d] != Long_val(Field(v_output_size, d))) {
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("fold: output spatial dimension mismatch");
    }
  }

  // Extract OCaml arrays into C arrays BEFORE releasing the runtime lock.
  long* c_output_size = (long*)calloc(K, sizeof(long));
  long* c_kernel_size = (long*)calloc(K, sizeof(long));
  long* c_stride = (long*)calloc(K, sizeof(long));
  long* c_dilation = (long*)calloc(K, sizeof(long));
  long* c_pad_before = (long*)calloc(K, sizeof(long));
  long* c_pad_after = (long*)calloc(K, sizeof(long));
  for (int d = 0; d < K; d++) {
    c_output_size[d] = Long_val(Field(v_output_size, d));
    c_kernel_size[d] = Long_val(Field(v_kernel_size, d));
    c_stride[d] = Long_val(Field(v_stride, d));
    c_dilation[d] = Long_val(Field(v_dilation, d));
    c_pad_before[d] = Long_val(Field(v_padding, 2 * d));
    c_pad_after[d] = Long_val(Field(v_padding, 2 * d + 1));
  }

  caml_enter_blocking_section();
  nx_c_fold_impl(&input, &output, K, c_output_size, c_kernel_size, c_stride,
                 c_dilation, c_pad_before, c_pad_after, leading_ndim, ops,
                 elem_size);
  caml_leave_blocking_section();

  free(c_output_size);
  free(c_kernel_size);
  free(c_stride);
  free(c_dilation);
  free(c_pad_before);
  free(c_pad_after);

  cleanup_ndarray(&input);
  cleanup_ndarray(&output);

  CAMLreturn(Val_unit);
}

// Bytecode wrappers for functions with >5 arguments
// These forward to the native versions and let them manage GC roots.
// OCaml expects these when the external is declared with two names
// (bytecode stub first, native stub second).
//
// unfold: 6 arguments
CAMLprim value caml_nx_op_unfold_bc(value* argv, int argn) {
  CAMLparam0();
  (void)argn;
  value ret =
      caml_nx_op_unfold(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
  CAMLreturn(ret);
}

// fold: 7 arguments
CAMLprim value caml_nx_op_fold_bc(value* argv, int argn) {
  CAMLparam0();
  (void)argn;
  value ret = caml_nx_op_fold(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6]);
  CAMLreturn(ret);
}
