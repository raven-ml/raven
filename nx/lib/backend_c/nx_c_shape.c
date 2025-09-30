// Pad and concatenate operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Type for fill kernel functions
typedef void (*fill_kernel_fn)(void *data, void *val, long offset);

// Helper to iterate over inner dimensions for fill operations
static inline void iterate_inner_dims_fill(const ndarray_t *z, long outer_idx,
                                           fill_kernel_fn kernel, void *val,
                                           void *z_data) {
  if (z->ndim <= 1) {
    kernel(z_data, val, outer_idx * z->strides[0]);
    return;
  }

  long z_base = outer_idx * z->strides[0];

  // Create temporary iterator for inner dimensions
  int inner_ndim = z->ndim - 1;
  int *coords = (int *)calloc(inner_ndim, sizeof(int));
  if (!coords) {
    caml_failwith("iterate_inner_dims_fill: allocation failed");
  }

  // Iterate over inner dimensions
  bool done = false;
  while (!done) {
    long z_off = z_base;

    for (int i = 0; i < inner_ndim; i++) {
      z_off += coords[i] * z->strides[i + 1];
    }

    kernel(z_data, val, z_off);

    // Advance to next position
    done = true;
    for (int i = inner_ndim - 1; i >= 0; i--) {
      coords[i]++;
      if (coords[i] < z->shape[i + 1]) {
        done = false;
        break;
      }
      coords[i] = 0;
    }
  }

  free(coords);
}

// Type definitions for pad and cat operations

// Fill operation type: fill ndarray with constant value
typedef void (*fill_op_t)(const ndarray_t *, void *);

// Copy operation type: copy src to dst with pad_before offsets (used for pad
// and cat)
typedef void (*copy_op_t)(const ndarray_t *, const ndarray_t *, long *);

// Dispatch tables for fill and copy
typedef struct {
  fill_op_t i8, u8, i16, u16, i32, i64, inat;
  fill_op_t f16, f32, f64;
  fill_op_t c32, c64;
  fill_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2, c16, qi8, qu8;
} fill_op_table;

typedef struct {
  copy_op_t i8, u8, i16, u16, i32, i64, inat;
  copy_op_t f16, f32, f64;
  copy_op_t c32, c64;
  copy_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2, c16, qi8, qu8;
} copy_op_table;

// Helper to get element size in bytes for memcpy eligibility (0 for
// unsupported)
static int get_elem_size(int kind) {
  switch (kind) {
    case CAML_BA_SINT8:
    case CAML_BA_UINT8:
    case NX_BA_BOOL:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
    case NX_BA_QINT8:
    case NX_BA_QUINT8:
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
    case CAML_BA_FLOAT64:
      return 8;
    case CAML_BA_COMPLEX32:
      return 8;
    case CAML_BA_COMPLEX64:
      return 16;
    case NX_BA_COMPLEX16:
      return 4;
    case NX_BA_INT4:
    case NX_BA_UINT4:
      return 0;  // Packed, no memcpy
    default:
      return 0;
  }
}

// is_contiguous is now defined in nx_c_shared.h

// Helper iterator for inner dimensions in copy operations
typedef void (*copy_kernel_fn)(void *, void *, long, long);

static inline void iterate_inner_dims_copy(const ndarray_t *src,
                                           const ndarray_t *dst, long outer_idx,
                                           copy_kernel_fn kernel,
                                           void *src_data, void *dst_data,
                                           long *pad_before) {
  if (src->ndim <= 1) {
    long src_base = outer_idx * src->strides[0];
    long dst_base = (outer_idx + pad_before[0]) * dst->strides[0];
    kernel(src_data, dst_data, src_base, dst_base);
    return;
  }

  long src_base = outer_idx * src->strides[0];
  long dst_base = (outer_idx + pad_before[0]) * dst->strides[0];
  int inner_ndim = src->ndim - 1;
  int *coords = (int *)calloc(inner_ndim, sizeof(int));
  if (!coords) {
    caml_failwith("iterate_inner_dims_copy: allocation failed");
  }

  bool done = false;
  while (!done) {
    long src_off = src_base;
    long dst_off = dst_base;

    for (int i = 0; i < inner_ndim; i++) {
      src_off += coords[i] * src->strides[i + 1];
      dst_off += (coords[i] + pad_before[i + 1]) * dst->strides[i + 1];
    }

    kernel(src_data, dst_data, src_off, dst_off);

    done = true;
    for (int i = inner_ndim - 1; i >= 0; i--) {
      coords[i]++;
      if (coords[i] < src->shape[i + 1]) {
        done = false;
        break;
      }
      coords[i] = 0;
    }
  }

  free(coords);
}

// Macro for standard fill operations
#define FILL_OP_KERNEL(name, T, suffix)                                  \
  static void nx_c_##name##_##suffix##_kernel(void *z_data, void *val_p, \
                                              long z_off) {              \
    T *z = (T *)z_data;                                                  \
    T val = *(T *)val_p;                                                 \
    z[z_off] = val;                                                      \
  }

#define FILL_OP_IMPL(name, T, suffix)                                          \
  static void nx_c_##name##_##suffix(const ndarray_t *z, void *val_p) {        \
    if (!z) {                                                                  \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    long total = total_elements_safe(z);                                       \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_contiguous(z)) {                                                    \
      _Pragma("omp parallel for simd if(total > 1000)") for (long i = 0;       \
                                                             i < total; i++) { \
        nx_c_##name##_##suffix##_kernel(z->data, val_p, z->offset + i);        \
      }                                                                        \
    } else if (z->shape[0] > 1 && total / z->shape[0] > 50) {                  \
      _Pragma("omp parallel for if(z->shape[0] > 4)") for (long i = 0;         \
                                                           i < z->shape[0];    \
                                                           i++) {              \
        iterate_inner_dims_fill(                                               \
            z, i, (fill_kernel_fn)nx_c_##name##_##suffix##_kernel, val_p,      \
            z->data);                                                          \
      }                                                                        \
    } else {                                                                   \
      nd_iterator_t it;                                                        \
      nd_iterator_init_safe(&it, z, z, z);                                     \
      do {                                                                     \
        long x_off, y_off, z_off;                                              \
        nd_iterator_get_offsets(&it, &x_off, &y_off, &z_off);                  \
        nx_c_##name##_##suffix##_kernel(z->data, val_p, z->offset + z_off);    \
      } while (nd_iterator_next(&it));                                         \
      nd_iterator_destroy(&it);                                                \
    }                                                                          \
  }

#define FILL_OP_FOR_TYPE(name, T, suffix) \
  FILL_OP_KERNEL(name, T, suffix)         \
  FILL_OP_IMPL(name, T, suffix)

#define GENERATE_FILL_OP(name)               \
  FILL_OP_FOR_TYPE(name, int8_t, i8)         \
  FILL_OP_FOR_TYPE(name, uint8_t, u8)        \
  FILL_OP_FOR_TYPE(name, int16_t, i16)       \
  FILL_OP_FOR_TYPE(name, uint16_t, u16)      \
  FILL_OP_FOR_TYPE(name, int32_t, i32)       \
  FILL_OP_FOR_TYPE(name, int64_t, i64)       \
  FILL_OP_FOR_TYPE(name, intnat, inat)       \
  FILL_OP_FOR_TYPE(name, float, f32)         \
  FILL_OP_FOR_TYPE(name, double, f64)        \
  FILL_OP_FOR_TYPE(name, complex32, c32)     \
  FILL_OP_FOR_TYPE(name, complex64, c64)     \
  FILL_OP_FOR_TYPE(name, caml_ba_qint8, qi8) \
  FILL_OP_FOR_TYPE(name, caml_ba_quint8, qu8)

// For low-precision fill
#define LOW_PREC_FILL_KERNEL(name, T, suffix, TO_FLOAT, FROM_FLOAT)      \
  static void nx_c_##name##_##suffix##_kernel(void *z_data, void *val_p, \
                                              long z_off) {              \
    T *z = (T *)z_data;                                                  \
    float val = *(float *)val_p;                                         \
    z[z_off] = FROM_FLOAT(val);                                          \
  }

#define LOW_PREC_FILL_IMPL(name, T, suffix) FILL_OP_IMPL(name, T, suffix)

// For complex16 fill
#define COMPLEX16_FILL_KERNEL(name)                               \
  static void nx_c_##name##_c16_kernel(void *z_data, void *val_p, \
                                       long z_off) {              \
    caml_ba_complex16 *z = (caml_ba_complex16 *)z_data;           \
    caml_ba_complex16 val = *(caml_ba_complex16 *)val_p;          \
    z[z_off] = val;                                               \
  }

// For int4 fill (packed, with saturation)
#define INT4_FILL_IMPL(name, signedness, suffix)                         \
  static void nx_c_##name##_##suffix##_kernel(void *z_data, void *val_p, \
                                              long z_off) {              \
    uint8_t *z = (uint8_t *)z_data;                                      \
    int val = *(int *)val_p;                                             \
    val = signedness ? CLAMP_I4(val) : CLAMP_U4(val);                    \
    uint8_t nib = (uint8_t)val & 0x0F;                                   \
    long byte_off = z_off / 2;                                           \
    int nib_off = z_off % 2;                                             \
    if (nib_off) {                                                       \
      z[byte_off] = (z[byte_off] & 0x0F) | (nib << 4);                   \
    } else {                                                             \
      z[byte_off] = (z[byte_off] & 0xF0) | nib;                          \
    }                                                                    \
  }                                                                      \
  FILL_OP_IMPL(name, uint8_t, suffix)  // Use uint8_t for packed

// Macro for standard copy operations
#define COPY_OP_KERNEL(name, T, suffix)                                       \
  static void nx_c_##name##_##suffix##_kernel(void *src_data, void *dst_data, \
                                              long src_off, long dst_off) {   \
    T *src = (T *)src_data;                                                   \
    T *dst = (T *)dst_data;                                                   \
    dst[dst_off] = src[src_off];                                              \
  }

#define COPY_OP_IMPL(name, T, suffix)                                          \
  static void nx_c_##name##_##suffix(const ndarray_t *src,                     \
                                     const ndarray_t *dst, long *pad_before) { \
    if (!src || !dst) {                                                        \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    long total = total_elements_safe(src);                                     \
    if (total == 0) return;                                                    \
                                                                               \
    /* Even if both are contiguous, we can't do a simple linear copy          \
       because the destination has different dimensions due to padding */      \
    if (false) {                                                               \
      /* Disabled - linear copy doesn't work for padding */                    \
    } else if (src->ndim > 0 && src->shape[0] > 1 &&                           \
               total / src->shape[0] > 50) {                                   \
      _Pragma("omp parallel for if(src->shape[0] > 4)") for (long i = 0;       \
                                                             i <               \
                                                             src->shape[0];    \
                                                             i++) {            \
        iterate_inner_dims_copy(src, dst, i, nx_c_##name##_##suffix##_kernel,  \
                                src->data, dst->data, pad_before);             \
      }                                                                        \
    } else {                                                                   \
      int ndim = src->ndim;                                                    \
      int *coords = (int *)calloc(ndim, sizeof(int));                          \
      if (!coords) {                                                           \
        caml_failwith("nx_c_" #name "_" #suffix ": allocation failed");        \
      }                                                                        \
      bool done = false;                                                       \
      while (!done) {                                                          \
        long src_off = 0;                                                      \
        long dst_off = 0;                                                      \
        for (int d = 0; d < ndim; d++) {                                       \
          src_off += (long)coords[d] * src->strides[d];                        \
          dst_off += ((long)coords[d] + pad_before[d]) * dst->strides[d];      \
        }                                                                      \
        nx_c_##name##_##suffix##_kernel(src->data, dst->data,                  \
                                        src->offset + src_off,                 \
                                        dst->offset + dst_off);                \
        done = true;                                                           \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          coords[d]++;                                                         \
          if (coords[d] < src->shape[d]) {                                     \
            done = false;                                                      \
            break;                                                             \
          }                                                                    \
          coords[d] = 0;                                                       \
        }                                                                      \
      }                                                                        \
      free(coords);                                                            \
    }                                                                          \
  }

#define COPY_OP_FOR_TYPE(name, T, suffix) \
  COPY_OP_KERNEL(name, T, suffix)         \
  COPY_OP_IMPL(name, T, suffix)

#define GENERATE_COPY_OP(name)               \
  COPY_OP_FOR_TYPE(name, int8_t, i8)         \
  COPY_OP_FOR_TYPE(name, uint8_t, u8)        \
  COPY_OP_FOR_TYPE(name, int16_t, i16)       \
  COPY_OP_FOR_TYPE(name, uint16_t, u16)      \
  COPY_OP_FOR_TYPE(name, int32_t, i32)       \
  COPY_OP_FOR_TYPE(name, int64_t, i64)       \
  COPY_OP_FOR_TYPE(name, intnat, inat)       \
  COPY_OP_FOR_TYPE(name, float, f32)         \
  COPY_OP_FOR_TYPE(name, double, f64)        \
  COPY_OP_FOR_TYPE(name, complex32, c32)     \
  COPY_OP_FOR_TYPE(name, complex64, c64)     \
  COPY_OP_FOR_TYPE(name, caml_ba_qint8, qi8) \
  COPY_OP_FOR_TYPE(name, caml_ba_quint8, qu8)

// For low-precision copy (bitwise copy)
#define LOW_PREC_COPY_KERNEL(name, T, suffix) COPY_OP_KERNEL(name, T, suffix)
#define LOW_PREC_COPY_IMPL(name, T, suffix) COPY_OP_IMPL(name, T, suffix)

// For complex16 copy
#define COMPLEX16_COPY_KERNEL(name) COPY_OP_KERNEL(name, caml_ba_complex16, c16)

// For int4 copy (packed)
#define INT4_COPY_IMPL(name, signedness, suffix)                               \
  static void nx_c_##name##_##suffix##_kernel(void *src_data, void *dst_data,  \
                                              long src_off, long dst_off) {    \
    uint8_t *src = (uint8_t *)src_data;                                        \
    uint8_t *dst = (uint8_t *)dst_data;                                        \
    long byte_off_src = src_off / 2;                                           \
    int nib_off_src = src_off % 2;                                             \
    int a = nib_off_src                                                        \
                ? (signedness ? (int8_t)(src[byte_off_src] >> 4)               \
                              : (src[byte_off_src] >> 4) & 0x0F)               \
                : (signedness ? (int8_t)((src[byte_off_src] & 0x0F) << 4) >> 4 \
                              : src[byte_off_src] & 0x0F);                     \
    uint8_t nib = (uint8_t)a & 0x0F;                                           \
    long byte_off_dst = dst_off / 2;                                           \
    int nib_off_dst = dst_off % 2;                                             \
    if (nib_off_dst) {                                                         \
      dst[byte_off_dst] = (dst[byte_off_dst] & 0x0F) | (nib << 4);             \
    } else {                                                                   \
      dst[byte_off_dst] = (dst[byte_off_dst] & 0xF0) | nib;                    \
    }                                                                          \
  }                                                                            \
  static void nx_c_##name##_##suffix(const ndarray_t *src,                     \
                                     const ndarray_t *dst, long *pad_before) { \
    long total = total_elements_safe(src);                                     \
    if (total == 0) return;                                                    \
                                                                               \
    if (src->ndim > 0 && src->shape[0] > 1 && total / src->shape[0] > 50) {    \
      _Pragma("omp parallel for if(src->shape[0] > 4)") for (long i = 0;       \
                                                             i <               \
                                                             src->shape[0];    \
                                                             i++) {            \
        iterate_inner_dims_copy(src, dst, i, nx_c_##name##_##suffix##_kernel,  \
                                src->data, dst->data, pad_before);             \
      }                                                                        \
    } else {                                                                   \
      int ndim = src->ndim;                                                    \
      int *coords = (int *)calloc(ndim, sizeof(int));                          \
      if (!coords) {                                                           \
        caml_failwith("nx_c_" #name "_" #suffix ": allocation failed");        \
      }                                                                        \
      bool done = false;                                                       \
      while (!done) {                                                          \
        long src_off = 0;                                                      \
        long dst_off = 0;                                                      \
        for (int d = 0; d < ndim; d++) {                                       \
          src_off += (long)coords[d] * src->strides[d];                        \
          dst_off += ((long)coords[d] + pad_before[d]) * dst->strides[d];      \
        }                                                                      \
        nx_c_##name##_##suffix##_kernel(src->data, dst->data,                  \
                                        src_off + src->offset,                 \
                                        dst_off + dst->offset);                \
        done = true;                                                           \
        for (int d = ndim - 1; d >= 0; d--) {                                  \
          coords[d]++;                                                         \
          if (coords[d] < src->shape[d]) {                                     \
            done = false;                                                      \
            break;                                                             \
          }                                                                    \
          coords[d] = 0;                                                       \
        }                                                                      \
      }                                                                        \
      free(coords);                                                            \
    }                                                                          \
  }

// Generate fill and copy for all ops
GENERATE_FILL_OP(fill)
LOW_PREC_FILL_KERNEL(fill, uint16_t, f16, , float_to_half)
LOW_PREC_FILL_IMPL(fill, uint16_t, f16)
LOW_PREC_FILL_KERNEL(fill, caml_ba_bfloat16, bf16, , float_to_bfloat16)
LOW_PREC_FILL_IMPL(fill, caml_ba_bfloat16, bf16)
LOW_PREC_FILL_KERNEL(fill, caml_ba_fp8_e4m3, f8e4m3, , float_to_fp8_e4m3)
LOW_PREC_FILL_IMPL(fill, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_FILL_KERNEL(fill, caml_ba_fp8_e5m2, f8e5m2, , float_to_fp8_e5m2)
LOW_PREC_FILL_IMPL(fill, caml_ba_fp8_e5m2, f8e5m2)
COMPLEX16_FILL_KERNEL(fill)
FILL_OP_IMPL(fill, caml_ba_complex16, c16)
INT4_FILL_IMPL(fill, 1, i4)
INT4_FILL_IMPL(fill, 0, u4)
FILL_OP_FOR_TYPE(fill, caml_ba_bool, bool_)

// Build dispatch table for fill operations
static const fill_op_table fill_table = {.i8 = nx_c_fill_i8,
                                         .u8 = nx_c_fill_u8,
                                         .i16 = nx_c_fill_i16,
                                         .u16 = nx_c_fill_u16,
                                         .i32 = nx_c_fill_i32,
                                         .i64 = nx_c_fill_i64,
                                         .inat = nx_c_fill_inat,
                                         .f16 = nx_c_fill_f16,
                                         .f32 = nx_c_fill_f32,
                                         .f64 = nx_c_fill_f64,
                                         .c32 = nx_c_fill_c32,
                                         .c64 = nx_c_fill_c64,
                                         .bf16 = nx_c_fill_bf16,
                                         .bool_ = nx_c_fill_bool_,
                                         .i4 = nx_c_fill_i4,
                                         .u4 = nx_c_fill_u4,
                                         .f8e4m3 = nx_c_fill_f8e4m3,
                                         .f8e5m2 = nx_c_fill_f8e5m2,
                                         .c16 = nx_c_fill_c16,
                                         .qi8 = nx_c_fill_qi8,
                                         .qu8 = nx_c_fill_qu8};

GENERATE_COPY_OP(copy)
LOW_PREC_COPY_KERNEL(copy, uint16_t, f16)
LOW_PREC_COPY_IMPL(copy, uint16_t, f16)
LOW_PREC_COPY_KERNEL(copy, caml_ba_bfloat16, bf16)
LOW_PREC_COPY_IMPL(copy, caml_ba_bfloat16, bf16)
LOW_PREC_COPY_KERNEL(copy, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_COPY_IMPL(copy, caml_ba_fp8_e4m3, f8e4m3)
LOW_PREC_COPY_KERNEL(copy, caml_ba_fp8_e5m2, f8e5m2)
LOW_PREC_COPY_IMPL(copy, caml_ba_fp8_e5m2, f8e5m2)
COMPLEX16_COPY_KERNEL(copy)
COPY_OP_IMPL(copy, caml_ba_complex16, c16)
INT4_COPY_IMPL(copy, 1, i4)
INT4_COPY_IMPL(copy, 0, u4)
COPY_OP_FOR_TYPE(copy, caml_ba_bool, bool_)

static const copy_op_table copy_table = {.i8 = nx_c_copy_i8,
                                         .u8 = nx_c_copy_u8,
                                         .i16 = nx_c_copy_i16,
                                         .u16 = nx_c_copy_u16,
                                         .i32 = nx_c_copy_i32,
                                         .i64 = nx_c_copy_i64,
                                         .inat = nx_c_copy_inat,
                                         .f16 = nx_c_copy_f16,
                                         .f32 = nx_c_copy_f32,
                                         .f64 = nx_c_copy_f64,
                                         .c32 = nx_c_copy_c32,
                                         .c64 = nx_c_copy_c64,
                                         .bf16 = nx_c_copy_bf16,
                                         .bool_ = nx_c_copy_bool_,
                                         .i4 = nx_c_copy_i4,
                                         .u4 = nx_c_copy_u4,
                                         .f8e4m3 = nx_c_copy_f8e4m3,
                                         .f8e5m2 = nx_c_copy_f8e5m2,
                                         .c16 = nx_c_copy_c16,
                                         .qi8 = nx_c_copy_qi8,
                                         .qu8 = nx_c_copy_qu8};

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_pad(value v_input, value v_pads, value v_fill,
                           value v_output) {
  CAMLparam4(v_input, v_pads, v_fill, v_output);
  ndarray_t input = extract_ndarray(v_input);
  ndarray_t output = extract_ndarray(v_output);
  int ndim = input.ndim;
  if (ndim != output.ndim) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("pad: ndim mismatch");
  }
  long *pad_before = (long *)malloc(ndim * sizeof(long));
  long *pad_after = (long *)malloc(ndim * sizeof(long));
  if (!pad_before || !pad_after) {
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("pad: allocation failed");
  }
  // v_pads is a flat array: [before_0, after_0, before_1, after_1, ...]
  for (int i = 0; i < ndim; i++) {
    pad_before[i] = Long_val(Field(v_pads, 2 * i));
    pad_after[i] = Long_val(Field(v_pads, 2 * i + 1));
    if (pad_before[i] < 0 || pad_after[i] < 0) {
      free(pad_before);
      free(pad_after);
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("pad: negative padding");
    }
    if (input.shape[i] + pad_before[i] + pad_after[i] != output.shape[i]) {
      free(pad_before);
      free(pad_after);
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("pad: shape mismatch");
    }
  }
  free(pad_after);  // Not used for copy, only validation

  value v_input_data = Field(v_input, FFI_TENSOR_DATA);
  struct caml_ba_array *ba = Caml_ba_array_val(v_input_data);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  value v_output_data = Field(v_output, FFI_TENSOR_DATA);
  int kind_out = Caml_ba_array_val(v_output_data)->flags & CAML_BA_KIND_MASK;
  if (kind != kind_out) {
    free(pad_before);
    cleanup_ndarray(&input);
    cleanup_ndarray(&output);
    caml_failwith("pad: dtype mismatch");
  }

  fill_op_t fill_op = NULL;
  copy_op_t copy_op = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      fill_op = fill_table.i8;
      copy_op = copy_table.i8;
      break;
    case CAML_BA_UINT8:
      fill_op = fill_table.u8;
      copy_op = copy_table.u8;
      break;
    case CAML_BA_SINT16:
      fill_op = fill_table.i16;
      copy_op = copy_table.i16;
      break;
    case CAML_BA_UINT16:
      fill_op = fill_table.u16;
      copy_op = copy_table.u16;
      break;
    case CAML_BA_INT32:
      fill_op = fill_table.i32;
      copy_op = copy_table.i32;
      break;
    case CAML_BA_INT64:
      fill_op = fill_table.i64;
      copy_op = copy_table.i64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      fill_op = fill_table.inat;
      copy_op = copy_table.inat;
      break;
    case CAML_BA_FLOAT16:
      fill_op = fill_table.f16;
      copy_op = copy_table.f16;
      break;
    case CAML_BA_FLOAT32:
      fill_op = fill_table.f32;
      copy_op = copy_table.f32;
      break;
    case CAML_BA_FLOAT64:
      fill_op = fill_table.f64;
      copy_op = copy_table.f64;
      break;
    case CAML_BA_COMPLEX32:
      fill_op = fill_table.c32;
      copy_op = copy_table.c32;
      break;
    case CAML_BA_COMPLEX64:
      fill_op = fill_table.c64;
      copy_op = copy_table.c64;
      break;
    case NX_BA_BFLOAT16:
      fill_op = fill_table.bf16;
      copy_op = copy_table.bf16;
      break;
    case NX_BA_BOOL:
      fill_op = fill_table.bool_;
      copy_op = copy_table.bool_;
      break;
    case NX_BA_INT4:
      fill_op = fill_table.i4;
      copy_op = copy_table.i4;
      break;
    case NX_BA_UINT4:
      fill_op = fill_table.u4;
      copy_op = copy_table.u4;
      break;
    case NX_BA_FP8_E4M3:
      fill_op = fill_table.f8e4m3;
      copy_op = copy_table.f8e4m3;
      break;
    case NX_BA_FP8_E5M2:
      fill_op = fill_table.f8e5m2;
      copy_op = copy_table.f8e5m2;
      break;
    case NX_BA_COMPLEX16:
      fill_op = fill_table.c16;
      copy_op = copy_table.c16;
      break;
    case NX_BA_QINT8:
      fill_op = fill_table.qi8;
      copy_op = copy_table.qi8;
      break;
    case NX_BA_QUINT8:
      fill_op = fill_table.qu8;
      copy_op = copy_table.qu8;
      break;
    default:
      free(pad_before);
      cleanup_ndarray(&input);
      cleanup_ndarray(&output);
      caml_failwith("pad: unsupported dtype");
  }

  // Convert fill_value to C type
  union {
    int8_t i8;
    uint8_t u8;
    int16_t i16;
    uint16_t u16;
    int32_t i32;
    int64_t i64;
    intnat inat;
    float f32;
    double f64;
    complex32 c32;
    complex64 c64;
    uint16_t f16;
    caml_ba_bfloat16 bf16;
    caml_ba_fp8_e4m3 f8e4m3;
    caml_ba_fp8_e5m2 f8e5m2;
    caml_ba_complex16 c16;
    uint8_t bool_val;
    int i4_val;
    caml_ba_qint8 qi8;
    caml_ba_quint8 qu8;
  } fill_c;
  void *fill_p = &fill_c;
  switch (kind) {
    case CAML_BA_SINT8:
      fill_c.i8 = (int8_t)Long_val(v_fill);
      break;
    case CAML_BA_UINT8:
      fill_c.u8 = (uint8_t)Long_val(v_fill);
      break;
    case CAML_BA_SINT16:
      fill_c.i16 = (int16_t)Long_val(v_fill);
      break;
    case CAML_BA_UINT16:
      fill_c.u16 = (uint16_t)Long_val(v_fill);
      break;
    case CAML_BA_INT32:
      fill_c.i32 = Int32_val(v_fill);
      break;
    case CAML_BA_INT64:
      fill_c.i64 = Int64_val(v_fill);
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      fill_c.inat = Long_val(v_fill);
      break;
    case CAML_BA_FLOAT32:
      fill_c.f32 = (float)Double_val(v_fill);
      break;
    case CAML_BA_FLOAT64:
      fill_c.f64 = Double_val(v_fill);
      break;
    case CAML_BA_COMPLEX32:
      // For complex types, v_fill is a Complex.t record {re: float; im: float}
      if (Is_block(v_fill)) {
        // Complex record - use Double_field to access float fields directly
        fill_c.c32 = (float)Double_field(v_fill, 0) +
                     I * (float)Double_field(v_fill, 1);
      } else {
        // Should not happen for complex types, but handle gracefully
        fill_c.c32 = 0.0f + I * 0.0f;
      }
      break;
    case CAML_BA_COMPLEX64:
      if (Is_block(v_fill)) {
        // Complex record - use Double_field to access float fields directly
        fill_c.c64 = Double_field(v_fill, 0) + I * Double_field(v_fill, 1);
      } else {
        // Should not happen for complex types, but handle gracefully
        fill_c.c64 = 0.0 + I * 0.0;
      }
      break;
    case CAML_BA_FLOAT16:
      fill_c.f16 = float_to_half((float)Double_val(v_fill));
      break;
    case NX_BA_BFLOAT16:
      fill_c.bf16 = float_to_bfloat16((float)Double_val(v_fill));
      break;
    case NX_BA_FP8_E4M3:
      fill_c.f8e4m3 = float_to_fp8_e4m3((float)Double_val(v_fill));
      break;
    case NX_BA_FP8_E5M2:
      fill_c.f8e5m2 = float_to_fp8_e5m2((float)Double_val(v_fill));
      break;
    case NX_BA_COMPLEX16:
      if (Is_block(v_fill)) {
        // Complex record - use Double_field to access float fields directly
        fill_c.c16.re = float_to_half((float)Double_field(v_fill, 0));
        fill_c.c16.im = float_to_half((float)Double_field(v_fill, 1));
      } else {
        // Should not happen for complex types, but handle gracefully
        fill_c.c16.re = float_to_half(0.0f);
        fill_c.c16.im = float_to_half(0.0f);
      }
      break;
    case NX_BA_BOOL:
      fill_c.bool_val = Bool_val(v_fill) ? 1 : 0;
      break;
    case NX_BA_INT4:
      fill_c.i4_val = CLAMP_I4(Long_val(v_fill));
      break;
    case NX_BA_UINT4:
      fill_c.i4_val = CLAMP_U4(Long_val(v_fill));
      break;
    case NX_BA_QINT8:
      fill_c.qi8 = (caml_ba_qint8)Long_val(v_fill);
      break;
    case NX_BA_QUINT8:
      fill_c.qu8 = (caml_ba_quint8)Long_val(v_fill);
      break;
  }

  caml_enter_blocking_section();
  fill_op(&output, fill_p);
  copy_op(&input, &output, pad_before);
  caml_leave_blocking_section();

  free(pad_before);
  cleanup_ndarray(&input);
  cleanup_ndarray(&output);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_cat(value v_inputs, value v_axis, value v_output) {
  CAMLparam3(v_inputs, v_axis, v_output);
  int axis = Int_val(v_axis);
  ndarray_t output = extract_ndarray(v_output);
  int ndim = output.ndim;
  if (axis < 0 || axis >= ndim) {
    cleanup_ndarray(&output);
    caml_failwith("cat: invalid axis");
  }

  value v_first_data = Field(Field(v_inputs, 0), FFI_TENSOR_DATA);
  struct caml_ba_array *ba = Caml_ba_array_val(v_first_data);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  value v_output_data = Field(v_output, FFI_TENSOR_DATA);
  int kind_out = Caml_ba_array_val(v_output_data)->flags & CAML_BA_KIND_MASK;
  if (kind != kind_out) {
    cleanup_ndarray(&output);
    caml_failwith("cat: dtype mismatch");
  }

  copy_op_t copy_op = NULL;
  switch (kind) {
    case CAML_BA_SINT8:
      copy_op = copy_table.i8;
      break;
    case CAML_BA_UINT8:
      copy_op = copy_table.u8;
      break;
    case CAML_BA_SINT16:
      copy_op = copy_table.i16;
      break;
    case CAML_BA_UINT16:
      copy_op = copy_table.u16;
      break;
    case CAML_BA_INT32:
      copy_op = copy_table.i32;
      break;
    case CAML_BA_INT64:
      copy_op = copy_table.i64;
      break;
    case CAML_BA_CAML_INT:
    case CAML_BA_NATIVE_INT:
      copy_op = copy_table.inat;
      break;
    case CAML_BA_FLOAT16:
      copy_op = copy_table.f16;
      break;
    case CAML_BA_FLOAT32:
      copy_op = copy_table.f32;
      break;
    case CAML_BA_FLOAT64:
      copy_op = copy_table.f64;
      break;
    case CAML_BA_COMPLEX32:
      copy_op = copy_table.c32;
      break;
    case CAML_BA_COMPLEX64:
      copy_op = copy_table.c64;
      break;
    case NX_BA_BFLOAT16:
      copy_op = copy_table.bf16;
      break;
    case NX_BA_BOOL:
      copy_op = copy_table.bool_;
      break;
    case NX_BA_INT4:
      copy_op = copy_table.i4;
      break;
    case NX_BA_UINT4:
      copy_op = copy_table.u4;
      break;
    case NX_BA_FP8_E4M3:
      copy_op = copy_table.f8e4m3;
      break;
    case NX_BA_FP8_E5M2:
      copy_op = copy_table.f8e5m2;
      break;
    case NX_BA_COMPLEX16:
      copy_op = copy_table.c16;
      break;
    case NX_BA_QINT8:
      copy_op = copy_table.qi8;
      break;
    case NX_BA_QUINT8:
      copy_op = copy_table.qu8;
      break;
    default:
      cleanup_ndarray(&output);
      caml_failwith("cat: unsupported dtype");
  }

  long *pad_before = (long *)malloc(ndim * sizeof(long));
  if (!pad_before) {
    cleanup_ndarray(&output);
    caml_failwith("cat: allocation failed");
  }

  long current = 0;
  value tail = v_inputs;
  while (tail != Val_int(0)) {  // Empty list is Val_int(0) in OCaml
    value v_in = Field(tail, 0);
    ndarray_t in = extract_ndarray(v_in);
    if (in.ndim != ndim) {
      free(pad_before);
      cleanup_ndarray(&in);
      cleanup_ndarray(&output);
      caml_failwith("cat: ndim mismatch");
    }
    for (int i = 0; i < ndim; i++) {
      pad_before[i] = (i == axis) ? current : 0;
      if (i != axis && in.shape[i] != output.shape[i]) {
        free(pad_before);
        cleanup_ndarray(&in);
        cleanup_ndarray(&output);
        caml_failwith("cat: shape mismatch");
      }
    }
    copy_op(&in, &output, pad_before);
    current += in.shape[axis];
    cleanup_ndarray(&in);
    tail = Field(tail, 1);
  }
  if (current != output.shape[axis]) {
    free(pad_before);
    cleanup_ndarray(&output);
    caml_failwith("cat: concatenated size mismatch");
  }

  free(pad_before);
  cleanup_ndarray(&output);
  CAMLreturn(Val_unit);
}