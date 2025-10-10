// Ternary operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Type definitions for ternary operations
typedef void (*ternary_op_t)(const ndarray_t *, const ndarray_t *,
                             const ndarray_t *, ndarray_t *);

// Dispatch table for each type
typedef struct {
  ternary_op_t i8, u8, i16, u16, i32, i64, inat;
  ternary_op_t f16, f32, f64;
  ternary_op_t c32, c64;
  ternary_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2, c16, qi8, qu8;
} ternary_op_table;

// Iterator for ternary operations (4 arrays)
typedef struct {
  int ndim;
  int *shape;
  int *coords;
  int *cond_strides;
  int *x_strides;
  int *y_strides;
  int *z_strides;
} nd_iterator_ternary_t;

// Check if all 4 arrays are fully contiguous
static inline bool is_fully_contiguous_ternary(const ndarray_t *cond,
                                               const ndarray_t *x,
                                               const ndarray_t *y,
                                               const ndarray_t *z) {
  if (!cond || !x || !y || !z || cond->ndim != x->ndim || x->ndim != y->ndim ||
      y->ndim != z->ndim)
    return false;
  if (cond->ndim == 0) return true;

  // Check C-contiguous layout
  int expected_stride = 1;
  for (int i = cond->ndim - 1; i >= 0; i--) {
    if (cond->strides[i] != expected_stride ||
        x->strides[i] != expected_stride || y->strides[i] != expected_stride ||
        z->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= cond->shape[i];
  }
  return true;
}

static inline void nd_iterator_init_ternary(nd_iterator_ternary_t *it,
                                            const ndarray_t *cond,
                                            const ndarray_t *x,
                                            const ndarray_t *y,
                                            const ndarray_t *z) {
  if (!it || !cond || !x || !y || !z) {
    caml_failwith("nd_iterator_init_ternary: null pointer");
  }
  if (cond->ndim != x->ndim || x->ndim != y->ndim || y->ndim != z->ndim) {
    caml_failwith("nd_iterator_init_ternary: dimension mismatch");
  }
  it->ndim = cond->ndim;
  it->shape = cond->shape;
  it->coords = (int *)calloc(cond->ndim, sizeof(int));
  it->cond_strides = cond->strides;
  it->x_strides = x->strides;
  it->y_strides = y->strides;
  it->z_strides = z->strides;
  if (!it->coords) {
    caml_failwith("nd_iterator_init_ternary: allocation failed");
  }
}

static inline void nd_iterator_get_offsets_ternary(
    const nd_iterator_ternary_t *it, long *cond_off, long *x_off, long *y_off,
    long *z_off) {
  *cond_off = 0;
  *x_off = 0;
  *y_off = 0;
  *z_off = 0;
  for (int i = 0; i < it->ndim; i++) {
    *cond_off += it->coords[i] * it->cond_strides[i];
    *x_off += it->coords[i] * it->x_strides[i];
    *y_off += it->coords[i] * it->y_strides[i];
    *z_off += it->coords[i] * it->z_strides[i];
  }
}

static inline bool nd_iterator_next_ternary(nd_iterator_ternary_t *it) {
  for (int i = it->ndim - 1; i >= 0; i--) {
    it->coords[i]++;
    if (it->coords[i] < it->shape[i]) {
      return true;
    }
    it->coords[i] = 0;
  }
  return false;
}

static inline void nd_iterator_destroy_ternary(nd_iterator_ternary_t *it) {
  if (it && it->coords) {
    free(it->coords);
    it->coords = NULL;
  }
}

// Macro to generate all standard type variants for an operation
#define GENERATE_TERNARY_OP(name)                     \
  TERNARY_OP_FOR_TYPE(name, int8_t, i8)               \
  TERNARY_OP_FOR_TYPE(name, uint8_t, u8)              \
  TERNARY_OP_FOR_TYPE(name, int16_t, i16)             \
  TERNARY_OP_FOR_TYPE(name, uint16_t, u16)            \
  TERNARY_OP_FOR_TYPE(name, int32_t, i32)             \
  TERNARY_OP_FOR_TYPE(name, int64_t, i64)             \
  TERNARY_OP_FOR_TYPE(name, intnat, inat)             \
  TERNARY_OP_FOR_TYPE(name, float, f32)               \
  TERNARY_OP_FOR_TYPE(name, double, f64)              \
  TERNARY_OP_FOR_TYPE(name, complex32, c32)           \
  TERNARY_OP_FOR_TYPE(name, complex64, c64)           \
  TERNARY_OP_FOR_TYPE(name, uint16_t, f16)            \
  TERNARY_OP_FOR_TYPE(name, caml_ba_bfloat16, bf16)   \
  TERNARY_OP_FOR_TYPE(name, caml_ba_fp8_e4m3, f8e4m3) \
  TERNARY_OP_FOR_TYPE(name, caml_ba_fp8_e5m2, f8e5m2) \
  TERNARY_OP_FOR_TYPE(name, caml_ba_complex16, c16)   \
  TERNARY_OP_FOR_TYPE(name, caml_ba_bool, bool_)      \
  TERNARY_OP_FOR_TYPE(name, caml_ba_qint8, qi8)       \
  TERNARY_OP_FOR_TYPE(name, caml_ba_quint8, qu8)

// Macro to build dispatch table
#define BUILD_DISPATCH_TABLE(name)               \
  static const ternary_op_table name##_table = { \
      .i8 = nx_c_##name##_i8,                    \
      .u8 = nx_c_##name##_u8,                    \
      .i16 = nx_c_##name##_i16,                  \
      .u16 = nx_c_##name##_u16,                  \
      .i32 = nx_c_##name##_i32,                  \
      .i64 = nx_c_##name##_i64,                  \
      .inat = nx_c_##name##_inat,                \
      .f16 = nx_c_##name##_f16,                  \
      .f32 = nx_c_##name##_f32,                  \
      .f64 = nx_c_##name##_f64,                  \
      .c32 = nx_c_##name##_c32,                  \
      .c64 = nx_c_##name##_c64,                  \
      .bf16 = nx_c_##name##_bf16,                \
      .bool_ = nx_c_##name##_bool_,              \
      .i4 = nx_c_##name##_i4,                    \
      .u4 = nx_c_##name##_u4,                    \
      .f8e4m3 = nx_c_##name##_f8e4m3,            \
      .f8e5m2 = nx_c_##name##_f8e5m2,            \
      .c16 = nx_c_##name##_c16,                  \
      .qi8 = nx_c_##name##_qi8,                  \
      .qu8 = nx_c_##name##_qu8}

// Helper to iterate over inner dimensions with a kernel function for ternary
// operations
typedef void (*kernel_fn)(void *, void *, void *, void *, long, long, long,
                          long);

static inline void iterate_inner_dims_ternary(
    const ndarray_t *cond, const ndarray_t *x, const ndarray_t *y,
    const ndarray_t *z, long outer_idx, kernel_fn kernel, void *cond_data,
    void *x_data, void *y_data, void *z_data) {
  if (x->ndim <= 1) {
    kernel(cond_data, x_data, y_data, z_data, outer_idx * cond->strides[0],
           outer_idx * x->strides[0], outer_idx * y->strides[0],
           outer_idx * z->strides[0]);
    return;
  }

  long cond_base = outer_idx * cond->strides[0];
  long x_base = outer_idx * x->strides[0];
  long y_base = outer_idx * y->strides[0];
  long z_base = outer_idx * z->strides[0];

  // Create temporary iterator for inner dimensions
  int inner_ndim = x->ndim - 1;
  int *coords = (int *)calloc(inner_ndim, sizeof(int));
  if (!coords) {
    caml_failwith("iterate_inner_dims_ternary: allocation failed");
  }

  // Iterate over inner dimensions
  bool done = false;
  while (!done) {
    long cond_off = cond_base;
    long x_off = x_base;
    long y_off = y_base;
    long z_off = z_base;

    for (int i = 0; i < inner_ndim; i++) {
      cond_off += coords[i] * cond->strides[i + 1];
      x_off += coords[i] * x->strides[i + 1];
      y_off += coords[i] * y->strides[i + 1];
      z_off += coords[i] * z->strides[i + 1];
    }

    kernel(cond_data, x_data, y_data, z_data, cond_off, x_off, y_off, z_off);

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

// Generic ternary operation kernel
#define TERNARY_OP_KERNEL(name, T, suffix)                       \
  static void nx_c_##name##_##suffix##_kernel(                   \
      void *cond_data, void *x_data, void *y_data, void *z_data, \
      long cond_off, long x_off, long y_off, long z_off) {       \
    uint8_t *cond = (uint8_t *)cond_data;                        \
    T *x = (T *)x_data;                                          \
    T *y = (T *)y_data;                                          \
    T *z = (T *)z_data;                                          \
    z[z_off] = cond[cond_off] ? x[x_off] : y[y_off];             \
  }

// Generic ternary operation implementation
#define TERNARY_OP_IMPL(name, T, suffix)                                       \
  static void nx_c_##name##_##suffix(const ndarray_t *cond,                    \
                                     const ndarray_t *x, const ndarray_t *y,   \
                                     ndarray_t *z) {                           \
    if (!cond || !x || !y || !z) {                                             \
      caml_failwith("nx_c_" #name "_" #suffix ": null pointer");               \
    }                                                                          \
    long total = total_elements_safe(x);                                       \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_fully_contiguous_ternary(cond, x, y, z)) {                          \
      _Pragma("omp parallel for simd if(total > 1000)") for (long i = 0;       \
                                                             i < total; i++) { \
        nx_c_##name##_##suffix##_kernel(cond->data, x->data, y->data, z->data, \
                                        cond->offset + i, x->offset + i,       \
                                        y->offset + i, z->offset + i);         \
      }                                                                        \
    } else if (x->shape[0] > 1 && total / x->shape[0] > 50) {                  \
      _Pragma("omp parallel for if(x->shape[0] > 4)") for (long i = 0;         \
                                                           i < x->shape[0];    \
                                                           i++) {              \
        iterate_inner_dims_ternary(cond, x, y, z, i,                           \
                                   nx_c_##name##_##suffix##_kernel,            \
                                   cond->data, x->data, y->data, z->data);     \
      }                                                                        \
    } else {                                                                   \
      nd_iterator_ternary_t it;                                                \
      nd_iterator_init_ternary(&it, cond, x, y, z);                            \
      do {                                                                     \
        long cond_off, x_off, y_off, z_off;                                    \
        nd_iterator_get_offsets_ternary(&it, &cond_off, &x_off, &y_off,        \
                                        &z_off);                               \
        nx_c_##name##_##suffix##_kernel(                                       \
            cond->data, x->data, y->data, z->data, cond->offset + cond_off,    \
            x->offset + x_off, y->offset + y_off, z->offset + z_off);          \
      } while (nd_iterator_next_ternary(&it));                                 \
      nd_iterator_destroy_ternary(&it);                                        \
    }                                                                          \
  }

// Macro to generate both kernel and implementation for an operation
#define TERNARY_OP_FOR_TYPE(name, T, suffix) \
  TERNARY_OP_KERNEL(name, T, suffix)         \
  TERNARY_OP_IMPL(name, T, suffix)

// Special implementation for int4 (packed, unpack/select/pack)
#define INT4_WHERE_IMPL(signedness, suffix)                                    \
  static void nx_c_where_##suffix##_kernel(                                    \
      void *cond_data, void *x_data, void *y_data, void *z_data,               \
      long cond_off, long x_off, long y_off, long z_off) {                     \
    uint8_t *cond = (uint8_t *)cond_data;                                      \
    uint8_t *x = (uint8_t *)x_data;                                            \
    uint8_t *y = (uint8_t *)y_data;                                            \
    uint8_t *z = (uint8_t *)z_data;                                            \
    long byte_off = z_off / 2;                                                 \
    int nib_off = z_off % 2;                                                   \
    uint8_t *src = cond[cond_off] ? x : y;                                     \
    long src_byte_off = (cond[cond_off] ? x_off : y_off) / 2;                  \
    int src_nib_off = (cond[cond_off] ? x_off : y_off) % 2;                    \
    int a = src_nib_off                                                        \
                ? (signedness ? (int8_t)(src[src_byte_off] >> 4)               \
                              : (src[src_byte_off] >> 4) & 0x0F)               \
                : (signedness ? (int8_t)((src[src_byte_off] & 0x0F) << 4) >> 4 \
                              : src[src_byte_off] & 0x0F);                     \
    uint8_t nib = (uint8_t)a & 0x0F;                                           \
    if (nib_off) {                                                             \
      z[byte_off] = (z[byte_off] & 0x0F) | (nib << 4);                         \
    } else {                                                                   \
      z[byte_off] = (z[byte_off] & 0xF0) | nib;                                \
    }                                                                          \
  }                                                                            \
  static void nx_c_where_##suffix(const ndarray_t *cond, const ndarray_t *x,   \
                                  const ndarray_t *y, ndarray_t *z) {          \
    if (!cond || !x || !y || !z) {                                             \
      caml_failwith("nx_c_where_" #suffix ": null pointer");                   \
    }                                                                          \
    long total = total_elements_safe(x);                                       \
    if (total == 0) return;                                                    \
                                                                               \
    if (is_fully_contiguous_ternary(cond, x, y, z)) {                          \
      void *cond_data = cond->data + cond->offset;                             \
      void *x_data = x->data + x->offset;                                      \
      void *y_data = y->data + y->offset;                                      \
      void *z_data = z->data + z->offset;                                      \
      _Pragma("omp parallel for if(total > 10000)") for (long i = 0;           \
                                                         i < total; i++) {     \
        nx_c_where_##suffix##_kernel(cond_data, x_data, y_data, z_data, i, i,  \
                                     i, i);                                    \
      }                                                                        \
    } else {                                                                   \
      nd_iterator_ternary_t it;                                                \
      nd_iterator_init_ternary(&it, cond, x, y, z);                            \
      void *cond_data = cond->data;                                            \
      void *x_data = x->data;                                                  \
      void *y_data = y->data;                                                  \
      void *z_data = z->data;                                                  \
      do {                                                                     \
        long cond_off, x_off, y_off, z_off;                                    \
        nd_iterator_get_offsets_ternary(&it, &cond_off, &x_off, &y_off,        \
                                        &z_off);                               \
        nx_c_where_##suffix##_kernel(                                          \
            cond_data, x_data, y_data, z_data, cond->offset + cond_off,        \
            x->offset + x_off, y->offset + y_off, z->offset + z_off);          \
      } while (nd_iterator_next_ternary(&it));                                 \
      nd_iterator_destroy_ternary(&it);                                        \
    }                                                                          \
  }

// Generate for where
GENERATE_TERNARY_OP(where)
INT4_WHERE_IMPL(1, i4)
INT4_WHERE_IMPL(0, u4)
BUILD_DISPATCH_TABLE(where);

// Generic dispatch function for ternary operations
static void dispatch_ternary_op(value v_cond, value v_x, value v_y, value v_z,
                                const ternary_op_table *table,
                                const char *op_name) {
  // Extract ndarrays from FFI tensors
  ndarray_t cond = extract_ndarray(v_cond);
  ndarray_t x = extract_ndarray(v_x);
  ndarray_t y = extract_ndarray(v_y);
  ndarray_t z = extract_ndarray(v_z);

  // Check shapes match
  if (cond.ndim != x.ndim || cond.ndim != y.ndim || cond.ndim != z.ndim) {
    cleanup_ndarray(&cond);
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("shape mismatch");
  }
  for (int i = 0; i < cond.ndim; i++) {
    if (cond.shape[i] != x.shape[i] || cond.shape[i] != y.shape[i] ||
        cond.shape[i] != z.shape[i]) {
      cleanup_ndarray(&cond);
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("shape mismatch");
    }
  }

  // Get bigarray kind from the data fields
  value v_cond_data = Field(v_cond, FFI_TENSOR_DATA);
  value v_x_data = Field(v_x, FFI_TENSOR_DATA);
  value v_y_data = Field(v_y, FFI_TENSOR_DATA);
  value v_z_data = Field(v_z, FFI_TENSOR_DATA);

  struct caml_ba_array *ba_cond = Caml_ba_array_val(v_cond_data);
  int kind_cond = nx_ba_get_kind(ba_cond);

  // Assume condition is bool or uint8
  if (kind_cond != NX_BA_BOOL && kind_cond != CAML_BA_UINT8) {
    cleanup_ndarray(&cond);
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("condition must be bool or uint8");
  }

  struct caml_ba_array *ba_x = Caml_ba_array_val(v_x_data);
  int kind = nx_ba_get_kind(ba_x);

  // Check kinds match for x, y, z
  int kind_y = nx_ba_get_kind(Caml_ba_array_val(v_y_data));
  int kind_z = nx_ba_get_kind(Caml_ba_array_val(v_z_data));
  if (kind != kind_y || kind != kind_z) {
    cleanup_ndarray(&cond);
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith("dtype mismatch");
  }

  // Select operation based on dtype
  ternary_op_t op = NULL;
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
      cleanup_ndarray(&cond);
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("dispatch_ternary_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    cleanup_ndarray(&cond);
    cleanup_ndarray(&x);
    cleanup_ndarray(&y);
    cleanup_ndarray(&z);
    caml_failwith(msg);
  }

  // Enter blocking section for potentially long computation
  caml_enter_blocking_section();
  op(&cond, &x, &y, &z);
  caml_leave_blocking_section();

  // Clean up if heap allocated
  cleanup_ndarray(&cond);
  cleanup_ndarray(&x);
  cleanup_ndarray(&y);
  cleanup_ndarray(&z);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_where(value v_cond, value v_x, value v_y, value v_z) {
  CAMLparam4(v_cond, v_x, v_y, v_z);
  dispatch_ternary_op(v_cond, v_x, v_y, v_z, &where_table, "where");
  CAMLreturn(Val_unit);
}
