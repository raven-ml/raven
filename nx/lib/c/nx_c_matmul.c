// Matrix multiplication for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Type definitions for matmul operations
typedef void (*matmul_op_t)(const ndarray_t *, const ndarray_t *, ndarray_t *);

// Dispatch table for each type
typedef struct {
  matmul_op_t i8, u8, i16, u16, i32, i64, inat;
  matmul_op_t f16, f32, f64;
  matmul_op_t c32, c64;
  matmul_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2, c16, qi8, qu8;
} matmul_op_table;

// Macro to generate all standard type variants for matmul
#define GENERATE_MATMUL_OP(suffix, T, ACCUM_T, CAST) \
  MATMUL_OP_FOR_TYPE(suffix, T, ACCUM_T, CAST)

// Helper to iterate over batch dimensions with a kernel function for matmul
typedef void (*matmul_kernel_t)(void *, long, long, long, void *, long, long,
                                long, void *, long, long, long, long, long,
                                long);

static inline void iterate_batch(
    const long *batch_shape, int batch_nd, const long *batch_strides_a,
    const long *batch_strides_b, const long *batch_strides_c, void *a_data,
    void *b_data, void *c_data, long a_off, long b_off, long c_off, long a_rs,
    long a_cs, long b_rs, long b_cs, long c_rs, long c_cs, long m, long k,
    long n, matmul_kernel_t kernel) {
  if (batch_nd <= 0) {
    kernel(a_data, a_off, a_rs, a_cs, b_data, b_off, b_rs, b_cs, c_data, c_off,
           c_rs, c_cs, m, k, n);
    return;
  }

  int *coords = (int *)calloc(batch_nd, sizeof(int));
  if (!coords) {
    caml_failwith("iterate_batch: allocation failed");
  }

  bool done = false;
  while (!done) {
    long a_batch_off = a_off;
    long b_batch_off = b_off;
    long c_batch_off = c_off;

    for (int i = 0; i < batch_nd; i++) {
      a_batch_off += coords[i] * batch_strides_a[i];
      b_batch_off += coords[i] * batch_strides_b[i];
      c_batch_off += coords[i] * batch_strides_c[i];
    }

    kernel(a_data, a_batch_off, a_rs, a_cs, b_data, b_batch_off, b_rs, b_cs,
           c_data, c_batch_off, c_rs, c_cs, m, k, n);

    // Advance to next position
    done = true;
    for (int i = batch_nd - 1; i >= 0; i--) {
      coords[i]++;
      if (coords[i] < batch_shape[i]) {
        done = false;
        break;
      }
      coords[i] = 0;
    }
  }

  free(coords);
}

// Generic matmul kernel
#define MATMUL_OP_KERNEL(suffix, T, ACCUM_T, CAST)                            \
  static void nx_c_matmul_##suffix##_kernel(                                  \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,           \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,  \
      long c_cs, long m, long k, long n) {                                    \
    T *a = (T *)a_data;                                                       \
    T *b = (T *)b_data;                                                       \
    T *c = (T *)c_data;                                                       \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)") for (long i = 0; \
                                                                  i < m;      \
                                                                  i++) {      \
      for (long j = 0; j < n; j++) {                                          \
        ACCUM_T sum = 0;                                                      \
        for (long p = 0; p < k; p++) {                                        \
          sum += (ACCUM_T)a[a_off + i * a_rs + p * a_cs] *                    \
                 (ACCUM_T)b[b_off + p * b_rs + j * b_cs];                     \
        }                                                                     \
        c[c_off + i * c_rs + j * c_cs] = CAST(sum);                           \
      }                                                                       \
    }                                                                         \
  }

// Generic matmul implementation
#define MATMUL_OP_IMPL(suffix, ELEM_SIZE)                                      \
  static void nx_c_matmul_##suffix(const ndarray_t *a, const ndarray_t *b,     \
                                   ndarray_t *c) {                             \
    if (!a || !b || !c) {                                                      \
      caml_failwith("nx_c_matmul_" #suffix ": null pointer");                  \
    }                                                                          \
    int nd = a->ndim > b->ndim ? a->ndim : b->ndim;                            \
    if (c->ndim != nd) {                                                       \
      caml_failwith("nx_c_matmul_" #suffix ": output ndim mismatch");          \
    }                                                                          \
    if (a->ndim < 2 || b->ndim < 2) {                                          \
      caml_failwith("nx_c_matmul_" #suffix ": input ndim < 2");                \
    }                                                                          \
    long m = a->shape[a->ndim - 2];                                            \
    long k = a->shape[a->ndim - 1];                                            \
    long kk = b->shape[b->ndim - 2];                                           \
    long n = b->shape[b->ndim - 1];                                            \
    if (k != kk) {                                                             \
      caml_failwith("nx_c_matmul_" #suffix ": inner dimension mismatch");      \
    }                                                                          \
    if (c->shape[c->ndim - 2] != m || c->shape[c->ndim - 1] != n) {            \
      caml_failwith("nx_c_matmul_" #suffix ": output shape mismatch");         \
    }                                                                          \
    int batch_nd = nd - 2;                                                     \
    long *batch_shape = (long *)calloc(batch_nd, sizeof(long));                \
    long *batch_strides_a = (long *)calloc(batch_nd, sizeof(long));            \
    long *batch_strides_b = (long *)calloc(batch_nd, sizeof(long));            \
    long *batch_strides_c = (long *)calloc(batch_nd, sizeof(long));            \
    if (!batch_shape || !batch_strides_a || !batch_strides_b ||                \
        !batch_strides_c) {                                                    \
      caml_failwith("nx_c_matmul_" #suffix ": allocation failed");             \
    }                                                                          \
    int a_batch_offset = nd - a->ndim;                                         \
    int b_batch_offset = nd - b->ndim;                                         \
    for (int i = 0; i < batch_nd; i++) {                                       \
      long sa = 1, sb = 1;                                                     \
      long stra = 0, strb = 0;                                                 \
      if (i >= a_batch_offset) {                                               \
        int a_i = i - a_batch_offset;                                          \
        sa = a->shape[a_i];                                                    \
        stra = a->strides[a_i];                                                \
      }                                                                        \
      if (i >= b_batch_offset) {                                               \
        int b_i = i - b_batch_offset;                                          \
        sb = b->shape[b_i];                                                    \
        strb = b->strides[b_i];                                                \
      }                                                                        \
      if (sa != sb && sa != 1 && sb != 1) {                                    \
        free(batch_shape);                                                     \
        free(batch_strides_a);                                                 \
        free(batch_strides_b);                                                 \
        free(batch_strides_c);                                                 \
        caml_failwith("nx_c_matmul_" #suffix ": batch shape mismatch");        \
      }                                                                        \
      long s = sa > sb ? sa : sb;                                              \
      batch_shape[i] = s;                                                      \
      batch_strides_a[i] = (sa == 1) ? 0 : stra;                               \
      batch_strides_b[i] = (sb == 1) ? 0 : strb;                               \
      batch_strides_c[i] = c->strides[i];                                      \
      if (c->shape[i] != s) {                                                  \
        free(batch_shape);                                                     \
        free(batch_strides_a);                                                 \
        free(batch_strides_b);                                                 \
        free(batch_strides_c);                                                 \
        caml_failwith("nx_c_matmul_" #suffix ": output batch shape mismatch"); \
      }                                                                        \
    }                                                                          \
    long a_rs = a->strides[a->ndim - 2];                                       \
    long a_cs = a->strides[a->ndim - 1];                                       \
    long b_rs = b->strides[b->ndim - 2];                                       \
    long b_cs = b->strides[b->ndim - 1];                                       \
    long c_rs = c->strides[c->ndim - 2];                                       \
    long c_cs = c->strides[c->ndim - 1];                                       \
    void *a_data = (char *)a->data + (ELEM_SIZE ? a->offset * ELEM_SIZE : a->offset / 2);  \
    void *b_data = (char *)b->data + (ELEM_SIZE ? b->offset * ELEM_SIZE : b->offset / 2);  \
    void *c_data = (char *)c->data + (ELEM_SIZE ? c->offset * ELEM_SIZE : c->offset / 2);  \
    caml_enter_blocking_section();                                             \
    iterate_batch(batch_shape, batch_nd, batch_strides_a, batch_strides_b,     \
                  batch_strides_c, a_data, b_data, c_data, 0, 0, 0, a_rs,      \
                  a_cs, b_rs, b_cs, c_rs, c_cs, m, k, n,                       \
                  nx_c_matmul_##suffix##_kernel);                              \
    caml_leave_blocking_section();                                             \
    free(batch_shape);                                                         \
    free(batch_strides_a);                                                     \
    free(batch_strides_b);                                                     \
    free(batch_strides_c);                                                     \
  }

// Macro to generate both kernel and implementation for matmul
#define MATMUL_OP_FOR_TYPE(suffix, T, ACCUM_T, CAST) \
  MATMUL_OP_KERNEL(suffix, T, ACCUM_T, CAST)         \
  MATMUL_OP_IMPL(suffix, sizeof(T))

// Low-precision float kernel (convert to float for mul/acc)
#define LOW_PREC_MATMUL_KERNEL(suffix, T, TO_FLOAT, FROM_FLOAT)               \
  static void nx_c_matmul_##suffix##_kernel(                                  \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,           \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,  \
      long c_cs, long m, long k, long n) {                                    \
    T *a = (T *)a_data;                                                       \
    T *b = (T *)b_data;                                                       \
    T *c = (T *)c_data;                                                       \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)") for (long i = 0; \
                                                                  i < m;      \
                                                                  i++) {      \
      for (long j = 0; j < n; j++) {                                          \
        float sum = 0.0f;                                                     \
        for (long p = 0; p < k; p++) {                                        \
          float aa = TO_FLOAT(a[a_off + i * a_rs + p * a_cs]);                \
          float bb = TO_FLOAT(b[b_off + p * b_rs + j * b_cs]);                \
          sum += aa * bb;                                                     \
        }                                                                     \
        c[c_off + i * c_rs + j * c_cs] = FROM_FLOAT(sum);                     \
      }                                                                       \
    }                                                                         \
  }

// For low-precision, use the impl with the special kernel
#define LOW_PREC_MATMUL_IMPL(suffix, T) MATMUL_OP_IMPL(suffix, sizeof(T))

// Complex16 matmul kernel
#define COMPLEX16_MATMUL_KERNEL                                               \
  static void nx_c_matmul_c16_kernel(                                         \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,           \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,  \
      long c_cs, long m, long k, long n) {                                    \
    caml_ba_complex16 *a = (caml_ba_complex16 *)a_data;                       \
    caml_ba_complex16 *b = (caml_ba_complex16 *)b_data;                       \
    caml_ba_complex16 *c = (caml_ba_complex16 *)c_data;                       \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)") for (long i = 0; \
                                                                  i < m;      \
                                                                  i++) {      \
      for (long j = 0; j < n; j++) {                                          \
        complex32 sum = 0;                                                    \
        for (long p = 0; p < k; p++) {                                        \
          complex32 aa =                                                      \
              complex16_to_complex32(a[a_off + i * a_rs + p * a_cs]);         \
          complex32 bb =                                                      \
              complex16_to_complex32(b[b_off + p * b_rs + j * b_cs]);         \
          sum += aa * bb;                                                     \
        }                                                                     \
        c[c_off + i * c_rs + j * c_cs] = complex32_to_complex16(sum);         \
      }                                                                       \
    }                                                                         \
  }

// Special implementation for int4 (packed, unpack/mul/acc/pack with saturation)
#define INT4_MATMUL_IMPL(signedness, suffix)                                   \
  static void nx_c_matmul_##suffix##_kernel(                                   \
      void *a_data, long a_off, long a_rs, long a_cs, void *b_data,            \
      long b_off, long b_rs, long b_cs, void *c_data, long c_off, long c_rs,   \
      long c_cs, long m, long k, long n) {                                     \
    uint8_t *a = (uint8_t *)a_data;                                            \
    uint8_t *b = (uint8_t *)b_data;                                            \
    uint8_t *c = (uint8_t *)c_data;                                            \
    _Pragma("omp parallel for collapse(2) if(m * n > 1000)") for (long i = 0;  \
                                                                  i < m;       \
                                                                  i++) {       \
      for (long j = 0; j < n; j++) {                                           \
        int32_t sum = 0;                                                       \
        for (long p = 0; p < k; p++) {                                         \
          long a_idx = a_off + i * a_rs + p * a_cs;                            \
          long a_byte_off = a_idx / 2;                                         \
          int a_nib_off = a_idx % 2;                                           \
          int aa =                                                             \
              a_nib_off                                                        \
                  ? (signedness ? (int8_t)(a[a_byte_off] >> 4)                 \
                                : ((a[a_byte_off] >> 4) & 0x0F))               \
                  : (signedness ? (int8_t)(((a[a_byte_off] & 0x0F) << 4) >> 4) \
                                : (a[a_byte_off] & 0x0F));                     \
          long b_idx = b_off + p * b_rs + j * b_cs;                            \
          long b_byte_off = b_idx / 2;                                         \
          int b_nib_off = b_idx % 2;                                           \
          int bb =                                                             \
              b_nib_off                                                        \
                  ? (signedness ? (int8_t)(b[b_byte_off] >> 4)                 \
                                : ((b[b_byte_off] >> 4) & 0x0F))               \
                  : (signedness ? (int8_t)(((b[b_byte_off] & 0x0F) << 4) >> 4) \
                                : (b[b_byte_off] & 0x0F));                     \
          sum += aa * bb;                                                      \
        }                                                                      \
        int res = signedness ? CLAMP_I4(sum) : CLAMP_U4(sum);                  \
        uint8_t nib = (uint8_t)res & 0x0F;                                     \
        long c_idx = c_off + i * c_rs + j * c_cs;                              \
        long c_byte_off = c_idx / 2;                                           \
        int c_nib_off = c_idx % 2;                                             \
        if (c_nib_off) {                                                       \
          c[c_byte_off] = (c[c_byte_off] & 0x0F) | (nib << 4);                 \
        } else {                                                               \
          c[c_byte_off] = (c[c_byte_off] & 0xF0) | nib;                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  MATMUL_OP_IMPL(suffix, 0)  /* int4 offset is in nibbles, handled in kernel */

// Generate for integer types with wider accumulation
GENERATE_MATMUL_OP(i8, int8_t, int64_t, (int8_t))
GENERATE_MATMUL_OP(u8, uint8_t, uint64_t, (uint8_t))
GENERATE_MATMUL_OP(i16, int16_t, int64_t, (int16_t))
GENERATE_MATMUL_OP(u16, uint16_t, uint64_t, (uint16_t))
GENERATE_MATMUL_OP(i32, int32_t, int64_t, (int32_t))
GENERATE_MATMUL_OP(i64, int64_t, int64_t, (int64_t))
GENERATE_MATMUL_OP(inat, intnat, int64_t, (intnat))
GENERATE_MATMUL_OP(qi8, caml_ba_qint8, int64_t, (caml_ba_qint8))
GENERATE_MATMUL_OP(qu8, caml_ba_quint8, uint64_t, (caml_ba_quint8))
GENERATE_MATMUL_OP(bool_, caml_ba_bool, uint64_t, (caml_ba_bool))

// Float types with same-type accumulation
GENERATE_MATMUL_OP(f32, float, float, )
GENERATE_MATMUL_OP(f64, double, double, )

// Complex types with same-type accumulation
GENERATE_MATMUL_OP(c32, complex32, complex32, )
GENERATE_MATMUL_OP(c64, complex64, complex64, )

// Low-precision floats
LOW_PREC_MATMUL_KERNEL(f16, uint16_t, half_to_float, float_to_half)
LOW_PREC_MATMUL_IMPL(f16, uint16_t)
LOW_PREC_MATMUL_KERNEL(bf16, caml_ba_bfloat16, bfloat16_to_float,
                       float_to_bfloat16)
LOW_PREC_MATMUL_IMPL(bf16, caml_ba_bfloat16)
LOW_PREC_MATMUL_KERNEL(f8e4m3, caml_ba_fp8_e4m3, fp8_e4m3_to_float,
                       float_to_fp8_e4m3)
LOW_PREC_MATMUL_IMPL(f8e4m3, caml_ba_fp8_e4m3)
LOW_PREC_MATMUL_KERNEL(f8e5m2, caml_ba_fp8_e5m2, fp8_e5m2_to_float,
                       float_to_fp8_e5m2)
LOW_PREC_MATMUL_IMPL(f8e5m2, caml_ba_fp8_e5m2)

// Complex16
COMPLEX16_MATMUL_KERNEL
MATMUL_OP_IMPL(c16, sizeof(caml_ba_complex16))

// Int4/Uint4
INT4_MATMUL_IMPL(1, i4)
INT4_MATMUL_IMPL(0, u4)

// Build dispatch table
#define BUILD_DISPATCH_TABLE(name)                                             \
  static const matmul_op_table name##_table = {.i8 = nx_c_##name##_i8,         \
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

BUILD_DISPATCH_TABLE(matmul);

// Generic dispatch function for matmul operations
static void dispatch_matmul_op(value v_a, value v_b, value v_c,
                               const matmul_op_table *table,
                               const char *op_name) {
  // Extract ndarrays from FFI tensors
  ndarray_t A = extract_ndarray(v_a);
  ndarray_t B = extract_ndarray(v_b);
  ndarray_t C = extract_ndarray(v_c);

  // Get bigarray kind from the data field
  value v_a_data = Field(v_a, FFI_TENSOR_DATA);
  value v_b_data = Field(v_b, FFI_TENSOR_DATA);
  value v_c_data = Field(v_c, FFI_TENSOR_DATA);

  struct caml_ba_array *ba = Caml_ba_array_val(v_a_data);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  // Check kinds match for b and c
  int kind_b = Caml_ba_array_val(v_b_data)->flags & CAML_BA_KIND_MASK;
  int kind_c = Caml_ba_array_val(v_c_data)->flags & CAML_BA_KIND_MASK;
  if (kind != kind_b || kind != kind_c) {
    cleanup_ndarray(&A);
    cleanup_ndarray(&B);
    cleanup_ndarray(&C);
    caml_failwith("dtype mismatch");
  }

  // Select operation based on dtype
  matmul_op_t op = NULL;
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
      cleanup_ndarray(&A);
      cleanup_ndarray(&B);
      cleanup_ndarray(&C);
      caml_failwith("dispatch_matmul_op: unsupported dtype");
  }

  if (!op) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: operation not supported for dtype",
             op_name);
    cleanup_ndarray(&A);
    cleanup_ndarray(&B);
    cleanup_ndarray(&C);
    caml_failwith(msg);
  }

  // Perform the operation
  op(&A, &B, &C);

  // Clean up if heap allocated
  cleanup_ndarray(&A);
  cleanup_ndarray(&B);
  cleanup_ndarray(&C);
}

// ============================================================================
// OCaml FFI Stubs
// ============================================================================

CAMLprim value caml_nx_matmul(value v_a, value v_b, value v_c) {
  CAMLparam3(v_a, v_b, v_c);
  dispatch_matmul_op(v_a, v_b, v_c, &matmul_table, "matmul");
  CAMLreturn(Val_unit);
}