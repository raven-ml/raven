// PRNG operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <stdint.h>
#include <string.h>

#include "nx_c_shared.h"

// Type definitions for binary operations (reused structure from binary ops)
typedef void (*binary_op_t)(const ndarray_t *, const ndarray_t *, ndarray_t *);

// Dispatch table for each type (only int32 supported for threefry)
typedef struct {
  binary_op_t i8, u8, i16, u16, i32, i64, u32, u64, inat;
  binary_op_t f16, f32, f64;
  binary_op_t c32, c64;
  binary_op_t bf16, bool_, i4, u4, f8e4m3, f8e5m2;
} binary_op_table;

// Threefry2x32 definitions
typedef uint32_t u32_t;

typedef struct {
  u32_t v[2];
} tfry_ctr_t;

typedef tfry_ctr_t tfry_key_t;

#define ROTL_32(x, r) (((x) << (r)) | ((x) >> (32u - (r))))

static tfry_ctr_t threefry2x32(tfry_key_t key, tfry_ctr_t ctr) {
  tfry_ctr_t X;
  u32_t ks[3];
  ks[0] = key.v[0];
  ks[1] = key.v[1];
  ks[2] = 0x1BD11BDA ^ ks[0] ^ ks[1];

  u32_t X0 = ctr.v[0] + ks[0];
  u32_t X1 = ctr.v[1] + ks[1];

  const int rots[4] = {13, 17, 13, 8};
  const int shifts[4] = {16, 13, 16, 13};

  for (int r = 0; r < 20; r++) {
    X0 += X1;
    X1 = ROTL_32(X1, rots[r % 4]);
    X1 ^= X0;
    X0 = ROTL_32(X0, shifts[r % 4]);
    if ((r + 1) % 4 == 0) {
      int s = (r + 1) / 4;
      X0 += ks[s % 3];
      X1 += ks[(s + 1) % 3] + (u32_t)s;
    }
  }

  X.v[0] = X0;
  X.v[1] = X1;
  return X;
}

// Threefry implementation for int32 (only supported type)
static void nx_c_threefry_i32(const ndarray_t *key_p, const ndarray_t *ctr_p,
                              ndarray_t *out_p) {
  if (!key_p || !ctr_p || !out_p) {
    caml_failwith("nx_c_threefry_i32: null pointer");
  }

  ndarray_t key = *key_p;
  ndarray_t ctr = *ctr_p;
  ndarray_t out = *out_p;

  // Dimension check already done before blocking section in dispatch_binary_op

  long total_vectors = total_elements_safe(&key) / 2;
  if (total_vectors == 0) return;

  long last_stride_key = key.strides[key.ndim - 1];
  long last_stride_ctr = ctr.strides[ctr.ndim - 1];
  long last_stride_out = out.strides[out.ndim - 1];

  int prefix_ndim = key.ndim - 1;
  key.ndim = prefix_ndim;
  ctr.ndim = prefix_ndim;
  out.ndim = prefix_ndim;

  if (is_fully_contiguous(key_p, ctr_p, out_p) &&
      key_p->strides[key_p->ndim - 1] == 1 &&
      ctr_p->strides[ctr_p->ndim - 1] == 1 &&
      out_p->strides[out_p->ndim - 1] == 1) {
    _Pragma(
        "omp parallel for simd if(total_vectors > 1000)") for (long i = 0;
                                                               i <
                                                               total_vectors;
                                                               i++) {
      long off = i * 2;
      tfry_key_t k;
      tfry_ctr_t c;
      int32_t *key_data = (int32_t *)key.data;
      int32_t *ctr_data = (int32_t *)ctr.data;
      int32_t *out_data = (int32_t *)out.data;
      k.v[0] = (u32_t)key_data[key.offset + off];
      k.v[1] = (u32_t)key_data[key.offset + off + 1];
      c.v[0] = (u32_t)ctr_data[ctr.offset + off];
      c.v[1] = (u32_t)ctr_data[ctr.offset + off + 1];
      tfry_ctr_t res = threefry2x32(k, c);
      out_data[out.offset + off] = (int32_t)res.v[0];
      out_data[out.offset + off + 1] = (int32_t)res.v[1];
    }
  } else {
    nd_iterator_t it;
    nd_iterator_init_safe(&it, &key, &ctr, &out);
    do {
      long key_base, ctr_base, out_base;
      nd_iterator_get_offsets(&it, &key_base, &ctr_base, &out_base);
      long key_off0 = key.offset + key_base;
      long key_off1 = key_off0 + last_stride_key;
      long ctr_off0 = ctr.offset + ctr_base;
      long ctr_off1 = ctr_off0 + last_stride_ctr;
      long out_off0 = out.offset + out_base;
      long out_off1 = out_off0 + last_stride_out;
      tfry_key_t k;
      tfry_ctr_t c;
      int32_t *key_data = (int32_t *)key.data;
      int32_t *ctr_data = (int32_t *)ctr.data;
      int32_t *out_data = (int32_t *)out.data;
      k.v[0] = (u32_t)key_data[key_off0];
      k.v[1] = (u32_t)key_data[key_off1];
      c.v[0] = (u32_t)ctr_data[ctr_off0];
      c.v[1] = (u32_t)ctr_data[ctr_off1];
      tfry_ctr_t res = threefry2x32(k, c);
      out_data[out_off0] = (int32_t)res.v[0];
      out_data[out_off1] = (int32_t)res.v[1];
    } while (nd_iterator_next(&it));
    nd_iterator_destroy(&it);
  }
}

// Build dispatch table (only i32 supported)
static const binary_op_table threefry_table = {.i8 = NULL,
                                               .u8 = NULL,
                                               .i16 = NULL,
                                               .u16 = NULL,
                                               .i32 = nx_c_threefry_i32,
                                               .i64 = NULL,
                                               .u32 = nx_c_threefry_i32,
                                               .u64 = NULL,
                                               .inat = NULL,
                                               .f16 = NULL,
                                               .f32 = NULL,
                                               .f64 = NULL,
                                               .c32 = NULL,
                                               .c64 = NULL,
                                               .bf16 = NULL,
                                               .bool_ = NULL,
                                               .i4 = NULL,
                                               .u4 = NULL,
                                               .f8e4m3 = NULL,
                                               .f8e5m2 = NULL};

// Reuse dispatch from binary (compatible structure)
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

  // For threefry, validate that last dimension is 2 before blocking section
  if (strcmp(op_name, "threefry") == 0) {
    if (x.ndim < 1 || x.shape[x.ndim - 1] != 2 ||
        y.shape[y.ndim - 1] != 2 || z.shape[z.ndim - 1] != 2) {
      cleanup_ndarray(&x);
      cleanup_ndarray(&y);
      cleanup_ndarray(&z);
      caml_failwith("threefry: last dimension must be 2");
    }
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

CAMLprim value caml_nx_threefry(value v_x, value v_y, value v_z) {
  CAMLparam3(v_x, v_y, v_z);
  dispatch_binary_op(v_x, v_y, v_z, &threefry_table, "threefry");
  CAMLreturn(Val_unit);
}
