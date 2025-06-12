/* nx_cblas_stubs.c - OCaml stubs for CBLAS backend */

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdlib.h>

/* =========================== Type Definitions =========================== */

typedef struct {
  void *data;
  int ndim;
  int *shape;
  int *strides;
  int offset;
} strided_array_t;

/* =========================== External Function Declarations
 * =========================== */

/* From nx_cblas_ops_impl.c */
extern void nx_cblas_init_strided_array(strided_array_t *arr, void *data,
                                        int ndim, const int *shape,
                                        const int *strides, int offset);
extern void nx_cblas_free_strided_array(strided_array_t *arr);

/* Float32 operations */
extern void nx_cblas_add_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_sub_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_mul_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_div_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_max_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_pow_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_mod_f32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_neg_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_sqrt_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_sin_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_exp2_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_log2_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_recip_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_copy_f32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_cmplt_f32(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_cmpne_f32(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_reduce_sum_f32(const strided_array_t *x, void *result);
extern void nx_cblas_reduce_max_f32(const strided_array_t *x, void *result);
extern void nx_cblas_reduce_min_f32(const strided_array_t *x, void *result);

/* Float64 operations */
extern void nx_cblas_add_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_sub_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_mul_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_div_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_max_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_pow_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_mod_f64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_neg_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_neg_i32(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_neg_i64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_sqrt_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_sin_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_exp2_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_log2_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_recip_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_copy_f64(const strided_array_t *x, strided_array_t *z);
extern void nx_cblas_cmplt_f64(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_cmplt_i32(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_cmplt_i64(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_cmpne_f64(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_cmpne_i32(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_cmpne_i64(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_reduce_sum_f64(const strided_array_t *x, void *result);
extern void nx_cblas_reduce_max_f64(const strided_array_t *x, void *result);
extern void nx_cblas_reduce_min_f64(const strided_array_t *x, void *result);
extern void nx_cblas_reduce_prod_f32(const strided_array_t *x, void *result);
extern void nx_cblas_reduce_prod_f64(const strided_array_t *x, void *result);

/* Integer operations */
extern void nx_cblas_add_i32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_add_i64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_sub_i32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_sub_i64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_mul_i32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_mul_i64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_div_i32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_div_i64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_max_i32(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_max_i64(const strided_array_t *x, const strided_array_t *y,
                             strided_array_t *z);
extern void nx_cblas_idiv_i32(const strided_array_t *x,
                              const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_idiv_i64(const strided_array_t *x,
                              const strided_array_t *y, strided_array_t *z);

/* Bitwise operations */
extern void nx_cblas_xor_int32(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_xor_int64(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_xor_uint8(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_xor_uint16(const strided_array_t *x,
                                const strided_array_t *y, strided_array_t *z);

extern void nx_cblas_or_int32(const strided_array_t *x,
                              const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_or_int64(const strided_array_t *x,
                              const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_or_uint8(const strided_array_t *x,
                              const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_or_uint16(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);

extern void nx_cblas_and_int32(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_and_int64(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_and_uint8(const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_and_uint16(const strided_array_t *x,
                                const strided_array_t *y, strided_array_t *z);

/* Ternary operations */
extern void nx_cblas_where_f32(const strided_array_t *cond,
                               const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_where_f64(const strided_array_t *cond,
                               const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_where_i32(const strided_array_t *cond,
                               const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);
extern void nx_cblas_where_i64(const strided_array_t *cond,
                               const strided_array_t *x,
                               const strided_array_t *y, strided_array_t *z);

/* Pad operations */
extern void nx_cblas_pad_f32(const strided_array_t *x, strided_array_t *z,
                             const int *pad_config, float fill_value);
extern void nx_cblas_pad_f64(const strided_array_t *x, strided_array_t *z,
                             const int *pad_config, double fill_value);

/* Cast operations */
extern void nx_cblas_cast_f32_to_f64(const strided_array_t *x,
                                     strided_array_t *z);
extern void nx_cblas_cast_f64_to_f32(const strided_array_t *x,
                                     strided_array_t *z);

/* Threefry operation */
extern void nx_cblas_threefry(const strided_array_t *key,
                              const strided_array_t *counter,
                              strided_array_t *z);

/* Gather operations */
extern void nx_cblas_gather_f32(const strided_array_t *data,
                                const strided_array_t *indices, int axis,
                                strided_array_t *z);
extern void nx_cblas_gather_f64(const strided_array_t *data,
                                const strided_array_t *indices, int axis,
                                strided_array_t *z);

/* Scatter operations */
extern void nx_cblas_scatter_f32(const strided_array_t *data_template,
                                 const strided_array_t *indices,
                                 const strided_array_t *updates, int axis,
                                 strided_array_t *z);
extern void nx_cblas_scatter_f64(const strided_array_t *data_template,
                                 const strided_array_t *indices,
                                 const strided_array_t *updates, int axis,
                                 strided_array_t *z);
extern void nx_cblas_scatter_i32(const strided_array_t *data_template,
                                 const strided_array_t *indices,
                                 const strided_array_t *updates, int axis,
                                 strided_array_t *z);
extern void nx_cblas_scatter_i64(const strided_array_t *data_template,
                                 const strided_array_t *indices,
                                 const strided_array_t *updates, int axis,
                                 strided_array_t *z);

/* Generic copy for other types */
extern void nx_cblas_copy_generic(const strided_array_t *x, strided_array_t *z,
                                  size_t elem_size);

/* =========================== Helper Macros =========================== */

/* Use the built-in macro from bigarray.h instead of redefining it */

/* =========================== Helper Functions =========================== */

static size_t get_element_size(int kind) {
  switch (kind) {
    case CAML_BA_FLOAT32:
      return sizeof(float);
    case CAML_BA_FLOAT64:
      return sizeof(double);
    case CAML_BA_SINT8:
      return sizeof(int8_t);
    case CAML_BA_UINT8:
      return sizeof(uint8_t);
    case CAML_BA_SINT16:
      return sizeof(int16_t);
    case CAML_BA_UINT16:
      return sizeof(uint16_t);
    case CAML_BA_INT32:
      return sizeof(int32_t);
    case CAML_BA_INT64:
      return sizeof(int64_t);
    case CAML_BA_CAML_INT:
      return sizeof(intnat);
    case CAML_BA_NATIVE_INT:
      return sizeof(intnat);
    case CAML_BA_COMPLEX32:
      return 2 * sizeof(float);
    case CAML_BA_COMPLEX64:
      return 2 * sizeof(double);
    default:
      return 0;
  }
}

/* Initialize strided array info from OCaml values */
static void init_strided_array(strided_array_t *arr, value vData, int ndim,
                               value vShape, value vStrides, value vOffset) {
  struct caml_ba_array *ba = Caml_ba_array_val(vData);
  void *data = ba->data;
  int offset = Int_val(vOffset);

  int *shape = (int *)malloc(ndim * sizeof(int));
  int *strides = (int *)malloc(ndim * sizeof(int));

  for (int i = 0; i < ndim; i++) {
    shape[i] = Int_val(Field(vShape, i));
    strides[i] = Int_val(Field(vStrides, i));
  }

  nx_cblas_init_strided_array(arr, data, ndim, shape, strides, offset);

  free(shape);
  free(strides);
}

/* =========================== Binary Operations =========================== */

#define DEFINE_BINARY_STUB(name)                                           \
  CAMLprim value nx_##name(value vNdim, value vShape, value vX,            \
                           value vXStrides, value vXOffset, value vY,      \
                           value vYStrides, value vYOffset, value vZ,      \
                           value vZStrides, value vZOffset) {              \
    CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);                      \
    CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);                        \
                                                                           \
    int ndim = Int_val(vNdim);                                             \
    strided_array_t x, y, z;                                               \
                                                                           \
    init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);         \
    init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);         \
    init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);         \
                                                                           \
    caml_enter_blocking_section();                                         \
                                                                           \
    struct caml_ba_array *ba = Caml_ba_array_val(vX);                      \
    int kind = ba->flags & CAML_BA_KIND_MASK;                              \
                                                                           \
    if (kind == CAML_BA_FLOAT32) {                                         \
      nx_cblas_##name##_f32(&x, &y, &z);                                   \
    } else if (kind == CAML_BA_FLOAT64) {                                  \
      nx_cblas_##name##_f64(&x, &y, &z);                                   \
    } else {                                                               \
      caml_leave_blocking_section();                                       \
      nx_cblas_free_strided_array(&x);                                     \
      nx_cblas_free_strided_array(&y);                                     \
      nx_cblas_free_strided_array(&z);                                     \
      caml_failwith(#name ": unsupported dtype");                          \
    }                                                                      \
                                                                           \
    caml_leave_blocking_section();                                         \
                                                                           \
    nx_cblas_free_strided_array(&x);                                       \
    nx_cblas_free_strided_array(&y);                                       \
    nx_cblas_free_strided_array(&z);                                       \
                                                                           \
    CAMLreturn(Val_unit);                                                  \
  }                                                                        \
                                                                           \
  CAMLprim value nx_##name##_bc(value *argv, int argn) {                   \
    return nx_##name(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], \
                     argv[6], argv[7], argv[8], argv[9], argv[10]);        \
  }

/* Custom implementations for operations that support integers */

CAMLprim value nx_add(value vNdim, value vShape, value vX,
                     value vXStrides, value vXOffset, value vY,
                     value vYStrides, value vYOffset, value vZ,
                     value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_add_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_add_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_add_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_add_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("add: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_add_bc(value *argv, int argn) {
  return nx_add(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], argv[8], argv[9], argv[10]);
}

CAMLprim value nx_sub(value vNdim, value vShape, value vX,
                     value vXStrides, value vXOffset, value vY,
                     value vYStrides, value vYOffset, value vZ,
                     value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_sub_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_sub_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_sub_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_sub_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("sub: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_sub_bc(value *argv, int argn) {
  return nx_sub(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], argv[8], argv[9], argv[10]);
}

CAMLprim value nx_mul(value vNdim, value vShape, value vX,
                     value vXStrides, value vXOffset, value vY,
                     value vYStrides, value vYOffset, value vZ,
                     value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_mul_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_mul_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_mul_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_mul_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("mul: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_mul_bc(value *argv, int argn) {
  return nx_mul(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], argv[8], argv[9], argv[10]);
}

CAMLprim value nx_div(value vNdim, value vShape, value vX,
                     value vXStrides, value vXOffset, value vY,
                     value vYStrides, value vYOffset, value vZ,
                     value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_div_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_div_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_div_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_div_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("div: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_div_bc(value *argv, int argn) {
  return nx_div(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], argv[8], argv[9], argv[10]);
}

CAMLprim value nx_max(value vNdim, value vShape, value vX,
                     value vXStrides, value vXOffset, value vY,
                     value vYStrides, value vYOffset, value vZ,
                     value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_max_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_max_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_max_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_max_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("max: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_max_bc(value *argv, int argn) {
  return nx_max(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], argv[8], argv[9], argv[10]);
}
DEFINE_BINARY_STUB(pow)
DEFINE_BINARY_STUB(mod)

/* =========================== Unary Operations =========================== */

#define DEFINE_UNARY_STUB(name)                                            \
  CAMLprim value nx_##name(value vNdim, value vShape, value vX,            \
                           value vXStrides, value vXOffset, value vZ,      \
                           value vZStrides, value vZOffset) {              \
    CAMLparam5(vShape, vX, vXStrides, vZ, vZStrides);                      \
    CAMLxparam1(vZOffset);                                                 \
                                                                           \
    int ndim = Int_val(vNdim);                                             \
    strided_array_t x, z;                                                  \
                                                                           \
    init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);         \
    init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);         \
                                                                           \
    caml_enter_blocking_section();                                         \
                                                                           \
    struct caml_ba_array *ba = Caml_ba_array_val(vX);                      \
    int kind = ba->flags & CAML_BA_KIND_MASK;                              \
                                                                           \
    if (kind == CAML_BA_FLOAT32) {                                         \
      nx_cblas_##name##_f32(&x, &z);                                       \
    } else if (kind == CAML_BA_FLOAT64) {                                  \
      nx_cblas_##name##_f64(&x, &z);                                       \
    } else {                                                               \
      caml_leave_blocking_section();                                       \
      nx_cblas_free_strided_array(&x);                                     \
      nx_cblas_free_strided_array(&z);                                     \
      caml_failwith(#name ": unsupported dtype");                          \
    }                                                                      \
                                                                           \
    caml_leave_blocking_section();                                         \
                                                                           \
    nx_cblas_free_strided_array(&x);                                       \
    nx_cblas_free_strided_array(&z);                                       \
                                                                           \
    CAMLreturn(Val_unit);                                                  \
  }                                                                        \
                                                                           \
  CAMLprim value nx_##name##_bc(value *argv, int argn) {                   \
    return nx_##name(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], \
                     argv[6], argv[7]);                                    \
  }

/* Special neg that handles integers */
CAMLprim value nx_neg(value vNdim, value vShape, value vX,
                     value vXStrides, value vXOffset, value vZ,
                     value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vZ, vZStrides);
  CAMLxparam1(vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_neg_f32(&x, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_neg_f64(&x, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_neg_i32(&x, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_neg_i64(&x, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&z);
    caml_failwith("neg: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_neg_bc(value *argv, int argn) {
  return nx_neg(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
               argv[6], argv[7]);
}
DEFINE_UNARY_STUB(sqrt)
DEFINE_UNARY_STUB(sin)
DEFINE_UNARY_STUB(exp2)
DEFINE_UNARY_STUB(log2)
DEFINE_UNARY_STUB(recip)

/* =========================== Copy Operation =========================== */

CAMLprim value nx_copy(value vNdim, value vShape, value vX, value vXStrides,
                       value vXOffset, value vZ, value vZStrides,
                       value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vZ, vZStrides);
  CAMLxparam1(vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_copy_f32(&x, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_copy_f64(&x, &z);
  } else {
    size_t elem_size = get_element_size(kind);
    nx_cblas_copy_generic(&x, &z, elem_size);
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_copy_bc(value *argv, int argn) {
  return nx_copy(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                 argv[7]);
}

/* =========================== Comparison Operations ===========================
 */

#define DEFINE_CMP_STUB(name)                                              \
  CAMLprim value nx_##name(value vNdim, value vShape, value vX,            \
                           value vXStrides, value vXOffset, value vY,      \
                           value vYStrides, value vYOffset, value vZ,      \
                           value vZStrides, value vZOffset) {              \
    CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);                      \
    CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);                        \
                                                                           \
    int ndim = Int_val(vNdim);                                             \
    strided_array_t x, y, z;                                               \
                                                                           \
    init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);         \
    init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);         \
    init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);         \
                                                                           \
    caml_enter_blocking_section();                                         \
                                                                           \
    struct caml_ba_array *ba = Caml_ba_array_val(vX);                      \
    int kind = ba->flags & CAML_BA_KIND_MASK;                              \
                                                                           \
    if (kind == CAML_BA_FLOAT32) {                                         \
      nx_cblas_##name##_f32(&x, &y, &z);                                   \
    } else if (kind == CAML_BA_FLOAT64) {                                  \
      nx_cblas_##name##_f64(&x, &y, &z);                                   \
    } else {                                                               \
      caml_leave_blocking_section();                                       \
      nx_cblas_free_strided_array(&x);                                     \
      nx_cblas_free_strided_array(&y);                                     \
      nx_cblas_free_strided_array(&z);                                     \
      caml_failwith(#name ": unsupported dtype");                          \
    }                                                                      \
                                                                           \
    caml_leave_blocking_section();                                         \
                                                                           \
    nx_cblas_free_strided_array(&x);                                       \
    nx_cblas_free_strided_array(&y);                                       \
    nx_cblas_free_strided_array(&z);                                       \
                                                                           \
    CAMLreturn(Val_unit);                                                  \
  }                                                                        \
                                                                           \
  CAMLprim value nx_##name##_bc(value *argv, int argn) {                   \
    return nx_##name(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], \
                     argv[6], argv[7], argv[8], argv[9], argv[10]);        \
  }

/* Custom implementations for comparison operations that support integers */

CAMLprim value nx_cmplt(value vNdim, value vShape, value vX,
                       value vXStrides, value vXOffset, value vY,
                       value vYStrides, value vYOffset, value vZ,
                       value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_cmplt_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_cmplt_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_cmplt_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_cmplt_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("cmplt: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_cmplt_bc(value *argv, int argn) {
  return nx_cmplt(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                  argv[6], argv[7], argv[8], argv[9], argv[10]);
}

CAMLprim value nx_cmpne(value vNdim, value vShape, value vX,
                       value vXStrides, value vXOffset, value vY,
                       value vYStrides, value vYOffset, value vZ,
                       value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_cmpne_f32(&x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_cmpne_f64(&x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_cmpne_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_cmpne_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("cmpne: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_cmpne_bc(value *argv, int argn) {
  return nx_cmpne(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                  argv[6], argv[7], argv[8], argv[9], argv[10]);
}

/* =========================== Reduction Operations ===========================
 */

CAMLprim value nx_reduce_sum(value vNdim, value vShape, value vX,
                             value vXStrides, value vXOffset, value vKeepDims,
                             value vZ) {
  CAMLparam4(vShape, vX, vXStrides, vZ);

  int ndim = Int_val(vNdim);
  strided_array_t x;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_reduce_sum_f32(&x, Caml_ba_data_val(vZ));
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_reduce_sum_f64(&x, Caml_ba_data_val(vZ));
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    caml_failwith("reduce_sum: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_reduce_sum_bc(value *argv, int argn) {
  return nx_reduce_sum(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                       argv[6]);
}

CAMLprim value nx_reduce_max(value vNdim, value vShape, value vX,
                             value vXStrides, value vXOffset, value vKeepDims,
                             value vZ) {
  CAMLparam4(vShape, vX, vXStrides, vZ);

  int ndim = Int_val(vNdim);
  strided_array_t x;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_reduce_max_f32(&x, Caml_ba_data_val(vZ));
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_reduce_max_f64(&x, Caml_ba_data_val(vZ));
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    caml_failwith("reduce_max: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_reduce_max_bc(value *argv, int argn) {
  return nx_reduce_max(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                       argv[6]);
}

CAMLprim value nx_reduce_min(value vNdim, value vShape, value vX,
                             value vXStrides, value vXOffset, value vKeepDims,
                             value vZ) {
  CAMLparam4(vShape, vX, vXStrides, vZ);

  int ndim = Int_val(vNdim);
  strided_array_t x;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_reduce_min_f32(&x, Caml_ba_data_val(vZ));
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_reduce_min_f64(&x, Caml_ba_data_val(vZ));
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    caml_failwith("reduce_min: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_reduce_min_bc(value *argv, int argn) {
  return nx_reduce_min(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                       argv[6]);
}

/* =========================== Product Reduction =========================== */

CAMLprim value nx_reduce_prod(value vNdim, value vShape, value vX,
                              value vXStrides, value vXOffset, value vKeepDims,
                              value vZ) {
  CAMLparam4(vShape, vX, vXStrides, vZ);

  int ndim = Int_val(vNdim);
  strided_array_t x;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_reduce_prod_f32(&x, Caml_ba_data_val(vZ));
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_reduce_prod_f64(&x, Caml_ba_data_val(vZ));
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    caml_failwith("reduce_prod: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_reduce_prod_bc(value *argv, int argn) {
  return nx_reduce_prod(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                        argv[6]);
}

/* =========================== Integer Division =========================== */

CAMLprim value nx_idiv(value vNdim, value vShape, value vX, value vXStrides,
                       value vXOffset, value vY, value vYStrides,
                       value vYOffset, value vZ, value vZStrides,
                       value vZOffset) {
  CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);
  CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t x, y, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_INT32) {
    nx_cblas_idiv_i32(&x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_idiv_i64(&x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("idiv: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_idiv_bc(value *argv, int argn) {
  return nx_idiv(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                 argv[7], argv[8], argv[9], argv[10]);
}

/* =========================== Bitwise Operations =========================== */

#define DEFINE_BITWISE_STUB(name)                                          \
  CAMLprim value nx_##name(value vNdim, value vShape, value vX,            \
                           value vXStrides, value vXOffset, value vY,      \
                           value vYStrides, value vYOffset, value vZ,      \
                           value vZStrides, value vZOffset) {              \
    CAMLparam5(vShape, vX, vXStrides, vY, vYStrides);                      \
    CAMLxparam4(vYOffset, vZ, vZStrides, vZOffset);                        \
                                                                           \
    int ndim = Int_val(vNdim);                                             \
    strided_array_t x, y, z;                                               \
                                                                           \
    init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);         \
    init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);         \
    init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);         \
                                                                           \
    caml_enter_blocking_section();                                         \
                                                                           \
    struct caml_ba_array *ba = Caml_ba_array_val(vX);                      \
    int kind = ba->flags & CAML_BA_KIND_MASK;                              \
                                                                           \
    if (kind == CAML_BA_INT32) {                                           \
      nx_cblas_##name##_int32(&x, &y, &z);                                 \
    } else if (kind == CAML_BA_INT64) {                                    \
      nx_cblas_##name##_int64(&x, &y, &z);                                 \
    } else if (kind == CAML_BA_UINT8) {                                    \
      nx_cblas_##name##_uint8(&x, &y, &z);                                 \
    } else if (kind == CAML_BA_UINT16) {                                   \
      nx_cblas_##name##_uint16(&x, &y, &z);                                \
    } else {                                                               \
      caml_leave_blocking_section();                                       \
      nx_cblas_free_strided_array(&x);                                     \
      nx_cblas_free_strided_array(&y);                                     \
      nx_cblas_free_strided_array(&z);                                     \
      caml_failwith(#name ": unsupported dtype");                          \
    }                                                                      \
                                                                           \
    caml_leave_blocking_section();                                         \
                                                                           \
    nx_cblas_free_strided_array(&x);                                       \
    nx_cblas_free_strided_array(&y);                                       \
    nx_cblas_free_strided_array(&z);                                       \
                                                                           \
    CAMLreturn(Val_unit);                                                  \
  }                                                                        \
                                                                           \
  CAMLprim value nx_##name##_bc(value *argv, int argn) {                   \
    return nx_##name(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], \
                     argv[6], argv[7], argv[8], argv[9], argv[10]);        \
  }

DEFINE_BITWISE_STUB(xor)
DEFINE_BITWISE_STUB(or)
DEFINE_BITWISE_STUB(and)

/* =========================== WHERE Operation =========================== */

CAMLprim value nx_where(value vNdim, value vShape, value vCond,
                        value vCondStrides, value vCondOffset, value vX,
                        value vXStrides, value vXOffset, value vY,
                        value vYStrides, value vYOffset, value vZ,
                        value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vCond, vCondStrides, vX, vXStrides);
  CAMLxparam5(vXOffset, vY, vYStrides, vYOffset, vZ);
  CAMLxparam2(vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t cond, x, y, z;

  init_strided_array(&cond, vCond, ndim, vShape, vCondStrides, vCondOffset);
  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&y, vY, ndim, vShape, vYStrides, vYOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_where_f32(&cond, &x, &y, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_where_f64(&cond, &x, &y, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_where_i32(&cond, &x, &y, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_where_i64(&cond, &x, &y, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&cond);
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&y);
    nx_cblas_free_strided_array(&z);
    caml_failwith("where: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&cond);
  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&y);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_where_bc(value *argv, int argn) {
  return nx_where(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                  argv[7], argv[8], argv[9], argv[10], argv[11], argv[12],
                  argv[13]);
}

/* =========================== PAD Operation =========================== */

CAMLprim value nx_pad(value vNdim, value vShape, value vX, value vXStrides,
                      value vXOffset, value vZ, value vZStrides, value vZOffset,
                      value vPadConfig, value vFillValue) {
  CAMLparam5(vShape, vX, vXStrides, vZ, vZStrides);
  CAMLxparam3(vZOffset, vPadConfig, vFillValue);

  int ndim = Int_val(vNdim);
  strided_array_t x, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);

  /* Compute output shape from input shape and padding */
  int *out_shape = (int *)malloc(ndim * sizeof(int));
  int *pad_config = (int *)malloc(2 * ndim * sizeof(int));

  for (int i = 0; i < ndim; i++) {
    pad_config[i * 2] = Int_val(Field(Field(vPadConfig, i), 0));
    pad_config[i * 2 + 1] = Int_val(Field(Field(vPadConfig, i), 1));
    out_shape[i] = x.shape[i] + pad_config[i * 2] + pad_config[i * 2 + 1];
  }

  /* Create output array descriptor */
  value vOutShape = caml_alloc_small(ndim, 0);
  for (int i = 0; i < ndim; i++) {
    Field(vOutShape, i) = Val_int(out_shape[i]);
  }

  init_strided_array(&z, vZ, ndim, vOutShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vX);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    float fill_value = Double_field(vFillValue, 0);
    nx_cblas_pad_f32(&x, &z, pad_config, fill_value);
  } else if (kind == CAML_BA_FLOAT64) {
    double fill_value = Double_field(vFillValue, 0);
    nx_cblas_pad_f64(&x, &z, pad_config, fill_value);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&z);
    free(out_shape);
    free(pad_config);
    caml_failwith("pad: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&z);
  free(out_shape);
  free(pad_config);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_pad_bc(value *argv, int argn) {
  return nx_pad(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                argv[7], argv[8], argv[9]);
}

/* =========================== CAST Operation =========================== */

CAMLprim value nx_cast(value vNdim, value vShape, value vX, value vXStrides,
                       value vXOffset, value vZ, value vZStrides,
                       value vZOffset, value vFromKind, value vToKind) {
  CAMLparam5(vShape, vX, vXStrides, vZ, vZStrides);
  CAMLxparam3(vZOffset, vFromKind, vToKind);

  int ndim = Int_val(vNdim);
  strided_array_t x, z;

  init_strided_array(&x, vX, ndim, vShape, vXStrides, vXOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  int from_kind = Int_val(vFromKind);
  int to_kind = Int_val(vToKind);

  if (from_kind == CAML_BA_FLOAT32 && to_kind == CAML_BA_FLOAT64) {
    nx_cblas_cast_f32_to_f64(&x, &z);
  } else if (from_kind == CAML_BA_FLOAT64 && to_kind == CAML_BA_FLOAT32) {
    nx_cblas_cast_f64_to_f32(&x, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&x);
    nx_cblas_free_strided_array(&z);
    caml_failwith("cast: unsupported dtype conversion");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&x);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_cast_bc(value *argv, int argn) {
  return nx_cast(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6],
                 argv[7], argv[8], argv[9]);
}

/* =========================== THREEFRY Operation =========================== */

CAMLprim value nx_threefry(value vNdim, value vShape, value vKey,
                           value vKeyStrides, value vKeyOffset, value vCounter,
                           value vCounterStrides, value vCounterOffset,
                           value vZ, value vZStrides, value vZOffset) {
  CAMLparam5(vShape, vKey, vKeyStrides, vCounter, vCounterStrides);
  CAMLxparam4(vCounterOffset, vZ, vZStrides, vZOffset);

  int ndim = Int_val(vNdim);
  strided_array_t key, counter, z;

  init_strided_array(&key, vKey, ndim, vShape, vKeyStrides, vKeyOffset);
  init_strided_array(&counter, vCounter, ndim, vShape, vCounterStrides,
                     vCounterOffset);
  init_strided_array(&z, vZ, ndim, vShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  nx_cblas_threefry(&key, &counter, &z);

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&key);
  nx_cblas_free_strided_array(&counter);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_threefry_bc(value *argv, int argn) {
  return nx_threefry(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                     argv[6], argv[7], argv[8], argv[9], argv[10]);
}

/* =========================== GATHER Operation =========================== */

CAMLprim value nx_gather(value vDataNdim, value vDataShape, value vData,
                         value vDataStrides, value vDataOffset,
                         value vIndicesShape, value vIndices,
                         value vIndicesStrides, value vIndicesOffset,
                         value vAxis, value vZ, value vZStrides,
                         value vZOffset) {
  CAMLparam5(vDataShape, vData, vDataStrides, vIndicesShape, vIndices);
  CAMLxparam5(vIndicesStrides, vIndicesOffset, vZ, vZStrides, vZOffset);

  int data_ndim = Int_val(vDataNdim);
  int axis = Int_val(vAxis);
  strided_array_t data, indices, z;

  init_strided_array(&data, vData, data_ndim, vDataShape, vDataStrides,
                     vDataOffset);
  init_strided_array(&indices, vIndices, data_ndim, vIndicesShape,
                     vIndicesStrides, vIndicesOffset);
  init_strided_array(&z, vZ, data_ndim, vIndicesShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vData);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_gather_f32(&data, &indices, axis, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_gather_f64(&data, &indices, axis, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&data);
    nx_cblas_free_strided_array(&indices);
    nx_cblas_free_strided_array(&z);
    caml_failwith("gather: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&data);
  nx_cblas_free_strided_array(&indices);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_gather_bc(value *argv, int argn) {
  return nx_gather(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                   argv[6], argv[7], argv[8], argv[9], argv[10], argv[11],
                   argv[12]);
}

/* =========================== SCATTER Operation =========================== */

CAMLprim value nx_scatter(value vTemplateNdim, value vTemplateShape,
                          value vTemplate, value vTemplateStrides,
                          value vTemplateOffset, value vIndicesShape,
                          value vIndices, value vIndicesStrides,
                          value vIndicesOffset, value vUpdates,
                          value vUpdatesStrides, value vUpdatesOffset,
                          value vAxis, value vZ, value vZStrides,
                          value vZOffset) {
  CAMLparam5(vTemplateShape, vTemplate, vTemplateStrides, vIndicesShape,
             vIndices);
  CAMLxparam5(vIndicesStrides, vUpdates, vUpdatesStrides, vUpdatesOffset, vZ);
  CAMLxparam2(vZStrides, vZOffset);

  int ndim = Int_val(vTemplateNdim);
  int axis = Int_val(vAxis);
  strided_array_t data_template, indices, updates, z;

  init_strided_array(&data_template, vTemplate, ndim, vTemplateShape,
                     vTemplateStrides, vTemplateOffset);
  init_strided_array(&indices, vIndices, ndim, vIndicesShape, vIndicesStrides,
                     vIndicesOffset);
  init_strided_array(&updates, vUpdates, ndim, vIndicesShape, vUpdatesStrides,
                     vUpdatesOffset);
  init_strided_array(&z, vZ, ndim, vTemplateShape, vZStrides, vZOffset);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vTemplate);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_scatter_f32(&data_template, &indices, &updates, axis, &z);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_scatter_f64(&data_template, &indices, &updates, axis, &z);
  } else if (kind == CAML_BA_INT32) {
    nx_cblas_scatter_i32(&data_template, &indices, &updates, axis, &z);
  } else if (kind == CAML_BA_INT64) {
    nx_cblas_scatter_i64(&data_template, &indices, &updates, axis, &z);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&data_template);
    nx_cblas_free_strided_array(&indices);
    nx_cblas_free_strided_array(&updates);
    nx_cblas_free_strided_array(&z);
    caml_failwith("scatter: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&data_template);
  nx_cblas_free_strided_array(&indices);
  nx_cblas_free_strided_array(&updates);
  nx_cblas_free_strided_array(&z);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_scatter_bc(value *argv, int argn) {
  return nx_scatter(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                    argv[6], argv[7], argv[8], argv[9], argv[10], argv[11],
                    argv[12], argv[13], argv[14], argv[15]);
}

/* =========================== MATMUL Operation =========================== */

extern void nx_cblas_matmul_f32(const strided_array_t *a, const strided_array_t *b, strided_array_t *c);
extern void nx_cblas_matmul_f64(const strided_array_t *a, const strided_array_t *b, strided_array_t *c);

CAMLprim value nx_matmul(value vNdimA, value vShapeA, value vA, value vStridesA,
                         value vOffsetA, value vShapeB, value vB, value vStridesB,
                         value vOffsetB, value vShapeC, value vC, value vStridesC,
                         value vOffsetC) {
  CAMLparam5(vShapeA, vA, vStridesA, vShapeB, vB);
  CAMLxparam5(vStridesB, vShapeC, vC, vStridesC, vOffsetC);

  int ndim_a = Int_val(vNdimA);
  int ndim_b = Wosize_val(vShapeB);
  int ndim_c = Wosize_val(vShapeC);
  strided_array_t a, b, c;

  init_strided_array(&a, vA, ndim_a, vShapeA, vStridesA, vOffsetA);
  init_strided_array(&b, vB, ndim_b, vShapeB, vStridesB, vOffsetB);
  init_strided_array(&c, vC, ndim_c, vShapeC, vStridesC, vOffsetC);

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vA);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_matmul_f32(&a, &b, &c);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_matmul_f64(&a, &b, &c);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&a);
    nx_cblas_free_strided_array(&b);
    nx_cblas_free_strided_array(&c);
    caml_failwith("matmul: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&a);
  nx_cblas_free_strided_array(&b);
  nx_cblas_free_strided_array(&c);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_matmul_bc(value *argv, int argn) {
  return nx_matmul(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                   argv[6], argv[7], argv[8], argv[9], argv[10], argv[11],
                   argv[12]);
}

/* =========================== UNFOLD Operation =========================== */

extern void nx_cblas_unfold_f32(const strided_array_t *input, strided_array_t *output,
                                int n_spatial, const int *kernel_size, const int *stride,
                                const int *dilation, const int *padding);
extern void nx_cblas_unfold_f64(const strided_array_t *input, strided_array_t *output,
                                int n_spatial, const int *kernel_size, const int *stride,
                                const int *dilation, const int *padding);

CAMLprim value nx_unfold(value vInputNdim, value vInputShape, value vInput,
                         value vInputStrides, value vInputOffset, value vOutputShape,
                         value vOutput, value vOutputStrides, value vOutputOffset,
                         value vNSpatial, value vKernelSize, value vStride,
                         value vDilation, value vPadding) {
  CAMLparam5(vInputShape, vInput, vInputStrides, vOutputShape, vOutput);
  CAMLxparam5(vOutputStrides, vKernelSize, vStride, vDilation, vPadding);

  int input_ndim = Int_val(vInputNdim);
  int output_ndim = Wosize_val(vOutputShape);
  int n_spatial = Int_val(vNSpatial);
  strided_array_t input, output;

  init_strided_array(&input, vInput, input_ndim, vInputShape, vInputStrides, vInputOffset);
  init_strided_array(&output, vOutput, output_ndim, vOutputShape, vOutputStrides, vOutputOffset);

  // Extract kernel_size, stride, dilation, padding arrays
  int *kernel_size = (int *)malloc(n_spatial * sizeof(int));
  int *stride = (int *)malloc(n_spatial * sizeof(int));
  int *dilation = (int *)malloc(n_spatial * sizeof(int));
  int *padding = (int *)malloc(n_spatial * 2 * sizeof(int));

  for (int i = 0; i < n_spatial; i++) {
    kernel_size[i] = Int_val(Field(vKernelSize, i));
    stride[i] = Int_val(Field(vStride, i));
    dilation[i] = Int_val(Field(vDilation, i));
    value v_pair = Field(vPadding, i);
    padding[i * 2] = Int_val(Field(v_pair, 0));
    padding[i * 2 + 1] = Int_val(Field(v_pair, 1));
  }

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vInput);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_unfold_f32(&input, &output, n_spatial, kernel_size, stride, dilation, padding);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_unfold_f64(&input, &output, n_spatial, kernel_size, stride, dilation, padding);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&input);
    nx_cblas_free_strided_array(&output);
    free(kernel_size);
    free(stride);
    free(dilation);
    free(padding);
    caml_failwith("unfold: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&input);
  nx_cblas_free_strided_array(&output);
  free(kernel_size);
  free(stride);
  free(dilation);
  free(padding);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_unfold_bc(value *argv, int argn) {
  return nx_unfold(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                   argv[6], argv[7], argv[8], argv[9], argv[10], argv[11],
                   argv[12], argv[13]);
}

/* =========================== FOLD Operation =========================== */

extern void nx_cblas_fold_f32(const strided_array_t *input, strided_array_t *output,
                              int n_spatial, const int *output_size, const int *kernel_size,
                              const int *stride, const int *dilation, const int *padding);
extern void nx_cblas_fold_f64(const strided_array_t *input, strided_array_t *output,
                              int n_spatial, const int *output_size, const int *kernel_size,
                              const int *stride, const int *dilation, const int *padding);

CAMLprim value nx_fold(value vInputNdim, value vInputShape, value vInput,
                       value vInputStrides, value vInputOffset, value vOutputShape,
                       value vOutput, value vOutputStrides, value vOutputOffset,
                       value vNSpatial, value vOutputSize, value vKernelSize,
                       value vStride, value vDilation, value vPadding) {
  CAMLparam5(vInputShape, vInput, vInputStrides, vOutputShape, vOutput);
  CAMLxparam5(vOutputStrides, vOutputSize, vKernelSize, vStride, vDilation);
  CAMLxparam1(vPadding);

  int input_ndim = Int_val(vInputNdim);
  int output_ndim = Wosize_val(vOutputShape);
  int n_spatial = Int_val(vNSpatial);
  strided_array_t input, output;

  init_strided_array(&input, vInput, input_ndim, vInputShape, vInputStrides, vInputOffset);
  init_strided_array(&output, vOutput, output_ndim, vOutputShape, vOutputStrides, vOutputOffset);

  // Extract output_size, kernel_size, stride, dilation, padding arrays
  int *output_size = (int *)malloc(n_spatial * sizeof(int));
  int *kernel_size = (int *)malloc(n_spatial * sizeof(int));
  int *stride = (int *)malloc(n_spatial * sizeof(int));
  int *dilation = (int *)malloc(n_spatial * sizeof(int));
  int *padding = (int *)malloc(n_spatial * 2 * sizeof(int));

  for (int i = 0; i < n_spatial; i++) {
    output_size[i] = Int_val(Field(vOutputSize, i));
    kernel_size[i] = Int_val(Field(vKernelSize, i));
    stride[i] = Int_val(Field(vStride, i));
    dilation[i] = Int_val(Field(vDilation, i));
    value v_pair = Field(vPadding, i);
    padding[i * 2] = Int_val(Field(v_pair, 0));
    padding[i * 2 + 1] = Int_val(Field(v_pair, 1));
  }

  caml_enter_blocking_section();

  struct caml_ba_array *ba = Caml_ba_array_val(vInput);
  int kind = ba->flags & CAML_BA_KIND_MASK;

  if (kind == CAML_BA_FLOAT32) {
    nx_cblas_fold_f32(&input, &output, n_spatial, output_size, kernel_size, stride, dilation, padding);
  } else if (kind == CAML_BA_FLOAT64) {
    nx_cblas_fold_f64(&input, &output, n_spatial, output_size, kernel_size, stride, dilation, padding);
  } else {
    caml_leave_blocking_section();
    nx_cblas_free_strided_array(&input);
    nx_cblas_free_strided_array(&output);
    free(output_size);
    free(kernel_size);
    free(stride);
    free(dilation);
    free(padding);
    caml_failwith("fold: unsupported dtype");
  }

  caml_leave_blocking_section();

  nx_cblas_free_strided_array(&input);
  nx_cblas_free_strided_array(&output);
  free(output_size);
  free(kernel_size);
  free(stride);
  free(dilation);
  free(padding);

  CAMLreturn(Val_unit);
}

CAMLprim value nx_fold_bc(value *argv, int argn) {
  return nx_fold(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                 argv[6], argv[7], argv[8], argv[9], argv[10], argv[11],
                 argv[12], argv[13], argv[14]);
}