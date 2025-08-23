// Memory operations for nx C backend

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>

#include "nx_c_shared.h"

// Helper to copy an int array (shape or strides)
static value copy_int_array(value v_old) {
  CAMLparam1(v_old);
  int n = Wosize_val(v_old);
  value v_new = caml_alloc(n, 0);
  for (int i = 0; i < n; i++) {
    Store_field(v_new, i, Field(v_old, i));
  }
  CAMLreturn(v_new);
}

// Helper to create a new tensor value
static value create_tensor_value(value v_shape, value v_strides, value v_data,
                                 long offset) {
  CAMLparam3(v_shape, v_strides, v_data);
  CAMLlocal1(v_new);
  v_new = caml_alloc(4, 0);
  Store_field(v_new, FFI_TENSOR_DATA, v_data);
  Store_field(v_new, FFI_TENSOR_SHAPE, v_shape);
  Store_field(v_new, FFI_TENSOR_STRIDES, v_strides);
  Store_field(v_new, FFI_TENSOR_OFFSET, Val_long(offset));
  CAMLreturn(v_new);
}

// Helper to set standard C-contiguous strides
static void set_standard_strides(int *strides, int *shape, int ndim) {
  if (ndim == 0) return;
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

// Helper to check if two ndarrays have the same shape
static bool same_shape(const ndarray_t *a, const ndarray_t *b) {
  if (a->ndim != b->ndim) return false;
  for (int i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) return false;
  }
  return true;
}

// Helper to check if an ndarray is C-contiguous (row-major, no gaps)
static bool is_c_contiguous(const ndarray_t *nd) {
  if (nd->ndim == 0) return true;
  long s = 1;
  for (int i = nd->ndim - 1; i >= 0; i--) {
    if (nd->strides[i] != s) return false;
    s *= nd->shape[i];
  }
  return true;
}

// Helper to get element size in bytes based on kind
static long get_element_size(int kind) {
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
    case NX_BA_COMPLEX16:
      return 2;
    case CAML_BA_INT32:
    case CAML_BA_FLOAT32:
      return 4;
    case CAML_BA_COMPLEX32:
      return 8;  // 2 * float32
    case CAML_BA_INT64:
    case CAML_BA_FLOAT64:
      return 8;
    case CAML_BA_COMPLEX64:
      return 16;  // 2 * float64
    case CAML_BA_NATIVE_INT:
    case CAML_BA_CAML_INT:
      return sizeof(intnat);
    case NX_BA_INT4:
    case NX_BA_UINT4:
      // Special handling required; size not used for memcpy
      caml_failwith("get_element_size: int4/uint4 not supported for size");
    default:
      caml_failwith("get_element_size: unsupported kind");
  }
  return 0;  // Unreachable
}

// Core copy function: copies data from src to dst (assumes same shape and
// dtype)
static void nx_c_copy(const ndarray_t *src, const ndarray_t *dst, int kind) {
  if (!src || !dst) caml_failwith("nx_c_copy: null pointer");
  if (!same_shape(src, dst)) caml_failwith("nx_c_copy: shape mismatch");

  long total = total_elements_safe(src);
  if (total == 0) return;

  if (kind == NX_BA_INT4 || kind == NX_BA_UINT4) {
    bool signedness = (kind == NX_BA_INT4);
    nd_copy_iterator_t it;
    nd_copy_iterator_init(&it, src, dst);
    do {
      long src_off, dst_off;
      nd_copy_iterator_get_offsets(&it, &src_off, &dst_off);
      long abs_src_off = src->offset + src_off;
      long byte_off = abs_src_off / 2;
      int nib_off = abs_src_off % 2;
      uint8_t *sdata = (uint8_t *)src->data;
      int val;
      if (nib_off) {
        val = signedness ? (int8_t)(sdata[byte_off] >> 4)
                         : (sdata[byte_off] >> 4) & 0x0F;
      } else {
        val = signedness ? (int8_t)((sdata[byte_off] & 0x0F) << 4) >> 4
                         : sdata[byte_off] & 0x0F;
      }
      long abs_dst_off = dst->offset + dst_off;
      byte_off = abs_dst_off / 2;
      nib_off = abs_dst_off % 2;
      uint8_t *ddata = (uint8_t *)dst->data;
      uint8_t nib = (uint8_t)val & 0x0F;
      if (nib_off) {
        ddata[byte_off] = (ddata[byte_off] & 0x0F) | (nib << 4);
      } else {
        ddata[byte_off] = (ddata[byte_off] & 0xF0) | nib;
      }
    } while (nd_copy_iterator_next(&it));
    nd_copy_iterator_destroy(&it);
  } else {
    long elsize = get_element_size(kind);
    // Cannot use memcpy if src has broadcasts (zero strides)
    bool src_has_broadcast = false;
    for (int i = 0; i < src->ndim; i++) {
      if (src->strides[i] == 0 && src->shape[i] > 1) {
        src_has_broadcast = true;
        break;
      }
    }
    bool cont = !src_has_broadcast && is_c_contiguous(src) && is_c_contiguous(dst);
    if (cont) {
      memcpy((char *)dst->data + dst->offset * elsize,
             (char *)src->data + src->offset * elsize, total * elsize);
    } else {
      nd_copy_iterator_t it;
      nd_copy_iterator_init(&it, src, dst);
      do {
        long src_off, dst_off;
        nd_copy_iterator_get_offsets(&it, &src_off, &dst_off);
        memcpy((char *)dst->data + (dst->offset + dst_off) * elsize,
               (char *)src->data + (src->offset + src_off) * elsize, elsize);
      } while (nd_copy_iterator_next(&it));
      nd_copy_iterator_destroy(&it);
    }
  }
}

// FFI stub for assign (in-place copy)
CAMLprim value caml_nx_assign(value v_src, value v_dst) {
  CAMLparam2(v_src, v_dst);
  ndarray_t src = extract_ndarray(v_src);
  ndarray_t dst = extract_ndarray(v_dst);
  struct caml_ba_array *ba_src =
      Caml_ba_array_val(Field(v_src, FFI_TENSOR_DATA));
  struct caml_ba_array *ba_dst =
      Caml_ba_array_val(Field(v_dst, FFI_TENSOR_DATA));
  int kind_src = ba_src->flags & CAML_BA_KIND_MASK;
  int kind_dst = ba_dst->flags & CAML_BA_KIND_MASK;
  if (kind_src != kind_dst) {
    cleanup_ndarray(&src);
    cleanup_ndarray(&dst);
    caml_failwith("caml_nx_assign: dtype mismatch");
  }
  if (!same_shape(&src, &dst)) {
    cleanup_ndarray(&src);
    cleanup_ndarray(&dst);
    caml_failwith("caml_nx_assign: shape mismatch");
  }
  caml_enter_blocking_section();
  nx_c_copy(&src, &dst, kind_src);
  caml_leave_blocking_section();
  cleanup_ndarray(&src);
  cleanup_ndarray(&dst);
  CAMLreturn(Val_unit);
}

// Helper to create a contiguous tensor (shared or copied)
static value make_contiguous(value v_src, bool force_copy) {
  CAMLparam1(v_src);
  CAMLlocal4(v_new_data, v_new_shape, v_new_strides, v_new);
  ndarray_t src = extract_ndarray(v_src);
  struct caml_ba_array *ba = Caml_ba_array_val(Field(v_src, FFI_TENSOR_DATA));
  int flags = ba->flags;
  int kind = flags & CAML_BA_KIND_MASK;
  long total = total_elements_safe(&src);
  bool can_share = !force_copy && is_c_contiguous(&src) && src.offset == 0;
  if (can_share) {
    v_new_data = Field(v_src, FFI_TENSOR_DATA);
    v_new_shape = copy_int_array(Field(v_src, FFI_TENSOR_SHAPE));
    v_new_strides = copy_int_array(Field(v_src, FFI_TENSOR_STRIDES));
    v_new = create_tensor_value(v_new_shape, v_new_strides, v_new_data, 0);
  } else {
    long dim = total;
    if (kind == NX_BA_INT4 || kind == NX_BA_UINT4) {
      dim = (total + 1) / 2;
      flags = (flags & ~CAML_BA_KIND_MASK) |
              CAML_BA_UINT8;  // Use byte array for packed
    }
    v_new_data = caml_ba_alloc(flags, 1, NULL, &dim);
    v_new_shape = copy_int_array(Field(v_src, FFI_TENSOR_SHAPE));
    v_new_strides = caml_alloc(src.ndim, 0);
    int strides[32];  // Stack buffer for strides - use int not long
    // Calculate C-contiguous strides
    if (src.ndim > 0) {
      int stride = 1;
      for (int i = src.ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= src.shape[i];
      }
    }
    for (int i = 0; i < src.ndim; i++) {
      Store_field(v_new_strides, i, Val_long(strides[i]));
    }
    ndarray_t dst = {0};
    dst.data = Caml_ba_data_val(v_new_data);
    dst.ndim = src.ndim;
    dst.shape = src.shape;  // Can reuse since it's temporary
    dst.strides = strides;  // Now types match correctly
    dst.offset = 0;
    caml_enter_blocking_section();
    nx_c_copy(&src, &dst, kind);
    caml_leave_blocking_section();
    v_new = create_tensor_value(v_new_shape, v_new_strides, v_new_data, 0);
  }
  cleanup_ndarray(&src);
  CAMLreturn(v_new);
}

// FFI stub for copy (always own buffer)
CAMLprim value caml_nx_copy(value v_src) {
  return make_contiguous(v_src, true);
}

// FFI stub for contiguous (may share buffer)
CAMLprim value caml_nx_contiguous(value v_src) {
  return make_contiguous(v_src, false);
}
