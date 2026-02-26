/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include "nx_buffer_stubs.h"

#include <caml/fail.h>
#include <stdlib.h>
#include <string.h>

/* External declarations for standard bigarray functions */
extern value caml_ba_get_N(value vb, value *vind, int nind);
extern value caml_ba_set_N(value vb, value *vind, int nargs);
extern value caml_ba_blit(value vsrc, value vdst);
extern CAMLprim value caml_ba_fill(value vb, value vinit);

/*---------------------------------------------------------------------------
   Helpers
  ---------------------------------------------------------------------------*/

static int nx_buffer_base_kind(int kind) {
  switch (kind) {
    case NX_BA_BFLOAT16:    return CAML_BA_FLOAT16;
    case NX_BA_BOOL:        return CAML_BA_UINT8;
    case NX_BA_INT4:        return CAML_BA_UINT8;
    case NX_BA_UINT4:       return CAML_BA_UINT8;
    case NX_BA_FP8_E4M3:   return CAML_BA_UINT8;
    case NX_BA_FP8_E5M2:   return CAML_BA_UINT8;
    case NX_BA_UINT32:      return CAML_BA_INT32;
    case NX_BA_UINT64:      return CAML_BA_INT64;
    default:                return -1;
  }
}

/* Byte size per element for extended kinds. Returns 0 for int4/uint4
   (which pack 2 per byte) and -1 for unknown kinds. */
static int nx_buffer_element_byte_size(int kind) {
  switch (kind) {
    case NX_BA_BFLOAT16:    return 2;
    case NX_BA_BOOL:        return 1;
    case NX_BA_FP8_E4M3:   return 1;
    case NX_BA_FP8_E5M2:   return 1;
    case NX_BA_UINT32:      return 4;
    case NX_BA_UINT64:      return 8;
    case NX_BA_INT4:        return 0;
    case NX_BA_UINT4:       return 0;
    default:                return -1;
  }
}

static value nx_buffer_alloc_with_kind(int kind, int layout_flag, int num_dims,
                                       intnat *dim, void *data) {
  int base_kind = nx_buffer_is_extended_kind(kind)
                      ? nx_buffer_base_kind(kind)
                      : kind;
  if (base_kind < 0) caml_failwith("Unknown extended bigarray kind");
  int flags = base_kind | layout_flag | CAML_BA_MANAGED;
  value res = caml_ba_alloc(flags, num_dims, data, dim);
  struct caml_ba_array *ba = Caml_ba_array_val(res);
  ba->flags = nx_buffer_store_extended_kind(ba->flags, kind);
  return res;
}

/* Overflow-safe multiplication */
static int umul_overflow(uintnat a, uintnat b, uintnat *res) {
  if (b != 0 && a > (uintnat)(-1) / b) return 1;
  *res = a * b;
  return 0;
}

static uintnat nx_buffer_num_elts_from_dims(int num_dims, intnat *dim) {
  uintnat num_elts = 1;
  for (int i = 0; i < num_dims; i++) {
    if (umul_overflow(num_elts, dim[i], &num_elts))
      caml_raise_out_of_memory();
  }
  return num_elts;
}

static uintnat nx_buffer_num_elts(struct caml_ba_array *b) {
  uintnat num_elts = 1;
  for (int i = 0; i < b->num_dims; i++)
    num_elts *= b->dim[i];
  return num_elts;
}

static intnat nx_buffer_offset(struct caml_ba_array *b, intnat *index) {
  intnat offset = 0;
  switch ((enum caml_ba_layout)(b->flags & CAML_BA_LAYOUT_MASK)) {
    case CAML_BA_C_LAYOUT:
      for (int i = 0; i < b->num_dims; i++) {
        if ((uintnat)index[i] >= (uintnat)b->dim[i])
          caml_array_bound_error();
        offset = offset * b->dim[i] + index[i];
      }
      break;
    case CAML_BA_FORTRAN_LAYOUT:
      for (int i = b->num_dims - 1; i >= 0; i--) {
        if ((uintnat)(index[i] - 1) >= (uintnat)b->dim[i])
          caml_array_bound_error();
        offset = offset * b->dim[i] + (index[i] - 1);
      }
      break;
  }
  return offset;
}

/*---------------------------------------------------------------------------
   Creation
  ---------------------------------------------------------------------------*/

#define CREATE_BA_FUNCTION(name, type_enum, bytes_per_elem)                    \
  CAMLprim value caml_nx_buffer_create_##name(value vlayout, value vdim) {     \
    CAMLparam2(vlayout, vdim);                                                 \
    CAMLlocal1(res);                                                           \
                                                                               \
    int num_dims = Wosize_val(vdim);                                           \
    intnat dim[CAML_BA_MAX_NUM_DIMS];                                          \
    for (int i = 0; i < num_dims; i++)                                         \
      dim[i] = Long_val(Field(vdim, i));                                       \
                                                                               \
    uintnat num_elts = nx_buffer_num_elts_from_dims(num_dims, dim);            \
    uintnat size;                                                              \
    if (umul_overflow(num_elts, (bytes_per_elem), &size))                      \
      caml_raise_out_of_memory();                                              \
                                                                               \
    void *data = calloc(1, size);                                              \
    if (data == NULL && size != 0) caml_raise_out_of_memory();                 \
                                                                               \
    int layout_flag = Caml_ba_layout_val(vlayout);                             \
    res = nx_buffer_alloc_with_kind((type_enum), layout_flag, num_dims, dim,   \
                                    data);                                     \
    CAMLreturn(res);                                                           \
  }

CREATE_BA_FUNCTION(bfloat16, NX_BA_BFLOAT16, 2)
CREATE_BA_FUNCTION(bool, NX_BA_BOOL, 1)
CREATE_BA_FUNCTION(float8_e4m3, NX_BA_FP8_E4M3, 1)
CREATE_BA_FUNCTION(float8_e5m2, NX_BA_FP8_E5M2, 1)
CREATE_BA_FUNCTION(uint32, NX_BA_UINT32, 4)
CREATE_BA_FUNCTION(uint64, NX_BA_UINT64, 8)

/* Int4/uint4 pack 2 values per byte */
static value nx_buffer_create_int4(int kind, value vlayout, value vdim) {
  CAMLparam2(vlayout, vdim);
  CAMLlocal1(res);
  int num_dims = Wosize_val(vdim);
  intnat dim[CAML_BA_MAX_NUM_DIMS];
  for (int i = 0; i < num_dims; i++)
    dim[i] = Long_val(Field(vdim, i));
  uintnat num_elts = nx_buffer_num_elts_from_dims(num_dims, dim);
  uintnat size = (num_elts + 1) / 2;
  void *data = calloc(1, size);
  if (data == NULL && size != 0) caml_raise_out_of_memory();
  int layout_flag = Caml_ba_layout_val(vlayout);
  res = nx_buffer_alloc_with_kind(kind, layout_flag, num_dims, dim, data);
  CAMLreturn(res);
}

CAMLprim value caml_nx_buffer_create_int4_signed(value vlayout, value vdim) {
  return nx_buffer_create_int4(NX_BA_INT4, vlayout, vdim);
}

CAMLprim value caml_nx_buffer_create_int4_unsigned(value vlayout, value vdim) {
  return nx_buffer_create_int4(NX_BA_UINT4, vlayout, vdim);
}

/*---------------------------------------------------------------------------
   Element access
  ---------------------------------------------------------------------------*/

CAMLprim value caml_nx_buffer_get(value vb, value vind) {
  CAMLparam2(vb, vind);
  CAMLlocal1(res);
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  int num_dims = Wosize_val(vind);
  if (num_dims != b->num_dims)
    caml_invalid_argument("Bigarray.get: wrong number of indices");

  intnat index[CAML_BA_MAX_NUM_DIMS];
  for (int i = 0; i < num_dims; i++)
    index[i] = Long_val(Field(vind, i));
  intnat offset = nx_buffer_offset(b, index);

  int kind = nx_buffer_get_kind(b);
  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    value args[CAML_BA_MAX_NUM_DIMS];
    for (int i = 0; i < num_dims; i++)
      args[i] = Field(vind, i);
    CAMLreturn(caml_ba_get_N(vb, args, num_dims));
  }

  switch (kind) {
    case NX_BA_BFLOAT16:
      res = caml_copy_double(
          (double)bfloat16_to_float(((uint16_t *)b->data)[offset]));
      break;
    case NX_BA_BOOL:
      res = Val_bool(((uint8_t *)b->data)[offset]);
      break;
    case NX_BA_INT4: {
      uint8_t byte = ((uint8_t *)b->data)[offset / 2];
      int val;
      if (offset % 2 == 0)
        val = (int8_t)((byte & 0x0F) << 4) >> 4; /* Sign extend lower nibble */
      else
        val = (int8_t)(byte & 0xF0) >> 4; /* Sign extend upper nibble */
      res = Val_int(val);
      break;
    }
    case NX_BA_UINT4: {
      uint8_t byte = ((uint8_t *)b->data)[offset / 2];
      int val;
      if (offset % 2 == 0)
        val = byte & 0x0F;
      else
        val = (byte >> 4) & 0x0F;
      res = Val_int(val);
      break;
    }
    case NX_BA_FP8_E4M3:
      res = caml_copy_double(
          (double)fp8_e4m3_to_float(((uint8_t *)b->data)[offset]));
      break;
    case NX_BA_FP8_E5M2:
      res = caml_copy_double(
          (double)fp8_e5m2_to_float(((uint8_t *)b->data)[offset]));
      break;
    case NX_BA_UINT32:
      res = caml_copy_int32(((uint32_t *)b->data)[offset]);
      break;
    case NX_BA_UINT64:
      res = caml_copy_int64(((uint64_t *)b->data)[offset]);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
  CAMLreturn(res);
}

CAMLprim value caml_nx_buffer_set(value vb, value vind, value newval) {
  CAMLparam3(vb, vind, newval);
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  int num_dims = Wosize_val(vind);
  if (num_dims != b->num_dims)
    caml_invalid_argument("Bigarray.set: wrong number of indices");

  intnat index[CAML_BA_MAX_NUM_DIMS];
  for (int i = 0; i < num_dims; i++)
    index[i] = Long_val(Field(vind, i));
  intnat offset = nx_buffer_offset(b, index);

  int kind = nx_buffer_get_kind(b);
  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    value args[CAML_BA_MAX_NUM_DIMS + 1];
    for (int i = 0; i < num_dims; i++)
      args[i] = Field(vind, i);
    args[num_dims] = newval;
    caml_ba_set_N(vb, args, num_dims + 1);
    CAMLreturn(Val_unit);
  }

  switch (kind) {
    case NX_BA_BFLOAT16:
      ((uint16_t *)b->data)[offset] =
          float_to_bfloat16((float)Double_val(newval));
      break;
    case NX_BA_BOOL:
      ((uint8_t *)b->data)[offset] = Bool_val(newval);
      break;
    case NX_BA_INT4: {
      int val = Int_val(newval);
      if (val > 7) val = 7;
      if (val < -8) val = -8;
      uint8_t nibble = val & 0x0F;
      uint8_t *byte_ptr = &((uint8_t *)b->data)[offset / 2];
      if (offset % 2 == 0)
        *byte_ptr = (*byte_ptr & 0xF0) | nibble;
      else
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4);
      break;
    }
    case NX_BA_UINT4: {
      int val = Int_val(newval);
      if (val > 15) val = 15;
      if (val < 0) val = 0;
      uint8_t nibble = val & 0x0F;
      uint8_t *byte_ptr = &((uint8_t *)b->data)[offset / 2];
      if (offset % 2 == 0)
        *byte_ptr = (*byte_ptr & 0xF0) | nibble;
      else
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4);
      break;
    }
    case NX_BA_FP8_E4M3:
      ((uint8_t *)b->data)[offset] =
          float_to_fp8_e4m3((float)Double_val(newval));
      break;
    case NX_BA_FP8_E5M2:
      ((uint8_t *)b->data)[offset] =
          float_to_fp8_e5m2((float)Double_val(newval));
      break;
    case NX_BA_UINT32:
      ((uint32_t *)b->data)[offset] = Int32_val(newval);
      break;
    case NX_BA_UINT64:
      ((uint64_t *)b->data)[offset] = Int64_val(newval);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
  CAMLreturn(Val_unit);
}

/* Unsafe 1D get — flat offset, no bounds check */
static value nx_buffer_unsafe_get_ext(struct caml_ba_array *b, intnat offset) {
  int kind = nx_buffer_get_kind(b);
  switch (kind) {
    case NX_BA_BFLOAT16:
      return caml_copy_double(
          (double)bfloat16_to_float(((uint16_t *)b->data)[offset]));
    case NX_BA_BOOL:
      return Val_bool(((uint8_t *)b->data)[offset]);
    case NX_BA_INT4: {
      uint8_t byte = ((uint8_t *)b->data)[offset / 2];
      int val;
      if (offset % 2 == 0)
        val = (int8_t)((byte & 0x0F) << 4) >> 4;
      else
        val = (int8_t)(byte & 0xF0) >> 4;
      return Val_int(val);
    }
    case NX_BA_UINT4: {
      uint8_t byte = ((uint8_t *)b->data)[offset / 2];
      int val;
      if (offset % 2 == 0)
        val = byte & 0x0F;
      else
        val = (byte >> 4) & 0x0F;
      return Val_int(val);
    }
    case NX_BA_FP8_E4M3:
      return caml_copy_double(
          (double)fp8_e4m3_to_float(((uint8_t *)b->data)[offset]));
    case NX_BA_FP8_E5M2:
      return caml_copy_double(
          (double)fp8_e5m2_to_float(((uint8_t *)b->data)[offset]));
    case NX_BA_UINT32:
      return caml_copy_int32(((uint32_t *)b->data)[offset]);
    case NX_BA_UINT64:
      return caml_copy_int64(((uint64_t *)b->data)[offset]);
    default:
      caml_failwith("Unsupported bigarray kind");
  }
}

CAMLprim value caml_nx_buffer_unsafe_get(value vb, value vi) {
  CAMLparam1(vb);
  CAMLlocal1(res);
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  intnat i = Long_val(vi);
  int kind = nx_buffer_get_kind(b);
  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    /* For standard kinds, use Bigarray.Array1.unsafe_get semantics:
       direct data access without bounds checking */
    extern value caml_ba_get_1(value vb, value vind);
    CAMLreturn(caml_ba_get_1(vb, vi));
  }
  res = nx_buffer_unsafe_get_ext(b, i);
  CAMLreturn(res);
}

/* Unsafe 1D set — flat offset, no bounds check */
static void nx_buffer_unsafe_set_ext(struct caml_ba_array *b, intnat offset,
                                     value newval) {
  int kind = nx_buffer_get_kind(b);
  switch (kind) {
    case NX_BA_BFLOAT16:
      ((uint16_t *)b->data)[offset] =
          float_to_bfloat16((float)Double_val(newval));
      break;
    case NX_BA_BOOL:
      ((uint8_t *)b->data)[offset] = Bool_val(newval);
      break;
    case NX_BA_INT4: {
      int val = Int_val(newval);
      if (val > 7) val = 7;
      if (val < -8) val = -8;
      uint8_t nibble = val & 0x0F;
      uint8_t *byte_ptr = &((uint8_t *)b->data)[offset / 2];
      if (offset % 2 == 0)
        *byte_ptr = (*byte_ptr & 0xF0) | nibble;
      else
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4);
      break;
    }
    case NX_BA_UINT4: {
      int val = Int_val(newval);
      if (val > 15) val = 15;
      if (val < 0) val = 0;
      uint8_t nibble = val & 0x0F;
      uint8_t *byte_ptr = &((uint8_t *)b->data)[offset / 2];
      if (offset % 2 == 0)
        *byte_ptr = (*byte_ptr & 0xF0) | nibble;
      else
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4);
      break;
    }
    case NX_BA_FP8_E4M3:
      ((uint8_t *)b->data)[offset] =
          float_to_fp8_e4m3((float)Double_val(newval));
      break;
    case NX_BA_FP8_E5M2:
      ((uint8_t *)b->data)[offset] =
          float_to_fp8_e5m2((float)Double_val(newval));
      break;
    case NX_BA_UINT32:
      ((uint32_t *)b->data)[offset] = Int32_val(newval);
      break;
    case NX_BA_UINT64:
      ((uint64_t *)b->data)[offset] = Int64_val(newval);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
}

CAMLprim value caml_nx_buffer_unsafe_set(value vb, value vi, value newval) {
  CAMLparam2(vb, newval);
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  intnat i = Long_val(vi);
  int kind = nx_buffer_get_kind(b);
  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    extern value caml_ba_set_1(value vb, value vind, value newval);
    caml_ba_set_1(vb, vi, newval);
    CAMLreturn(Val_unit);
  }
  nx_buffer_unsafe_set_ext(b, i, newval);
  CAMLreturn(Val_unit);
}

/*---------------------------------------------------------------------------
   Kind query
  ---------------------------------------------------------------------------*/

CAMLprim value caml_nx_buffer_kind(value vb) {
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  int kind = nx_buffer_get_kind(b);

  /* Map to GADT constructor index (19 constructors) */
  switch (kind) {
    case CAML_BA_FLOAT16:    return Val_int(0);
    case CAML_BA_FLOAT32:    return Val_int(1);
    case CAML_BA_FLOAT64:    return Val_int(2);
    case NX_BA_BFLOAT16:     return Val_int(3);
    case NX_BA_FP8_E4M3:    return Val_int(4);
    case NX_BA_FP8_E5M2:    return Val_int(5);
    case CAML_BA_SINT8:      return Val_int(6);
    case CAML_BA_UINT8:      return Val_int(7);
    case CAML_BA_SINT16:     return Val_int(8);
    case CAML_BA_UINT16:     return Val_int(9);
    case CAML_BA_INT32:      return Val_int(10);
    case NX_BA_UINT32:       return Val_int(11);
    case CAML_BA_INT64:      return Val_int(12);
    case NX_BA_UINT64:       return Val_int(13);
    case NX_BA_INT4:         return Val_int(14);
    case NX_BA_UINT4:        return Val_int(15);
    case CAML_BA_COMPLEX32:  return Val_int(16);
    case CAML_BA_COMPLEX64:  return Val_int(17);
    case NX_BA_BOOL:         return Val_int(18);
    default:
      caml_failwith("Unknown bigarray kind");
  }
}

/*---------------------------------------------------------------------------
   Bulk operations
  ---------------------------------------------------------------------------*/

CAMLprim value caml_nx_buffer_blit(value vsrc, value vdst) {
  CAMLparam2(vsrc, vdst);
  struct caml_ba_array *src = Caml_ba_array_val(vsrc);
  struct caml_ba_array *dst = Caml_ba_array_val(vdst);

  int src_kind = nx_buffer_get_kind(src);
  int dst_kind = nx_buffer_get_kind(dst);
  if (src_kind != dst_kind)
    caml_invalid_argument("Nx_buffer.blit: arrays have different kinds");
  if (src->num_dims != dst->num_dims)
    caml_invalid_argument("Nx_buffer.blit: arrays have different dimensions");

  uintnat num_elts = 1;
  for (int i = 0; i < src->num_dims; i++) {
    if (src->dim[i] != dst->dim[i])
      caml_invalid_argument("Nx_buffer.blit: arrays have different dimensions");
    num_elts *= src->dim[i];
  }

  if (src_kind >= CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    int elem_size = nx_buffer_element_byte_size(src_kind);
    size_t byte_size;
    if (elem_size > 0)
      byte_size = num_elts * elem_size;
    else
      byte_size = (num_elts + 1) / 2; /* int4/uint4 */
    memcpy(dst->data, src->data, byte_size);
  } else {
    caml_ba_blit(vsrc, vdst);
  }

  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_buffer_fill(value vb, value vinit) {
  CAMLparam2(vb, vinit);
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  int kind = nx_buffer_get_kind(b);

  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    caml_ba_fill(vb, vinit);
    CAMLreturn(Val_unit);
  }

  uintnat num_elts = nx_buffer_num_elts(b);

  switch (kind) {
    case NX_BA_BFLOAT16: {
      uint16_t init = float_to_bfloat16((float)Double_val(vinit));
      uint16_t *p = (uint16_t *)b->data;
      for (uintnat i = 0; i < num_elts; i++)
        p[i] = init;
      break;
    }
    case NX_BA_BOOL: {
      uint8_t init = Bool_val(vinit);
      memset(b->data, init, num_elts);
      break;
    }
    case NX_BA_INT4: {
      int val = Int_val(vinit);
      val = val < -8 ? -8 : val > 7 ? 7 : val;
      uint8_t nibble = (uint8_t)val & 0x0F;
      uint8_t packed = (nibble << 4) | nibble;
      memset(b->data, packed, (num_elts + 1) / 2);
      break;
    }
    case NX_BA_UINT4: {
      int val = Int_val(vinit);
      val = val < 0 ? 0 : val > 15 ? 15 : val;
      uint8_t nibble = (uint8_t)val & 0x0F;
      uint8_t packed = (nibble << 4) | nibble;
      memset(b->data, packed, (num_elts + 1) / 2);
      break;
    }
    case NX_BA_FP8_E4M3: {
      uint8_t init = float_to_fp8_e4m3((float)Double_val(vinit));
      memset(b->data, init, num_elts);
      break;
    }
    case NX_BA_FP8_E5M2: {
      uint8_t init = float_to_fp8_e5m2((float)Double_val(vinit));
      memset(b->data, init, num_elts);
      break;
    }
    case NX_BA_UINT32: {
      uint32_t init = Int32_val(vinit);
      uint32_t *p = (uint32_t *)b->data;
      for (uintnat i = 0; i < num_elts; i++)
        p[i] = init;
      break;
    }
    case NX_BA_UINT64: {
      uint64_t init = Int64_val(vinit);
      uint64_t *p = (uint64_t *)b->data;
      for (uintnat i = 0; i < num_elts; i++)
        p[i] = init;
      break;
    }
    default:
      caml_failwith("Unknown extended bigarray kind in fill");
  }

  CAMLreturn(Val_unit);
}

/*---------------------------------------------------------------------------
   Bytes blit ([@@noalloc] — no OCaml allocation, no exceptions)
  ---------------------------------------------------------------------------*/

CAMLprim value caml_nx_buffer_blit_from_bytes(value vbytes, value vsrc_off,
                                              value vdst, value vdst_off,
                                              value vlen) {
  struct caml_ba_array *dst = Caml_ba_array_val(vdst);
  size_t len = (size_t)Long_val(vlen);
  uint8_t *dst_ptr = (uint8_t *)dst->data + (size_t)Long_val(vdst_off);
  const uint8_t *src_ptr =
      (const uint8_t *)Bytes_val(vbytes) + (size_t)Long_val(vsrc_off);
  memcpy(dst_ptr, src_ptr, len);
  return Val_unit;
}

CAMLprim value caml_nx_buffer_blit_to_bytes(value vsrc, value vsrc_off,
                                            value vbytes, value vdst_off,
                                            value vlen) {
  struct caml_ba_array *src = Caml_ba_array_val(vsrc);
  size_t len = (size_t)Long_val(vlen);
  const uint8_t *src_ptr =
      (const uint8_t *)src->data + (size_t)Long_val(vsrc_off);
  uint8_t *dst_ptr = (uint8_t *)Bytes_val(vbytes) + (size_t)Long_val(vdst_off);
  memcpy(dst_ptr, src_ptr, len);
  return Val_unit;
}
