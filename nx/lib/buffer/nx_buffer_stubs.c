/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include "nx_buffer_stubs.h"

#include <caml/fail.h>
#include <stdlib.h>
#include <string.h>

/* External declarations for standard bigarray functions */
extern value caml_ba_get_N(value vb, value* vind, int nind);
extern value caml_ba_set_N(value vb, value* vind, int nargs);
extern value caml_ba_blit(value vsrc, value vdst);

static int nx_ba_base_kind(int kind) {
  switch (kind) {
    case NX_BA_BFLOAT16:
      return CAML_BA_FLOAT16;
    case NX_BA_BOOL:
      return CAML_BA_UINT8;
    case NX_BA_INT4:
    case NX_BA_UINT4:
    case NX_BA_FP8_E4M3:
    case NX_BA_FP8_E5M2:
      return CAML_BA_UINT8;
    case NX_BA_UINT32:
      return CAML_BA_INT32;
    case NX_BA_UINT64:
      return CAML_BA_INT64;
    default:
      return -1;
  }
}

static value caml_nx_ba_alloc_with_kind(int kind, int layout_flag, int num_dims,
                                        intnat* dim, void* data) {
  int base_kind = nx_ba_is_extended_kind(kind) ? nx_ba_base_kind(kind) : kind;
  if (base_kind < 0) caml_failwith("Unknown extended bigarray kind");
  int flags = base_kind | layout_flag | CAML_BA_MANAGED;
  value res = caml_ba_alloc(flags, num_dims, data, dim);
  struct caml_ba_array* ba = Caml_ba_array_val(res);
  ba->flags = nx_ba_store_extended_kind(ba->flags, kind);
  return res;
}

/* Helper for overflow-safe multiplication */
static int umul_overflow(uintnat a, uintnat b, uintnat* res) {
  if (b != 0 && a > (uintnat)(-1) / b) return 1;
  *res = a * b;
  return 0;
}

/* Calculate total number of elements from dimensions */
static uintnat nx_ba_num_elts_from_dims(int num_dims, intnat* dim) {
  uintnat num_elts = 1;
  for (int i = 0; i < num_dims; i++) {
    if (umul_overflow(num_elts, dim[i], &num_elts)) caml_raise_out_of_memory();
  }
  return num_elts;
}

/* Helper macro to create bigarray creation functions */
#define CREATE_BA_FUNCTION(name, type_enum, bytes_per_elem)                   \
  CAMLprim value caml_nx_ba_create_##name(value vlayout, value vdim) {        \
    CAMLparam2(vlayout, vdim);                                                \
    CAMLlocal1(res);                                                          \
                                                                              \
    int num_dims = Wosize_val(vdim);                                          \
    intnat dim[CAML_BA_MAX_NUM_DIMS];                                         \
                                                                              \
    for (int i = 0; i < num_dims; i++) {                                      \
      dim[i] = Long_val(Field(vdim, i));                                      \
    }                                                                         \
                                                                              \
    uintnat num_elts = nx_ba_num_elts_from_dims(num_dims, dim);               \
    uintnat size;                                                             \
    if (umul_overflow(num_elts, (bytes_per_elem), &size))                     \
      caml_raise_out_of_memory();                                             \
                                                                              \
    void* data = calloc(1, size);                                             \
    if (data == NULL && size != 0) caml_raise_out_of_memory();                \
                                                                              \
    int layout_flag = Caml_ba_layout_val(vlayout);                            \
    res = caml_nx_ba_alloc_with_kind((type_enum), layout_flag, num_dims, dim, \
                                     data);                                   \
                                                                              \
    CAMLreturn(res);                                                          \
  }

/* Create functions for each new type */
CREATE_BA_FUNCTION(bfloat16, NX_BA_BFLOAT16, 2)
CREATE_BA_FUNCTION(bool, NX_BA_BOOL, 1)
CREATE_BA_FUNCTION(float8_e4m3, NX_BA_FP8_E4M3, 1)
CREATE_BA_FUNCTION(float8_e5m2, NX_BA_FP8_E5M2, 1)
CREATE_BA_FUNCTION(uint32, NX_BA_UINT32, 4)
CREATE_BA_FUNCTION(uint64, NX_BA_UINT64, 8)

/* Special handling for int4/uint4 which pack 2 values per byte */
CAMLprim value caml_nx_ba_create_int4_signed(value vlayout, value vdim) {
  CAMLparam2(vlayout, vdim);
  CAMLlocal1(res);
  int num_dims = Wosize_val(vdim);
  intnat dim[CAML_BA_MAX_NUM_DIMS];
  intnat original_dim[CAML_BA_MAX_NUM_DIMS];
  for (int i = 0; i < num_dims; i++) {
    original_dim[i] = Long_val(Field(vdim, i));
    dim[i] = original_dim[i];
  }
  uintnat num_elts = nx_ba_num_elts_from_dims(num_dims, dim);
  /* For int4, we pack 2 values per byte, so divide by 2 (round up) */
  uintnat size = (num_elts + 1) / 2;
  void* data = calloc(1, size);
  if (data == NULL && size != 0) caml_raise_out_of_memory();
  int layout_flag = Caml_ba_layout_val(vlayout);
  /* Pass original dimensions to caml_ba_alloc and tag the extended kind for
   * consumers */
  res = caml_nx_ba_alloc_with_kind(NX_BA_INT4, layout_flag, num_dims,
                                   original_dim, data);
  CAMLreturn(res);
}

CAMLprim value caml_nx_ba_create_int4_unsigned(value vlayout, value vdim) {
  CAMLparam2(vlayout, vdim);
  CAMLlocal1(res);
  int num_dims = Wosize_val(vdim);
  intnat dim[CAML_BA_MAX_NUM_DIMS];
  intnat original_dim[CAML_BA_MAX_NUM_DIMS];
  for (int i = 0; i < num_dims; i++) {
    original_dim[i] = Long_val(Field(vdim, i));
    dim[i] = original_dim[i];
  }
  uintnat num_elts = nx_ba_num_elts_from_dims(num_dims, dim);
  /* For uint4, we pack 2 values per byte, so divide by 2 (round up) */
  uintnat size = (num_elts + 1) / 2;
  void* data = calloc(1, size);
  if (data == NULL && size != 0) caml_raise_out_of_memory();
  int layout_flag = Caml_ba_layout_val(vlayout);
  /* Pass original dimensions to caml_ba_alloc and tag the extended kind for
   * consumers */
  res = caml_nx_ba_alloc_with_kind(NX_BA_UINT4, layout_flag, num_dims,
                                   original_dim, data);
  CAMLreturn(res);
}

/* Compute offset for bigarray element */
static intnat nx_ba_offset(struct caml_ba_array* b, intnat* index) {
  intnat offset = 0;
  switch ((enum caml_ba_layout)(b->flags & CAML_BA_LAYOUT_MASK)) {
    case CAML_BA_C_LAYOUT:
      /* C-style layout: row major, indices start at 0 */
      for (int i = 0; i < b->num_dims; i++) {
        if ((uintnat)index[i] >= (uintnat)b->dim[i]) caml_array_bound_error();
        offset = offset * b->dim[i] + index[i];
      }
      break;
    case CAML_BA_FORTRAN_LAYOUT:
      /* Fortran-style layout: column major, indices start at 1 */
      for (int i = b->num_dims - 1; i >= 0; i--) {
        if ((uintnat)(index[i] - 1) >= (uintnat)b->dim[i])
          caml_array_bound_error();
        offset = offset * b->dim[i] + (index[i] - 1);
      }
      break;
  }
  return offset;
}

/* Generic get function for extended types */
CAMLprim value caml_nx_ba_get_generic(value vb, value vind) {
  CAMLparam2(vb, vind);
  CAMLlocal1(res);
  struct caml_ba_array* b = Caml_ba_array_val(vb);
  intnat index[CAML_BA_MAX_NUM_DIMS];
  intnat offset;
  int num_dims = Wosize_val(vind);
  /* Check number of indices = number of dimensions of array */
  if (num_dims != b->num_dims)
    caml_invalid_argument("Bigarray.get: wrong number of indices");
  /* Compute offset and check bounds */
  for (int i = 0; i < b->num_dims; i++) index[i] = Long_val(Field(vind, i));
  offset = nx_ba_offset(b, index);
  /* Perform read based on kind */
  int kind = nx_ba_get_kind(b);
  /* Handle standard types first */
  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    /* Use standard bigarray get - we need to build the arguments */
    value args[CAML_BA_MAX_NUM_DIMS + 1];
    args[0] = vb;
    for (int i = 0; i < num_dims; i++) {
      args[i + 1] = Field(vind, i);
    }
    CAMLreturn(caml_ba_get_N(vb, args + 1, num_dims));
  }
  /* Handle extended types */
  switch (kind) {
    case NX_BA_BFLOAT16:
      res = caml_copy_double(
          (double)bfloat16_to_float(((uint16_t*)b->data)[offset]));
      break;
    case NX_BA_BOOL:
      res = Val_bool(((uint8_t*)b->data)[offset]);
      break;
    case NX_BA_INT4: {
      uint8_t byte = ((uint8_t*)b->data)[offset / 2];
      int val;
      if (offset % 2 == 0) {
        val = (int8_t)((byte & 0x0F) << 4) >> 4; /* Sign extend lower 4 bits */
      } else {
        val = (int8_t)(byte & 0xF0) >> 4; /* Sign extend upper 4 bits */
      }
      res = Val_int(val);
      break;
    }
    case NX_BA_UINT4: {
      uint8_t byte = ((uint8_t*)b->data)[offset / 2];
      int val;
      if (offset % 2 == 0) {
        val = byte & 0x0F; /* Lower 4 bits */
      } else {
        val = (byte >> 4) & 0x0F; /* Upper 4 bits */
      }
      res = Val_int(val);
      break;
    }
    case NX_BA_FP8_E4M3:
      res = caml_copy_double(
          (double)fp8_e4m3_to_float(((uint8_t*)b->data)[offset]));
      break;
    case NX_BA_FP8_E5M2:
      res = caml_copy_double(
          (double)fp8_e5m2_to_float(((uint8_t*)b->data)[offset]));
      break;
    case NX_BA_UINT32:
      res = caml_copy_int32(((uint32_t*)b->data)[offset]);
      break;
    case NX_BA_UINT64:
      res = caml_copy_int64(((uint64_t*)b->data)[offset]);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
  CAMLreturn(res);
}

/* Generic set function for extended types */
CAMLprim value caml_nx_ba_set_generic(value vb, value vind, value newval) {
  CAMLparam3(vb, vind, newval);
  struct caml_ba_array* b = Caml_ba_array_val(vb);
  intnat index[CAML_BA_MAX_NUM_DIMS];
  intnat offset;
  int num_dims = Wosize_val(vind);
  /* Check number of indices = number of dimensions of array */
  if (num_dims != b->num_dims)
    caml_invalid_argument("Bigarray.set: wrong number of indices");
  /* Compute offset and check bounds */
  for (int i = 0; i < b->num_dims; i++) index[i] = Long_val(Field(vind, i));
  offset = nx_ba_offset(b, index);
  /* Perform write based on kind */
  int kind = nx_ba_get_kind(b);
  /* Handle standard types first */
  if (kind < CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    /* Use standard bigarray set */
    value args[CAML_BA_MAX_NUM_DIMS + 2];
    args[0] = vb;
    for (int i = 0; i < num_dims; i++) {
      args[i + 1] = Field(vind, i);
    }
    args[num_dims + 1] = newval;
    caml_ba_set_N(vb, args + 1, num_dims + 1);
    CAMLreturn(Val_unit);
  }
  /* Handle extended types */
  switch (kind) {
    case NX_BA_BFLOAT16:
      ((uint16_t*)b->data)[offset] =
          float_to_bfloat16((float)Double_val(newval));
      break;
    case NX_BA_BOOL:
      ((uint8_t*)b->data)[offset] = Bool_val(newval);
      break;
    case NX_BA_INT4: {
      int val = Int_val(newval);
      /* Clamp to [-8, 7] for signed 4-bit */
      if (val > 7) val = 7;
      if (val < -8) val = -8;
      uint8_t nibble = val & 0x0F; /* Two's complement representation */
      uint8_t* byte_ptr = &((uint8_t*)b->data)[offset / 2];
      if (offset % 2 == 0) {
        *byte_ptr = (*byte_ptr & 0xF0) | nibble; /* Set lower 4 bits */
      } else {
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4); /* Set upper 4 bits */
      }
      break;
    }
    case NX_BA_UINT4: {
      int val = Int_val(newval);
      /* Clamp to [0, 15] for unsigned 4-bit */
      if (val > 15) val = 15;
      if (val < 0) val = 0;
      uint8_t nibble = val & 0x0F;
      uint8_t* byte_ptr = &((uint8_t*)b->data)[offset / 2];
      if (offset % 2 == 0) {
        *byte_ptr = (*byte_ptr & 0xF0) | nibble; /* Set lower 4 bits */
      } else {
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4); /* Set upper 4 bits */
      }
      break;
    }
    case NX_BA_FP8_E4M3:
      ((uint8_t*)b->data)[offset] =
          float_to_fp8_e4m3((float)Double_val(newval));
      break;
    case NX_BA_FP8_E5M2:
      ((uint8_t*)b->data)[offset] =
          float_to_fp8_e5m2((float)Double_val(newval));
      break;
    case NX_BA_UINT32:
      ((uint32_t*)b->data)[offset] = Int32_val(newval);
      break;
    case NX_BA_UINT64:
      ((uint64_t*)b->data)[offset] = Int64_val(newval);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_ba_blit_from_bytes(value vbytes, value vsrc_off,
                                          value vdst, value vdst_off,
                                          value vlen) {
  CAMLparam5(vbytes, vsrc_off, vdst, vdst_off, vlen);
  struct caml_ba_array* dst = Caml_ba_array_val(vdst);
  size_t len = (size_t)Long_val(vlen);
  uint8_t* dst_ptr = (uint8_t*)dst->data + (size_t)Long_val(vdst_off);
  const uint8_t* src_ptr =
      (const uint8_t*)Bytes_val(vbytes) + (size_t)Long_val(vsrc_off);
  memcpy(dst_ptr, src_ptr, len);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_ba_blit_to_bytes(value vsrc, value vsrc_off,
                                        value vbytes, value vdst_off,
                                        value vlen) {
  CAMLparam5(vsrc, vsrc_off, vbytes, vdst_off, vlen);
  struct caml_ba_array* src = Caml_ba_array_val(vsrc);
  size_t len = (size_t)Long_val(vlen);
  const uint8_t* src_ptr =
      (const uint8_t*)src->data + (size_t)Long_val(vsrc_off);
  uint8_t* dst_ptr = (uint8_t*)Bytes_val(vbytes) + (size_t)Long_val(vdst_off);
  memcpy(dst_ptr, src_ptr, len);
  CAMLreturn(Val_unit);
}

/* Get the extended kind of a bigarray */
CAMLprim value caml_nx_ba_kind(value vb) {
  struct caml_ba_array* b = Caml_ba_array_val(vb);
  int kind = nx_ba_get_kind(b);

  /* Map to GADT constructor index (19 constructors) */
  switch (kind) {
    case CAML_BA_FLOAT16:
      return Val_int(0); /* Float16 */
    case CAML_BA_FLOAT32:
      return Val_int(1); /* Float32 */
    case CAML_BA_FLOAT64:
      return Val_int(2); /* Float64 */
    case NX_BA_BFLOAT16:
      return Val_int(3); /* Bfloat16 */
    case NX_BA_FP8_E4M3:
      return Val_int(4); /* Float8_e4m3 */
    case NX_BA_FP8_E5M2:
      return Val_int(5); /* Float8_e5m2 */
    case CAML_BA_SINT8:
      return Val_int(6); /* Int8_signed */
    case CAML_BA_UINT8:
      return Val_int(7); /* Int8_unsigned */
    case CAML_BA_SINT16:
      return Val_int(8); /* Int16_signed */
    case CAML_BA_UINT16:
      return Val_int(9); /* Int16_unsigned */
    case CAML_BA_INT32:
      return Val_int(10); /* Int32 */
    case NX_BA_UINT32:
      return Val_int(11); /* Uint32 */
    case CAML_BA_INT64:
      return Val_int(12); /* Int64 */
    case NX_BA_UINT64:
      return Val_int(13); /* Uint64 */
    case NX_BA_INT4:
      return Val_int(14); /* Int4_signed */
    case NX_BA_UINT4:
      return Val_int(15); /* Int4_unsigned */
    case CAML_BA_COMPLEX32:
      return Val_int(16); /* Complex32 */
    case CAML_BA_COMPLEX64:
      return Val_int(17); /* Complex64 */
    case NX_BA_BOOL:
      return Val_int(18); /* Bool */
    default:
      caml_failwith("Unknown bigarray kind");
  }
}

/* Blit implementation for extended types */
CAMLprim value caml_nx_ba_blit(value vsrc, value vdst) {
  CAMLparam2(vsrc, vdst);
  struct caml_ba_array* src = Caml_ba_array_val(vsrc);
  struct caml_ba_array* dst = Caml_ba_array_val(vdst);

  /* Check that kinds match */
  int src_kind = nx_ba_get_kind(src);
  int dst_kind = nx_ba_get_kind(dst);
  if (src_kind != dst_kind) {
    caml_invalid_argument("caml_nx_ba_blit: arrays have different kinds");
  }

  /* Check that dimensions match */
  if (src->num_dims != dst->num_dims) {
    caml_invalid_argument("caml_nx_ba_blit: arrays have different dimensions");
  }

  /* Get total number of elements */
  uintnat num_elts = 1;
  for (int i = 0; i < src->num_dims; i++) {
    if (src->dim[i] != dst->dim[i]) {
      caml_invalid_argument(
          "caml_nx_ba_blit: arrays have different dimensions");
    }
    num_elts *= src->dim[i];
  }

  /* Check if this is an extended type */
  if (src_kind >= CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    /* For extended types, get element size and use memcpy */
    size_t byte_size;
    switch (src_kind) {
      case NX_BA_BFLOAT16:
        byte_size = num_elts * 2;
        break;
      case NX_BA_BOOL:
        byte_size = num_elts;
        break;
      case NX_BA_INT4:
      case NX_BA_UINT4:
        /* int4/uint4 pack 2 elements per byte */
        byte_size = (num_elts + 1) / 2;
        break;
      case NX_BA_FP8_E4M3:
        byte_size = num_elts;
        break;
      case NX_BA_FP8_E5M2:
        byte_size = num_elts;
        break;
      case NX_BA_UINT32:
        byte_size = num_elts * 4;
        break;
      case NX_BA_UINT64:
        byte_size = num_elts * 8;
        break;
      default:
        caml_failwith("Unknown extended bigarray kind in blit");
    }
    memcpy(dst->data, src->data, byte_size);
  } else {
    /* For standard types, use the standard blit */
    caml_ba_blit(vsrc, vdst);
  }

  CAMLreturn(Val_unit);
}

/* Helper function to get number of elements */
static uintnat nx_ba_num_elts(struct caml_ba_array* b) {
  uintnat num_elts = 1;
  for (int i = 0; i < b->num_dims; i++) {
    num_elts *= b->dim[i];
  }
  return num_elts;
}

/* Fill implementation for extended types */
CAMLprim value caml_nx_ba_fill(value vb, value vinit) {
  CAMLparam2(vb, vinit);
  struct caml_ba_array* b = Caml_ba_array_val(vb);
  int kind = nx_ba_get_kind(b);

  /* Check if this is an extended type */
  if (kind >= CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    uintnat num_elts = nx_ba_num_elts(b);

    switch (kind) {
      case NX_BA_BFLOAT16: {
        float fval = (float)Double_val(vinit);
        uint16_t init = float_to_bfloat16(fval);
        uint16_t* p = (uint16_t*)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_BOOL: {
        uint8_t init = Bool_val(vinit);
        uint8_t* p = (uint8_t*)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_INT4: {
        /* Int4 needs special handling as 2 values are packed per byte */
        int val = Int_val(vinit);
        /* Clamp to [-8, 7] */
        val = val < -8 ? -8 : val > 7 ? 7 : val;
        uint8_t nibble = (uint8_t)val & 0x0F; /* Two's complement */
        uint8_t packed = (nibble << 4) | nibble;
        uint8_t* p = (uint8_t*)b->data;
        uintnat bytes = (num_elts + 1) / 2;
        for (uintnat i = 0; i < bytes; i++) {
          p[i] = packed;
        }
        break;
      }
      case NX_BA_UINT4: {
        /* UInt4 needs special handling as 2 values are packed per byte */
        int val = Int_val(vinit);
        /* Clamp to [0, 15] */
        val = val < 0 ? 0 : val > 15 ? 15 : val;
        uint8_t nibble = (uint8_t)val & 0x0F;
        uint8_t packed = (nibble << 4) | nibble;
        uint8_t* p = (uint8_t*)b->data;
        uintnat bytes = (num_elts + 1) / 2;
        for (uintnat i = 0; i < bytes; i++) {
          p[i] = packed;
        }
        break;
      }
      case NX_BA_FP8_E4M3: {
        uint8_t init = float_to_fp8_e4m3((float)Double_val(vinit));
        uint8_t* p = (uint8_t*)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_FP8_E5M2: {
        uint8_t init = float_to_fp8_e5m2((float)Double_val(vinit));
        uint8_t* p = (uint8_t*)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_UINT32: {
        uint32_t init = Int32_val(vinit);
        uint32_t* p = (uint32_t*)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_UINT64: {
        uint64_t init = Int64_val(vinit);
        uint64_t* p = (uint64_t*)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      default:
        caml_failwith("Unknown extended bigarray kind in fill");
    }
  } else {
    /* Use standard fill for regular types */
    extern CAMLprim value caml_ba_fill(value vb, value vinit);
    caml_ba_fill(vb, vinit);
  }

  CAMLreturn(Val_unit);
}
