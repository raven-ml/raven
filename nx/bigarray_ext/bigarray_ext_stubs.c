#include "bigarray_ext_stubs.h"
#include <caml/fail.h>
#include <stdlib.h>
#include <string.h>

/* External declarations for standard bigarray functions */
extern value caml_ba_get_N(value vb, value * vind, int nind);
extern value caml_ba_set_N(value vb, value * vind, int nargs);
extern value caml_ba_blit(value vsrc, value vdst);

/* Element sizes for our extended types, aligning with stdlib caml_ba_element_size[] */
int caml_ba_extended_element_size[] = {
    [NX_BA_BFLOAT16 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 2, /* bfloat16 */
    [NX_BA_BOOL - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* bool */
    [NX_BA_INT4 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* int4_signed - use 1 byte (2 values packed), caml_ba_alloc will see 1 byte per "element" */
    [NX_BA_UINT4 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* int4_unsigned - use 1 byte (2 values packed) */
    [NX_BA_FP8_E4M3 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* fp8_e4m3 */
    [NX_BA_FP8_E5M2 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* fp8_e5m2 */
    [NX_BA_COMPLEX16 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 4, /* complex16 */
    [NX_BA_QINT8 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* qint8 */
    [NX_BA_QUINT8 - CAML_BA_FIRST_UNIMPLEMENTED_KIND] = 1, /* quint8 */
};

/* External reference to OCaml's element size table */
extern int caml_ba_element_size[];

/* Initialize element sizes in OCaml runtime's table */
__attribute__((constructor))
static void nx_ba_init(void) {
  /* Patch OCaml's caml_ba_element_size array with our extended types */
  for (int i = 0; i < (NX_BA_LAST_KIND - CAML_BA_FIRST_UNIMPLEMENTED_KIND); i++) {
    caml_ba_element_size[CAML_BA_FIRST_UNIMPLEMENTED_KIND + i] = caml_ba_extended_element_size[i];
  }
}

/* Helper for overflow-safe multiplication */
static int umul_overflow(uintnat a, uintnat b, uintnat *res) {
  if (b != 0 && a > (uintnat)(-1) / b) return 1;
  *res = a * b;
  return 0;
}

/* Calculate total number of elements from dimensions */
static uintnat nx_ba_num_elts_from_dims(int num_dims, intnat *dim) {
  uintnat num_elts = 1;
  for (int i = 0; i < num_dims; i++) {
    if (umul_overflow(num_elts, dim[i], &num_elts))
      caml_raise_out_of_memory();
  }
  return num_elts;
}

/* Helper macro to create bigarray creation functions */
#define CREATE_BA_FUNCTION(name, type_enum, bytes_per_elem) \
  CAMLprim value caml_nx_ba_create_##name(value vlayout, value vdim) { \
    CAMLparam2(vlayout, vdim); \
    CAMLlocal1(res); \
                                                                       \
    int num_dims = Wosize_val(vdim); \
    intnat dim[CAML_BA_MAX_NUM_DIMS]; \
                                                                       \
    for (int i = 0; i < num_dims; i++) { \
      dim[i] = Long_val(Field(vdim, i)); \
    } \
                                                                       \
    uintnat num_elts = nx_ba_num_elts_from_dims(num_dims, dim); \
    uintnat size; \
    if (umul_overflow(num_elts, (bytes_per_elem), &size)) \
      caml_raise_out_of_memory(); \
                                                                       \
    void *data = calloc(1, size); \
    if (data == NULL && size != 0) caml_raise_out_of_memory(); \
                                                                       \
    int layout_flag = Caml_ba_layout_val(vlayout); \
    int flags = (type_enum) | layout_flag | CAML_BA_MANAGED; \
    res = caml_ba_alloc(flags, num_dims, data, dim); \
                                                                       \
    CAMLreturn(res); \
  }

/* Create functions for each new type */
CREATE_BA_FUNCTION(bfloat16, NX_BA_BFLOAT16, 2)
CREATE_BA_FUNCTION(bool, NX_BA_BOOL, 1)
CREATE_BA_FUNCTION(float8_e4m3, NX_BA_FP8_E4M3, 1)
CREATE_BA_FUNCTION(float8_e5m2, NX_BA_FP8_E5M2, 1)
CREATE_BA_FUNCTION(complex16, NX_BA_COMPLEX16, 4) /* 2 x float16 */
CREATE_BA_FUNCTION(qint8, NX_BA_QINT8, 1)
CREATE_BA_FUNCTION(quint8, NX_BA_QUINT8, 1)

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
  void *data = calloc(1, size);
  if (data == NULL && size != 0) caml_raise_out_of_memory();
  int layout_flag = Caml_ba_layout_val(vlayout);
  int flags = NX_BA_INT4 | layout_flag | CAML_BA_MANAGED;
  /* Pass original dimensions to caml_ba_alloc - the element size of 1 in caml_ba_element_size
     will make it compute size correctly */
  res = caml_ba_alloc(flags, num_dims, data, original_dim);
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
  void *data = calloc(1, size);
  if (data == NULL && size != 0) caml_raise_out_of_memory();
  int layout_flag = Caml_ba_layout_val(vlayout);
  int flags = NX_BA_UINT4 | layout_flag | CAML_BA_MANAGED;
  /* Pass original dimensions to caml_ba_alloc - the element size of 1 in caml_ba_element_size
     will make it compute size correctly */
  res = caml_ba_alloc(flags, num_dims, data, original_dim);
  CAMLreturn(res);
}

/* Compute offset for bigarray element */
static intnat nx_ba_offset(struct caml_ba_array *b, intnat *index) {
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
  struct caml_ba_array *b = Caml_ba_array_val(vb);
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
  int kind = b->flags & CAML_BA_KIND_MASK;
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
          (double)bfloat16_to_float(((uint16_t *)b->data)[offset]));
      break;
    case NX_BA_BOOL:
      res = Val_bool(((uint8_t *)b->data)[offset]);
      break;
    case NX_BA_INT4: {
      uint8_t byte = ((uint8_t *)b->data)[offset / 2];
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
      uint8_t byte = ((uint8_t *)b->data)[offset / 2];
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
          (double)fp8_e4m3_to_float(((uint8_t *)b->data)[offset]));
      break;
    case NX_BA_FP8_E5M2:
      res = caml_copy_double(
          (double)fp8_e5m2_to_float(((uint8_t *)b->data)[offset]));
      break;
    case NX_BA_COMPLEX16: {
      uint16_t *p = ((uint16_t *)b->data) + offset * 2;
      float real = half_to_float(p[0]);
      float imag = half_to_float(p[1]);
      res = caml_alloc_small(2 * Double_wosize, Double_array_tag);
      Store_double_flat_field(res, 0, (double)real);
      Store_double_flat_field(res, 1, (double)imag);
      break;
    }
    case NX_BA_QINT8:
      res = Val_int(((int8_t *)b->data)[offset]);
      break;
    case NX_BA_QUINT8:
      res = Val_int(((uint8_t *)b->data)[offset]);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
  CAMLreturn(res);
}

/* Generic set function for extended types */
CAMLprim value caml_nx_ba_set_generic(value vb, value vind, value newval) {
  CAMLparam3(vb, vind, newval);
  struct caml_ba_array *b = Caml_ba_array_val(vb);
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
  int kind = b->flags & CAML_BA_KIND_MASK;
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
      ((uint16_t *)b->data)[offset] =
          float_to_bfloat16((float)Double_val(newval));
      break;
    case NX_BA_BOOL:
      ((uint8_t *)b->data)[offset] = Bool_val(newval);
      break;
    case NX_BA_INT4: {
      int val = Int_val(newval);
      /* Clamp to [-8, 7] for signed 4-bit */
      if (val > 7) val = 7;
      if (val < -8) val = -8;
      uint8_t nibble = val & 0x0F;  /* Two's complement representation */
      uint8_t *byte_ptr = &((uint8_t *)b->data)[offset / 2];
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
      uint8_t *byte_ptr = &((uint8_t *)b->data)[offset / 2];
      if (offset % 2 == 0) {
        *byte_ptr = (*byte_ptr & 0xF0) | nibble; /* Set lower 4 bits */
      } else {
        *byte_ptr = (*byte_ptr & 0x0F) | (nibble << 4); /* Set upper 4 bits */
      }
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
    case NX_BA_COMPLEX16: {
      uint16_t *p = ((uint16_t *)b->data) + offset * 2;
      p[0] = float_to_half((float)Double_flat_field(newval, 0));
      p[1] = float_to_half((float)Double_flat_field(newval, 1));
      break;
    }
    case NX_BA_QINT8:
      ((int8_t *)b->data)[offset] = Int_val(newval);
      break;
    case NX_BA_QUINT8:
      ((uint8_t *)b->data)[offset] = Int_val(newval);
      break;
    default:
      caml_failwith("Unsupported bigarray kind");
  }
  CAMLreturn(Val_unit);
}

/* Get the extended kind of a bigarray */
CAMLprim value caml_nx_ba_kind(value vb) {
  struct caml_ba_array *b = Caml_ba_array_val(vb);
  int kind = b->flags & CAML_BA_KIND_MASK;
 
  /* Map standard kinds to our extended kind values */
  switch (kind) {
    case CAML_BA_FLOAT32: return Val_int(0); /* Float32 */
    case CAML_BA_FLOAT64: return Val_int(1); /* Float64 */
    case CAML_BA_SINT8: return Val_int(2); /* Int8_signed */
    case CAML_BA_UINT8: return Val_int(3); /* Int8_unsigned */
    case CAML_BA_SINT16: return Val_int(4); /* Int16_signed */
    case CAML_BA_UINT16: return Val_int(5); /* Int16_unsigned */
    case CAML_BA_INT32: return Val_int(6); /* Int32 */
    case CAML_BA_INT64: return Val_int(7); /* Int64 */
    case CAML_BA_CAML_INT: return Val_int(8); /* Int */
    case CAML_BA_NATIVE_INT: return Val_int(9); /* Nativeint */
    case CAML_BA_COMPLEX32: return Val_int(10); /* Complex32 */
    case CAML_BA_COMPLEX64: return Val_int(11); /* Complex64 */
    case CAML_BA_CHAR: return Val_int(12); /* Char */
    case CAML_BA_FLOAT16: return Val_int(13); /* Float16 */
    case NX_BA_BFLOAT16: return Val_int(14); /* Bfloat16 */
    case NX_BA_BOOL: return Val_int(15); /* Bool */
    case NX_BA_INT4: return Val_int(16); /* Int4_signed */
    case NX_BA_UINT4: return Val_int(17); /* Int4_unsigned */
    case NX_BA_FP8_E4M3: return Val_int(18); /* Float8_e4m3 */
    case NX_BA_FP8_E5M2: return Val_int(19); /* Float8_e5m2 */
    case NX_BA_COMPLEX16: return Val_int(20); /* Complex16 */
    case NX_BA_QINT8: return Val_int(21); /* Qint8 */
    case NX_BA_QUINT8: return Val_int(22); /* Quint8 */
    default:
      caml_failwith("Unknown bigarray kind");
  }
}

/* Blit implementation for extended types */
CAMLprim value caml_nx_ba_blit(value vsrc, value vdst) {
  CAMLparam2(vsrc, vdst);
  struct caml_ba_array *src = Caml_ba_array_val(vsrc);
  struct caml_ba_array *dst = Caml_ba_array_val(vdst);
 
  /* Check that kinds match */
  int src_kind = src->flags & CAML_BA_KIND_MASK;
  int dst_kind = dst->flags & CAML_BA_KIND_MASK;
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
      caml_invalid_argument("caml_nx_ba_blit: arrays have different dimensions");
    }
    num_elts *= src->dim[i];
  }
 
  /* Check if this is an extended type */
  if (src_kind >= CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    /* For extended types, get element size and use memcpy */
    size_t byte_size;
    switch (src_kind) {
      case NX_BA_BFLOAT16: byte_size = num_elts * 2; break;
      case NX_BA_BOOL: byte_size = num_elts; break;
      case NX_BA_INT4:
      case NX_BA_UINT4: 
        /* int4/uint4 pack 2 elements per byte */
        byte_size = (num_elts + 1) / 2; 
        break;
      case NX_BA_FP8_E4M3: byte_size = num_elts; break;
      case NX_BA_FP8_E5M2: byte_size = num_elts; break;
      case NX_BA_COMPLEX16: byte_size = num_elts * 4; break;
      case NX_BA_QINT8: byte_size = num_elts; break;
      case NX_BA_QUINT8: byte_size = num_elts; break;
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
static uintnat nx_ba_num_elts(struct caml_ba_array * b) {
  uintnat num_elts = 1;
  for (int i = 0; i < b->num_dims; i++) {
    num_elts *= b->dim[i];
  }
  return num_elts;
}

/* Fill implementation for extended types */
CAMLprim value caml_nx_ba_fill(value vb, value vinit) {
  CAMLparam2(vb, vinit);
  struct caml_ba_array * b = Caml_ba_array_val(vb);
  int kind = b->flags & CAML_BA_KIND_MASK;
 
  /* Check if this is an extended type */
  if (kind >= CAML_BA_FIRST_UNIMPLEMENTED_KIND) {
    uintnat num_elts = nx_ba_num_elts(b);
   
    switch (kind) {
      case NX_BA_BFLOAT16: {
        float fval = (float)Double_val(vinit);
        uint16_t init = float_to_bfloat16(fval);
        uint16_t *p = (uint16_t *)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_BOOL: {
        uint8_t init = Bool_val(vinit);
        uint8_t *p = (uint8_t *)b->data;
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
        uint8_t *p = (uint8_t *)b->data;
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
        uint8_t *p = (uint8_t *)b->data;
        uintnat bytes = (num_elts + 1) / 2;
        for (uintnat i = 0; i < bytes; i++) {
          p[i] = packed;
        }
        break;
      }
      case NX_BA_FP8_E4M3: {
        uint8_t init = float_to_fp8_e4m3((float)Double_val(vinit));
        uint8_t *p = (uint8_t *)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_FP8_E5M2: {
        uint8_t init = float_to_fp8_e5m2((float)Double_val(vinit));
        uint8_t *p = (uint8_t *)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_COMPLEX16: {
        uint16_t re = float_to_half((float)Double_flat_field(vinit, 0));
        uint16_t im = float_to_half((float)Double_flat_field(vinit, 1));
        uint16_t *p = (uint16_t *)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[2*i] = re;
          p[2*i+1] = im;
        }
        break;
      }
      case NX_BA_QINT8: {
        int8_t init = Int_val(vinit);
        int8_t *p = (int8_t *)b->data;
        for (uintnat i = 0; i < num_elts; i++) {
          p[i] = init;
        }
        break;
      }
      case NX_BA_QUINT8: {
        uint8_t init = Int_val(vinit);
        uint8_t *p = (uint8_t *)b->data;
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