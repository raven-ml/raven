#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include "../../bigarray_ext/bigarray_ext_stubs.h"

/* Create a bigarray from an external pointer without copying.
   This works for both standard and extended types. */
CAMLprim value caml_metal_ba_from_ptr(value vkind, value vlayout, value vdim, value vptr) {
  CAMLparam4(vkind, vlayout, vdim, vptr);
  CAMLlocal1(res);
  
  int kind = Int_val(vkind);
  int layout = Int_val(vlayout);
  intnat dim = Long_val(vdim);
  void *data = (void *)Nativeint_val(vptr);
  
  /* Create a bigarray that points to the existing memory
     without copying. The CAML_BA_EXTERNAL flag means we don't
     own the memory and won't free it. */
  int flags = kind | layout | CAML_BA_EXTERNAL;
  
  res = caml_ba_alloc(flags, 1, data, &dim);
  
  CAMLreturn(res);
}

/* Get the raw integer value of a bigarray kind for extended types */
CAMLprim value caml_metal_kind_to_int(value vkind) {
  /* The OCaml kind constructors map to these values - must match bigarray_ext_stubs.h */
  int ocaml_constructor = Int_val(vkind);
  
  /* Map OCaml constructor values to bigarray kind flags */
  switch(ocaml_constructor) {
    case 0: return Val_int(CAML_BA_FLOAT32);        /* Float32 */
    case 1: return Val_int(CAML_BA_FLOAT64);        /* Float64 */
    case 2: return Val_int(CAML_BA_SINT8);          /* Int8_signed */
    case 3: return Val_int(CAML_BA_UINT8);          /* Int8_unsigned */
    case 4: return Val_int(CAML_BA_SINT16);         /* Int16_signed */
    case 5: return Val_int(CAML_BA_UINT16);         /* Int16_unsigned */
    case 6: return Val_int(CAML_BA_INT32);          /* Int32 */
    case 7: return Val_int(CAML_BA_INT64);          /* Int64 */
    case 8: return Val_int(CAML_BA_CAML_INT);       /* Int */
    case 9: return Val_int(CAML_BA_NATIVE_INT);     /* Nativeint */
    case 10: return Val_int(CAML_BA_COMPLEX32);     /* Complex32 */
    case 11: return Val_int(CAML_BA_COMPLEX64);     /* Complex64 */
    case 12: return Val_int(CAML_BA_CHAR);          /* Char */
    case 13: return Val_int(CAML_BA_FLOAT16);       /* Float16 */
    /* Extended types - these need to match bigarray_ext_stubs.h */
    case 14: return Val_int(NX_BA_BFLOAT16);        /* Bfloat16 */
    case 15: return Val_int(NX_BA_BOOL);            /* Bool */
    case 16: return Val_int(NX_BA_INT4);            /* Int4_signed */
    case 17: return Val_int(NX_BA_UINT4);           /* Int4_unsigned */
    case 18: return Val_int(NX_BA_FP8_E4M3);        /* Float8_e4m3 */
    case 19: return Val_int(NX_BA_FP8_E5M2);        /* Float8_e5m2 */
    case 20: return Val_int(NX_BA_COMPLEX16);       /* Complex16 */
    case 21: return Val_int(NX_BA_QINT8);           /* Qint8 */
    case 22: return Val_int(NX_BA_QUINT8);          /* Quint8 */
    default: return Val_int(ocaml_constructor);     /* Unknown - pass through */
  }
}