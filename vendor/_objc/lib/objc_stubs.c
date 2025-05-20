#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/config.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <ffi.h>
#include <objc/message.h>
#include <objc/runtime.h>

typedef void *ObjC_Id_t;
typedef void *ObjC_Sel_t;
typedef void *ObjC_Class_t;

#define Val_ptr(p) caml_copy_nativeint((intnat)(p))
#define Ptr_val(v) ((void *)Nativeint_val(v))

CAMLprim value caml_objc_getClass(value v_name) {
  CAMLparam1(v_name);
  const char *name = String_val(v_name);
  ObjC_Class_t cls = (ObjC_Class_t)objc_getClass(name); /* may be NULL */
  CAMLreturn(Val_ptr(cls));
}

CAMLprim value caml_sel_registerName(value v_name) {
  CAMLparam1(v_name);
  const char *name = String_val(v_name);
  ObjC_Sel_t sel = (ObjC_Sel_t)sel_registerName(name);
  CAMLreturn(Val_ptr(sel));
}

CAMLprim value caml_sel_getName(value v_sel) {
  CAMLparam1(v_sel);
  ObjC_Sel_t sel = (ObjC_Sel_t)Ptr_val(v_sel);
  if (sel == NULL) caml_failwith("caml_sel_getName: NULL selector");
  CAMLreturn(caml_copy_string(sel_getName((SEL)sel)));
}

CAMLprim value caml_class_getName(value v_cls) {
  CAMLparam1(v_cls);
  ObjC_Class_t cls = (ObjC_Class_t)Ptr_val(v_cls);
  if (cls == NULL) caml_failwith("caml_class_getName: NULL class");
  CAMLreturn(caml_copy_string(class_getName((Class)cls)));
}

CAMLprim value caml_object_getClass(value v_obj) {
  CAMLparam1(v_obj);
  ObjC_Id_t obj = (ObjC_Id_t)Ptr_val(v_obj);
  ObjC_Class_t cls =
      (ObjC_Class_t)object_getClass((id)obj); /* returns nil => 0 */
  CAMLreturn(Val_ptr(cls));
}

static ffi_type *ffi_type_of_tag(int tag) {
  switch (tag) {
    case 0:
      return &ffi_type_void; /* Void   */
    case 1:                  /* Char   */
    case 2:                  /* SChar  */
    case 3:
      return &ffi_type_schar; /* UChar  uses same rep */
    case 4:
      return &ffi_type_uint8; /* Bool   */
    case 5:
      return &ffi_type_sint16; /* Short  */
    case 6:
      return &ffi_type_uint16; /* UShort */
    case 7:
      return &ffi_type_sint32; /* Int    */
    case 8:
      return &ffi_type_uint32; /* UInt   */
    case 9:
      return &ffi_type_sint64; /* Long   */
    case 10:
      return &ffi_type_uint64; /* ULong  */
    case 11:
      return &ffi_type_sint64; /* LLong  */
    case 12:
      return &ffi_type_uint64; /* ULLong */
    case 13:
      return &ffi_type_float; /* Float  */
    case 14:
      return &ffi_type_double; /* Double */
    case 15:
      return &ffi_type_pointer; /* Pointer*/
    default:
      caml_failwith("msg_send: unknown tag");
  }
  return NULL; /* not reached */
}

CAMLprim value caml_msg_send_ffi(value v_self, value v_sel, value v_arg_tags,
                                 value v_arg_data, value v_ret_tag) {
  CAMLparam5(v_self, v_sel, v_arg_tags, v_arg_data, v_ret_tag);

  /* 1.  Extract arguments from OCaml values */
  long self = Nativeint_val(v_self);
  long sel = Nativeint_val(v_sel);
  int ret_tag = Int_val(v_ret_tag);

  int nargs = Wosize_val(v_arg_tags);
  if (nargs != Wosize_val(v_arg_data))
    caml_failwith("msg_send: tag/data length mismatch");

  /* 2.  Build ffi_type** and void* arrays */
  ffi_type **arg_types =
      alloca(sizeof(ffi_type *) * (nargs + 2)); /* self + _cmd + user args */
  void **arg_values = alloca(sizeof(void *) * (nargs + 2));

  long *raw_data = alloca(sizeof(long) * (nargs + 2));
  /* put self and _cmd first */
  raw_data[0] = self;
  raw_data[1] = sel;
  arg_values[0] = &raw_data[0];
  arg_values[1] = &raw_data[1];
  arg_types[0] = ffi_type_of_tag(15); /* pointer */
  arg_types[1] = ffi_type_of_tag(15); /* pointer */

  for (int i = 0; i < nargs; i++) {
    int tag_i = Int_val(Field(v_arg_tags, i));
    long data_i = Nativeint_val(Field(v_arg_data, i));
    raw_data[i + 2] = data_i;
    arg_values[i + 2] = &raw_data[i + 2];
    arg_types[i + 2] = ffi_type_of_tag(tag_i);
  }

  /* 3.  Prepare CIF */
  ffi_cif cif;
  ffi_type *rtype = ffi_type_of_tag(ret_tag);
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, nargs + 2, rtype, arg_types) !=
      FFI_OK)
    caml_failwith("msg_send: ffi_prep_cif failed");

  /* 4.  Call */
  long result_long = 0; /* big enough for all scalar results & pointers */
  ffi_call(&cif, FFI_FN(objc_msgSend), &result_long, arg_values);

  CAMLreturn(caml_copy_nativeint(result_long));
}