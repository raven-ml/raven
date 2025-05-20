// objc_stubs.c
// OCaml FFI C stubs. Calls the C API defined in objc_impl.h.

// OCaml FFI Headers
#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/config.h>  // For intnat type
#include <caml/custom.h>  // Will be needed for custom blocks for Id.t, Sel.t, Class.t
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "objc_impl.h"  // Our C API implemented in objc_impl.m

// Helper Macros: OCaml nativeint <-> C pointer (ObjC_Id_t, etc.)
// These remain the same as before, as OCaml side still uses nativeint.
#define Ocaml_nativeint_to_c_api_ptr(v_ml) ((void *)Nativeint_val(v_ml))
#define C_api_ptr_to_ocaml_nativeint(p_c) (caml_copy_nativeint((intnat)(p_c)))

// Stub Implementations for Core Runtime Functions (calling objc_impl layer)

CAMLprim value caml_objc_getClass(value ocaml_className) {
  CAMLparam1(ocaml_className);
  const char *c_className = String_val(ocaml_className);

  // Call the C API function from objc_impl.m
  ObjC_Class_t impl_class = objc_impl_getClass(c_className);

  CAMLreturn(C_api_ptr_to_ocaml_nativeint(impl_class));
}

CAMLprim value caml_sel_registerName(value ocaml_selectorName) {
  CAMLparam1(ocaml_selectorName);
  const char *c_selectorName = String_val(ocaml_selectorName);

  ObjC_Sel_t impl_selector = objc_impl_registerName(c_selectorName);

  CAMLreturn(C_api_ptr_to_ocaml_nativeint(impl_selector));
}

CAMLprim value caml_sel_getName(value ocaml_selector) {
  CAMLparam1(ocaml_selector);
  ObjC_Sel_t impl_sel =
      (ObjC_Sel_t)Ocaml_nativeint_to_c_api_ptr(ocaml_selector);

  // The objc_impl_getNameFromSelector handles NULL input for impl_sel if
  // needed, but an OCaml-side check (as in your runtime.ml) is still good. Here
  // we rely on the OCaml wrapper to check for null before calling. If we want
  // this C stub to be robust against OCaml sending null:
  if (impl_sel == NULL) {
    caml_failwith("caml_sel_getName (C stub): received NULL selector");
  }

  const char *c_name = objc_impl_getNameFromSelector(impl_sel);

  if (c_name == NULL ||
      c_name[0] == '\0') {  // Handle NULL or empty string from impl
    CAMLreturn(caml_copy_string(""));
  }
  CAMLreturn(caml_copy_string(c_name));
}

CAMLprim value caml_class_getName(value ocaml_class) {
  CAMLparam1(ocaml_class);
  ObjC_Class_t impl_class =
      (ObjC_Class_t)Ocaml_nativeint_to_c_api_ptr(ocaml_class);

  if (impl_class == NULL) {
    caml_failwith("caml_class_getName (C stub): received NULL class");
  }

  const char *c_name = objc_impl_getNameFromClass(impl_class);

  if (c_name == NULL || c_name[0] == '\0') {
    CAMLreturn(caml_copy_string(""));
  }
  CAMLreturn(caml_copy_string(c_name));
}

CAMLprim value caml_object_getClass(value ocaml_object) {
  CAMLparam1(ocaml_object);
  ObjC_Id_t impl_obj = (ObjC_Id_t)Ocaml_nativeint_to_c_api_ptr(ocaml_object);

  // objc_impl_getClassOfObject handles nil objects correctly.
  ObjC_Class_t impl_class = objc_impl_getClassOfObject(impl_obj);

  CAMLreturn(C_api_ptr_to_ocaml_nativeint(impl_class));
}
