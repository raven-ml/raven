#define CAML_NAME_SPACE  // Recommended for C stubs
#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <objc/message.h>  // For objc_msgSend, etc.
#include <objc/runtime.h>
#include <stdint.h>  // For intptr_t
#include <string.h>  // For strlen, etc.

// --- Helper: Custom block for ObjC pointers ---
// We store the ObjC pointer directly in the custom block.
// A finalizer will call release on it.

static void finalize_objc_ptr(value v_ptr) {
  // id obj = (id)Nativeint_val(Field(v_ptr, 0)); // If storing as Nativeint
  id obj = (id)Data_custom_val(v_ptr);
  if (obj != NULL) {  // In C, nil is (id)0
    // Call release if you are doing manual reference counting from OCaml.
    // If your ObjC code is ARC and you expect ARC to handle it,
    // this finalizer might do nothing or just log.
    // The original library implies manual retain/release.
    // objc_release(obj); // Be careful: only if OCaml "owned" a retain count
  }
}

static struct custom_operations objc_ptr_ops = {
    "ocaml.objc_ptr",           finalize_objc_ptr,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

// Wrap a C id (ObjC object pointer) into an OCaml custom block
static value caml_alloc_objc_ptr(id obj) {
  if (obj == NULL) {  // Represent nil as a specific OCaml value or handle
                      // appropriately
    // For simplicity here, we could return Val_NULL from OCaml,
    // or a globally allocated custom block representing nil.
    // Let's assume the OCaml side will have a `Objc_core.nil` value.
    // This function should not be called with C nil if OCaml has its own nil.
    // Or, handle it:
    return Val_nativeint(0);  // Or some other convention for nil
  }
  value v = caml_alloc_custom(&objc_ptr_ops, sizeof(id), 0, 1);
  *((id *)Data_custom_val(v)) = obj;
  return v;
}

// Extract C id from OCaml value
// Add error checking: ensure v is a custom block of the right type.
static id get_objc_ptr(value v_ptr) {
  // If using Val_nativeint(0) for nil:
  if (v_ptr == Val_nativeint(0)) return NULL;
  return *((id *)Data_custom_val(v_ptr));
}
// Similar for Class, SEL, etc. They are often just typedefs for id or struct
// pointers.
#define get_objc_class(v) ((Class)get_objc_ptr(v))
#define get_objc_sel(v) ((SEL)get_objc_ptr(v))
#define get_objc_ivar(v) ((Ivar)get_objc_ptr(v))
#define get_objc_protocol(v) ((Protocol)get_objc_ptr(v))
#define get_objc_method(v) ((Method)get_objc_ptr(v))
#define get_imp(v) ((IMP)get_objc_ptr(v))  // IMPs are function pointers

// Store OCaml closures. A simple array for demonstration.
// A more robust solution would use a hash table or a resizable array.
#define MAX_CALLBACKS 128
static value ocaml_callbacks[MAX_CALLBACKS];
static int ocaml_callback_count = 0;

// --- Stubs for objc_core.mli functions ---

CAMLprim value caml_objc_getClass(value name_v) {
  CAMLparam1(name_v);
  CAMLlocal1(result_v);
  const char *name = String_val(name_v);
  Class cls = objc_getClass(name);
  result_v = caml_alloc_objc_ptr((id)cls);  // Cast Class to id for the wrapper
  CAMLreturn(result_v);
}

CAMLprim value caml_sel_registerName(value name_v) {
  CAMLparam1(name_v);
  CAMLlocal1(result_v);
  const char *name = String_val(name_v);
  SEL sel = sel_registerName(name);
  result_v = caml_alloc_objc_ptr((id)sel);  // Cast SEL to id for the wrapper
  CAMLreturn(result_v);
}

CAMLprim value caml_class_getName(value class_v) {
  CAMLparam1(class_v);
  Class cls = get_objc_class(class_v);
  const char *name = class_getName(cls);
  CAMLreturn(caml_copy_string(name ? name : ""));
}

// Example for objc_msgSend (simple version: id, SEL -> id)
CAMLprim value caml_objc_msgSend_id_sel(value receiver_v, value selector_v) {
  CAMLparam2(receiver_v, selector_v);
  CAMLlocal1(result_v);
  id receiver = get_objc_ptr(receiver_v);
  SEL selector = get_objc_sel(selector_v);

  // The actual objc_msgSend call. Needs careful casting for the function
  // pointer.
  id (*msg_send_func)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
  id result = msg_send_func(receiver, selector);

  result_v = caml_alloc_objc_ptr(result);
  CAMLreturn(result_v);
}

CAMLprim value caml_objc_msgSend_void_id_sel(value receiver_v,
                                             value selector_v) {
  CAMLparam2(receiver_v, selector_v);
  id receiver = get_objc_ptr(receiver_v);
  SEL selector = get_objc_sel(selector_v);
  void (*msg_send_func)(id, SEL) = (void (*)(id, SEL))objc_msgSend;
  msg_send_func(receiver, selector);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_objc_msgSend_id_sel_string(value receiver_v,
                                               value selector_v,
                                               value str_arg_v) {
  CAMLparam3(receiver_v, selector_v, str_arg_v);
  CAMLlocal1(result_v);
  id receiver = get_objc_ptr(receiver_v);
  SEL selector = get_objc_sel(selector_v);
  const char *str_arg = String_val(str_arg_v);  // Assuming arg is a C string

  id (*msg_send_func)(id, SEL, const char *) =
      (id (*)(id, SEL, const char *))objc_msgSend;
  id result = msg_send_func(receiver, selector, str_arg);

  result_v = caml_alloc_objc_ptr(result);
  CAMLreturn(result_v);
}

CAMLprim value caml_objc_retain(value obj_v) {
  CAMLparam1(obj_v);
  id obj = get_objc_ptr(obj_v);
  // In a non-ARC ObjC environment, or if you need to explicitly manage retain
  // counts for objects passed between OCaml and ObjC. If using ARC for ObjC
  // code, this might be bridged automatically. The original library implies
  // manual management. id result = objc_retain(obj); For this sketch, let's
  // assume retain does its job and returns the same pointer. If it can return a
  // different pointer (e.g. tagged pointers, though unlikely for retain), you'd
  // need to re-wrap: return caml_alloc_objc_ptr(result);
  if (obj) objc_retain(obj);  // objc_retain is typed as (id) -> id
  CAMLreturn(obj_v);          // Return the same OCaml value
}

CAMLprim value caml_objc_release(value obj_v) {
  CAMLparam1(obj_v);
  id obj = get_objc_ptr(obj_v);
  if (obj) objc_release(obj);
  CAMLreturn(Val_unit);
}

// ... Implement ALL other 'external' functions from objc_core.mli ...
// This is a lot of boilerplate.

// For class_addMethod, the IMP needs to be a C function pointer.
// If the implementation is an OCaml function, you need a C forwarder.

// C function that will be the actual IMP. It calls the OCaml closure.
// This is a generic forwarder. It assumes a certain signature for the OCaml
// callback. You'd need different forwarders for different OCaml function
// signatures. The `handle` (an int) would be an index into ocaml_callbacks.
static id ocaml_method_forwarder(id self, SEL _cmd /*, ... args */) {
  CAMLparam0();  // Must be called if any CAMLlocal is used or OCaml API is
                 // called.
  CAMLlocal3(res_v, self_v, cmd_v);  // Local roots for OCaml values

  // This is highly simplified. You need to know which OCaml function to call.
  // One way: the IMP pointer itself could be a struct containing the C func ptr
  // and the OCaml callback handle. Or, use a global lookup based on the C func
  // ptr. Let's assume for now the _cmd (or some other mechanism) helps find the
  // handle. This is a placeholder for a complex mechanism.
  int ocaml_callback_handle =
      0;  // FIXME: How to get this handle?
          // It might be associated with the IMP itself when registered.

  if (ocaml_callback_handle < 0 ||
      ocaml_callback_handle >= ocaml_callback_count) {
    caml_failwith("ocaml_method_forwarder: Invalid OCaml callback handle");
  }

  value ocaml_func = ocaml_callbacks[ocaml_callback_handle];

  self_v = caml_alloc_objc_ptr(self);
  cmd_v = caml_alloc_objc_ptr((id)_cmd);

  // This assumes the OCaml function takes (id -> sel -> id)
  // You'd need to marshall arguments and return types correctly.
  // This is where libffi would be useful if Ctypes isn't used.
  res_v = caml_callback2(ocaml_func, self_v, cmd_v);

  CAMLreturnT(id, get_objc_ptr(res_v));  // Return extracted id
}

CAMLprim value caml_class_addMethod_bc(value cls_v, value sel_v, value imp_v,
                                       value enc_v) {
  CAMLparam4(cls_v, sel_v, imp_v, enc_v);
  Class cls = get_objc_class(cls_v);
  SEL sel = get_objc_sel(sel_v);
  const char *enc = String_val(enc_v);

  // Here, imp_v is an OCaml value representing the IMP.
  // If it's an OCaml closure, imp_v would be a handle to it,
  // and we'd pass a C forwarder function pointer as the actual IMP.
  IMP imp_ptr;
  // This is where you'd look up or cast the 'imp_v'
  // If imp_v is an OCaml value representing a C function pointer (e.g., from
  // another C lib) imp_ptr = (IMP)Nativeint_val(imp_v); If imp_v is a handle to
  // an OCaml function to be wrapped: int callback_idx = Int_val(imp_v); //
  // Assuming imp_v is an int index imp_ptr = (IMP)ocaml_method_forwarder; //
  // But the forwarder needs to know which OCaml func! This part is tricky. The
  // `imp` in `Objc_core.mli` for `_class_addMethod` should probably be a C
  // function pointer type, possibly obtained via `_create_ocaml_imp_forwarder`.
  imp_ptr = get_imp(imp_v);  // Assuming imp_v directly holds the C function
                             // pointer via caml_alloc_objc_ptr

  BOOL success = class_addMethod(cls, sel, imp_ptr, enc);
  CAMLreturn(Val_bool(success));
}
CAMLprim value caml_class_addMethod(value cls_v, value sel_v, value imp_v,
                                    value enc_v) {
  return caml_class_addMethod_bc(cls_v, sel_v, imp_v, enc_v);
}

CAMLprim value caml_create_ocaml_imp_forwarder(value ocaml_method_imp_v) {
  CAMLparam1(ocaml_method_imp_v);
  // ocaml_method_imp_v is the OCaml closure. We need to store it and return a
  // C function pointer that knows how to call it.
  // This requires a unique C forwarder for each OCaml closure, or a generic
  // forwarder that can look up the closure.
  // For simplicity, let's assume we have a pool of pre-defined forwarders or
  // use a technique like trampolines if on an architecture that supports it
  // (harder). A common approach is to pass the ocaml_method_imp_v (the OCaml
  // 'value') as user data to a generic C forwarder, if the API allows
  // (class_addMethod doesn't directly). Another is to generate these forwarders
  // using libffi, or have a small number of them.

  // Storing the OCaml callback (MUST BE ROOTED!)
  if (ocaml_callback_count >= MAX_CALLBACKS) {
    caml_failwith("Max OCaml callbacks reached");
  }
  caml_register_global_root(&ocaml_callbacks[ocaml_callback_count]);
  ocaml_callbacks[ocaml_callback_count] = ocaml_method_imp_v;

  // This is a placeholder. The actual IMP needs to be a C function
  // that knows to call ocaml_callbacks[ocaml_callback_count].
  // This is the most complex part of replacing Ctypes for callbacks.
  // One way: the 'imp' type in OCaml for ocaml_method_imp is actually an int
  // index. The C forwarder (e.g. ocaml_method_forwarder) would need to be
  // modified to somehow receive this index. This usually means the IMP
  // registered is not 'ocaml_method_forwarder' directly, but a dynamically
  // generated thunk or one that retrieves the index from thread-local storage
  // or an associated object. For now, returning a dummy pointer that represents
  // this concept.
  IMP c_forwarder_ptr = (IMP)
      ocaml_method_forwarder;  // This is too simple.
                               // The forwarder needs to know WHICH callback.
                               // Let's imagine ocaml_method_imp_v carries this
                               // index for the C side. Or, the IMP itself
                               // stores this index for the forwarder.

  // For a truly dynamic system without Ctypes/libffi for callbacks, you might
  // end up writing small bits of assembly for trampolines or using a very
  // limited set of predefined forwarder signatures.

  ocaml_callback_count++;
  CAMLreturn(
      caml_alloc_objc_ptr((id)c_forwarder_ptr));  // Wrap the C function pointer
}

// Helper for getting the pointer value from Objc_core.id etc.
// This should not be exposed to typical OCaml users.
CAMLprim value caml_objc_ptr_to_nativeint(value ptr_v) {
  CAMLparam1(ptr_v);
  CAMLreturn(caml_copy_nativeint((intptr_t)get_objc_ptr(ptr_v)));
}
CAMLprim value caml_nativeint_to_objc_ptr(value ni_v) {
  CAMLparam1(ni_v);
  CAMLreturn(caml_alloc_objc_ptr((id)Nativeint_val(ni_v)));
}

// Ivar access:
CAMLprim value caml_ivar_getOffset(value ivar_v) {
  CAMLparam1(ivar_v);
  Ivar ivar = get_objc_ivar(ivar_v);
  ptrdiff_t offset = ivar_getOffset(ivar);
  CAMLreturn(caml_copy_nativeint((intptr_t)offset));
}

CAMLprim value caml_object_getInstanceVariable_ptr(value obj_v, value name_v) {
  CAMLparam2(obj_v, name_v);
  CAMLlocal1(pair_v);
  id obj = get_objc_ptr(obj_v);
  const char *name = String_val(name_v);
  void *ptr_to_ivar_val = NULL;  // To hold the address of the ivar's value
  Ivar ivar = object_getInstanceVariable(obj, name, &ptr_to_ivar_val);

  pair_v = caml_alloc_tuple(2);
  Store_field(pair_v, 0, caml_copy_nativeint((intptr_t)ptr_to_ivar_val));
  Store_field(pair_v, 1, caml_alloc_objc_ptr((id)ivar));
  CAMLreturn(pair_v);
}

CAMLprim value caml_ptr_read_id(value address_v) {
  CAMLparam1(address_v);
  id *ptr = (id *)Nativeint_val(address_v);
  if (!ptr) caml_failwith("caml_ptr_read_id: null address");
  // This assumes the id at *ptr is valid and we want to return it to OCaml.
  // If *ptr is an object, it might need to be retained if OCaml is to own a
  // reference. For now, just wrap it.
  CAMLreturn(caml_alloc_objc_ptr(*ptr));
}

CAMLprim value caml_ptr_write_id(value address_v, value id_to_write_v) {
  CAMLparam2(address_v, id_to_write_v);
  id *ptr = (id *)Nativeint_val(address_v);
  id val_to_write = get_objc_ptr(id_to_write_v);
  if (!ptr) caml_failwith("caml_ptr_write_id: null address");

  // This is a direct pointer write. If managing memory manually (e.g.
  // properties):
  // 1. Retain val_to_write (new value).
  // 2. Release the old value that was at *ptr.
  // 3. Update the pointer: *ptr = val_to_write.
  // This logic is usually part of setter implementations.
  // For this raw write, we just do the assignment. The caller must manage
  // retains/releases.
  *ptr = val_to_write;
  CAMLreturn(Val_unit);
}

// Initializer to get NSGlobalBlock class, etc.
// Called from OCaml at startup.
CAMLprim value caml_objc_core_init(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(tuple, v_nsstring, v_nsglobalblock);
  // In C, nil is (id)0. We can represent this as Nativeint 0 in OCaml for
  // Objc_core.nil
  Store_field(tuple, 0, Val_nativeint(0));  // nil

  // For well-known classes, get them once
  Class ns_object_cls = objc_getClass("NSObject");
  Class ns_string_cls = objc_getClass("NSString");
  Class ns_global_block_cls =
      objc_getClass("__NSGlobalBlock");  // Or NSBlock / _NSConcreteGlobalBlock

  // It's better to make these constants available from OCaml directly via
  // external or store them in OCaml global references after this init. For this
  // example, returning a tuple.
  tuple = caml_alloc_tuple(4);
  Store_field(tuple, 0, caml_alloc_objc_ptr((id)ns_object_cls));
  Store_field(tuple, 1, caml_alloc_objc_ptr((id)ns_string_cls));
  Store_field(tuple, 2, caml_alloc_objc_ptr((id)ns_global_block_cls));
  Store_field(tuple, 3, Val_nativeint(0));  // For nil representation

  CAMLreturn(tuple);
}