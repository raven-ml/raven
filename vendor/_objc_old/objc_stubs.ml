type untyped_id
and untyped_class
and untyped_selector
and untyped_method
and untyped_ivar
and untyped_protocol
and untyped_imp
and untyped_objc_super
and ocaml_closure_representation

(* This is a common trick to satisfy the type checker for abstract types when
   the actual definition comes from C. *)
module Objc_stubs_placeholder = struct
  type id_t
  type class_t
  type sel_t
  type method_t
  type ivar_t
  type protocol_t
  type imp_t
  type super_t
  type closure_rep_t
end

(* All 'val' in mli become 'external' here, pointing to C stub names *)
external initialize_raw :
  unit -> untyped_id * untyped_class * untyped_class * untyped_class
  = "caml_objc_ffi_initialize"

external objc_getClass : string -> untyped_class = "caml_objc_getClass"
external objc_getMetaClass : string -> untyped_class = "caml_objc_getMetaClass"
external objc_getProtocol : string -> untyped_protocol = "caml_objc_getProtocol"

external objc_registerClassPair : untyped_class -> unit
  = "caml_objc_registerClassPair"

external objc_allocateClassPair_bc :
  untyped_class -> string -> int -> untyped_class
  = "caml_objc_allocateClassPair_bc"

external objc_allocateClassPair_nat :
  untyped_class -> string -> int -> untyped_class
  = "caml_objc_allocateClassPair"

let objc_allocateClassPair ~superclass ~name ~extra_bytes =
  objc_allocateClassPair_nat superclass name
    extra_bytes (* or _bc depending on OCaml version/convention *)

external class_getName : untyped_class -> string = "caml_class_getName"

external class_getSuperclass : untyped_class -> untyped_class
  = "caml_class_getSuperclass"

external class_addMethod_bc :
  untyped_class -> untyped_selector -> untyped_imp -> string -> bool
  = "caml_class_addMethod_bc"

external class_addMethod_nat :
  untyped_class -> untyped_selector -> untyped_imp -> string -> bool
  = "caml_class_addMethod"

let class_addMethod cls sel imp enc = class_addMethod_nat cls sel imp enc

external class_addIvar_bc :
  untyped_class -> string -> int -> int -> string -> bool
  = "caml_class_addIvar_bc"

external class_addIvar_nat :
  untyped_class -> string -> int -> int -> string -> bool = "caml_class_addIvar"

let class_addIvar cls name size align enc =
  class_addIvar_nat cls name size align enc

external class_getInstanceVariable : untyped_class -> string -> untyped_ivar
  = "caml_class_getInstanceVariable"

external class_getMethodImplementation :
  untyped_class -> untyped_selector -> untyped_imp
  = "caml_class_getMethodImplementation"

external class_conformsToProtocol_bc : untyped_class -> untyped_protocol -> bool
  = "caml_class_conformsToProtocol_bc"

external class_conformsToProtocol_nat :
  untyped_class -> untyped_protocol -> bool = "caml_class_conformsToProtocol"

let class_conformsToProtocol cls proto = class_conformsToProtocol_nat cls proto

external class_createInstance_bc : untyped_class -> int -> untyped_id
  = "caml_class_createInstance_bc"

external class_createInstance_nat : untyped_class -> int -> untyped_id
  = "caml_class_createInstance"

let class_createInstance cls extra_bytes =
  class_createInstance_nat cls extra_bytes

external object_getClass : untyped_id -> untyped_class = "caml_object_getClass"

external object_getIvar : untyped_id -> untyped_ivar -> untyped_id
  = "caml_object_getIvar"

external object_setIvar : untyped_id -> untyped_ivar -> untyped_id -> unit
  = "caml_object_setIvar"

external object_getInstanceVariable_ptr :
  untyped_id -> string -> nativeint * untyped_ivar
  = "caml_object_getInstanceVariable_ptr"

external sel_registerName : string -> untyped_selector = "caml_sel_registerName"
external sel_getName : untyped_selector -> string = "caml_sel_getName"

external sel_isEqual : untyped_selector -> untyped_selector -> bool
  = "caml_sel_isEqual"

external ivar_getOffset : untyped_ivar -> nativeint = "caml_ivar_getOffset"
external ivar_getName : untyped_ivar -> string = "caml_ivar_getName"

external ivar_getTypeEncoding : untyped_ivar -> string
  = "caml_ivar_getTypeEncoding"

external method_getName : untyped_method -> untyped_selector
  = "caml_method_getName"

external method_getTypeEncoding : untyped_method -> string
  = "caml_method_getTypeEncoding"

external method_getImplementation : untyped_method -> untyped_imp
  = "caml_method_getImplementation"

external protocol_getName : untyped_protocol -> string = "caml_protocol_getName"
external retain : untyped_id -> untyped_id = "caml_objc_retain"
external release : untyped_id -> unit = "caml_objc_release"
external autorelease : untyped_id -> untyped_id = "caml_objc_autorelease"

external msgSend_id : untyped_id -> untyped_selector -> untyped_id
  = "caml_msgSend_id"

external msgSend_void : untyped_id -> untyped_selector -> unit
  = "caml_msgSend_void"

external msgSend_id_id :
  untyped_id -> untyped_selector -> untyped_id -> untyped_id
  = "caml_msgSend_id_id"

external msgSend_id_bool_bc :
  untyped_id -> untyped_selector -> bool -> untyped_id
  = "caml_msgSend_id_bool_bc"

external msgSend_id_bool_nat :
  untyped_id -> untyped_selector -> bool -> untyped_id = "caml_msgSend_id_bool"

let msgSend_id_bool obj sel b = msgSend_id_bool_nat obj sel b

external msgSend_void_bool_bc : untyped_id -> untyped_selector -> bool -> unit
  = "caml_msgSend_void_bool_bc"

external msgSend_void_bool_nat : untyped_id -> untyped_selector -> bool -> unit
  = "caml_msgSend_void_bool"

let msgSend_void_bool obj sel b = msgSend_void_bool_nat obj sel b

external msgSend_id_string :
  untyped_id -> untyped_selector -> string -> untyped_id
  = "caml_msgSend_id_string"

external msgSend_string : untyped_id -> untyped_selector -> string
  = "caml_msgSend_string"

external msgSend_bool_class_bc :
  untyped_id -> untyped_selector -> untyped_class -> bool
  = "caml_msgSend_bool_class_bc"

external msgSend_bool_class_nat :
  untyped_id -> untyped_selector -> untyped_class -> bool
  = "caml_msgSend_bool_class"

let msgSend_bool_class obj sel cls = msgSend_bool_class_nat obj sel cls

external objc_super_construct_bc :
  untyped_id -> untyped_class -> untyped_objc_super
  = "caml_objc_super_construct_bc"

external objc_super_construct_nat :
  untyped_id -> untyped_class -> untyped_objc_super
  = "caml_objc_super_construct"

let objc_super_construct recv sup_cls = objc_super_construct_nat recv sup_cls

external msgSendSuper_id : untyped_objc_super -> untyped_selector -> untyped_id
  = "caml_msgSendSuper_id"

external ptr_read_int : nativeint -> int
  = "%nativeint_load_int" (* Example using intrinsic *)

external ptr_write_int : nativeint -> int -> unit
  = "%nativeint_store_int" (* Example using intrinsic *)

external ptr_read_id : nativeint -> untyped_id = "caml_ptr_read_id"
external ptr_write_id : nativeint -> untyped_id -> unit = "caml_ptr_write_id"

external create_ocaml_imp_forwarder_raw :
  ocaml_closure_representation ->
  int ->
  untyped_imp * ocaml_closure_representation
  = "caml_create_ocaml_imp_forwarder_bc" "caml_create_ocaml_imp_forwarder"

let create_ocaml_imp_forwarder ocaml_func arity_id =
  create_ocaml_imp_forwarder_raw
    (ocaml_func : ocaml_closure_representation)
    arity_id

external unsafe_ptr_to_nativeint : untyped_id -> nativeint
  = "caml_unsafe_ptr_to_nativeint"

external unsafe_nativeint_to_ptr : nativeint -> untyped_id
  = "caml_unsafe_nativeint_to_ptr"

let is_raw_nil ptr = unsafe_ptr_to_nativeint ptr

(* Globals initialized once *)
let ( _nil_id_raw,
      _nsobject_class_raw,
      _nsstring_class_raw,
      _nsglobalblock_class_raw ) =
  initialize_raw ()

let initialize () = ()
(* Expose a unit function if explicit init is desired by user, otherwise it's
   internal *)
