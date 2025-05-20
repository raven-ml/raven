open Runtime_types

external caml_objc_getClass : string -> Class.t = "caml_objc_getClass"
external caml_sel_registerName : string -> Sel.t = "caml_sel_registerName"
external caml_sel_getName : Sel.t -> string = "caml_sel_getName"
external caml_class_getName : Class.t -> string = "caml_class_getName"
external caml_object_getClass : Id.t -> Class.t = "caml_object_getClass"

(* Retrieves an Objective-C Class object by its name. Returns Class.null if the
   class is not found (behavior of objc_getClass). *)
let get_class name =
  if name = "" then invalid_arg "get_class: class name cannot be empty";
  caml_objc_getClass name

(* Registers a selector name with the Objective-C runtime. *)
let register_selector name =
  if name = "" then
    invalid_arg "register_selector: selector name cannot be empty";
  caml_sel_registerName name

(* Retrieves the string name of a given selector. *)
let get_selector_name sel =
  if Sel.is_null sel then failwith "get_selector_name: selector is null";
  caml_sel_getName sel

(* Retrieves the string name of a given class. *)
let get_class_name cls =
  if Class.is_null cls then failwith "get_class_name: class is null";
  caml_class_getName cls

(* Retrieves the Class object of an Objective-C instance. *)
let get_object_class obj =
  (* In Objective-C, object_getClass(nil) returns Nil (a null Class pointer).
     The C stub 'caml_object_getClass' handles this by returning Class.null if
     the input 'obj' is Id.null, so no explicit Id.is_null check here. *)
  caml_object_getClass obj
