(** Top-level module for Objective-C interoperability. *)

(** {1 Core Types} *)

(** Opaque type representing any Objective-C object instance. *)
type id

(** Opaque type representing an Objective-C class. *)
type 'a class_t constraint 'a = [< `Object] (* 'a is the instance type *)

(** Opaque type representing an Objective-C selector (method name). *)
type selector

(** Opaque type representing an Objective-C method implementation (IMP). *)
type imp

(** Opaque type representing an Objective-C protocol. *)
type protocol

(** Opaque type representing an Objective-C instance variable (Ivar). *)
type ivar

(** Opaque type representing an Objective-C method description. *)
type method_desc

(** The null Objective-C object pointer. *)
val nil : id

(** Checks if an Objective-C object pointer is null. *)
val is_nil : id -> bool

(** Represents an Objective-C type for use in method signatures and encodings. *)
module Type : sig
  type _ t =
    | Id : id t
    | Class : 'a class_t t
    | Sel : selector t
    | Imp : imp t
    | Void : unit t
    | Bool : bool t
    | Char : char t
    | Int : int t
    | Int32 : int32 t
    | Int64 : int64 t
    | Nativeint : nativeint t
    | Float : float t
    | String : string t (* C string, char* *)
    | Pointer : 'a t -> 'a option t (* C pointer to a type, optional for nil *)
    | Block : id t (* Represents a generic block pointer *)
    | Struct : (* Details for struct handling would be complex here without Ctypes *)
        string (* name *) ->
        (unit -> 'a) (* constructor *) ->
        ('a -> Runtime_ffi.untyped_id) (* to_generic_pointer *) ->
        (Runtime_ffi.untyped_id -> 'a) (* from_generic_pointer *) ->
        'a t
    (* Add more specific types as needed, e.g., for common C structs if handled. *)

  type (_, _) tlist =
    | [] : ('r, 'r) tlist
    | (::) : 'a t * ('b, 'r) tlist -> ('a -> 'b, 'r) tlist

  val id : id t
  val class_ : 'a class_t t
  val selector : selector t
  val imp : imp t
  val void : unit t
  val bool : bool t
  val int : int t
  val string_ptr : string t (* char* *)
  (* ... other convenience constructors ... *)

  val encode_value : 'a t -> string
  val encode_method : args:(_, _) tlist -> return:'r t -> string
end


(** {1 Runtime Operations} *)

(** Obtain a class by its name. Returns [None] if not found. *)
val get_class : string -> 'a class_t option

(** Get the name of a class. *)
val class_name : 'a class_t -> string

(** Get the superclass of a class. Returns [None] for a root class. *)
val superclass : 'a class_t -> 'b class_t option

(** Register a selector with the runtime. *)
val selector : string -> selector

(** Get the name of a selector. *)
val selector_name : selector -> string


(** {1 Object Lifecycle and Interaction} *)

(** Allocate an instance of a class (sends `alloc`). Does not initialize. *)
val alloc : 'a class_t -> id

(** Initialize an object (sends `init`).
    Assumes the object is freshly allocated or needs reinitialization. *)
val init : id -> id

(** A convenience function to create a new object: `alloc` then `init`.
    The resulting object's memory will be managed by OCaml's GC via a finalizer
    that calls `release`.
*)
val new_object : class_name:string -> id

(** Retain an Objective-C object (increments its reference count). *)
val retain : id -> id

(** Release an Objective-C object (decrements its reference count). *)
val release : id -> unit

(** Autorelease an Objective-C object. *)
val autorelease : id -> id

(** Associates a finalizer with the OCaml [id] value that calls [release]
    on the Objective-C object when the OCaml value is garbage collected.
    Returns the original [id]. Idempotent. *)
val manage_with_gc : id -> id

(** Get the class of an object. *)
val object_class : id -> 'a class_t


(** {1 Message Sending} *)
(* This is the most challenging part without Ctypes/libffi for general cases.
   We provide a few common, typed versions. *)

(** Send a message to an object that returns an [id].
    Example: `[object description]` *)
val perform_selector_returning_id : target:id -> cmd:selector -> id

(** Send a message to an object with one [id] argument, returning an [id].
    Example: `[array objectAtIndex:index_obj]` *)
val perform_selector_id_arg_returning_id : target:id -> cmd:selector -> arg1:id -> id

(** Send a message to an object with one [bool] argument, returning [void].
    Example: `[object setHidden:YES]` *)
val perform_selector_bool_arg_returning_void : target:id -> cmd:selector -> arg1:bool -> unit

(** Send a message to an object with one C string argument, returning an [id].
    Example: `[NSString stringWithUTF8String:"hello"]` (if called on NSString class) *)
val perform_selector_string_arg_returning_id : target:id -> cmd:selector -> arg1:string -> id

(** Send a message to an object that returns a C string.
    The returned string is copied into an OCaml string.
    Example: `[object aCStringReturningMethod]` *)
val perform_selector_returning_string : target:id -> cmd:selector -> string

(** Send a message to the superclass of an object.
    This requires constructing an `objc_super` struct internally. *)
module Super : sig
  (** Send a message to the superclass of an object that returns an [id]. *)
  val perform_selector_returning_id : target:id -> cmd:selector -> id
  (* ... other common signatures for super calls ... *)
end

(** A more general, but less type-safe, mechanism for sending messages.
    Use with extreme caution. The caller is responsible for ensuring the types match.
    The `imp` is usually obtained via `class_getMethodImplementation`.
    The OCaml function type `('target_t -> 'sel_t -> 'args -> 'ret)` must exactly match
    the C function pointer signature of the IMP. *)
val invoke_method_implementation :
  imp:imp ->
  fn_typ:('target_t -> 'sel_t -> 'args -> 'ret) Type.fn_signature (* A way to describe fn signature for safety? *) ->
  target:id ->
  selector:selector -> (* Passed as the _cmd argument *)
  (* How to pass variadic arguments here without Ctypes?
     Maybe a fixed number of Obj.t and the C stub sorts it out based on fn_typ?
     Or restrict to a set of pre-defined arities/types.
     For this sketch, let's assume specific arity functions or a highly unsafe variant.
  *)
  'a (* This needs to be more concrete. Perhaps return Obj.t and require Obj.magic *)

(* Alternative for invoke_method_implementation:
   Provide stubs for common arities if libffi is not used in C stubs:
   val invoke_imp_0_args_ret_id : imp -> id -> selector -> id
   val invoke_imp_1_arg_id_ret_id : imp -> id -> selector -> id -> id
   ... etc.
*)


(** {1 String Conversion} *)

(** Create a new NSString object from an OCaml string.
    The resulting NSString is managed by OCaml's GC. *)
val new_string : string -> id

(** Convert an Objective-C object (preferably an NSString) to an OCaml string.
    If the object is not an NSString, its `description` method is called first.
    Returns an empty string if the object is `nil` or conversion fails. *)
val string_of_object : id -> string


(** {1 Class Definition} *)

module Property : sig
  type attribute =
    | Readonly
    | Readwrite (* Default *)
    | Retain
    | Copy
    | Assign (* Default for non-object types, non-default for object types *)
    | Nonatomic
    | Atomic (* Default *)
    | Setter of selector
    | Getter of selector
    (* | Dynamic *) (* Implies @dynamic, user provides methods *)

  type 'a t = {
    name : string;
    typ : 'a Type.t;
    attributes : attribute list;
  }

  val make :
    name:string ->
    typ:'a Type.t ->
    ?attributes:attribute list ->
    unit -> 'a t
end

module Ivar : sig
  type 'a t = {
    name : string;
    typ : 'a Type.t;
  }
  val make : name:string -> typ:'a Type.t -> 'a t
end

module Method : sig
  (** Represents an OCaml function intended as a method implementation. *)
  type ocaml_implementation =
    | Meth0_ret_id of (id -> selector -> id)
    | Meth0_ret_void of (id -> selector -> unit)
    | Meth1_id_ret_id of (id -> selector -> id -> id)
    | Meth1_id_ret_void of (id -> selector -> id -> unit)
    (* Add more variants for common signatures.
       The types (id, selector) are fixed as the first two arguments.
       'a Type.t and 'b Type.t, etc., would describe the subsequent args/return. *)

  type t = {
    selector : selector;
    encoding : string; (* Or derived from types *)
    implementation : ocaml_implementation; (* This will be translated to an imp *)
  }

  (** Define a method specification.
      The OCaml function must match the arguments implied by the selector
      and the type encoding. This is a point of potential runtime errors if mismatched.
  *)
  val make :
    selector:selector ->
    (* Instead of raw encoding, better to take Type.t list for args and return *)
    argument_types: (id Type.t -> selector Type.t -> ('a, 'b) Type.tlist) ->
    return_type: 'b Type.t ->
    implementation: (* OCaml function matching 'a -> 'b after self and cmd *)
      (id -> selector -> (* 'a args mapped here *) 'b) ->
    t
end

(** Defines a new class and registers it with the Objective-C runtime. *)
val define_class :
  name:string ->
  ?superclass:'a class_t ->
  ?ivars:'b Ivar.t list ->
  ?properties:'c Property.t list ->
  ?instance_methods:Method.t list ->
  ?class_methods:Method.t list ->
  ?protocols:protocol list ->
  unit -> 'd class_t


(** {1 Instance Variable (Ivar) Access} *)
(* These are low-level and potentially unsafe if types mismatch. *)

(** Get the value of an instance variable. *)
val get_ivar : owner:id -> name:string -> typ:'a Type.t -> 'a

(** Set the value of an instance variable. *)
val set_ivar : owner:id -> name:string -> typ:'a Type.t -> value:'a -> unit


(** {1 Property Access} *)
(* Assumes properties are KVC-compliant or have standard getters/setters. *)

(** Get the value of a property.
    Sends the standard getter message (e.g., `[owner propertyName]`). *)
val get_property : owner:id -> name:string -> typ:'a Type.t -> 'a

(** Set the value of a property.
    Sends the standard setter message (e.g., `[owner setPropertyName:value]`). *)
val set_property : owner:id -> name:string -> typ:'a Type.t -> value:'a -> unit


(** {1 Blocks} *)

(** Create an Objective-C block from an OCaml function.
    The block and the OCaml closure are managed by OCaml's GC.
    The signature of the OCaml function must match the expected block signature.
    This is another area where C FFI for callbacks is complex.
*)
val create_block :
  (* Example: a block taking an int and returning void *)
  fn:(int -> unit) ->
  arg_types:(('a, 'b) Type.tlist) -> (* Describes args after the implicit block self *)
  return_type:('b Type.t) ->
  id (* Returns the block as an id *)


(** {1 Exception Handling} *)

(** Represents an Objective-C exception caught by the OCaml bridge. *)
exception NSException of { name : string; reason : string; user_info : id option }

(** Executes a function [f] and translates any Objective-C exceptions
    thrown during its execution into the [NSException] OCaml exception. *)
val try_catch_objc_exceptions : (unit -> 'a) -> 'a


(** {1 Platform and Architecture} *)

module Platform : sig
  type t = MacOS | IOS | Catalyst | GNUStep
  val current : t
end

module Arch : sig
  type t = Amd64 | Arm64
  val current : t
end