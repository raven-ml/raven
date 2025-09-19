(* In your toplevel module (e.g., Objc.ml) *)

(* Core modules from other parts of the library or generated files *)
module Platform = Platform
module Inspect = Inspect
module Objc_type = Objc_type
module Define = Define
module C = Objc_c (* Assuming this is your Ctypes bindings root *)

(* Re-export Ctypes types for convenience *)
type objc_object = C.Types.objc_object
type objc_class = C.Types.objc_class
type class_t = C.Types.class_t
type object_t = C.Types.object_t
type imp_t = C.Types.imp_t
type selector_t = C.Types.selector_t
type protocol_t = C.Types.protocol_t
type ivar_t = C.Types.ivar_t
type _Enc = C.Types._Enc (* If this is a distinct type alias *)

(* Ctypes.typ values *)
let id = C.Types.id
let _Class = C.Types._Class
let _SEL = C.Types._SEL
let _IMP = C.Types._IMP

(* let _Enc = C.Types._Enc (* If this is also a Ctypes.typ, careful with name
   shadowing if different from the type alias above *) *)
let _Protocol = C.Types._Protocol
let _Ivar = C.Types._Ivar

(* These seem redundant if you already have id, _Class above, but harmless *)
(* let objc_object = C.Types.objc_object *)
(* let objc_class = C.Types.objc_class *)

(* Re-export Objc runtime functions from C.Functions.Objc *)
let get_class = C.Functions.Objc.get_class
let get_meta_class = C.Functions.Objc.get_meta_class
let get_protocol = C.Functions.Objc.get_protocol
let register_class = C.Functions.Objc.register_class
let get_class_list = C.Functions.Objc.get_class_list
let get_protocol_list = C.Functions.Objc.get_protocol_list
let allocate_class = C.Functions.Objc.allocate_class

(* Re-export from the Runtime module *)

(* Basic utilities & common operations *)
let selector = Runtime.selector
let string_of_selector = Runtime.string_of_selector
let to_selector = Runtime.to_selector
let nil = Runtime.nil
let is_nil = Runtime.is_nil

(* Object lifecycle *)
let alloc = Runtime.alloc
let new_ = Runtime._new_
let init = Runtime.init
let copy = Runtime._copy_
let retain = Runtime.retain
let release = Runtime.release
let autorelease = Runtime.autorelease
let gc_autorelease = Runtime.gc_autorelease
let alloc_object = Runtime.alloc_object
let new_object = Runtime.new_object

(* String conversion *)
let new_string = Runtime.new_string
let to_string = Runtime.to_string

(** {1 High-Level Message Sending}
    These functions use [Objc_type.t] for type safety and abstraction over
    Objective-C method type encodings. This is the recommended way to send
    messages for most use cases. *)

let msg_send = Runtime.msg_send
let msg_super = Runtime.msg_super

(** {1 Low-Level Message Sending}
    These functions provide more direct access to the underlying [objc_msgSend]
    family of functions. They require manual specification of [Ctypes.typ] for
    arguments and return values. Use these when the high-level functions are not
    suitable, or for performance-critical sections where [Objc_type] overhead is
    a concern, or when dealing with very complex/unusual method signatures not
    easily representable by [Objc_type]. *)
module Low_level = struct
  (** Sends a message with a user-specified Ctypes signature. Corresponds to
      [objc_msgSend]. *)
  let send_message = Runtime.Raw.send

  (** Sends a message, potentially releasing the OCaml runtime lock. Corresponds
      to [objc_msgSend]. *)
  let send_message_gilesc = Runtime.Raw.send_gilesc

  (** Sends a message to the superclass of an instance. Corresponds to
      [objc_msgSendSuper]. *)
  let send_super_message = Runtime.Raw.send_super

  (** Sends a message with a struct return value, handling architecture-specific
      ABI. Corresponds to [objc_msgSend_stret] or [objc_msgSend] on
      architectures where they are equivalent for certain struct sizes. *)
  let send_message_stret = Runtime.Raw.send_stret

  (** Shortcut for [send_message] with signature [id -> SEL -> returning id]. *)
  let send_message_returning_id = Runtime.Raw.send_returning_id

  (** Shortcut for [send_message] with signature
      [id -> SEL -> id -> returning void]. *)
  let send_message_with_id_arg = Runtime.Raw.send_with_id_arg

  (* You could also re-export _SEL and id here if they are often needed when
     constructing the `typ` argument for the functions above, though they are
     already in the toplevel scope. let _SEL = _SEL let id = id *)
end

(* High-level ivar access *)
let get_ivar = Runtime.get_ivar
let set_ivar = Runtime.set_ivar

(* Property access *)
let get_property = Runtime.get_property
let set_property = Runtime.set_property

(* Re-export modules from Runtime *)
module Sel = Runtime.Sel
module Object = Runtime.Object
module Class = Runtime.Class
module Property = Runtime.Property
module Bitmask = Runtime.Bitmask
module Block_descriptor = Runtime.Block_descriptor
module Block = Runtime.Block
module Method = Runtime.Method
module Ivar = Runtime.Ivar (* This is the one with define *)

(* Exception handling *)
exception CamlNSException = Runtime.CamlNSException

let set_uncaught_exception_handler = Runtime.set_uncaught_exception_handler

let default_uncaught_exception_handler =
  Runtime.default_uncaught_exception_handler

let setup_default_uncaught_exception_handler =
  Runtime.setup_default_uncaught_exception_handler

(* Setup exception handler by default when this module is initialized *)
let () = Runtime.setup_default_uncaught_exception_handler ()
