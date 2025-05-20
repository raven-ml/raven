open C.Types
open Ctypes
open Foreign
open Unsigned
module Platform = Platform
module Protocol = C.Functions.Protocol
module Inspect = Inspect
module Objc_type = Objc_type
module Define = Define

module Sel = struct
  include C.Functions.Sel

  let register_typed_name =
    match Platform.current with
    | GNUStep ->
        foreign "sel_registerTypedName_np" (string @-> _Enc @-> returning _SEL)
    | _ -> fun _ _ -> assert false
end

module Object = struct
  include C.Functions.Object

  (** Reads the value of an Id instance variable in an object. *)
  let get_ivar ~self ~ivar =
    foreign "object_getIvar" (id @-> _Ivar @-> returning id) self ivar

  (** Sets the value of an instance variable in an object. object_setIvar is
      faster than object_setInstanceVariable if the Ivar for the instance
      variable is already known. *)
  let set_ivar ~self ~ivar value =
    foreign "object_setIvar"
      (id @-> _Ivar @-> id @-> returning void)
      self ivar value

  (** Obtains the value of an instance variable of a class instance. *)
  let get_instance_variable ~self ~name ~value_ptr ~typ =
    foreign "object_getInstanceVariable"
      (id @-> string @-> ptr typ @-> returning _Ivar)
      self name value_ptr

  (** Changes the value of an instance variable of a class instance. *)
  let set_instance_variable ~self ~name ~value =
    foreign "object_setInstanceVariable"
      (id @-> string @-> ptr void @-> returning _Ivar)
      self name value

  (** Returns pointer to an ivar in object [self] *)
  let ivar_ptr ~self ~ivar_name =
    C.Functions.Class.get_instance_variable (get_class self) ivar_name
    |> C.Functions.Ivar.get_offset |> Ptrdiff.to_nativeint
    |> Nativeint.add (self |> to_voidp |> raw_address_of_ptr)
    |> ptr_of_raw_address

  (** Returns the ivar name capitalized, prefixed by "set", suffixed by ":" *)
  let setter_name_of_ivar ivar_name =
    "set" ^ String.capitalize_ascii ivar_name ^ ":"
end

(** Registers a method with the Objective-C runtime system, maps the method name
    to a selector, and returns the selector value. If the method name has
    already been registered, this function simply returns the selector. *)
let selector = Sel.register_name

(** Returns the selector name as string. *)
let string_of_selector = Sel.get_name

let to_selector p = to_voidp p |> coerce (ptr void) _SEL

module Objc = struct
  include C.Functions.Objc
  include C.Types
  include Ctypes
  include Unsigned
  include Signed

  (** This module extends Ctypes with types and functions specific to
      Objective-C. *)

  module Objc_super = struct
    type t

    let t : t structure typ = structure "objc_super"
    let receiver = field t "receiver" id
    let super_class = field t "super_class" _Class
    let () = seal t

    let make self =
      let self_class = Object.get_class self in
      let sup_cls = C.Functions.Class.get_superclass self_class
      and d = make t in
      setf d receiver self;
      (if C.Functions.Class.is_meta_class self_class then
         Object.get_class sup_cls
       else sup_cls)
      |> setf d super_class;
      allocate t d
  end

  (** Sends a message with a simple return value to an instance of a class. *)
  let msg_send ~self ~cmd ~typ =
    foreign "objc_msgSend" (id @-> _SEL @-> typ) self cmd

  (** Sends a message with a simple return value to an instance of a class. *)
  let msg_send' ?(release_runtime_lock = false) ~self ~typ cmd =
    foreign ~release_runtime_lock "objc_msgSend"
      (id @-> _SEL @-> typ)
      self (selector cmd)

  (** Sends a message with a simple return value to the superclass of an
      instance of a class. *)
  let msg_send_super ~self ~cmd ~typ =
    match Platform.current with
    | GNUStep ->
        let self_class = Object.get_class self in
        let imp =
          C.Functions.Class.get_method_implementation
            (C.Functions.Class.get_superclass self_class)
            cmd
        in
        let imp_fun = coerce _IMP (funptr (id @-> _SEL @-> typ)) imp in
        imp_fun self cmd
    | _ ->
        let objc_super = Objc_super.make self in
        foreign "objc_msgSendSuper"
          (ptr Objc_super.t @-> _SEL @-> typ)
          objc_super cmd

  (** Shortcut for type [void @-> id] *)
  let msg_send_vo = msg_send ~typ:(returning id)

  (** Shortcut for type [id @-> void] *)
  let msg_send_ov = msg_send ~typ:(id @-> returning void)

  (** Sends a message with a data-structure return value to an instance of a
      class. *)
  let msg_send_stret ~self ~cmd ~typ ~return_type =
    match Arch.current with
    | Amd64 -> (
        match sizeof return_type with
        | 2 | 4 | 8 | 16 -> msg_send ~self ~cmd ~typ
        | _ -> foreign "objc_msgSend_stret" (id @-> _SEL @-> typ) self cmd)
    | Arm64 -> msg_send ~self ~cmd ~typ

  (** Returns a pointer to the C value named by [name] or [null] if the symbol
      cannot be resolved. *)
  let foreign_value_or_null name ty =
    try foreign_value name ty with _ -> from_voidp ty null
end

let nsstring_class = Objc.get_class "NSString"
let nil = coerce (ptr void) Objc.id null
let is_nil = is_null

(** Returns a new instance of the receiving class. *)
let alloc self = Objc.msg_send_vo ~self ~cmd:(selector "alloc")

(** Allocates a new instance of the receiving class, sends it an init message,
    and returns the initialized object. *)
let _new_ self = Objc.msg_send_vo ~self ~cmd:(selector "new")

(** Implemented by subclasses to initialize a new object (the receiver)
    immediately after memory for it has been allocated. *)
let init self = Objc.msg_send_vo ~self ~cmd:(selector "init")

(** Returns the object returned by copyWithZone: *)
let _copy_ self = Objc.msg_send_vo ~self ~cmd:(selector "copy")

(** Increments the receiver’s reference count. *)
let retain self = Objc.msg_send_vo ~self ~cmd:(selector "retain")

(** Decrements the receiver’s reference count. *)
let release self =
  Objc.msg_send ~self ~cmd:(selector "release") ~typ:(returning void)

(** Decrements the receiver’s retain count at the end of the current autorelease
    pool block. *)
let autorelease self =
  Objc.msg_send ~self ~cmd:(selector "autorelease") ~typ:(returning void)

(** Release ObjC object when OCaml ptr is garbage collected. *)
let gc_autorelease self =
  Gc.finalise release self;
  self

(** Allocates an object given a class name. *)
let alloc_object class_name = alloc (Objc.get_class class_name)

(** Allocates an object and sends it [init] and [gc_autorelease]. *)
let new_object class_name = alloc_object class_name |> init |> gc_autorelease

(** Creates a new NSString object autoreleased by OCaml's GC. *)
let new_string str =
  Objc.msg_send ~self:nsstring_class
    ~cmd:(selector "stringWithUTF8String:")
    ~typ:(string @-> returning id)
    str
  |> gc_autorelease

(** Converts a NSString instance to an OCaml string. If the instance is not an
    NSString, it will be sent the message [NSObject.description] first. Prefer
    this to [NSString._UTF8String] as sending "UTF8String" to nil will crash
    your program. *)
let to_string self =
  let conv instance =
    Objc.msg_send ~self:instance ~cmd:(selector "UTF8String")
      ~typ:(returning string)
  in
  if is_nil self then ""
  else if
    Objc.msg_send ~self
      ~cmd:(selector "isKindOfClass:")
      ~typ:(_Class @-> returning bool)
      nsstring_class
  then conv self
  else
    conv (Objc.msg_send ~self ~cmd:(selector "description") ~typ:(returning id))

(** Sends a message with a simple return value to an instance of a class. *)
let msg_send cmd ~self ~args ~return =
  let typ = Objc_type.method_typ ~args return in
  Objc.msg_send ~self ~cmd ~typ

(** Sends a message with a simple return value to the superclass of an instance
    of a class. *)
let msg_super cmd ~self ~args ~return =
  let typ = Objc_type.method_typ ~args return in
  Objc.msg_send_super ~self ~cmd ~typ

(** Returns the value of an ivar reading it directly. *)
let get_ivar : type a. string -> a Objc_type.t -> object_t -> a =
 fun ivar_name t self ->
  match t with
  | Objc_type.Id ->
      let ivar =
        C.Functions.Class.get_instance_variable (Object.get_class self)
          ivar_name
      in
      Object.get_ivar ~self ~ivar
  | _ ->
      let typ = Objc_type.value_typ t in
      !@(Object.ivar_ptr ~self ~ivar_name |> coerce (ptr void) (ptr typ))

(** Sets the value of an ivar writing it directly. *)
let set_ivar : type a. string -> a -> a Objc_type.t -> object_t -> unit =
 fun ivar_name value t self ->
  match t with
  | Objc_type.Id ->
      let ivar =
        C.Functions.Class.get_instance_variable (Object.get_class self)
          ivar_name
      and typ = Objc_type.value_typ t in
      Object.set_ivar ~self ~ivar (coerce typ id value)
  | _ ->
      let typ = Objc_type.value_typ t in
      Object.ivar_ptr ~self ~ivar_name |> from_voidp typ <-@ value

let get_property prop_name typ self =
  Objc.(msg_send ~self ~cmd:(selector prop_name) ~typ:(returning typ))

let set_property prop_name value typ self =
  let cmd = selector (Object.setter_name_of_ivar prop_name) in
  Objc.(msg_send ~self ~cmd ~typ:(typ @-> returning void)) value

module Property = struct
  open Objc
  open Object
  open Define

  (** Get the value of a property. *)
  let get prop_name typ = get_property prop_name (Objc_type.value_typ typ)

  (** Set the value of a property. *)
  let set prop_name value typ =
    set_property prop_name value (Objc_type.value_typ typ)

  (** Getter for non-object values. *)
  let getter ~typ ~enc ivar_name =
    let cmd = selector ivar_name
    and imp self _cmd = !@(ivar_ptr ~self ~ivar_name |> from_voidp typ) in
    method_spec ~cmd ~typ:(returning typ) ~enc imp

  (** Setter for non-object values. *)
  let setter ?(notify_change = false) ~typ ~enc ivar_name =
    let cmd = selector (setter_name_of_ivar ivar_name)
    and key = new_string ivar_name in
    let imp self _cmd value =
      if notify_change then
        msg_send_ov ~self ~cmd:(selector "willChangeValueForKey:") key;

      ivar_ptr ~self ~ivar_name |> from_voidp typ <-@ value;

      if notify_change then
        msg_send_ov ~self ~cmd:(selector "didChangeValueForKey:") key
    in
    method_spec ~cmd ~typ:(typ @-> returning void) ~enc imp

  (** Getter for object values. *)
  let obj_getter ~typ ~enc ivar_name =
    let cmd = selector ivar_name
    and imp self _cmd =
      let ivar =
        C.Functions.Class.get_instance_variable (Object.get_class self)
          ivar_name
      in
      Object.get_ivar ~self ~ivar
    in
    method_spec ~cmd ~typ:(returning typ) ~enc imp

  (** Setter for object values. *)
  let obj_setter ?(assign = false) ?(copy = false) ?(notify_change = false) ~typ
      ~enc ivar_name =
    let cmd = selector (setter_name_of_ivar ivar_name)
    and key = new_string ivar_name in
    let imp self _cmd value =
      if (not assign) && not copy then value |> retain |> ignore;

      (* release old object *)
      let ivar =
        C.Functions.Class.get_instance_variable (Object.get_class self)
          ivar_name
      in
      assert (not (is_null ivar));
      Object.get_ivar ~self ~ivar |> release;

      if notify_change then
        msg_send_ov ~self ~cmd:(selector "willChangeValueForKey:") key;

      Object.set_ivar ~self ~ivar (if copy then _copy_ value else value);

      if notify_change then
        msg_send_ov ~self ~cmd:(selector "didChangeValueForKey:") key
    in
    method_spec ~cmd ~typ:(typ @-> returning void) ~enc imp

  (** Define a property getter and setter (unless [readonly] is [true]). *)
  let accessor_methods : type a.
      ?assign:bool ->
      ?copy:bool ->
      ?readonly:bool ->
      ?notify_change:bool ->
      string ->
      a Objc_type.t ->
      method_spec' list =
   fun ?(assign = false) ?(copy = false) ?(readonly = false)
       ?(notify_change = false) ivar_name t ->
    let typ = Objc_type.value_typ t and enc = Objc_type.encode_value t in
    match t with
    | Objc_type.Id ->
        if readonly then [ obj_getter ~typ ~enc ivar_name ]
        else
          [
            obj_getter ~typ ~enc ivar_name;
            obj_setter ~assign ~copy ~notify_change ~typ ~enc ivar_name;
          ]
    | _ ->
        if readonly then [ getter ~typ ~enc ivar_name ]
        else
          [
            getter ~typ ~enc ivar_name;
            setter ~typ ~enc ~notify_change ivar_name;
          ]

  (** Define a property with an ivar, getter, and setter (unless [readonly] is
      [true]). *)
  let define ?(retain = true) ?(copy = false) ?(readonly = false)
      ?(notify_change = false) name typ =
    prop_spec ~retain ~copy ~readonly ~notify_change ~typ name
end

module Class = struct
  include C.Functions.Class

  (* keep alive OCaml closures for defined methods *)
  let defined_method_roots = ref []

  let alignment_of_size size =
    let open Float in
    Size_t.to_int size |> of_int |> log2 |> round |> to_int |> UInt8.of_int

  let create_instance ?(extra_bytes = Size_t.of_int 0) cls =
    create_instance cls extra_bytes

  (** Adds a new method to a class with a given name and implementation. *)
  let add_method ~self ~cmd ~typ ~enc ?(runtime_lock = false)
      ?(thread_registration = false) imp =
    let method_t = id @-> _SEL @-> typ in
    let ty =
      _Class @-> _SEL
      @-> funptr ~runtime_lock ~thread_registration method_t
      @-> _Enc @-> returning bool
    in
    foreign "class_addMethod" ty self cmd imp enc

  (** Adds a new instance variable to a class. *)
  let add_ivar ~self ~name ~size ~enc =
    foreign "class_addIvar"
      (_Class @-> string @-> size_t @-> uint8_t @-> _Enc @-> returning bool)
      self name size (alignment_of_size size) enc

  (** Defines a new class and registers it with the Objective-C runtime. *)
  let define ?(superclass = C.Functions.Objc.get_class "NSObject")
      ?(protocols = []) ?(ivars = []) ?(properties = []) ?(methods = [])
      ?(class_methods = []) name =
    let self = C.Functions.Objc.allocate_class ~superclass name in
    let add_method'
        (Define.MethodSpec
           { cmd; typ; imp; enc; runtime_lock; thread_registration }) =
      (match Platform.current with
      | GNUStep ->
          let cmd = Sel.register_typed_name (Sel.get_name cmd) enc in
          assert (
            add_method ~self ~runtime_lock ~thread_registration ~cmd ~typ ~enc
              imp)
      | _ ->
          assert (
            add_method ~self ~runtime_lock ~thread_registration ~cmd ~typ ~enc
              imp));
      defined_method_roots := Root.create imp :: !defined_method_roots
    in
    assert (not (is_null self));

    methods |> List.iter add_method';

    properties
    |> List.iter
         (fun
           (Define.PropSpec
              { name; typ = t; retain; copy; readonly; notify_change })
         ->
           let typ = Objc_type.value_typ t
           and enc = Objc_type.encode_value t
           and assign = not retain in
           let size = Size_t.of_int (sizeof typ) in
           assert (add_ivar ~self ~name ~size ~enc);
           Property.accessor_methods ~assign ~copy ~readonly ~notify_change name
             t
           |> List.iter add_method');

    protocols
    |> List.iter (fun proto ->
           assert (not (is_null proto));
           assert (add_protocol self proto));

    ivars
    |> List.iter (fun (Define.IvarSpec { name; typ; enc }) ->
           let size = Size_t.of_int (sizeof typ) in
           assert (add_ivar ~self ~name ~size ~enc));

    C.Functions.Objc.register_class self;

    if List.length class_methods > 0 then (
      let metaclass = C.Functions.Objc.get_meta_class name in
      assert (not (is_null metaclass));
      class_methods
      |> List.iter
         @@
         fun (Define.MethodSpec
                { cmd; typ; imp; enc; runtime_lock; thread_registration })
         ->
         assert (
           add_method ~self:metaclass ~runtime_lock ~thread_registration ~cmd
             ~typ ~enc imp);
         defined_method_roots := Root.create imp :: !defined_method_roots);

    self
end

module Bitmask = struct
  (** Represents a set of options or flags using a single integer value. *)

  let of_list = List.fold_left Int.logor Int.zero
  let of_list' = List.fold_left ULLong.logor ULLong.zero
  let ( + ) = Int.logor
end

module Block_descriptor = struct
  type t

  let t : t structure typ = structure "Block_descriptor"
  let reserved = field t "reserved" ullong
  let size = field t "size" ullong
  let () = seal t

  let make sz =
    let d = make t in
    setf d reserved (ULLong.of_int 0);
    setf d size (ULLong.of_int sz);
    d
end

module Block = struct
  (** Represents a single task or unit of behavior. Blocks are Objective-C
      objects, which means they can be added to collections like [NSArray] or
      [NSDictionary]. *)

  type t

  let t : t structure typ = structure "Block_literal_1"
  let isa = field t "isa" _Class
  let flags = field t "flags" int
  let reserved = field t "reserved" int
  let invoke = field t "invoke" (ptr void)
  let descriptor = field t "descriptor" (ptr Block_descriptor.t)
  let () = seal t
  let size = sizeof t
  let desc_ptr = allocate Block_descriptor.t (Block_descriptor.make size)
  let block_is_global = Int.(shift_left one 28)
  let self = Objc.get_class "__NSGlobalBlock"

  let make' f =
    let b = make t in
    setf b isa self;
    setf b descriptor desc_ptr;
    setf b invoke f;
    setf b flags block_is_global;
    allocate t b |> coerce (ptr t) (ptr void)

  (** Create a global block which encapsulates the code for execution at a later
      time. *)
  let make ?(thread_registration = false) ?(runtime_lock = false) f ~args
      ~return =
    let typ = Objc_type.method_typ ~args:(Objc_type.id :: args) return in
    make' (coerce (funptr ~thread_registration ~runtime_lock typ) (ptr void) f)
end

module Method = struct
  include C.Functions.Method

  let invoke ~typ ~self m =
    foreign "method_invoke" (id @-> _Method @-> typ) self m

  let define ~cmd ~args ~return ?(runtime_lock = false)
      ?(thread_registration = false) imp =
    let typ = Objc_type.method_typ ~args return
    and enc = Objc_type.encode_method ~args return in
    Define.method_spec ~cmd ~typ ~enc ~runtime_lock ~thread_registration imp
end

module Ivar = struct
  include C.Functions.Ivar

  let define name typ =
    let typ = Objc_type.value_typ typ and enc = Objc_type.encode_value typ in
    Define.ivar_spec ~name ~typ ~enc
end

(* Exception handling *)

exception CamlNSException of string * string

(** Changes the top-level error handler. *)
let set_uncaught_exception_handler =
  foreign ~stub:true "NSSetUncaughtExceptionHandler"
    (funptr (id @-> returning void) @-> returning void)

let default_uncaught_exception_handler ex =
  let open Objc in
  let name = msg_send ~self:ex ~cmd:(selector "name") ~typ:(returning id)
  and reason = msg_send ~self:ex ~cmd:(selector "reason") ~typ:(returning id)
  and to_string self =
    msg_send ~self ~cmd:(selector "UTF8String") ~typ:(returning string)
  in
  Printf.eprintf "CamlNSException: %s -- %s\n%!" (to_string name)
    (to_string reason);
  raise @@ CamlNSException (to_string name, to_string reason)

let () =
  match Sys.backend_type with
  | Native -> (
      try set_uncaught_exception_handler default_uncaught_exception_handler
      with _ -> ())
  | _ -> ()
