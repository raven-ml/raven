open Ctypes
open Foreign
open Type_description.Types

(* module Types = Types_generated

module Functions (F : Ctypes.FOREIGN) = struct
  open Types
  open F *)

module Functions = struct
  module Objc = struct
    (** Returns the class definition of a specified class. *)
    let get_class = foreign "objc_getClass" (string @-> returning _Class)

    (** Returns the metaclass definition of a specified class. *)
    let get_meta_class =
      foreign "objc_getMetaClass" (string @-> returning _Class)

    let get_protocol =
      foreign "objc_getProtocol" (string @-> returning _Protocol)

    (** Registers a class that was allocated using [allocate_class]. *)
    let register_class =
      foreign "objc_registerClassPair" (_Class @-> returning void)

    (** Obtains the list of registered class definitions.
        [get_class_list buffer buffer_count] {e buffer} An array of Class
        values. On output, each Class value points to one class definition, up
        to either bufferCount or the total number of registered classes,
        whichever is less. You can pass NULL to obtain the total number of
        registered class definitions without actually retrieving any class
        definitions.

        @param buffer_count
          An integer value. Pass the number of pointers for which you have
          allocated space in buffer. On return, this function fills in only this
          number of elements. If this number is less than the number of
          registered classes, this function returns an arbitrary subset of the
          registered classes. *)
    let get_class_list =
      foreign "objc_getClassList"
        (ptr (ptr objc_class) @-> int @-> returning int)

    (** Returns an array of all the protocols known to the runtime. *)
    let get_protocol_list =
      foreign "objc_copyProtocolList" (ptr uint @-> returning (ptr _Protocol))

    (** Creates a new class and metaclass. extra_bytes: the number of bytes to
        allocate for indexed ivars at the end of the class and metaclass
        objects. *)
    let allocate_class ?(extra_bytes = Unsigned.Size_t.of_int 0) ~superclass
        name =
      foreign "objc_allocateClassPair"
        (_Class @-> string @-> size_t @-> returning _Class)
        superclass name extra_bytes
  end

  module Class = struct
    (** Returns the name of a class. *)
    let get_name = foreign "class_getName" (_Class @-> returning string)

    (** Returns the superclass of a class. *)
    let get_superclass =
      foreign "class_getSuperclass" (_Class @-> returning _Class)

    (** Adds a protocol to a class. *)
    let add_protocol =
      foreign "class_addProtocol" (_Class @-> _Protocol @-> returning bool)

    (** Returns a Boolean value that indicates whether a class conforms to a
        given protocol. *)
    let conforms_to_protocol =
      foreign "class_conformsToProtocol"
        (_Class @-> _Protocol @-> returning bool)

    (** Creates an instance of a class, allocating memory for the class in the
        default malloc memory zone. extraBytes: An integer indicating the number
        of extra bytes to allocate. The additional bytes can be used to store
        additional instance variables beyond those defined in the class
        definition. *)
    let create_instance =
      foreign "class_createInstance" (_Class @-> size_t @-> returning id)

    (** Describes the instance methods implemented by a class.

        @param outCount
          On return, contains the length of the returned array. If outCount is
          NULL, the length is not returned. *)
    let copy_method_list =
      foreign "class_copyMethodList"
        (_Class @-> ptr uint @-> returning (ptr _Method))

    (** Returns a specified instance method for a given class. *)
    let get_instance_method =
      foreign "class_getInstanceMethod" (_Class @-> _SEL @-> returning _Method)

    (** Returns the Ivar for a specified instance variable of a given class. *)
    let get_instance_variable =
      foreign "class_getInstanceVariable" (_Class @-> string @-> returning _Ivar)

    (** Returns the function pointer that would be called if a particular
        message were sent to an instance of a class. *)
    let get_method_implementation =
      foreign "class_getMethodImplementation"
        (_Class @-> _SEL @-> returning _IMP)

    (** Returns a Boolean value that indicates whether a class object is a
        metaclass. *)
    let is_meta_class = foreign "class_isMetaClass" (_Class @-> returning bool)

    (** Describes the properties declared by a class. *)
    let copy_property_list =
      foreign "class_copyPropertyList"
        (_Class @-> ptr uint @-> returning (ptr Property_attribute.t))

    (** Adds a property to a class. *)
    let add_property =
      foreign "class_addProperty"
        (_Class @-> string @-> ptr Property_attribute.t @-> uint
       @-> returning bool)

    (** Returns a property with a given name of a given class. *)
    let get_property =
      foreign "class_getProperty"
        (_Class @-> string @-> returning (ptr Property_attribute.t))
  end

  module Object = struct
    (** Returns the class of an object. *)
    let get_class = foreign "object_getClass" (id @-> returning _Class)
  end

  module Sel = struct
    (** Registers a method with the Objective-C runtime system, maps the method
        name to a selector, and returns the selector value. If the method name
        has already been registered, this function simply returns the selector.
    *)
    let register_name = foreign "sel_registerName" (string @-> returning _SEL)

    (** Returns the name of the method specified by a given selector. *)
    let get_name = foreign "sel_getName" (_SEL @-> returning string)

    (** Returns a Boolean value that indicates whether two selectors are equal.
    *)
    let is_equal = foreign "sel_isEqual" (_SEL @-> _SEL @-> returning bool)
  end

  module Ivar = struct
    (** Returns the offset of an instance variable. *)
    let get_offset = foreign "ivar_getOffset" (_Ivar @-> returning ptrdiff_t)

    (** Returns the name of an instance variable. *)
    let get_name = foreign "ivar_getName" (_Ivar @-> returning string)

    (** Returns the type string of an instance variable. *)
    let get_type_encoding =
      foreign "ivar_getTypeEncoding" (_Ivar @-> returning _Enc)
  end

  module Method = struct
    (** Returns the name of a method. *)
    let get_name = foreign "method_getName" (_Method @-> returning _SEL)

    (** Returns a string describing a method's return type. *)
    let get_return_type =
      foreign "method_copyReturnType" (_Method @-> returning string)

    (** Returns the number of arguments accepted by a method. *)
    let get_number_of_arguments =
      foreign "method_getNumberOfArguments" (_Method @-> returning uint)

    (** Returns a string describing a single parameter type of a method. *)
    let get_argument_type =
      foreign "method_copyArgumentType" (_Method @-> uint @-> returning string)

    (** Returns a string describing a method's parameter and return types. *)
    let get_type_encoding =
      foreign "method_getTypeEncoding" (_Method @-> returning string)
  end

  module Protocol = struct
    (** Returns the name of a protocol. *)
    let get_name = foreign "protocol_getName" (_Protocol @-> returning string)

    (** Returns an array of method descriptions of methods meeting a given
        specification for a given protocol. *)
    let get_method_descriptions =
      foreign "protocol_copyMethodDescriptionList"
        (_Protocol @-> bool @-> bool @-> ptr uint
        @-> returning (ptr Method_description.t))
  end
end
