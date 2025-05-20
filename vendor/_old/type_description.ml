open Ctypes

(* module Types (F : Ctypes.TYPE) = struct
  open F *)

module Types = struct
  type objc_class

  let objc_class : objc_class structure typ = structure "objc_class"

  type objc_object

  let objc_object : objc_object structure typ = structure "objc_object"

  type objc_method

  let objc_method : objc_method structure typ = structure "objc_method"

  type objc_selector

  let objc_selector : objc_selector structure typ = structure "objc_selector"

  type objc_ivar

  let objc_ivar : objc_ivar structure typ = structure "objc_ivar"

  type objc_protocol

  let objc_protocol : objc_protocol structure typ = structure "objc_protocol"

  type class_t = objc_object structure ptr
  type object_t = objc_object structure ptr
  type imp_t = unit ptr
  type selector_t = objc_selector structure ptr
  type protocol_t = objc_object structure ptr
  type ivar_t = objc_ivar structure ptr
  type _Enc = string

  module Method_description = struct
    let t : [ `Method_desc ] structure typ = structure "objc_method_description"
    let name = field t "name" string
    let types = field t "types" string
    let () = seal t
    let name t = getf t name
    let types t = getf t types
  end

  module Property_attribute = struct
    let t : [ `Prop_attr ] structure typ = structure "objc_property_attribute_t"
    let name = field t "name" string
    let value = field t "value" string
    let () = seal t
    let name t = getf t name
    let value t = getf t value
  end

  let id = ptr objc_object
  let _Class = ptr objc_object
  let _SEL = ptr objc_selector
  let _IMP = ptr void
  let _Enc = string
  let _Protocol = ptr objc_object
  let _Ivar = ptr objc_ivar
  let _Method = ptr objc_method

  (* let c_id = constant "_C_ID" char
  let c_uchar = constant "_C_UCHR" char *)
end
