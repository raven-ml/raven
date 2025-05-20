module Id = Runtime_types.Id
module Sel = Runtime_types.Sel
module Class = Runtime_types.Class

(* C_types *)

let ( @-> ) = C_types.( @-> )
let returning = C_types.returning
let void = C_types.void
let char = C_types.char
let schar = C_types.schar
let uchar = C_types.uchar
let bool = C_types.bool
let short = C_types.short
let ushort = C_types.ushort
let int = C_types.int
let uint = C_types.uint
let long = C_types.long
let ulong = C_types.ulong
let llong = C_types.llong
let ullong = C_types.ullong
let float_ = C_types.float_
let double_ = C_types.double_
let objc_id = C_types.objc_id
let objc_sel = C_types.objc_sel
let objc_class = C_types.objc_class
let ptr = C_types.ptr

(* *)

let get_class = Runtime.get_class
let register_selector = Runtime.register_selector
let get_selector_name = Runtime.get_selector_name
let get_class_name = Runtime.get_class_name
let get_object_class = Runtime.get_object_class
let msg_send = Runtime.msg_send
