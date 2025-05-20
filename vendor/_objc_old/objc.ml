open Objc_stubs

let selector name = sel_registerName name
let get_class name = objc_getClass name
let ns_object_class = class_ns_object
let ns_string_class = class_ns_string
let alloc cls = objc_msgSend_id_sel cls (selector "alloc")
let init obj = objc_msgSend_id_sel obj (selector "init")
let retain obj = if is_nil obj then obj else objc_retain obj
let release obj = if is_nil obj then () else objc_release obj
let autorelease obj = if is_nil obj then obj else objc_autorelease obj

let gc_finalize_release obj =
  if not (is_nil obj) then
    Gc.finalise (fun o -> if not (is_nil o) then release o) obj

let new_object class_name =
  let cls = get_class class_name in
  if is_nil cls then nil (* Or raise error *)
  else
    let obj = alloc cls in
    if is_nil obj then nil
    else
      let initialized_obj = init obj in
      gc_finalize_release initialized_obj;
      initialized_obj

let new_string s =
  let ns_str =
    objc_msgSend_id_sel_string ns_string_class
      (selector "stringWithUTF8String:")
      s
  in
  gc_finalize_release ns_str;
  ns_str

let to_string (obj : id) : string =
  if is_nil obj then ""
  else
    (* Check if it's an NSString first *)
    let obj_cls = object_getClass obj in
    let is_nsstring =
      (* This equality check might be tricky with opaque types if not
         canonicalized *)
      obj_cls = ns_string_class
      || class_conformsToProtocol obj_cls (_objc_getProtocol "NSString")
      || sel_isEqual (_class_getName obj_cls) (_class_getName ns_string_class)
      (* Brittle equality *)
    in
    let target_obj =
      if is_nsstring then obj
      else objc_msgSend_id_sel obj (selector "description")
    in
    if is_nil target_obj then ""
    else objc_msgSend_string_id_sel target_obj (selector "UTF8String")

(* --- Simplified msg_send (examples) --- *)
module Msg = struct
  let send_id_sel ~self ~cmd = objc_msgSend_id_sel self cmd
  let send_id_sel_id ~self ~cmd ~arg1 = objc_msgSend_id_sel_id self cmd arg1
  let send_void_id_sel ~self ~cmd = objc_msgSend_void_id_sel self cmd
  (* Add more variants based on Objc_type.t and common patterns *)
end

(* --- Ivar access (simplified example) --- *)
let get_ivar_raw_ptr (obj : id) (ivar_name : string) : nativeint =
  let ptr_addr, ivar_obj = object_getInstanceVariable_ptr obj ivar_name in
  ptr_addr

let get_ivar_int (obj : id) (ivar_name : string) : int =
  let addr = get_ivar_raw_ptr obj ivar_name in
  if addr = Nativeint.zero then
    failwith ("Ivar not found or is null: " ^ ivar_name)
  else ptr_read_int addr

let get_ivar_id (obj : id) (ivar_name : string) : id =
  let addr = get_ivar_raw_ptr obj ivar_name in
  if addr = Nativeint.zero then
    failwith ("Ivar not found or is null: " ^ ivar_name)
  else ptr_read_id addr (* Assumes ptr_read_id handles potential retain *)

(* Setter is more complex due to memory management (retain/release old/new
   value) *)

(* --- Class Definition --- *)
module Class = struct
  include Objc_core (* Re-export low-level functions if needed *)

  let add_method ~cls ~sel ~imp ~enc = class_addMethod cls sel imp enc

  let add_ivar_raw ~cls ~name ~size ~alignment ~enc =
    class_addIvar cls name size alignment enc

  let define ?(superclass_name = "NSObject") ?(protocols = []) ?(ivars = [])
      ?(methods = []) name =
    let super_cls = get_class superclass_name in
    if is_nil super_cls then
      failwith ("Superclass not found: " ^ superclass_name);

    let new_cls =
      objc_allocateClassPair ~superclass:super_cls ~name ~extra_bytes:0
    in
    if is_nil new_cls then failwith ("Could not allocate class: " ^ name);

    (* Add methods *)
    List.iter
      (fun (method_name, ocaml_imp_func, typ_enc_str) ->
        let sel = selector method_name in
        (* The ocaml_imp_func needs to be transformed into a C IMP. This is the
           most complex part. The OCaml function must be rooted. A C stub
           (forwarder) calls the OCaml function. *)
        let c_imp = create_ocaml_imp_forwarder ocaml_imp_func in
        if not (add_method ~cls:new_cls ~sel ~imp:c_imp ~enc:typ_enc_str) then
          Printf.eprintf "Warning: Failed to add method %s to class %s\n"
            method_name name)
      methods;

    (* Add ivars - Objc_define.ivar_spec' would carry necessary info *)
    List.iter
      (fun (ivar_name, size, alignment, enc_str) ->
        if
          not
            (add_ivar_raw ~cls:new_cls ~name:ivar_name ~size ~alignment
               ~enc:enc_str)
        then
          Printf.eprintf "Warning: Failed to add ivar %s to class %s\n"
            ivar_name name)
      ivars;

    (* Add protocols *)
    (* List.iter (fun proto_name ->
      let proto = objc_getProtocol proto_name in
      if not (is_nil proto) then ignore (_class_addProtocol new_cls proto)
    ) protocols; *)
    (* class_addProtocol external not defined in sketch *)
    objc_registerClassPair new_cls;
    new_cls
end

(* Other modules like Define, Inspect, Block would be adapted similarly, relying
   on Objc_core and the new Objc_runtime. *)
