open Ctypes
open C.Types
open C.Functions

module Objc = struct
  include Objc

  (** Returns the names of all the loaded Objective-C frameworks and dynamic
      libraries. *)
  let copy_image_names =
    match Platform.current with
    | GNUStep -> fun _ -> failwith "Not supported"
    | _ ->
        Foreign.foreign "objc_copyImageNames"
          (ptr uint @-> returning (ptr string))

  (** Returns the names of all the classes within a specified library or
      framework. *)
  let copy_image_class_names =
    match Platform.current with
    | GNUStep -> fun _ -> failwith "Not supported"
    | _ ->
        Foreign.foreign "objc_copyClassNamesForImage"
          (string @-> ptr uint @-> returning (ptr string))

  (** Returns the name of the dynamic library a class originated from. *)
  let get_image_name =
    match Platform.current with
    | GNUStep -> fun _ -> failwith "Not supported"
    | _ -> Foreign.foreign "class_getImageName" (_Class @-> returning string)
end

let loaded_library_names ?(sorted = true) () =
  let count = allocate uint Unsigned.UInt.zero in
  let libsp = Objc.copy_image_names count in
  let libs =
    CArray.from_ptr libsp (Unsigned.UInt.to_int !@count) |> CArray.to_list
  in
  match sorted with
  | false -> libs
  | true ->
      let pub =
        libs
        |> List.filter
             (String.starts_with ~prefix:"/System/Library/Frameworks/")
        |> List.sort String.compare
      and priv =
        libs
        |> List.filter
             (String.starts_with ~prefix:"/System/Library/PrivateFrameworks/")
        |> List.sort String.compare
      and rest =
        libs
        |> List.filter (fun lib ->
               (not
                  (String.starts_with ~prefix:"/System/Library/Frameworks/" lib))
               && not
                    (String.starts_with
                       ~prefix:"/System/Library/PrivateFrameworks/" lib))
        |> List.sort String.compare
      in
      List.concat [ pub; priv; rest ]

let library_class_names ?(sorted = true) lib_name =
  let count = allocate uint Unsigned.UInt.zero in
  let namesp = Objc.copy_image_class_names lib_name count in
  let names =
    CArray.from_ptr namesp (Unsigned.UInt.to_int !@count) |> CArray.to_list
  in
  match sorted with false -> names | true -> List.sort String.compare names

let registered_classes_count () =
  Objc.get_class_list (from_voidp (ptr objc_class) null) 0

let registered_classes () =
  let count = registered_classes_count () in
  let buf = CArray.make (ptr objc_class) count in
  let _ = Objc.get_class_list (CArray.start buf) count in
  CArray.to_list buf

let registered_class_names () =
  registered_classes ()
  |> List.map (fun c -> Class.get_name (coerce (ptr objc_class) _Class c))
  |> List.sort String.compare

let registered_protocols () =
  let count = allocate uint Unsigned.UInt.zero in
  let p = Objc.get_protocol_list count in
  CArray.from_ptr p (Unsigned.UInt.to_int !@count) |> CArray.to_list

let registered_protocol_names () =
  registered_protocols () |> List.map Protocol.get_name
  |> List.sort String.compare

let methods cls =
  let count = allocate uint Unsigned.UInt.zero in
  let m = Class.copy_method_list cls count in
  CArray.from_ptr m (Unsigned.UInt.to_int !@count) |> CArray.to_list

let method_names cls =
  methods cls |> List.map Method.get_name |> List.map Sel.get_name
  |> List.sort String.compare

let protocol_methods ?(required = false) ?(instance = true) proto =
  let count = allocate uint Unsigned.UInt.zero in
  let pm = Protocol.get_method_descriptions proto required instance count in
  CArray.from_ptr pm (Unsigned.UInt.to_int !@count) |> CArray.to_list

let protocol_method_names ?(required = false) ?(instance = true) proto =
  protocol_methods ~required ~instance proto
  |> List.map Method_description.name
  |> List.sort String.compare
