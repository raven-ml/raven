open Ctypes
open Objc_c.Types
open Objc_c.Functions

(* Helper for freeing memory allocated by C. Ideally, this 'free_mem' would be
   in a more central C bindings module, e.g., Objc_c.Stdlib.free or similar. *)
let free_mem : unit ptr -> unit =
  Foreign.foreign "free" (ptr void @-> returning void)

module Objc = struct
  include Objc_c.Functions.Objc

  (* Functions specific to Apple platforms, often involving 'copy' semantics
     that require manual memory management. *)

  (** Returns the names of all the loaded Objective-C frameworks and dynamic
      libraries. Caller must free the returned pointer if not NULL. *)
  let _objc_copyImageNames_raw =
    match Platform.current with
    | GNUStep -> fun _ -> failwith "Not supported"
    | _ ->
        Foreign.foreign "objc_copyImageNames"
          (ptr uint @-> returning (ptr string))

  (** Returns the names of all the classes within a specified library or
      framework. Caller must free the returned pointer if not NULL. *)
  let _objc_copyClassNamesForImage_raw =
    match Platform.current with
    | GNUStep -> fun _ -> failwith "Not supported"
    | _ ->
        Foreign.foreign "objc_copyClassNamesForImage"
          (string @-> ptr uint @-> returning (ptr string))

  (** Returns the name of the dynamic library a class originated from. This
      typically returns a const char*, not needing a free by caller. *)
  let get_image_name =
    match Platform.current with
    | GNUStep -> fun _ -> failwith "Not supported"
    | _ -> Foreign.foreign "class_getImageName" (_Class @-> returning string)
end

let loaded_library_names ?(sorted = true) () =
  match Platform.current with
  | GNUStep -> failwith "loaded_library_names: Not supported on GNUStep"
  | _ ->
      let count_ptr = allocate uint Unsigned.UInt.zero in
      let names_carray_ptr = Objc._objc_copyImageNames_raw count_ptr in
      if is_null names_carray_ptr then []
      else
        Fun.protect
          ~finally:(fun () -> free_mem (to_voidp names_carray_ptr))
          (fun () ->
            let count = Unsigned.UInt.to_int !@count_ptr in
            let libs =
              CArray.from_ptr names_carray_ptr count |> CArray.to_list
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
                       (String.starts_with
                          ~prefix:"/System/Library/PrivateFrameworks/")
                  |> List.sort String.compare
                and rest =
                  libs
                  |> List.filter (fun lib ->
                         (not
                            (String.starts_with
                               ~prefix:"/System/Library/Frameworks/" lib))
                         && not
                              (String.starts_with
                                 ~prefix:"/System/Library/PrivateFrameworks/"
                                 lib))
                  |> List.sort String.compare
                in
                List.concat [ pub; priv; rest ])

let library_class_names ?(sorted = true) lib_name =
  match Platform.current with
  | GNUStep -> failwith "library_class_names: Not supported on GNUStep"
  | _ ->
      let count_ptr = allocate uint Unsigned.UInt.zero in
      let names_carray_ptr =
        Objc._objc_copyClassNamesForImage_raw lib_name count_ptr
      in
      if is_null names_carray_ptr then []
      else
        Fun.protect
          ~finally:(fun () -> free_mem (to_voidp names_carray_ptr))
          (fun () ->
            let count = Unsigned.UInt.to_int !@count_ptr in
            let names =
              CArray.from_ptr names_carray_ptr count |> CArray.to_list
            in
            match sorted with
            | false -> names
            | true -> List.sort String.compare names)

let registered_classes_count () =
  Objc.get_class_list (from_voidp (ptr objc_class) null) 0

let registered_classes () =
  let count = registered_classes_count () in
  if count = 0 then []
  else
    let buf = CArray.make (ptr objc_class) count in
    (* objc_getClassList does not allocate memory for the buffer itself, it
       fills the provided buffer. No free needed for 'buf' in this manner. *)
    let _ = Objc.get_class_list (CArray.start buf) count in
    CArray.to_list buf

let registered_class_names () =
  registered_classes ()
  |> List.map (fun c -> Class.get_name (coerce (ptr objc_class) _Class c))
  |> List.sort String.compare

let registered_protocols () =
  let count_ptr = allocate uint Unsigned.UInt.zero in
  (* Objc.get_protocol_list is objc_copyProtocolList *)
  let protocols_carray_ptr = Objc.get_protocol_list count_ptr in
  if is_null protocols_carray_ptr then []
  else
    Fun.protect
      ~finally:(fun () -> free_mem (to_voidp protocols_carray_ptr))
      (fun () ->
        let count = Unsigned.UInt.to_int !@count_ptr in
        CArray.from_ptr protocols_carray_ptr count |> CArray.to_list)

let registered_protocol_names () =
  registered_protocols () |> List.map Protocol.get_name
  |> List.sort String.compare

let methods cls =
  let count_ptr = allocate uint Unsigned.UInt.zero in
  (* Class.copy_method_list is class_copyMethodList *)
  let methods_carray_ptr = Class.copy_method_list cls count_ptr in
  if is_null methods_carray_ptr then []
  else
    Fun.protect
      ~finally:(fun () -> free_mem (to_voidp methods_carray_ptr))
      (fun () ->
        let count = Unsigned.UInt.to_int !@count_ptr in
        CArray.from_ptr methods_carray_ptr count |> CArray.to_list)

let method_names cls =
  methods cls |> List.map Method.get_name |> List.map Sel.get_name
  |> List.sort String.compare

let protocol_methods ?(required = false) ?(instance = true) proto =
  let count_ptr = allocate uint Unsigned.UInt.zero in
  (* Protocol.get_method_descriptions is protocol_copyMethodDescriptionList *)
  let descs_carray_ptr =
    Protocol.get_method_descriptions proto required instance count_ptr
  in
  if is_null descs_carray_ptr then []
  else
    Fun.protect
      ~finally:(fun () -> free_mem (to_voidp descs_carray_ptr))
      (fun () ->
        let count = Unsigned.UInt.to_int !@count_ptr in
        CArray.from_ptr descs_carray_ptr count |> CArray.to_list)

let protocol_method_names ?(required = false) ?(instance = true) proto =
  protocol_methods ~required ~instance proto
  |> List.map Method_description.name
  |> List.sort String.compare
