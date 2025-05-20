open Runtime_types
open C_types

(* A compact description of C scalar types that libffi understands. *)
module Tag = struct
  (* Keep the OCaml view small but give each tag the integer expected by
     ffi_type_of_tag in objc_stubs.c (see the big switch-case there). *)
  type t =
    | Void (* 0 *)
    | U8 (* 1 – unsigned 8-bit (Char/Uchar) *)
    | S8 (* 2 – signed 8-bit (Schar) *)
    | Bool (* 4 – Objective-C BOOL (uint8) *)
    | U16 (* 6 *)
    | S16 (* 5 *)
    | U32 (* 8 *)
    | S32 (* 7 *)
    | U64 (* 12 *)
    | S64 (* 11 *)
    | F32 (* 13 *)
    | F64 (* 14 *)
    | Ptr (* 15 *)

  let to_int = function
    | Void -> 0
    | U8 -> 1
    | S8 -> 2
    | Bool -> 4
    | S16 -> 5
    | U16 -> 6
    | S32 -> 7
    | U32 -> 8
    | S64 -> 11
    | U64 -> 12
    | F32 -> 13
    | F64 -> 14
    | Ptr -> 15
end

let tag_of_typ : type a. a typ -> Tag.t = function
  | Void -> Void
  | Primitive Char | Primitive Uchar -> U8
  | Primitive Schar -> S8
  | Primitive Bool -> Bool
  | Primitive Short -> S16
  | Primitive Ushort -> U16
  | Primitive Int -> S32
  | Primitive Uint -> U32
  | Primitive Long | Primitive Llong -> S64
  | Primitive Ulong | Primitive Ullong -> U64
  | Primitive Float -> F32
  | Primitive Double -> F64
  | Primitive (Objc_id | Objc_sel | Objc_class) | Pointer _ | Abstract _ -> Ptr
  | OCaml _ | View _ ->
      invalid_arg "msg_send: cannot marshal OCaml/View types through FFI"

let encode_value : type a. a typ -> a -> nativeint =
 fun ty v ->
  match ty with
  | Void -> Nativeint.zero
  | Primitive Char -> Nativeint.of_int (Char.code v)
  | Primitive Uchar -> Nativeint.of_int (Char.code v)
  | Primitive Schar -> Nativeint.of_int (Char.code v)
  | Primitive Bool -> if v then Nativeint.one else Nativeint.zero
  | Primitive Short -> Nativeint.of_int v
  | Primitive Ushort -> Nativeint.of_int v
  | Primitive Int -> Nativeint.of_int v
  | Primitive Uint -> Nativeint.of_int v
  | Primitive Long -> v
  | Primitive Ulong -> v
  | Primitive Llong -> Int64.to_nativeint v
  | Primitive Ullong -> Int64.to_nativeint v
  | Primitive Float -> Nativeint.of_int32 (Int32.bits_of_float v)
  | Primitive Double ->
      let bits = Int64.bits_of_float v in
      Int64.to_nativeint bits
  | Primitive Objc_id -> v
  | Primitive Objc_sel -> v
  | Primitive Objc_class -> v
  | Pointer _ -> get_ptr_address v
  | Abstract _ -> v
  | OCaml _ | View _ -> assert false

let decode_value : type a. a typ -> nativeint -> a =
 fun ty raw ->
  match ty with
  | Void -> ()
  | Primitive Char -> Char.chr (Nativeint.to_int raw)
  | Primitive Uchar -> Char.chr (Nativeint.to_int raw)
  | Primitive Schar -> Char.chr (Nativeint.to_int raw)
  | Primitive Bool -> raw <> Nativeint.zero
  | Primitive Short -> Nativeint.to_int raw
  | Primitive Ushort -> Nativeint.to_int raw
  | Primitive Int -> Nativeint.to_int raw
  | Primitive Uint -> Nativeint.to_int raw
  | Primitive Long -> raw
  | Primitive Ulong -> raw
  | Primitive Llong -> Int64.of_nativeint raw
  | Primitive Ullong -> Int64.of_nativeint raw
  | Primitive Float -> Int32.float_of_bits (Nativeint.to_int32 raw)
  | Primitive Double -> Int64.float_of_bits (Int64.of_nativeint raw)
  | Primitive Objc_id -> raw
  | Primitive Objc_sel -> raw
  | Primitive Objc_class -> raw
  | Pointer pt -> make_ptr_value raw pt
  | Abstract _ -> raw
  | OCaml _ | View _ -> assert false

let rec gather : type a r.
    (a, r) fn -> a -> Tag.t list * nativeint array * r typ =
 fun fn args ->
  match fn with
  | Returns rt -> ([], [||], rt)
  | Function (ty, rest) ->
      let arg, rest_args = args in
      let tags_tail, data_tail, rt = gather rest rest_args in
      ( tag_of_typ ty :: tags_tail,
        Array.append [| encode_value ty arg |] data_tail,
        rt )

external call_raw :
  nativeint -> nativeint -> int array -> nativeint array -> int -> nativeint
  = "caml_msg_send_ffi"

let msg_send : type a r. self:Id.t -> sel:Sel.t -> (a, r) fn -> a -> r =
 fun ~self ~sel fn args ->
  let arg_tags, arg_data, ret_typ = gather fn args in
  let raw_res =
    call_raw (Id.to_nativeint self) (Sel.to_nativeint sel)
      (Array.of_list (List.map Tag.to_int arg_tags))
      arg_data
      (Tag.to_int (tag_of_typ ret_typ))
  in
  decode_value ret_typ raw_res
