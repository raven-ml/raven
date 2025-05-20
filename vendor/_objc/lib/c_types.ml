module Unsigned = struct
  type uint64 = int64

  let to_int64 x = x
  let of_int64 x = x
end

exception Undefined_property of string
exception Unsupported of string

let undefined_property fmt =
  Printf.ksprintf (fun s -> raise (Undefined_property s)) fmt

let unsupported fmt = Printf.ksprintf (fun s -> raise (Unsupported s)) fmt

type ('a, 'b) view_spec = {
  read : 'b -> 'a;
  write : 'a -> 'b;
  underlying_type : 'b typ;
}

and abstract_info = { aname : string; asize : int; aalignment : int }

and _ ocaml_type =
  | OCaml_string : string ocaml_type
  | OCaml_bytes : bytes ocaml_type

and _ prim =
  | Char : char prim
  | Schar : char prim
  | Uchar : char prim
  | Bool : bool prim
  | Short : int prim
  | Ushort : int prim
  | Int : int prim
  | Uint : int prim
  | Long : Nativeint.t prim
  | Ulong : Nativeint.t prim
  | Llong : int64 prim
  | Ullong : Unsigned.uint64 prim
  | Float : float prim
  | Double : float prim
  | Objc_id : Runtime_types.Id.t prim
  | Objc_sel : Runtime_types.Sel.t prim
  | Objc_class : Runtime_types.Class.t prim

and 'a ptr = Ptr_repr of { address : nativeint; pointee_type : 'a typ }
and 'a abstract_ptr = nativeint

and _ typ =
  | Void : unit typ
  | Primitive : 'a prim -> 'a typ
  | Pointer : 'a typ -> 'a ptr typ
  | OCaml : 'a ocaml_type -> 'a typ
  | Abstract : abstract_info -> 'a abstract_ptr typ
  | View : ('a, 'b) view_spec -> 'a typ

and (_, _) fn =
  (* Base case: a method that takes no additional Objective-C arguments. *)
  | Returns : 'rt typ -> (unit, 'rt) fn
  (* Recursive case: prepend one argument to the tuple-chain. *)
  | Function : 'arg typ * ('b, 'rt) fn -> ('arg * 'b, 'rt) fn

module Arch_details = struct
  let char_size = 1
  let short_size = 2
  let int_size = 4
  let long_size = Sys.int_size / 8
  let llong_size = 8
  let float_size = 4
  let double_size = 8
  let pointer_size = Sys.word_size / 8
  let char_align = 1
  let short_align = 2
  let int_align = 4
  let long_align = Sys.int_size / 8
  let llong_align = 8
  let float_align = 4
  let double_align = 8
  let pointer_align = Sys.word_size / 8
end

let sizeof_prim : type a. a prim -> int = function
  | Char | Schar | Uchar -> Arch_details.char_size
  | Bool -> Arch_details.char_size
  | Short | Ushort -> Arch_details.short_size
  | Int | Uint -> Arch_details.int_size
  | Long | Ulong -> Arch_details.long_size
  | Llong | Ullong -> Arch_details.llong_size
  | Float -> Arch_details.float_size
  | Double -> Arch_details.double_size
  | Objc_id | Objc_sel | Objc_class -> Arch_details.pointer_size

let alignment_prim : type a. a prim -> int = function
  | Char | Schar | Uchar -> Arch_details.char_align
  | Bool -> Arch_details.char_align
  | Short | Ushort -> Arch_details.short_align
  | Int | Uint -> Arch_details.int_align
  | Long | Ulong -> Arch_details.long_align
  | Llong | Ullong -> Arch_details.llong_align
  | Float -> Arch_details.float_align
  | Double -> Arch_details.double_align
  | Objc_id | Objc_sel | Objc_class -> Arch_details.pointer_align

let rec sizeof : type a. a typ -> int = function
  | Void -> undefined_property "sizeof(void)"
  | Primitive p -> sizeof_prim p
  | Pointer _ -> Arch_details.pointer_size
  | Abstract info -> info.asize
  | View v -> sizeof v.underlying_type
  | OCaml _ -> unsupported "sizeof cannot be applied to OCaml types directly"

let rec alignment : type a. a typ -> int = function
  | Void -> undefined_property "alignmentof(void)"
  | Primitive p -> alignment_prim p
  | Pointer _ -> Arch_details.pointer_align
  | Abstract info -> info.aalignment
  | View v -> alignment v.underlying_type
  | OCaml _ -> unsupported "alignment cannot be applied to OCaml types directly"

(* Standard type constructors *)
let void = Void
let char = Primitive Char
let schar = Primitive Schar
let uchar = Primitive Uchar
let bool = Primitive Bool
let short = Primitive Short
let ushort = Primitive Ushort
let int = Primitive Int
let uint = Primitive Uint
let long = Primitive Long
let ulong = Primitive Ulong
let llong = Primitive Llong
let ullong = Primitive Ullong
let float_ = Primitive Float
let double_ = Primitive Double
let objc_id = Primitive Objc_id
let objc_sel = Primitive Objc_sel
let objc_class = Primitive Objc_class
let ptr t = Pointer t

let make_ptr_value (address : nativeint) (pointee_type : 'a typ) : 'a ptr =
  Ptr_repr { address; pointee_type }

let get_ptr_address (ptr_val : 'a ptr) : nativeint =
  match ptr_val with Ptr_repr { address; _ } -> address

let abstract ~name ~size ~alignment =
  Abstract { aname = name; asize = size; aalignment = alignment }

let returning ty = Returns ty
let ( @-> ) arg_ty rest = Function (arg_ty, rest)
let view ~read ~write underlying_type = View { read; write; underlying_type }

(* OCaml type constructors *)
let ocaml_string : string typ = OCaml OCaml_string
let ocaml_bytes : bytes typ = OCaml OCaml_bytes
