open Ctypes
open C.Types

(** Represents an Objective-C type. *)
type _ t =
  | Id : object_t t
  | Class : object_t t
  | Sel : selector_t t
  | Void : unit t
  | Str : string t
  | Char : char t
  | Bool : bool t
  | Int : int t
  | Short : int t
  | Long : Signed.long t
  | LLong : Signed.llong t
  | ULLong : Unsigned.ullong t
  | Float : float t
  | Double : float t
  | Unknown : unit ptr t
  | Ptr : 'a t -> 'a ptr t
  | Arr : 'a t -> 'a carray t
  | Struc : 'a structure typ -> 'a structure t
  | Union : 'a t -> 'a union t
  | Imp : unit ptr t
  | Enc : _Enc t
  | Proto : protocol_t t
  | Ivar : ivar_t t

(** Represents a tuple of Objective-C types. *)
type (_, _) tlist =
  | [] : ('r, 'r) tlist
  | ( :: ) : 'a t * ('b, 'r) tlist -> ('a -> 'b, 'r) tlist

let rec ctype_of_t : type a. a t -> a typ = function
  | Id -> id
  | Class -> _Class
  | Sel -> _SEL
  | Void -> void
  | Str -> string
  | Char -> char
  | Bool -> bool
  | Int -> int
  | Short -> short
  | Long -> long
  | LLong -> llong
  | ULLong -> ullong
  | Float -> float
  | Double -> double
  | Unknown -> ptr void
  | Ptr ty -> ptr (ctype_of_t ty)
  | Arr ty -> array 0 (ctype_of_t ty)
  | Imp -> _IMP
  | Enc -> _Enc
  | Proto -> _Protocol
  | Ivar -> _Ivar
  | Struc s -> s
  | Union _ -> invalid_arg "not implemented"

(* Convenience constructors *)

let id = Id
let _Class = Class
let _SEL = Sel
let _IMP = Imp
let _Enc = Enc
let _Protocol = Proto
let _Ivar = Ivar
let void = Void
let string = Str
let char = Char
let bool = Bool
let int = Int
let short = Short
let long = Long
let llong = LLong
let ullong = ULLong
let float = Float
let double = Double
let unknown = Unknown
let ptr ty = Ptr ty
let array ty = Arr ty
let struc ty = Struc ty
let union ty = Union ty

let rec fn_of_tlist : type a b. b fn -> (a, b) tlist -> a fn =
 fun r -> function [] -> r | t :: xs -> ctype_of_t t @-> fn_of_tlist r xs

let byte_size_of_t : type a. a t -> int =
  let open Ctypes in
  function
  | Void -> sizeof void
  | Str -> sizeof string
  | Char -> sizeof char
  | Bool -> sizeof bool
  | Int -> sizeof int
  | Short -> sizeof short
  | Long -> sizeof long
  | LLong -> sizeof llong
  | ULLong -> sizeof ullong
  | Float -> sizeof float
  | Double -> sizeof double
  | Id | Class | Sel | Ptr _ | Imp | Enc | Proto | Ivar | Unknown ->
      sizeof (ptr void)
  | Struc s -> sizeof s
  | Arr _ | Union _ -> invalid_arg "not implemented"

let rec byte_size_of_tlist : type a b. (a, b) tlist -> int = function
  | [] -> 0
  | t :: xs -> byte_size_of_t t + byte_size_of_tlist xs

(** Returns the ctypes fn corresponding to the method type. *)
let method_typ ~args return = fn_of_tlist (returning (ctype_of_t return)) args

(** Returns the ctypes value typ corresponding to the value type. *)
let value_typ = ctype_of_t

(** The method accepts only the implicit [self] and [cmd] arguments. *)
let noargs = []

(* Functions dealing with the encoding of types in the Objective-C runtime. *)

let rec enc_of_t : type a. a t -> string = function
  | Id -> "@"
  | Class -> "#"
  | Sel -> ":"
  | Void -> "v"
  | Str -> "*"
  | Char -> "c"
  | Bool -> "C"
  | Int -> "i"
  | Short -> "s"
  | Long -> "l"
  | LLong -> "q"
  | ULLong -> "Q"
  | Float -> "f"
  | Double -> "d"
  | Unknown -> "?"
  | Ptr ty -> "^" ^ enc_of_t ty
  | Arr ty -> "[" ^ enc_of_t ty ^ "]"
  | Struc _ -> "{?}"
  | Union ty -> "(" ^ enc_of_t ty ^ ")"
  | Imp | Enc | Proto | Ivar -> "?"

let rec enc_of_tlist : type a b. int -> (a, b) tlist -> string =
 fun arg_offset -> function
  | [] -> ""
  | t :: xs ->
      enc_of_t t ^ string_of_int arg_offset
      ^ enc_of_tlist (arg_offset + byte_size_of_t t) xs

(** Returns the encoding of a method type in the Objective-C runtime. *)
let encode_method ~args return =
  enc_of_t return
  ^ string_of_int (byte_size_of_tlist args)
  ^ enc_of_tlist 0 (id :: _SEL :: args)

(** Returns the encoding of a value type in the Objective-C runtime. *)
let encode_value = enc_of_t
