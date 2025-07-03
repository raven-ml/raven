(** XLA Element Types *)

type t =
  | Invalid
  | Pred
  | S8
  | S16
  | S32
  | S64
  | U8
  | U16
  | U32
  | U64
  | F16
  | F32
  | F64
  | BF16
  | C64
  | C128
  | Tuple
  | Opaque_type
  | Token

let to_int = function
  | Invalid -> 0
  | Pred -> 1
  | S8 -> 2
  | U8 -> 3
  | S16 -> 4
  | U16 -> 16
  | S32 -> 5
  | S64 -> 6
  | U32 -> 7
  | U64 -> 8
  | F16 -> 9
  | F32 -> 11
  | F64 -> 12
  | BF16 -> 16
  | C64 -> 15
  | C128 -> 17
  | Tuple -> 13
  | Opaque_type -> 14
  | Token -> 18

let of_int = function
  | 0 -> Invalid
  | 1 -> Pred
  | 2 -> S8
  | 3 -> U8
  | 4 -> S16
  | 16 -> U16
  | 5 -> S32
  | 6 -> S64
  | 7 -> U32
  | 8 -> U64
  | 9 -> F16
  | 11 -> F32
  | 12 -> F64
  | 15 -> C64
  | 17 -> C128
  | 13 -> Tuple
  | 14 -> Opaque_type
  | 18 -> Token
  | _ -> Invalid

let to_string = function
  | Invalid -> "Invalid"
  | Pred -> "Pred"
  | S8 -> "S8"
  | S16 -> "S16"
  | S32 -> "S32"
  | S64 -> "S64"
  | U8 -> "U8"
  | U16 -> "U16"
  | U32 -> "U32"
  | U64 -> "U64"
  | F16 -> "F16"
  | F32 -> "F32"
  | F64 -> "F64"
  | BF16 -> "BF16"
  | C64 -> "C64"
  | C128 -> "C128"
  | Tuple -> "Tuple"
  | Opaque_type -> "Opaque_type"
  | Token -> "Token"
