(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  ---------------------------------------------------------------------------*)

type t = Dtype.bound

let int n = `Int (Z.of_int n)
let zero = int 0
let one = int 1

let integer = function
  | `Int n -> n
  | `Bool b -> if b then Z.one else Z.zero
  | `Float _ -> invalid_arg "Bound.integer: floating-point bound"

let to_int b =
  let n = integer b in
  if Z.fits_int n then Z.to_int n
  else invalid_arg "Bound.to_int: bound exceeds the host integer range"

let to_float = function
  | `Int n -> Z.to_float n
  | `Bool b -> if b then 1. else 0.
  | `Float f -> f

let compare_integer_float n f =
  if Float.is_nan f then invalid_arg "Bound.compare: NaN has no interval"
  else if f = infinity then -1
  else if f = neg_infinity then 1
  else
    let c = Z.compare n (Z.of_float f) in
    if c <> 0 || f = Float.trunc f then c else if f > 0. then -1 else 1

let compare a b =
  match a, b with
  | `Float x, `Float y ->
      if Float.is_nan x || Float.is_nan y then
        invalid_arg "Bound.compare: NaN has no interval";
      Float.compare x y
  | `Float f, b -> -(compare_integer_float (integer b) f)
  | a, `Float f -> compare_integer_float (integer a) f
  | a, b -> Z.compare (integer a) (integer b)

let equal a b = compare a b = 0
let lt a b = compare a b < 0
let le a b = compare a b <= 0
let min a b = if le a b then a else b
let max a b = if le a b then b else a

let arithmetic fi ff a b =
  match a, b with
  | `Float _, _ | _, `Float _ -> `Float (ff (to_float a) (to_float b))
  | _ -> `Int (fi (integer a) (integer b))

let add = arithmetic Z.add ( +. )
let sub = arithmetic Z.sub ( -. )
let mul = arithmetic Z.mul ( *. )
let neg = function `Float f -> `Float (-.f) | b -> `Int (Z.neg (integer b))
let succ b = add b one
let pred b = sub b one
let cdiv a b = `Int (Z.div (integer a) (integer b))
let floordiv a b = `Int (Z.fdiv (integer a) (integer b))
let floormod a b = sub a (mul (floordiv a b) b)
let lognot b = `Int (Z.lognot (integer b))
let shift_left a b = `Int (Z.shift_left (integer a) (to_int b))
let shift_right a b = `Int (Z.shift_right (integer a) (to_int b))

let round dtype b =
  if Dtype.is_float dtype then `Float (Dtype.truncate_float dtype (to_float b))
  else if Dtype.is_int dtype then
    match b with
    | `Float f when Float.is_finite f -> `Int (Z.of_float f)
    | _ -> b
  else b

let const dtype b =
  Const.of_view dtype
    (match b with `Bool b -> Const.Bool b | `Int n -> Const.Int n | `Float f -> Const.Float f)
