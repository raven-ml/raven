(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

open Import

include Float64x2_internal

type mask = int64x2#

external box : t -> float64x2 @@ portable = "%box_vec128"
external unbox : float64x2 @ local -> t @@ portable = "%unbox_vec128"

(* ───── Constants ───── *)

external const1 : float# -> t @@ portable
  = "caml_vec128_unreachable" "caml_float64x2_const1"
  [@@noalloc] [@@builtin]

external const : float# -> float# -> t @@ portable
  = "caml_vec128_unreachable" "caml_float64x2_const2"
  [@@noalloc] [@@builtin]

let[@inline] zero () = const1 #0.
let[@inline] one () = const1 #1.
let[@inline] sign64_mask () = Int64x2.const1 #0x8000000000000000L
let[@inline] absf64_mask () = Int64x2.const1 #0x7fffffffffffffffL

external of_int64x2 : int64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

(* ───── Operators ───── *)

let[@inline] ( + ) x y = add x y
let[@inline] ( - ) x y = sub x y
let[@inline] ( * ) x y = mul x y
let[@inline] ( / ) x y = div x y
let[@inline] horizontal_add x y = hadd x y

let[@inline] is_nan t = Int64x2.bitwise_not (cmeq t t)
let[@inline] is_not_nan t = cmeq t t

let[@inline] ( >= ) x y = cmge x y
let[@inline] ( <= ) x y = cmle x y
let[@inline] ( = ) x y = cmeq x y
let[@inline] ( > ) x y = cmgt x y
let[@inline] ( < ) x y = cmlt x y
let[@inline] ( <> ) x y = Int64x2.bitwise_not (cmeq x y)
let[@inline] equal x y = cmeq x y

(* ───── Bitwise Operations ───── *)

(* Negation and absolute value operate on floats via int64 bit manipulation:
   - neg: XOR with sign bit mask flips the sign
   - abs: AND with inverse sign mask clears the sign bit *)

let[@inline] neg x =
  Int64x2.(bitwise_xor (sign64_mask ()) (of_float64x2 x)) |> of_int64x2

let[@inline] abs x =
  Int64x2.(bitwise_and (absf64_mask ()) (of_float64x2 x)) |> of_int64x2

(* ───── Lane Operations ───── *)

external low_of : float# -> t @@ portable
  = "caml_vec128_unreachable" "caml_float64x2_low_of_float"
  [@@noalloc] [@@builtin]

external low_to : t -> float# @@ portable
  = "caml_vec128_unreachable" "caml_float64x2_low_to_float"
  [@@noalloc] [@@builtin]

external high_64_to_low_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_high_64_to_low_64"
  [@@noalloc] [@@unboxed] [@@builtin]

external low_64_to_high_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_low_64_to_high_64"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] set1 a =
  let a = low_of a in
  Int64x2.dup (Int64x2.of_float64x2 a) |> of_int64x2

let[@inline] set a b = low_64_to_high_64 (low_of a) (low_of b)

let[@inline] extract ~idx x =
  match idx with
  | #0L -> low_to x
  | #1L -> low_to (high_64_to_low_64 x x)
  | _ -> assert false

let[@inline] insert ~idx t a =
  match idx with
  | #0L -> low_64_to_high_64 (low_of a) (high_64_to_low_64 t t)
  | #1L -> low_64_to_high_64 t (low_of a)
  | _ -> assert false

let[@inline] extract0 x = low_to x
let[@inline] splat x = #(low_to x, low_to (high_64_to_low_64 x x))

let[@inline] movemask m = Int64x2.movemask m
let[@inline] bitmask m = m

let[@inline] round_nearest x = round_near x

(* ───── Interleave ───── *)

external interleave_high_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_high_64"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_64"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] interleave_upper ~even ~odd = interleave_high_64 even odd
let[@inline] interleave_lower ~even ~odd = interleave_low_64 even odd
let[@inline] duplicate_low x = Int64x2.dup (Int64x2.of_float64x2 x) |> of_int64x2

(* ───── Conversions ───── *)

external of_float32x4 : float32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int8x16 : int8x16# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int16x8 : int16x8# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int32x4 : int32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]


(* Use saturating narrowing conversion to match SSE behavior *)
let[@inline] cvt_int32x4 (t : t) : int32x4# =
  t |> round_current |> cvt_int64x2 |> Int64x2.cvt_int32x4_saturating

let[@inline] cvtt_int32x4 (t : t) : int32x4# =
  t |> cvtt_int64x2 |> Int64x2.cvt_int32x4_saturating

let[@inline] of_float32x4_bits x = of_float32x4 x
let[@inline] of_int8x16_bits x = of_int8x16 x
let[@inline] of_int16x8_bits x = of_int16x8 x
let[@inline] of_int32x4_bits x = of_int32x4 x
let[@inline] of_int64x2_bits x = of_int64x2 x
let[@inline] unsafe_of_float x = low_of x
let[@inline] cvt_of_int32x4 x = Int32x4.cvt_f64 x

(* ───── String Conversion ───── *)

let[@inline] to_string x =
  let #(a, b) = splat x in
  Printf.sprintf "(%.17g %.17g)" (Float_u.to_float a) (Float_u.to_float b)

let[@inline] of_string s =
  Scanf.sscanf s "(%g %g)" (fun a b ->
      set (Float_u.of_float a) (Float_u.of_float b) |> box)
  |> unbox

(* ───── Unboxed Array Load/Store ───── *)

module Array = struct
  external unsafe_get : (float# array[@local_opt]) @ read -> idx:int -> t
    = "%caml_unboxed_float_array_get128u#"

  external unsafe_set : (float# array[@local_opt]) -> idx:int -> t -> unit
    = "%caml_unboxed_float_array_set128u#"
end
