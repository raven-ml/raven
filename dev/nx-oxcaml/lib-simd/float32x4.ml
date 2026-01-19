(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

open Import

include Float32x4_internal

type mask = int32x4#

external box : t -> float32x4 @@ portable = "%box_vec128"
external unbox : float32x4 @ local -> t @@ portable = "%unbox_vec128"

(* ───── Constants ───── *)

external const1 : float32# -> t @@ portable
  = "caml_vec128_unreachable" "caml_float32x4_const1"
  [@@noalloc] [@@builtin]

external const : float32# -> float32# -> float32# -> float32# -> t @@ portable
  = "caml_vec128_unreachable" "caml_float32x4_const4"
  [@@noalloc] [@@builtin]

let[@inline] zero () = const1 #0.0s
let[@inline] one () = const1 #1.0s
let[@inline] sign32_mask () = Int32x4.const1 #0x80000000l
let[@inline] absf32_mask () = Int32x4.const1 #0x7fffffffl

external of_int32x4 : int32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

(* ───── Operators ───── *)

let[@inline] ( + ) x y = add x y
let[@inline] ( - ) x y = sub x y
let[@inline] ( * ) x y = mul x y
let[@inline] ( / ) x y = div x y
let[@inline] horizontal_add x y = hadd x y

let[@inline] is_nan t = Int32x4.bitwise_not (cmeq t t)
let[@inline] is_not_nan t = cmeq t t

let[@inline] ( >= ) x y = cmge x y
let[@inline] ( <= ) x y = cmle x y
let[@inline] ( = ) x y = cmeq x y
let[@inline] ( > ) x y = cmgt x y
let[@inline] ( < ) x y = cmlt x y
let[@inline] ( <> ) x y = Int32x4.bitwise_not (cmeq x y)
let[@inline] equal x y = cmeq x y

(* ───── Bitwise Operations ───── *)

(* Negation and absolute value operate on floats via int32 bit manipulation:
   - neg: XOR with sign bit mask flips the sign
   - abs: AND with inverse sign mask clears the sign bit *)

let[@inline] neg x =
  Int32x4.(bitwise_xor (sign32_mask ()) (of_float32x4 x)) |> of_int32x4

let[@inline] abs x =
  Int32x4.(bitwise_and (absf32_mask ()) (of_float32x4 x)) |> of_int32x4

(* ───── Lane Operations ───── *)

external low_of : float32# -> t @@ portable
  = "caml_vec128_unreachable" "caml_float32x4_low_of_float32"
  [@@noalloc] [@@builtin]

external low_to : t -> float32# @@ portable
  = "caml_vec128_unreachable" "caml_float32x4_low_to_float32"
  [@@noalloc] [@@builtin]

let[@inline] set1 a =
  let a = low_of a in
  Int32x4.dup (Int32x4.of_float32x4 a) |> of_int32x4

let[@inline] set a b c d =
  let a = Int32x4.of_float32x4 (low_of a) in
  let b = Int32x4.of_float32x4 (low_of b) in
  let c = Int32x4.of_float32x4 (low_of c) in
  let d = Int32x4.of_float32x4 (low_of d) in
  let ab = Int32x4.interleave_low_32 a b in
  let cd = Int32x4.interleave_low_32 c d in
  Int32x4.interleave_low_64 ab cd |> of_int32x4

let[@inline] extract ~idx x =
  match idx with
  | #0L -> low_to x
  | #1L ->
    let shuffled = Int32x4.dup_lane 1 (Int32x4.of_float32x4 x) |> of_int32x4 in
    low_to shuffled
  | #2L ->
    let shuffled = Int32x4.dup_lane 2 (Int32x4.of_float32x4 x) |> of_int32x4 in
    low_to shuffled
  | #3L ->
    let shuffled = Int32x4.dup_lane 3 (Int32x4.of_float32x4 x) |> of_int32x4 in
    low_to shuffled
  | _ -> assert false

let[@inline] insert ~idx t a =
  let a_vec = Int32x4.of_float32x4 (low_of a) in
  let a_bits = Int32x4.extract_neon 0 a_vec in
  let t_as_int = Int32x4.of_float32x4 t in
  let result =
    match idx with
    | #0L -> Int32x4.insert_neon 0 t_as_int a_bits
    | #1L -> Int32x4.insert_neon 1 t_as_int a_bits
    | #2L -> Int32x4.insert_neon 2 t_as_int a_bits
    | #3L -> Int32x4.insert_neon 3 t_as_int a_bits
    | _ -> assert false
  in
  of_int32x4 result

let[@inline] extract0 x = low_to x

let[@inline] splat x =
  #(low_to x, extract ~idx:#1L x, extract ~idx:#2L x, extract ~idx:#3L x)

let[@inline] movemask m = Int32x4.movemask m
let[@inline] bitmask m = m

let[@inline] round_nearest x = round_near x
let[@inline] iround_current x = cvt_int32x4 x

(* ───── Interleave ───── *)

let[@inline] interleave_upper ~even ~odd =
  Int32x4.interleave_high_32 (Int32x4.of_float32x4 even) (Int32x4.of_float32x4 odd)
  |> of_int32x4

let[@inline] interleave_lower ~even ~odd =
  Int32x4.interleave_low_32 (Int32x4.of_float32x4 even) (Int32x4.of_float32x4 odd)
  |> of_int32x4

let[@inline] duplicate_even x =
  Int32x4.duplicate_even (Int32x4.of_float32x4 x) |> of_int32x4

let[@inline] duplicate_odd x =
  Int32x4.duplicate_odd (Int32x4.of_float32x4 x) |> of_int32x4

external blend
  :  (Simd_types.Blend4.t[@untagged])
  -> t
  -> t
  -> t
  @@ portable
  = "caml_vec128_unreachable" "caml_sse41_vec128_blend_32"
  [@@noalloc] [@@builtin]

(* ───── Conversions ───── *)

external of_float64x2 : float64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int8x16 : int8x16# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int16x8 : int16x8# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]


external of_int64x2 : int64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

let[@inline] of_float64x2_bits x = of_float64x2 x
let[@inline] of_int8x16_bits x = of_int8x16 x
let[@inline] of_int16x8_bits x = of_int16x8 x
let[@inline] of_int32x4_bits x = of_int32x4 x
let[@inline] of_int64x2_bits x = of_int64x2 x
let[@inline] unsafe_of_float32 x = low_of x
let[@inline] of_int32x4 x = Int32x4.cvt_f32 x
let[@inline] of_float64x2 x = Float64x2.cvt_float32x4 x

(* ───── String Conversion ───── *)

let[@inline] to_string x =
  let f x = Float32.to_float (Float32_u.to_float32 x) in
  let #(a, b, c, d) = splat x in
  Printf.sprintf "(%.9g %.9g %.9g %.9g)" (f a) (f b) (f c) (f d)

let[@inline] of_string s =
  let f x = Float32_u.of_float32 (Float32.of_float x) in
  Scanf.sscanf s "(%g %g %g %g)" (fun a b c d -> set (f a) (f b) (f c) (f d) |> box)
  |> unbox

(* ───── Unboxed Array Load/Store ───── *)

module Array = struct
  external unsafe_get : (float32# array[@local_opt]) @ read -> idx:int -> t
    = "%caml_unboxed_float32_array_get128u#"

  external unsafe_set : (float32# array[@local_opt]) -> idx:int -> t -> unit
    = "%caml_unboxed_float32_array_set128u#"
end
