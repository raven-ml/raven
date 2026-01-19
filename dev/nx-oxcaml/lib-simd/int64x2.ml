(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

open Import

include Int64x2_internal

type mask = int64x2#

external box : t -> int64x2 @@ portable = "%box_vec128"
external unbox : int64x2 @ local -> t @@ portable = "%unbox_vec128"

(* ───── Constants ───── *)

external const1 : int64# -> t @@ portable
  = "caml_vec128_unreachable" "caml_int64x2_const1"
  [@@noalloc] [@@builtin]

external const : int64# -> int64# -> t @@ portable
  = "caml_vec128_unreachable" "caml_int64x2_const2"
  [@@noalloc] [@@builtin]

let[@inline] zero () = const1 #0L
let[@inline] one () = const1 #1L
let[@inline] all_ones () = const1 #0xffffffffffffffffL

(* ───── Operators ───── *)

let[@inline] ( + ) x y = add x y
let[@inline] ( - ) x y = sub x y
let[@inline] ( lor ) x y = bitwise_or x y
let[@inline] ( land ) x y = bitwise_and x y
let[@inline] ( lxor ) x y = bitwise_xor x y
let[@inline] lnot m = bitwise_xor (all_ones ()) m
let[@inline] landnot ~not y = bitwise_and (bitwise_not not) y

let[@inline] ( >= ) x y = bitwise_or (cmpgt x y) (cmpeq x y)
let[@inline] ( <= ) x y = bitwise_or (cmpgt y x) (cmpeq x y)
let[@inline] ( = ) x y = cmpeq x y
let[@inline] ( > ) x y = cmpgt x y
let[@inline] ( < ) x y = cmpgt y x
let[@inline] ( <> ) x y = bitwise_xor (all_ones ()) (cmpeq x y)
let[@inline] equal x y = cmpeq x y

(* ───── Interleave And Shuffle ───── *)

external interleave_high_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_high_64"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_64"
  [@@noalloc] [@@unboxed] [@@builtin]

external high_64_to_low_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_high_64_to_low_64"
  [@@noalloc] [@@unboxed] [@@builtin]

external low_64_to_high_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_low_64_to_high_64"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] interleave_upper ~even ~odd = interleave_high_64 even odd
let[@inline] interleave_lower ~even ~odd = interleave_low_64 even odd

external blend
  :  (Simd_types.Blend2.t[@untagged])
  -> t
  -> t
  -> t
  @@ portable
  = "caml_vec128_unreachable" "caml_sse41_vec128_blend_64"
  [@@noalloc] [@@builtin]

(* ───── Element Access ───── *)

external low_of : int64# -> t @@ portable
  = "caml_vec128_unreachable" "caml_int64x2_low_of_int64"
  [@@noalloc] [@@builtin]

external low_to : t -> int64# @@ portable
  = "caml_vec128_unreachable" "caml_int64x2_low_to_int64"
  [@@noalloc] [@@builtin]

let[@inline] set1 a = dup (low_of a)
let[@inline] set a b = low_64_to_high_64 (low_of a) (low_of b)

let[@inline] extract0 x = low_to x
let[@inline] splat x = #(low_to x, extract ~idx:#1L x)

(* ───── Conversions ───── *)

external of_float32x4 : float32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float64x2 : float64x2# -> t @@ portable
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

let[@inline] of_float32x4_bits x = of_float32x4 x
let[@inline] of_float64x2_bits x = of_float64x2 x
let[@inline] of_int8x16_bits x = of_int8x16 x
let[@inline] of_int16x8_bits x = of_int16x8 x
let[@inline] of_int32x4_bits x = of_int32x4 x

(* ───── String Conversion ───── *)

let[@inline] to_string x =
  let #(a, b) = splat x in
  Printf.sprintf "(%Ld %Ld)" (Int64_u.to_int64 a) (Int64_u.to_int64 b)

let[@inline] of_string s =
  Scanf.sscanf s "(%Ld %Ld)" (fun a b ->
      set (Int64_u.of_int64 a) (Int64_u.of_int64 b) |> box)
  |> unbox

(* ───── Unboxed Array Load/Store ───── *)

module Array = struct
  external unsafe_get : (int64# array[@local_opt]) @ read -> idx:int -> t
    = "%caml_unboxed_int64_array_get128u#"

  external unsafe_set : (int64# array[@local_opt]) -> idx:int -> t -> unit
    = "%caml_unboxed_int64_array_set128u#"
end
