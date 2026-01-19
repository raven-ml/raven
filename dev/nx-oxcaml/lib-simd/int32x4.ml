(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

open Import

include Int32x4_internal

type mask = int32x4#

external box : t -> int32x4 @@ portable = "%box_vec128"
external unbox : int32x4 @ local -> t @@ portable = "%unbox_vec128"

(* ───── Constants ───── *)

external const1 : int32# -> t @@ portable
  = "caml_vec128_unreachable" "caml_int32x4_const1"
  [@@noalloc] [@@builtin]

external const : int32# -> int32# -> int32# -> int32# -> t @@ portable
  = "caml_vec128_unreachable" "caml_int32x4_const4"
  [@@noalloc] [@@builtin]

let[@inline] zero () = const1 #0l
let[@inline] one () = const1 #1l
let[@inline] all_ones () = const1 #0xffffffffl

(* ───── Operators ───── *)

let[@inline] ( + ) x y = add x y
let[@inline] ( - ) x y = sub x y
let[@inline] mul_low_bits x y = mul_low x y
let[@inline] horizontal_add x y = hadd x y

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

external interleave_high_32 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_high_32"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_32 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_32"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_64 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_64"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] interleave_upper ~even ~odd = interleave_high_32 even odd
let[@inline] interleave_lower ~even ~odd = interleave_low_32 even odd

external blend
  :  (Simd_types.Blend4.t[@untagged])
  -> t
  -> t
  -> t
  @@ portable
  = "caml_vec128_unreachable" "caml_sse41_vec128_blend_32"
  [@@noalloc] [@@builtin]

(* ───── Element Access ───── *)

external low_of : int32# -> t @@ portable
  = "caml_vec128_unreachable" "caml_int32x4_low_of_int32"
  [@@noalloc] [@@builtin]

external low_to : t -> int32# @@ portable
  = "caml_vec128_unreachable" "caml_int32x4_low_to_int32"
  [@@noalloc] [@@builtin]

let[@inline] set1 a =
  let a = low_of a in
  dup a

let[@inline] set a b c d =
  let a = low_of a in
  let b = low_of b in
  let c = low_of c in
  let d = low_of d in
  let ba = interleave_low_32 a b in
  let dc = interleave_low_32 c d in
  interleave_low_64 ba dc

let[@inline] extract0 x = low_to x
let[@inline] splat x = #(extract0 x, extract ~idx:#1L x, extract ~idx:#2L x, extract ~idx:#3L x)

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

external of_int64x2 : int64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

let[@inline] of_float32x4_bits x = of_float32x4 x
let[@inline] of_float64x2_bits x = of_float64x2 x
let[@inline] of_int8x16_bits x = of_int8x16 x
let[@inline] of_int16x8_bits x = of_int16x8 x
let[@inline] of_int64x2_bits x = of_int64x2 x

let[@inline] cvt_f64 (t : t) : float64x2# = Int64x2.cvt_float64x2 (cvtsx_i64 t)

(* ───── String Conversion ───── *)

let[@inline] to_string x =
  let f = Int32_u.to_int32 in
  let #(a, b, c, d) = splat x in
  Printf.sprintf "(%ld %ld %ld %ld)" (f a) (f b) (f c) (f d)

let[@inline] of_string s =
  let f = Int32_u.of_int32 in
  Scanf.sscanf s "(%ld %ld %ld %ld)" (fun a b c d ->
      set (f a) (f b) (f c) (f d) |> box)
  |> unbox

(* ───── Unboxed Array Load/Store ───── *)

module Array = struct
  external unsafe_get : (int32# array[@local_opt]) @ read -> idx:int -> t
    = "%caml_unboxed_int32_array_get128u#"

  external unsafe_set : (int32# array[@local_opt]) -> idx:int -> t -> unit
    = "%caml_unboxed_int32_array_set128u#"
end
