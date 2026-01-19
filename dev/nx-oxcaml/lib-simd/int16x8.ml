(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

include Int16x8_internal

type mask = int16x8#

external box : t -> int16x8 @@ portable = "%box_vec128"
external unbox : int16x8 @ local -> t @@ portable = "%unbox_vec128"

(* ───── Constants ───── *)

let[@inline always] zero () = const1 #0S
let[@inline always] one () = const1 #1S
let[@inline always] all_ones () = const1 #0xFFFFS

(* ───── Operators ───── *)

let[@inline] ( + ) x y = add x y
let[@inline] ( - ) x y = sub x y
let[@inline] mul_low_bits x y = mul_low x y
let[@inline] horizontal_add x y = horizontal_add x y

let[@inline] ( lor ) x y = or_ x y
let[@inline] ( land ) x y = and_ x y
let[@inline] ( lxor ) x y = xor x y
let[@inline] lnot m = xor (all_ones ()) m
let[@inline] landnot ~not y = andnot ~not y

let[@inline] ( >= ) x y = or_ (cmpgt x y) (cmpeq x y)
let[@inline] ( <= ) x y = or_ (cmpgt y x) (cmpeq x y)
let[@inline] ( = ) x y = cmpeq x y
let[@inline] ( > ) x y = cmpgt x y
let[@inline] ( < ) x y = cmpgt y x
let[@inline] ( <> ) x y = xor (all_ones ()) (cmpeq x y)
let[@inline] equal x y = cmpeq x y

(* ───── Interleave ───── *)

let[@inline] interleave_upper ~even ~odd = interleave_high_16 even odd
let[@inline] interleave_lower ~even ~odd = interleave_low_16 even odd

(* ───── Element Access ───── *)

let[@inline] set1 a =
  let a = low_of a in
  dup a

let[@inline] extract0 x = low_to x

(* ───── Conversions ───── *)

external of_float32x4 : float32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float64x2 : float64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

let[@inline] of_float32x4_bits x = of_float32x4 x
let[@inline] of_float64x2_bits x = of_float64x2 x
let[@inline] of_int8x16_bits x = of_int8x16 x
let[@inline] of_int32x4_bits x = of_int32x4 x
let[@inline] of_int64x2_bits x = of_int64x2 x

