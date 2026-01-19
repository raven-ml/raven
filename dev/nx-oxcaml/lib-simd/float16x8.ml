(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

include Float16x8_internal

external box : t -> float16x8 @@ portable = "%box_vec128"
external unbox : float16x8 @ local -> t @@ portable = "%unbox_vec128"

(* ───── Interleave ───── *)

let[@inline] interleave_upper ~even ~odd = interleave_high_16 even odd
let[@inline] interleave_lower ~even ~odd = interleave_low_16 even odd

(* ───── Conversions ───── *)

let[@inline] of_float32x4_bits x = of_float32x4 x
let[@inline] of_float64x2_bits x = of_float64x2 x
let[@inline] of_int8x16_bits x = of_int8x16 x
let[@inline] of_int16x8_bits x = of_int16x8 x
let[@inline] of_int32x4_bits x = of_int32x4 x
let[@inline] of_int64x2_bits x = of_int64x2 x
