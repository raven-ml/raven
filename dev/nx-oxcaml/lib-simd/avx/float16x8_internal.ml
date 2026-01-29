(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

  include Ocaml_simd_sse.Float16x8

external of_float16x16
  :  float16x16#
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_vec256_low_to_vec128"
[@@noalloc] [@@builtin]

(* These intrinsics expect [0x0..0x7], but [%float_round] generates [0x8..0xC]. *)
module Round = struct
  type t = int64#

  let[@inline] nearest () = #0b000L
  let[@inline] current () = #0b100L
  let[@inline] down () = #0b001L
  let[@inline] up () = #0b010L
  let[@inline] toward_zero () = #0b011L
end

external of_float32x4
  :  Round.t
  -> float32x4#
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_f16c_cvt_float32x4_float16x8"
[@@noalloc] [@@builtin]

external of_float32x8
  :  Round.t
  -> float32x8#
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_f16c_cvt_float32x8_float16x8"
[@@noalloc] [@@builtin]

let[@inline] of_float32x4_nearest x = of_float32x4 (Round.nearest ()) x
let[@inline] of_float32x4_current x = of_float32x4 (Round.current ()) x
let[@inline] of_float32x4_down x = of_float32x4 (Round.down ()) x
let[@inline] of_float32x4_up x = of_float32x4 (Round.up ()) x
let[@inline] of_float32x4_toward_zero x = of_float32x4 (Round.toward_zero ()) x
let[@inline] of_float32x8_nearest x = of_float32x8 (Round.nearest ()) x
let[@inline] of_float32x8_current x = of_float32x8 (Round.current ()) x
let[@inline] of_float32x8_down x = of_float32x8 (Round.down ()) x
let[@inline] of_float32x8_up x = of_float32x8 (Round.up ()) x
let[@inline] of_float32x8_toward_zero x = of_float32x8 (Round.toward_zero ()) x