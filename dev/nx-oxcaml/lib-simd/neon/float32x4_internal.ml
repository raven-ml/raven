(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

type t = float32x4#

(* ───── Arithmetic ───── *)

external add : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_mul"
  [@@noalloc] [@@unboxed] [@@builtin]

external div : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_div"
  [@@noalloc] [@@unboxed] [@@builtin]

external sqrt : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_sqrt"
  [@@noalloc] [@@unboxed] [@@builtin]

external rcp : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_rcp"
  [@@noalloc] [@@unboxed] [@@builtin]

external rsqrt : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_rsqrt"
  [@@noalloc] [@@unboxed] [@@builtin]

external hadd : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_hadd"
  [@@noalloc] [@@unboxed] [@@builtin]

  external mul_add : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_fma"
  [@@noalloc] [@@unboxed] [@@builtin]

(* hsub, addsub, dp: Not available on ARM64 NEON.
   These are SSE3/SSE4.1 specific instructions. *)

(* ───── Min/Max ───── *)

external min : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_min"
  [@@noalloc] [@@unboxed] [@@builtin]

external max : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_max"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Comparison ───── *)

external cmeq : (t[@unboxed]) -> (t[@unboxed]) -> (int32x4#[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_cmeq"
  [@@noalloc] [@@builtin]

external cmge : (t[@unboxed]) -> (t[@unboxed]) -> (int32x4#[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_cmge"
  [@@noalloc] [@@builtin]

external cmgt : (t[@unboxed]) -> (t[@unboxed]) -> (int32x4#[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_cmgt"
  [@@noalloc] [@@builtin]

external cmle : (t[@unboxed]) -> (t[@unboxed]) -> (int32x4#[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_cmle"
  [@@noalloc] [@@builtin]

external cmlt : (t[@unboxed]) -> (t[@unboxed]) -> (int32x4#[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_cmlt"
  [@@noalloc] [@@builtin]

(* ───── Rounding ───── *)

external round_near : (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_float32x4_round_near"
  [@@noalloc] [@@builtin]

(* ───── Conversions ───── *)

external cvt_int32x4 : t -> int32x4# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_float32x4_to_int32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtt_int32x4 : t -> int32x4# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvtt_float32x4_to_int32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_float64x2 : t -> float64x2# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_float32x2_to_float64x2"
  [@@noalloc] [@@unboxed] [@@builtin]
