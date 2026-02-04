(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

type t = float64x2#

(* ───── Arithmetic ───── *)

external add : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_mul"
  [@@noalloc] [@@unboxed] [@@builtin]

external div : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_div"
  [@@noalloc] [@@unboxed] [@@builtin]

external sqrt : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_sqrt"
  [@@noalloc] [@@unboxed] [@@builtin]

external hadd : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse3_float64x2_hadd"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Min/Max ───── *)

external min : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_min"
  [@@noalloc] [@@unboxed] [@@builtin]

external max : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_max"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Comparison ───── *)

external cmeq : (t[@unboxed]) -> (t[@unboxed]) -> (int64x2#[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_cmpeq"
  [@@noalloc] [@@builtin]

external cmge : (t[@unboxed]) -> (t[@unboxed]) -> (int64x2#[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_cmpge"
  [@@noalloc] [@@builtin]

external cmgt : (t[@unboxed]) -> (t[@unboxed]) -> (int64x2#[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_cmpgt"
  [@@noalloc] [@@builtin]

external cmle : (t[@unboxed]) -> (t[@unboxed]) -> (int64x2#[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_cmple"
  [@@noalloc] [@@builtin]

external cmlt : (t[@unboxed]) -> (t[@unboxed]) -> (int64x2#[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_float64x2_cmplt"
  [@@noalloc] [@@builtin]

(* ───── FMA ───── *)

external mul_add : (t[@unboxed]) -> (t[@unboxed]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_fma_float64x2_fmadd"
  [@@noalloc]

(* ───── Rounding ───── *)

external round_near : (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse41_float64x2_round_near"
  [@@noalloc] [@@builtin]

external round_current : (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse41_float64x2_round_current"
  [@@noalloc] [@@builtin]

(* ───── Conversions ───── *)

external cvt_int64x2 : t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_avx512_cvt_float64x2_to_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtt_int64x2 : t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_avx512_cvtt_float64x2_to_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_float32x4 : t -> float32x4# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_cvt_float64x2_to_float32x2"
  [@@noalloc] [@@unboxed] [@@builtin]
