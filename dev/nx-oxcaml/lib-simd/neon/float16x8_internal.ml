(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

type t = float16x8#

(* ───── Casts ───── *)

external of_int8x16 : int8x16# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int16x8 : int16x8# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int32x4 : int32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int64x2 : int64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float32x4 : float32x4# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float64x2 : float64x2# -> t @@ portable
  = "caml_vec128_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

(* ───── Interleave ───── *)

external interleave_high_16 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_high_16"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_16 : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_16"
  [@@noalloc] [@@unboxed] [@@builtin]
