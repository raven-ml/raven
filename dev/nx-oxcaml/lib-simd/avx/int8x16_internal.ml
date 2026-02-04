(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

type t = int8x16#

(* ───── Constants and Lane Ops ───── *)

external low_of : int8# -> t @@ portable
  = "caml_sse2_unreachable" "caml_int8x16_low_of_int8"
  [@@noalloc] [@@builtin]

external low_to : t -> int8# @@ portable
  = "caml_sse2_unreachable" "caml_int8x16_low_to_int8"
  [@@noalloc] [@@builtin]

external const1 : int8# -> t @@ portable
  = "caml_sse2_unreachable" "caml_int8x16_const1"
  [@@noalloc] [@@builtin]

external dup : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_dup"
  [@@noalloc] [@@unboxed] [@@builtin]

external insert : (int[@untagged]) -> (t[@unboxed]) -> (int[@untagged]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int8x16_insert"
  [@@noalloc] [@@builtin]

external extract : (int[@untagged]) -> (t[@unboxed]) -> (int[@untagged]) @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int8x16_extract"
  [@@noalloc] [@@builtin]

(* ───── Casts ───── *)

external of_int16x8 : int16x8# -> t @@ portable
  = "caml_sse2_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int32x4 : int32x4# -> t @@ portable
  = "caml_sse2_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_int64x2 : int64x2# -> t @@ portable
  = "caml_sse2_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float16x8 : float16x8# -> t @@ portable
  = "caml_sse2_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float32x4 : float32x4# -> t @@ portable
  = "caml_sse2_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

external of_float64x2 : float64x2# -> t @@ portable
  = "caml_sse2_unreachable" "caml_vec128_cast"
  [@@noalloc] [@@builtin]

(* ───── Arithmetic ───── *)

external add : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external add_saturating : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_add_saturating"
  [@@noalloc] [@@unboxed] [@@builtin]

external add_saturating_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_add_saturating_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub_saturating : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_sub_saturating"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub_saturating_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_sub_saturating_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

external abs : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_ssse3_int8x16_abs"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] neg v = sub (sub v v) v

(* ───── Multiply with Horizontal Add ───── *)

external mul_horizontal_add_saturating : t -> t -> int16x8# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_mul_unsigned_hadd_saturating_int16x8"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Min/Max ───── *)

external max : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int8x16_max"
  [@@noalloc] [@@unboxed] [@@builtin]

external min : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int8x16_min"
  [@@noalloc] [@@unboxed] [@@builtin]

external max_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_max_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

external min_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_min_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Comparison ───── *)

external cmpeq : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_cmpeq"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpgt : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_cmpgt"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] cmpltz v = cmpgt (sub v v) v

(* ───── Bitwise ───── *)

external and_ : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_and"
  [@@noalloc] [@@unboxed] [@@builtin]

external or_ : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_or"
  [@@noalloc] [@@unboxed] [@@builtin]

external xor : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_xor"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] bitwise_not v = xor v (cmpeq v v)

let[@inline] andnot ~not:mask t = and_ (bitwise_not mask) t

(* ───── Movemask ───── *)

external movemask_8 : t -> int64# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_movemask"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Interleave ───── *)

external interleave_high_8 : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_simd_vec128_interleave_high_8"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_8 : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_simd_vec128_interleave_low_8"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_16 : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_simd_vec128_interleave_low_16"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_32 : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_simd_vec128_interleave_low_32"
  [@@noalloc] [@@unboxed] [@@builtin]

external interleave_low_64 : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_simd_vec128_interleave_low_64"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── mul_sign ───── *)

let[@inline] mul_sign a b =
  let sign_mask = cmpltz b in
  let negated = neg a in
  let zero_mask = cmpeq b (sub b b) in
  let result = or_ (and_ sign_mask negated) (and_ (bitwise_not sign_mask) a) in
  and_ (bitwise_not zero_mask) result

(* ───── Average ───── *)

external avg_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_avg_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── SAD (Sum of Absolute Differences) ───── *)

external sadu : t -> t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int8x16_sad_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Shuffle ───── *)

external shuffle_8 : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_ssse3_vec128_shuffle_8"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Blend ───── *)

let[@inline] blendv_8 a b mask =
  let selected_b = and_ b mask in
  let not_mask = bitwise_not mask in
  let selected_a = and_ a not_mask in
  or_ selected_a selected_b

(* ───── Conversions ───── *)

external cvtsx_i16 : t -> int16x8# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtsx_int8x16_int16x8"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtzx_i16 : t -> int16x8# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtzx_int8x16_int16x8"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtsx_i32 : t -> int32x4# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtsx_int8x16_int32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtzx_i32 : t -> int32x4# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtzx_int8x16_int32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtsx_i64 : t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtsx_int8x16_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtzx_i64 : t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtzx_int8x16_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]
