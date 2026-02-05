(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

type t = int32x4#

(* ───── Arithmetic ───── *)

external add : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external neg : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_neg"
  [@@noalloc] [@@unboxed] [@@builtin]

external abs : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_ssse3_int32x4_abs"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul_low : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_mul_low"
  [@@noalloc] [@@unboxed] [@@builtin]

external hadd : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_ssse3_int32x4_hadd"
  [@@noalloc] [@@unboxed] [@@builtin]

external hsub : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_ssse3_int32x4_hsub"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul_even : t -> t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_mul_even"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul_even_unsigned : t -> t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_mul_even_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Min/Max ───── *)

external min : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_min"
  [@@noalloc] [@@unboxed] [@@builtin]

external max : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_max"
  [@@noalloc] [@@unboxed] [@@builtin]

external min_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_min_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

external max_unsigned : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_max_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Bitwise ───── *)

external bitwise_or : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_or"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_and : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_and"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_xor : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_xor"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_not : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_not"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_andnot : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_vec128_andnot"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Comparison ───── *)

external cmpeq : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_cmpeq"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpgt : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_cmpgt"
  [@@noalloc] [@@unboxed] [@@builtin]

(* cmpltz emulated: compare against zero vector *)
let[@inline] cmpltz x = cmpgt (sub x x) x

(* mul_sign: result[i] = a[i] * sign(b[i]) *)
let[@inline] mul_sign a b =
  let sign_mask = cmpltz b in
  let negated = neg a in
  let zero_mask = cmpeq b (sub b b) in
  let result = bitwise_or (bitwise_and sign_mask negated) (bitwise_and (bitwise_not sign_mask) a) in
  bitwise_and (bitwise_not zero_mask) result

(* ───── Shifts ───── *)

external slli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_slli"
  [@@noalloc] [@@builtin]

external srli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_srli"
  [@@noalloc] [@@builtin]

external srai : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_srai"
  [@@noalloc] [@@builtin]

external sll : t -> int64x2# -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_sll"
  [@@noalloc] [@@unboxed] [@@builtin]

external srl : t -> int64x2# -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_srl"
  [@@noalloc] [@@unboxed] [@@builtin]

external sra : t -> int64x2# -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_sra"
  [@@noalloc] [@@unboxed] [@@builtin]

external dup : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_shuffle_0000"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] shift_left_logical x i = slli i x
let[@inline] shift_right_logical x i = srli i x
let[@inline] shift_right_arithmetic x i = srai i x

(* ───── Lane Operations ───── *)

external extract_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int32[@unboxed])
  @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_extract"
  [@@noalloc] [@@builtin]

external insert_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int32[@unboxed])
  -> (t[@unboxed])
  @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int32x4_insert"
  [@@noalloc] [@@builtin]

external dup_lane : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int32x4_dup_lane"
  [@@noalloc] [@@builtin]

external int32_to_int32u : int32 -> int32# @@ portable = "%unbox_int32"
external int32u_to_int32 : int32# -> int32 @@ portable = "%box_int32"

let[@inline] insert ~idx t x =
  match idx with
  | #0L -> insert_neon 0 t (int32u_to_int32 x)
  | #1L -> insert_neon 1 t (int32u_to_int32 x)
  | #2L -> insert_neon 2 t (int32u_to_int32 x)
  | #3L -> insert_neon 3 t (int32u_to_int32 x)
  | _ -> assert false

let[@inline] extract ~idx t =
  match idx with
  | #0L -> int32_to_int32u (extract_neon 0 t)
  | #1L -> int32_to_int32u (extract_neon 1 t)
  | #2L -> int32_to_int32u (extract_neon 2 t)
  | #3L -> int32_to_int32u (extract_neon 3 t)
  | _ -> assert false

external duplicate_odd : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse3_vec128_dup_odd_32"
  [@@noalloc] [@@unboxed] [@@builtin]

external duplicate_even : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse3_vec128_dup_even_32"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Conversions ───── *)

external cvt_f32 : t -> float32x4# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_cvt_int32x4_to_float32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtsx_i64 : t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtsx_int32x4_to_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtzx_i64 : t -> int64x2# @@ portable
  = "caml_sse2_unreachable" "caml_sse41_cvtzx_int32x4_to_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_si16 : t -> t -> int16x8# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_cvt_int32x4_int16x8_saturating"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_su16 : t -> t -> int16x8# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_cvt_int32x4_int16x8_saturating_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Blend ───── *)

(* blendv: select lanes from b where mask is set, else from a *)
let[@inline] blendv a b mask =
  let selected_b = bitwise_and b mask in
  let not_mask = bitwise_not mask in
  let selected_a = bitwise_and a not_mask in
  bitwise_or selected_a selected_b

(* ───── Movemask ───── *)

external movemask_native : t -> int @@ portable
  = "caml_sse2_unreachable" "caml_sse_vec128_movemask_32"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] movemask t = movemask_native t
