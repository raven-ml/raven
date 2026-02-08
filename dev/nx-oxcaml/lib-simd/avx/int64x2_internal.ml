(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

type t = int64x2#

(* ───── Arithmetic ───── *)

external add : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external neg : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_neg"
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

(* ───── Comparison ───── *)

external cmpeq : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int64x2_cmpeq"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpgt : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse42_int64x2_cmpgt"
  [@@noalloc] [@@unboxed] [@@builtin]

(* cmpltz emulated: compare against zero vector *)
let[@inline] cmpltz x = cmpgt (sub x x) x

(* ───── Shifts ───── *)

external slli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_slli"
  [@@noalloc] [@@builtin]

external srli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_srli"
  [@@noalloc] [@@builtin]

external sll : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_sll"
  [@@noalloc] [@@unboxed] [@@builtin]

external srl : t -> t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_srl"
  [@@noalloc] [@@unboxed] [@@builtin]

external dup : t -> t @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_dup"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] shift_left_logical x i = sll x (dup i)
let[@inline] shift_right_logical x i = srl x (dup i)
let[@inline] shift_right_arithmetic x i = srl x (dup (neg i))

(* ───── Lane Operations ───── *)

external extract_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int64[@unboxed])
  @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int64x2_extract"
  [@@noalloc] [@@builtin]

external insert_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int64[@unboxed])
  -> (t[@unboxed])
  @@ portable
  = "caml_sse2_unreachable" "caml_sse41_int64x2_insert"
  [@@noalloc] [@@builtin]

external dup_lane : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_sse2_unreachable" "caml_sse2_int64x2_dup_lane"
  [@@noalloc] [@@builtin]

external int64_to_int64u : int64 -> int64# @@ portable = "%unbox_int64"
external int64u_to_int64 : int64# -> int64 @@ portable = "%box_int64"

let[@inline] insert ~idx t x =
  match idx with
  | #0L -> insert_neon 0 t (int64u_to_int64 x)
  | #1L -> insert_neon 1 t (int64u_to_int64 x)
  | _ -> assert false

let[@inline] extract ~idx t =
  match idx with
  | #0L -> int64_to_int64u (extract_neon 0 t)
  | #1L -> int64_to_int64u (extract_neon 1 t)
  | _ -> assert false

(* ───── Conversions ───── *)

external cvt_float64x2 : t -> float64x2# @@ portable
  = "caml_sse2_unreachable" "caml_avx512_cvt_int64x2_to_float64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_int32x4 : t -> int32x4# @@ portable
  = "caml_sse2_unreachable" "caml_sse2_cvt_int64x2_to_int32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_int32x4_saturating : t -> int32x4# @@ portable
  = "caml_sse2_unreachable" "caml_avx512_cvt_int64x2_to_int32x4_saturating"
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
  = "caml_sse2_unreachable" "caml_sse2_vec128_movemask_64"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] movemask t = movemask_native t
