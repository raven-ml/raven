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
  = "caml_vec128_unreachable" "caml_neon_int64x2_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external neg : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_neg"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Bitwise ───── *)

external bitwise_or : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_or"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_and : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_and"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_xor : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_xor"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_not : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_not"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Comparison ───── *)

external cmpeq : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_cmpeq"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpgt : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_cmpgt"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpltz : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_cmpltz"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Shifts ───── *)

external slli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_slli"
  [@@noalloc] [@@builtin]

external srli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_srli"
  [@@noalloc] [@@builtin]

external ushl : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_ushl"
  [@@noalloc] [@@unboxed] [@@builtin]

external sshl : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_sshl"
  [@@noalloc] [@@unboxed] [@@builtin]

external dup : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_dup"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] shift_left_logical x i = ushl x (dup i)
let[@inline] shift_right_logical x i = ushl x (dup (neg i))
let[@inline] shift_right_arithmetic x i = sshl x (dup (neg i))

(* ───── Lane Operations ───── *)

external extract_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int64[@unboxed])
  @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_extract"
  [@@noalloc] [@@builtin]

external insert_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int64[@unboxed])
  -> (t[@unboxed])
  @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_insert"
  [@@noalloc] [@@builtin]

external dup_lane : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int64x2_dup_lane"
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

(* ───── Byte Shifts ───── *)

external shift_left_bytes
  :  (int[@untagged])
  -> (int8x16[@unboxed])
  -> (int8x16[@unboxed])
  @@ portable
  = "caml_vec128_unreachable" "caml_neon_vec128_shift_left_bytes"
  [@@noalloc] [@@builtin]

external shift_right_bytes
  :  (int[@untagged])
  -> (int8x16[@unboxed])
  -> (int8x16[@unboxed])
  @@ portable
  = "caml_vec128_unreachable" "caml_neon_vec128_shift_right_bytes"
  [@@noalloc] [@@builtin]

(* ───── Conversions ───── *)

external cvt_float64x2 : t -> float64x2# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_int64x2_to_float64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_int32x4 : t -> int32x4# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_int64x2_to_int32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_int32x4_saturating : t -> int32x4# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_int64x2_to_int32x4_low_saturating"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Blend ───── *)

(* blendv: select lanes from b where mask is set, else from a
   Emulated via: (b AND mask) OR (a AND NOT mask) *)
let[@inline] blendv a b mask =
  let selected_b = bitwise_and b mask in
  let not_mask = bitwise_not mask in
  let selected_a = bitwise_and a not_mask in
  bitwise_or selected_a selected_b

(* ───── Movemask Emulation ───── *)

(* NEON lacks native movemask; emulate by testing sign bits via cmpltz *)
let[@inline] movemask (t : t) =
  let mask = cmpltz t in
  let res = 0L in
  let lane_mask = Int64.logand (extract_neon 0 mask) Int64.one in
  let res = Int64.logor res (Int64.shift_left lane_mask 0) in
  let lane_mask = Int64.logand (extract_neon 1 mask) Int64.one in
  let res = Int64.logor res (Int64.shift_left lane_mask 1) in
  Int64.to_int res
