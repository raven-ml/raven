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
  = "caml_vec128_unreachable" "caml_neon_int32x4_add"
  [@@noalloc] [@@unboxed] [@@builtin]

external sub : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_sub"
  [@@noalloc] [@@unboxed] [@@builtin]

external neg : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_neg"
  [@@noalloc] [@@unboxed] [@@builtin]

external abs : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_abs"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul_low : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_mul_low"
  [@@noalloc] [@@unboxed] [@@builtin]

external hadd : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_hadd"
  [@@noalloc] [@@unboxed] [@@builtin]

external hsub : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_hsub"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul_even : t -> t -> int64x2# @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_mul_even"
  [@@noalloc] [@@unboxed] [@@builtin]

external mul_even_unsigned : t -> t -> int64x2# @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_mul_even_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Min/Max ───── *)

external min : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_min"
  [@@noalloc] [@@unboxed] [@@builtin]

external max : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_max"
  [@@noalloc] [@@unboxed] [@@builtin]

external min_unsigned : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_min_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

external max_unsigned : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_max_unsigned"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Bitwise ───── *)

external bitwise_or : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_or"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_and : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_and"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_xor : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_xor"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_not : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_not"
  [@@noalloc] [@@unboxed] [@@builtin]

external bitwise_andnot : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_andnot"
  [@@noalloc] [@@unboxed] [@@builtin]

(* ───── Comparison ───── *)

external cmpeq : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_cmpeq"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpgt : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_cmpgt"
  [@@noalloc] [@@unboxed] [@@builtin]

external cmpltz : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_cmpltz"
  [@@noalloc] [@@unboxed] [@@builtin]

(* mul_sign: result[i] = a[i] * sign(b[i]) *)
let[@inline] mul_sign a b =
  let sign_mask = cmpltz b in
  let negated = neg a in
  let zero_mask = cmpeq b (sub b b) in
  let result = bitwise_or (bitwise_and sign_mask negated) (bitwise_and (bitwise_not sign_mask) a) in
  bitwise_and (bitwise_not zero_mask) result

(* ───── Shifts ───── *)

external slli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_slli"
  [@@noalloc] [@@builtin]

external srli : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_srli"
  [@@noalloc] [@@builtin]

external srai : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_srai"
  [@@noalloc] [@@builtin]

external ushl : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_ushl"
  [@@noalloc] [@@unboxed] [@@builtin]

external sshl : t -> t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_sshl"
  [@@noalloc] [@@unboxed] [@@builtin]

external dup : t -> t @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_dup"
  [@@noalloc] [@@unboxed] [@@builtin]

let[@inline] shift_left_logical x i = ushl x (dup i)
let[@inline] shift_right_logical x i = ushl x (neg (dup i))
let[@inline] shift_right_arithmetic x i = sshl x (neg (dup i))

(* ───── Lane Operations ───── *)

external extract_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int32[@unboxed])
  @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_extract"
  [@@noalloc] [@@builtin]

external insert_neon
  :  (int[@untagged])
  -> (t[@unboxed])
  -> (int32[@unboxed])
  -> (t[@unboxed])
  @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_insert"
  [@@noalloc] [@@builtin]

external dup_lane : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
  = "caml_vec128_unreachable" "caml_neon_int32x4_dup_lane"
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

(* NEON lacks native duplicate_even/odd, emulate via extract/insert *)
let[@inline] duplicate_odd a =
  let res = a in
  let lane = extract_neon 1 a in
  let res = insert_neon 0 res lane in
  let lane = extract_neon 3 a in
  insert_neon 2 res lane

let[@inline] duplicate_even a =
  let res = a in
  let lane = extract_neon 0 a in
  let res = insert_neon 1 res lane in
  let lane = extract_neon 2 a in
  insert_neon 3 res lane

(* NEON lacks native shuffle_32; emulate via dup_lane/extract/insert.
   Match arms are required because dup_lane needs compile-time immediates. *)
let[@inline] shuffle (ctrl : int) a b =
  let ctrl0 = Stdlib.( land ) ctrl 3 in
  let ctrl1 = Stdlib.( land ) (Stdlib.( lsr ) ctrl 2) 3 in
  let ctrl2 = Stdlib.( land ) (Stdlib.( lsr ) ctrl 4) 3 in
  let ctrl3 = Stdlib.( land ) (Stdlib.( lsr ) ctrl 6) 3 in
  let dup_lane_a lane =
    match lane with 0 -> dup_lane 0 a | 1 -> dup_lane 1 a | 2 -> dup_lane 2 a | 3 -> dup_lane 3 a | _ -> a
  in
  let extract_a lane =
    match lane with 0 -> extract_neon 0 a | 1 -> extract_neon 1 a | 2 -> extract_neon 2 a | 3 -> extract_neon 3 a | _ -> extract_neon 0 a
  in
  let extract_b lane =
    match lane with 0 -> extract_neon 0 b | 1 -> extract_neon 1 b | 2 -> extract_neon 2 b | 3 -> extract_neon 3 b | _ -> extract_neon 0 b
  in
  let res = dup_lane_a ctrl0 in
  let dst1 = extract_a ctrl1 in
  let dst2 = extract_b ctrl2 in
  let dst3 = extract_b ctrl3 in
  let res = insert_neon 1 res dst1 in
  let res = insert_neon 2 res dst2 in
  insert_neon 3 res dst3

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

external cvt_f32 : t -> float32x4# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_int32x4_to_float32x4"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtsx_i64 : t -> int64x2# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvtsx_int32x4_to_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvtzx_i64 : t -> int64x2# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvtzx_int32x4_to_int64x2"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_si16 : t -> t -> int16x8# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_int32x4_int16x8_saturating"
  [@@noalloc] [@@unboxed] [@@builtin]

external cvt_su16 : t -> t -> int16x8# @@ portable
  = "caml_vec128_unreachable" "caml_neon_cvt_int32x4_int16x8_saturating_unsigned"
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
  let res = 0l in
  let lane_mask = Int32.logand (extract_neon 0 mask) Int32.one in
  let res = Int32.logor res (Int32.shift_left lane_mask 0) in
  let lane_mask = Int32.logand (extract_neon 1 mask) Int32.one in
  let res = Int32.logor res (Int32.shift_left lane_mask 1) in
  let lane_mask = Int32.logand (extract_neon 2 mask) Int32.one in
  let res = Int32.logor res (Int32.shift_left lane_mask 2) in
  let lane_mask = Int32.logand (extract_neon 3 mask) Int32.one in
  let res = Int32.logor res (Int32.shift_left lane_mask 3) in
  Int32.to_int res
