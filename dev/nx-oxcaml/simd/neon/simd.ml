(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

(* ARM64 NEON SIMD — Prefetch, Int64x2, Int32x4, Float64x2, Float32x4 *)

module Prefetch = struct
  external read : 'a -> (int[@untagged]) -> unit
    = "caml_prefetch_ignore" "caml_prefetch_read_high_val_offset_untagged"
    [@@noalloc] [@@builtin]
end

module Int64x2 = struct
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

  external bitwise_and : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_and"
    [@@noalloc] [@@unboxed] [@@builtin]

  external bitwise_or : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_or"
    [@@noalloc] [@@unboxed] [@@builtin]

  external bitwise_xor : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_xor"
    [@@noalloc] [@@unboxed] [@@builtin]

  external bitwise_not : t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int64x2_bitwise_not"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] ( land ) x y = bitwise_and x y
  let[@inline] ( lor ) x y = bitwise_or x y
  let[@inline] ( lxor ) x y = bitwise_xor x y

  (* ───── Comparison ───── *)

  external cmpgt : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int64x2_cmpgt"
    [@@noalloc] [@@unboxed] [@@builtin]

  (* ───── Blend ───── *)

  let[@inline] blendv a b mask =
    bitwise_or (bitwise_and b mask) (bitwise_and a (bitwise_not mask))

  (* ───── Constants ───── *)

  external const1 : int64# -> t @@ portable
    = "caml_vec128_unreachable" "caml_int64x2_const1"
    [@@noalloc] [@@builtin]

  let[@inline] zero () = const1 #0L
  let[@inline] one () = const1 #1L
  let[@inline] all_ones () = const1 #0xffffffffffffffffL

  (* ───── Lanes ───── *)

  external low_of : int64# -> t @@ portable
    = "caml_vec128_unreachable" "caml_int64x2_low_of_int64"
    [@@noalloc] [@@builtin]

  external low_to : t -> int64# @@ portable
    = "caml_vec128_unreachable" "caml_int64x2_low_to_int64"
    [@@noalloc] [@@builtin]

  external dup : t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int64x2_dup"
    [@@noalloc] [@@unboxed] [@@builtin]

  external low_64_to_high_64 : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_simd_vec128_low_64_to_high_64"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] set1 a = dup (low_of a)
  let[@inline] set a b = low_64_to_high_64 (low_of a) (low_of b)

  (* ───── Casts ───── *)

  external of_float64x2 : float64x2# -> t @@ portable
    = "caml_vec128_unreachable" "caml_vec128_cast"
    [@@noalloc] [@@builtin]

  (* ───── Array ───── *)

  module Array = struct
    external unsafe_get : (int64# array[@local_opt]) @ read -> idx:int -> t
      = "%caml_unboxed_int64_array_get128u#"

    external unsafe_set : (int64# array[@local_opt]) -> idx:int -> t -> unit
      = "%caml_unboxed_int64_array_set128u#"
  end
end

module Int32x4 = struct
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

  (* ───── Min/Max ───── *)

  external min : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_min"
    [@@noalloc] [@@unboxed] [@@builtin]

  external max : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_max"
    [@@noalloc] [@@unboxed] [@@builtin]

  (* ───── Bitwise ───── *)

  external bitwise_and : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_and"
    [@@noalloc] [@@unboxed] [@@builtin]

  external bitwise_or : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_or"
    [@@noalloc] [@@unboxed] [@@builtin]

  external bitwise_xor : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_xor"
    [@@noalloc] [@@unboxed] [@@builtin]

  external bitwise_not : t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_bitwise_not"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] ( land ) x y = bitwise_and x y
  let[@inline] ( lor ) x y = bitwise_or x y
  let[@inline] ( lxor ) x y = bitwise_xor x y

  (* ───── Constants ───── *)

  external const1 : int32# -> t @@ portable
    = "caml_vec128_unreachable" "caml_int32x4_const1"
    [@@noalloc] [@@builtin]

  let[@inline] zero () = const1 #0l
  let[@inline] one () = const1 #1l

  (* ───── Lanes ───── *)

  external low_of : int32# -> t @@ portable
    = "caml_vec128_unreachable" "caml_int32x4_low_of_int32"
    [@@noalloc] [@@builtin]

  external low_to : t -> int32# @@ portable
    = "caml_vec128_unreachable" "caml_int32x4_low_to_int32"
    [@@noalloc] [@@builtin]

  external dup : t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_dup"
    [@@noalloc] [@@unboxed] [@@builtin]

  external interleave_low_32 : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_32"
    [@@noalloc] [@@unboxed] [@@builtin]

  external interleave_low_64 : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_simd_vec128_interleave_low_64"
    [@@noalloc] [@@unboxed] [@@builtin]

  external dup_lane : (int[@untagged]) -> (t[@unboxed]) -> (t[@unboxed]) @@ portable
    = "caml_vec128_unreachable" "caml_neon_int32x4_dup_lane"
    [@@noalloc] [@@builtin]

  let[@inline] set1 a = dup (low_of a)

  let[@inline] set a b c d =
    let a = low_of a in
    let b = low_of b in
    let c = low_of c in
    let d = low_of d in
    let ba = interleave_low_32 a b in
    let dc = interleave_low_32 c d in
    interleave_low_64 ba dc

  (* ───── Casts ───── *)

  external of_float32x4 : float32x4# -> t @@ portable
    = "caml_vec128_unreachable" "caml_vec128_cast"
    [@@noalloc] [@@builtin]

  (* ───── Array ───── *)

  module Array = struct
    external unsafe_get : (int32# array[@local_opt]) @ read -> idx:int -> t
      = "%caml_unboxed_int32_array_get128u#"

    external unsafe_set : (int32# array[@local_opt]) -> idx:int -> t -> unit
      = "%caml_unboxed_int32_array_set128u#"
  end
end

module Float64x2 = struct
  type t = float64x2#

  (* ───── Arithmetic ───── *)

  external add : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_add"
    [@@noalloc] [@@unboxed] [@@builtin]

  external sub : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_sub"
    [@@noalloc] [@@unboxed] [@@builtin]

  external mul : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_mul"
    [@@noalloc] [@@unboxed] [@@builtin]

  external div : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_div"
    [@@noalloc] [@@unboxed] [@@builtin]

  external sqrt : t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_sqrt"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] mul_add a b c = add (mul a b) c

  external hadd : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_hadd"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] horizontal_add x y = hadd x y

  (* ───── Min/Max ───── *)

  external min : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_min"
    [@@noalloc] [@@unboxed] [@@builtin]

  external max : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float64x2_max"
    [@@noalloc] [@@unboxed] [@@builtin]

  (* ───── Constants ───── *)

  external const1 : float# -> t @@ portable
    = "caml_vec128_unreachable" "caml_float64x2_const1"
    [@@noalloc] [@@builtin]

  let[@inline] zero () = const1 #0.
  let[@inline] one () = const1 #1.

  (* ───── Bitwise (for neg/abs) ───── *)

  external of_int64x2 : int64x2# -> t @@ portable
    = "caml_vec128_unreachable" "caml_vec128_cast"
    [@@noalloc] [@@builtin]

  let[@inline] neg x =
    Int64x2.(bitwise_xor (const1 #0x8000000000000000L) (of_float64x2 x))
    |> of_int64x2

  let[@inline] abs x =
    Int64x2.(bitwise_and (const1 #0x7fffffffffffffffL) (of_float64x2 x))
    |> of_int64x2

  (* ───── Lanes ───── *)

  external low_of : float# -> t @@ portable
    = "caml_vec128_unreachable" "caml_float64x2_low_of_float"
    [@@noalloc] [@@builtin]

  external low_to : t -> float# @@ portable
    = "caml_vec128_unreachable" "caml_float64x2_low_to_float"
    [@@noalloc] [@@builtin]

  external low_64_to_high_64 : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_simd_vec128_low_64_to_high_64"
    [@@noalloc] [@@unboxed] [@@builtin]

  external high_64_to_low_64 : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_simd_vec128_high_64_to_low_64"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] set1 a =
    let a = low_of a in
    Int64x2.dup (Int64x2.of_float64x2 a) |> of_int64x2

  let[@inline] set a b = low_64_to_high_64 (low_of a) (low_of b)
  let[@inline] extract0 x = low_to x
  let[@inline] splat x = #(low_to x, low_to (high_64_to_low_64 x x))

  (* ───── Array ───── *)

  module Array = struct
    external unsafe_get : (float# array[@local_opt]) @ read -> idx:int -> t
      = "%caml_unboxed_float_array_get128u#"

    external unsafe_set : (float# array[@local_opt]) -> idx:int -> t -> unit
      = "%caml_unboxed_float_array_set128u#"
  end
end

module Float32x4 = struct
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

  let[@inline] mul_add a b c = add (mul a b) c

  external hadd : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float32x4_hadd"
    [@@noalloc] [@@unboxed] [@@builtin]

  let[@inline] horizontal_add x y = hadd x y

  (* ───── Min/Max ───── *)

  external min : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float32x4_min"
    [@@noalloc] [@@unboxed] [@@builtin]

  external max : t -> t -> t @@ portable
    = "caml_vec128_unreachable" "caml_neon_float32x4_max"
    [@@noalloc] [@@unboxed] [@@builtin]

  (* ───── Constants ───── *)

  external const1 : float32# -> t @@ portable
    = "caml_vec128_unreachable" "caml_float32x4_const1"
    [@@noalloc] [@@builtin]

  let[@inline] zero () = const1 #0.0s
  let[@inline] one () = const1 #1.0s

  (* ───── Bitwise (for neg/abs) ───── *)

  external of_int32x4 : int32x4# -> t @@ portable
    = "caml_vec128_unreachable" "caml_vec128_cast"
    [@@noalloc] [@@builtin]

  let[@inline] neg x =
    Int32x4.(bitwise_xor (const1 #0x80000000l) (of_float32x4 x)) |> of_int32x4

  let[@inline] abs x =
    Int32x4.(bitwise_and (const1 #0x7fffffffl) (of_float32x4 x)) |> of_int32x4

  (* ───── Lanes ───── *)

  external low_of : float32# -> t @@ portable
    = "caml_vec128_unreachable" "caml_float32x4_low_of_float32"
    [@@noalloc] [@@builtin]

  external low_to : t -> float32# @@ portable
    = "caml_vec128_unreachable" "caml_float32x4_low_to_float32"
    [@@noalloc] [@@builtin]

  let[@inline] set1 a =
    let a = low_of a in
    Int32x4.dup (Int32x4.of_float32x4 a) |> of_int32x4

  let[@inline] set a b c d =
    let a = Int32x4.of_float32x4 (low_of a) in
    let b = Int32x4.of_float32x4 (low_of b) in
    let c = Int32x4.of_float32x4 (low_of c) in
    let d = Int32x4.of_float32x4 (low_of d) in
    let ba = Int32x4.interleave_low_32 a b in
    let dc = Int32x4.interleave_low_32 c d in
    Int32x4.interleave_low_64 ba dc |> of_int32x4

  let[@inline] extract0 x = low_to x

  let[@inline] splat x =
    let as_i = Int32x4.of_float32x4 x in
    let lane1 = Int32x4.dup_lane 1 as_i |> of_int32x4 |> low_to in
    let lane2 = Int32x4.dup_lane 2 as_i |> of_int32x4 |> low_to in
    let lane3 = Int32x4.dup_lane 3 as_i |> of_int32x4 |> low_to in
    #(low_to x, lane1, lane2, lane3)

  (* ───── Array ───── *)

  module Array = struct
    external unsafe_get : (float32# array[@local_opt]) @ read -> idx:int -> t
      = "%caml_unboxed_float32_array_get128u#"

    external unsafe_set : (float32# array[@local_opt]) -> idx:int -> t -> unit
      = "%caml_unboxed_float32_array_set128u#"
  end
end
