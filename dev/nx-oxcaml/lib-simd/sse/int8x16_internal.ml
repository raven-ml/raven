type t = int8x16#

external low_of
  :  int8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int8x16_low_of_int8"
[@@noalloc] [@@builtin]

external low_to
  :  t
  -> int8#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int8x16_low_to_int8"
[@@noalloc] [@@builtin]

external insert
  :  idx:int64#
  -> t
  -> int8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int8x16_insert"
[@@noalloc] [@@builtin]

external extract
  :  idx:int64#
  -> t
  -> int8#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int8x16_extract"
[@@noalloc] [@@builtin]

external const1
  :  int8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int8x16_const1"
[@@noalloc] [@@builtin]

external of_int16x8
  :  int16x8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_int32x4
  :  int32x4#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_int64x2
  :  int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_float16x8
  :  float16x8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_float32x4
  :  float32x4#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_float64x2
  :  float64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external add
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_add"
[@@noalloc] [@@builtin]

external add_saturating
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_add_saturating"
[@@noalloc] [@@builtin]

external add_saturating_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_add_saturating_unsigned"
[@@noalloc] [@@builtin]

external sub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_sub"
[@@noalloc] [@@builtin]

external sub_saturating
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_sub_saturating"
[@@noalloc] [@@builtin]

external sub_saturating_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_sub_saturating_unsigned"
[@@noalloc] [@@builtin]

external mul_horizontal_add_saturating
  :  t
  -> t
  -> int16x8#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int8x16_mul_unsigned_hadd_saturating_int16x8"
[@@noalloc] [@@builtin]

external max
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int8x16_max"
[@@noalloc] [@@builtin]

external min
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int8x16_min"
[@@noalloc] [@@builtin]

external max_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_max_unsigned"
[@@noalloc] [@@builtin]

external min_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_min_unsigned"
[@@noalloc] [@@builtin]

external cmpeq
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_cmpeq"
[@@noalloc] [@@builtin]

external cmpgt
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_cmpgt"
[@@noalloc] [@@builtin]

external and_
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_and"
[@@noalloc] [@@builtin]

external andnot
  :  not:t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_andnot"
[@@noalloc] [@@builtin]

external or_ : t -> t -> t @@ portable = "ocaml_simd_sse_unreachable" "caml_sse_vec128_or"
[@@noalloc] [@@builtin]

external xor
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_xor"
[@@noalloc] [@@builtin]

external movemask_8
  :  t
  -> int64#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_movemask_8"
[@@noalloc] [@@builtin]

external interleave_high_8
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_high_8"
[@@noalloc] [@@builtin]

external interleave_low_8
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_low_8"
[@@noalloc] [@@builtin]

external interleave_low_16
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_low_16"
[@@noalloc] [@@builtin]

external interleave_low_32
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_interleave_low_32"
[@@noalloc] [@@builtin]

external interleave_low_64
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_low_64"
[@@noalloc] [@@builtin]

external abs : t -> t @@ portable = "ocaml_simd_sse_unreachable" "caml_ssse3_int8x16_abs"
[@@noalloc] [@@builtin]

external mul_sign
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int8x16_mulsign"
[@@noalloc] [@@builtin]

external avg_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_avg_unsigned"
[@@noalloc] [@@builtin]

external sadu
  :  t
  -> t
  -> int64x2#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int8x16_sad_unsigned"
[@@noalloc] [@@builtin]

external shuffle_8
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_vec128_shuffle_8"
[@@noalloc] [@@builtin]

external blendv_8
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_vec128_blendv_8"
[@@noalloc] [@@builtin]

external cvtsx_i16
  :  t
  -> int16x8#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtsx_int8x16_int16x8"
[@@noalloc] [@@builtin]

external cvtzx_i16
  :  t
  -> int16x8#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtzx_int8x16_int16x8"
[@@noalloc] [@@builtin]

external cvtsx_i32
  :  t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtsx_int8x16_int32x4"
[@@noalloc] [@@builtin]

external cvtzx_i32
  :  t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtzx_int8x16_int32x4"
[@@noalloc] [@@builtin]

external cvtsx_i64
  :  t
  -> int64x2#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtsx_int8x16_int64x2"
[@@noalloc] [@@builtin]

external cvtzx_i64
  :  t
  -> int64x2#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtzx_int8x16_int64x2"
[@@noalloc] [@@builtin]
