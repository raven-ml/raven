type t = int16x8#

external low_of
  :  int16#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int16x8_low_of_int16"
[@@noalloc] [@@builtin]

external low_to
  :  t
  -> int16#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int16x8_low_to_int16"
[@@noalloc] [@@builtin]

external insert
  :  idx:int64#
  -> t
  -> int16#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int16x8_insert"
[@@noalloc] [@@builtin]

external extract
  :  idx:int64#
  -> t
  -> int16#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int16x8_extract"
[@@noalloc] [@@builtin]

external of_int8x16
  :  int8x16#
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
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_add"
[@@noalloc] [@@builtin]

external const1
  :  int16#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int16x8_const1"
[@@noalloc] [@@builtin]

external add_saturating
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_add_saturating"
[@@noalloc] [@@builtin]

external add_saturating_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_add_saturating_unsigned"
[@@noalloc] [@@builtin]

external sub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_sub"
[@@noalloc] [@@builtin]

external sub_saturating
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_sub_saturating"
[@@noalloc] [@@builtin]

external sub_saturating_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_sub_saturating_unsigned"
[@@noalloc] [@@builtin]

external max
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_max"
[@@noalloc] [@@builtin]

external min
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_min"
[@@noalloc] [@@builtin]

external max_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int16x8_max_unsigned"
[@@noalloc] [@@builtin]

external min_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int16x8_min_unsigned"
[@@noalloc] [@@builtin]

external cmpeq
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_cmpeq"
[@@noalloc] [@@builtin]

external cmpgt
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_cmpgt"
[@@noalloc] [@@builtin]

external abs : t -> t @@ portable = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_abs"
[@@noalloc] [@@builtin]

external horizontal_add
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_hadd"
[@@noalloc] [@@builtin]

external sll
  :  t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_sll"
[@@noalloc] [@@builtin]

external srl
  :  t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_srl"
[@@noalloc] [@@builtin]

external sra
  :  t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_sra"
[@@noalloc] [@@builtin]

external horizontal_add_saturating
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_hadd_saturating"
[@@noalloc] [@@builtin]

external horizontal_sub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_hsub"
[@@noalloc] [@@builtin]

external horizontal_sub_saturating
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_hsub_saturating"
[@@noalloc] [@@builtin]

external mul_sign
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_mulsign"
[@@noalloc] [@@builtin]

external avg_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_avg_unsigned"
[@@noalloc] [@@builtin]

external minpos_unsigned
  :  t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int16x8_minpos_unsigned"
[@@noalloc] [@@builtin]

external mul_high
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_mul_high"
[@@noalloc] [@@builtin]

external mul_low
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_mul_low"
[@@noalloc] [@@builtin]

external mul_high_unsigned
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_mul_high_unsigned"
[@@noalloc] [@@builtin]

external mul_horizontal_add
  :  t
  -> t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int16x8_mul_hadd_int32x4"
[@@noalloc] [@@builtin]

external mul_round
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_int16x8_mul_round"
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

external shuffle_8
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_ssse3_vec128_shuffle_8"
[@@noalloc] [@@builtin]

external interleave_high_16
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_high_16"
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

external cvt_si8
  :  t
  -> t
  -> int8x16#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_cvt_int16x8_int8x16_saturating"
[@@noalloc] [@@builtin]

external cvt_su8
  :  t
  -> t
  -> int8x16#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_cvt_int16x8_int8x16_saturating_unsigned"
[@@noalloc] [@@builtin]

external cvtsx_i32
  :  t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtsx_int16x8_int32x4"
[@@noalloc] [@@builtin]

external cvtzx_i32
  :  t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtzx_int16x8_int32x4"
[@@noalloc] [@@builtin]

external cvtsx_i64
  :  t
  -> int64x2#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtsx_int16x8_int64x2"
[@@noalloc] [@@builtin]

external cvtzx_i64
  :  t
  -> int64x2#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_cvtzx_int16x8_int64x2"
[@@noalloc] [@@builtin]
